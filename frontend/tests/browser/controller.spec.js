import { test, expect } from './fixtures.js';

// A same-origin, test-only document imports the production controller directly.
// There is no app entry point, substitute controller, or successful render stub.
async function openController(page, { fallback = false, deferred = false } = {}) {
  await page.route('**/__qa/controller', (route) => route.fulfill({
    contentType: 'text/html',
    body: '<!doctype html><html><head><link rel="icon" href="data:,"></head><body><video id="video" muted playsinline></video></body></html>',
  }));
  await page.goto('/__qa/controller');
  await page.evaluate(async ({ fallback, deferred }) => {
    const { createCameraController } = await import('/src/camera.js');
    const video = document.getElementById('video');
    const probe = window.__controllerQA = {
      video, frames: [], states: [], errors: [], requests: [], retained: [],
      scheduled: 0, delivered: 0, canceled: [], pending: new Set(),
      release: null, stopOnFrame: '', deferred,
    };
    const media = navigator.mediaDevices;
    const realGetUserMedia = media.getUserMedia.bind(media);
    media.getUserMedia = async (constraints) => {
      probe.requests.push({
        precedingTrackStates: probe.retained.flatMap((entry) => entry.tracks.map((track) => track.readyState)) });
      const stream = await realGetUserMedia(constraints);
      probe.retained.push({ stream, tracks: stream.getTracks() });
      if (!probe.deferred) return stream;
      return new Promise((resolve) => { probe.release = () => resolve(stream); });
    };
    if (fallback) {
      Object.defineProperty(video, 'requestVideoFrameCallback', { value: undefined, configurable: true });
      Object.defineProperty(video, 'cancelVideoFrameCallback', { value: undefined, configurable: true });
      const request = window.requestAnimationFrame.bind(window);
      const cancel = window.cancelAnimationFrame.bind(window);
      window.requestAnimationFrame = (callback) => {
        probe.scheduled++;
        const id = request((now) => {
          probe.pending.delete(id);
          probe.delivered++;
          callback(now);
        });
        probe.pending.add(id);
        return id;
      };
      window.cancelAnimationFrame = (id) => {
        probe.canceled.push({ id, wasPending: probe.pending.delete(id), kind: 'animation' });
        cancel(id);
      };
    } else {
      const request = video.requestVideoFrameCallback.bind(video);
      const cancel = video.cancelVideoFrameCallback.bind(video);
      video.requestVideoFrameCallback = (callback) => {
        probe.scheduled++;
        const id = request((now, metadata) => {
          probe.pending.delete(id);
          probe.delivered++;
          callback(now, metadata);
        });
        probe.pending.add(id);
        return id;
      };
      video.cancelVideoFrameCallback = (id) => {
        probe.canceled.push({ id, wasPending: probe.pending.delete(id), kind: 'video' });
        cancel(id);
      };
    }
    probe.controller = createCameraController(video, {
      onFrame(frameVideo, metadata) {
        if (frameVideo !== video) throw new Error('Controller changed video identity');
        probe.frames.push({ mediaTime: metadata.mediaTime, presentedFrames: metadata.presentedFrames,
          width: metadata.width, height: metadata.height, currentTime: video.currentTime });
        if (probe.stopOnFrame) {
          const action = probe.stopOnFrame;
          probe.stopOnFrame = '';
          probe.controller[action]();
        }
      },
      onState(state) { probe.states.push(state); },
      onError(error) { probe.errors.push({ name: error.name, message: error.message }); },
    });
  }, { fallback, deferred });
}

async function startController(page, deviceId) {
  const priorFrames = await page.evaluate(() => window.__controllerQA.frames.length);
  expect(await page.evaluate((id) => window.__controllerQA.controller.start(id), deviceId)).toBe(true);
  await expect.poll(() => page.evaluate(() => window.__controllerQA.frames.length)).toBeGreaterThan(priorFrames + 1);
  const state = await page.evaluate(() => {
    const probe = window.__controllerQA;
    const stream = probe.video.srcObject;
    const tracks = stream.getTracks();
    return { muted: probe.video.muted, playsInline: probe.video.playsInline,
      width: probe.video.videoWidth, height: probe.video.videoHeight,
      deviceId: stream.getVideoTracks()[0].getSettings().deviceId,
      trackStates: tracks.map((track) => track.readyState), states: probe.states };
  });
  expect(state.muted).toBe(true);
  expect(state.playsInline).toBe(true);
  expect([state.width, state.height]).toEqual([320, 240]);
  expect(state.trackStates).toEqual(['live']);
  expect(state.states.at(-1)).toBe('live');
  return state;
}

async function controllerSnapshot(page) {
  return page.evaluate(() => {
    const probe = window.__controllerQA;
    return { frameCount: probe.frames.length, scheduled: probe.scheduled, delivered: probe.delivered,
      canceled: probe.canceled, pending: [...probe.pending], states: probe.states, errors: probe.errors,
      detached: probe.video.srcObject === null,
      tracks: probe.retained.map((entry) => entry.tracks.map((track) => track.readyState)),
      requests: probe.requests };
  });
}

async function quietInterval(page) {
  // Real timers permit several 15fps source frames and many display callbacks.
  // Helpers intentionally do not use RAF, which the fallback probe counts.
  await page.evaluate(() => new Promise((resolve) => setTimeout(resolve, 300)));
}

test('dispose cancels the real video callback, ends retained tracks, and cannot restart', async ({ page }) => {
  await openController(page);
  await startController(page);
  const disposed = await page.evaluate(() => {
    const probe = window.__controllerQA;
    const outstanding = [...probe.pending];
    probe.controller.dispose();
    probe.controller.dispose();
    return { outstanding, frameCount: probe.frames.length, scheduled: probe.scheduled };
  });
  expect(disposed.outstanding).toHaveLength(1);
  let snapshot = await controllerSnapshot(page);
  expect(snapshot.detached).toBe(true);
  expect(snapshot.tracks).toEqual([['ended']]);
  expect(snapshot.pending).toEqual([]);
  expect(snapshot.canceled).toContainEqual({ id: disposed.outstanding[0], wasPending: true, kind: 'video' });
  expect(snapshot.states.at(-1)).toBe('idle');
  await quietInterval(page);
  snapshot = await controllerSnapshot(page);
  expect(snapshot.frameCount).toBe(disposed.frameCount);
  expect(snapshot.scheduled).toBe(disposed.scheduled);
  expect(snapshot.errors).toEqual([]);
  expect(await page.evaluate(() => window.__controllerQA.controller.start())).toBe(false);
  expect(await page.evaluate(() => window.__controllerQA.controller.listDevices())).toEqual([]);
  expect((await controllerSnapshot(page)).requests).toHaveLength(1);
});

test('RAF fallback deduplicates unchanged currentTime and cancels its actual outstanding callback', async ({ page }) => {
  await openController(page, { fallback: true });
  await startController(page);
  const frozen = await page.evaluate(() => {
    const probe = window.__controllerQA;
    const mediaTime = probe.video.currentTime;
    // Freeze only the observed media clock. The acquired video/stream remains
    // real and playing; this isolates currentTime deduplication, not rendering.
    Object.defineProperty(probe.video, 'currentTime', { configurable: true, get: () => mediaTime });
    return { frameCount: probe.frames.length, delivered: probe.delivered };
  });
  await expect.poll(() => page.evaluate(() => window.__controllerQA.delivered)).toBeGreaterThan(frozen.delivered + 4);
  const steadyCount = await page.evaluate(() => window.__controllerQA.frames.length);
  expect(steadyCount - frozen.frameCount).toBeLessThanOrEqual(1);
  await quietInterval(page);
  expect(await page.evaluate(() => window.__controllerQA.frames.length)).toBe(steadyCount);
  await page.evaluate(() => { delete window.__controllerQA.video.currentTime; });
  await expect.poll(() => page.evaluate(() => window.__controllerQA.frames.length)).toBeGreaterThan(steadyCount + 1);
  const stopped = await page.evaluate(() => {
    const probe = window.__controllerQA;
    const outstanding = [...probe.pending];
    probe.controller.stop();
    return { outstanding, frames: probe.frames.length, scheduled: probe.scheduled };
  });
  expect(stopped.outstanding).toHaveLength(1);
  await quietInterval(page);
  const snapshot = await controllerSnapshot(page);
  expect(snapshot.pending).toEqual([]);
  expect(snapshot.canceled).toContainEqual({ id: stopped.outstanding[0], wasPending: true, kind: 'animation' });
  expect(snapshot.frameCount).toBe(stopped.frames);
  expect(snapshot.scheduled).toBe(stopped.scheduled);
  expect(snapshot.detached).toBe(true);
  expect(snapshot.tracks).toEqual([['ended']]);
  expect(snapshot.errors).toEqual([]);
  await page.evaluate(() => window.__controllerQA.controller.dispose());
});

for (const action of ['stop', 'dispose']) {
  test(`${action} inside onFrame prevents synchronous rescheduling`, async ({ page }) => {
    await openController(page);
    await startController(page);
    const before = await page.evaluate((method) => {
      const probe = window.__controllerQA;
      probe.stopOnFrame = method;
      return probe.frames.length;
    }, action);
    await expect.poll(() => page.evaluate(() => window.__controllerQA.states.at(-1))).toBe('idle');
    const stopped = await controllerSnapshot(page);
    expect(stopped.frameCount).toBe(before + 1);
    expect(stopped.pending).toEqual([]);
    expect(stopped.detached).toBe(true);
    expect(stopped.tracks).toEqual([['ended']]);
    await quietInterval(page);
    const after = await controllerSnapshot(page);
    expect(after.frameCount).toBe(stopped.frameCount);
    expect(after.scheduled).toBe(stopped.scheduled);
    expect(after.errors).toEqual([]);
    await page.evaluate(() => window.__controllerQA.controller.dispose());
  });
}

test('real device switching releases old tracks before exact-device acquisition and suppresses duplicate live starts', async ({ page }) => {
  await openController(page);
  const first = await startController(page);
  expect(await page.evaluate(() => window.__controllerQA.controller.start())).toBe(false);
  const devices = await page.evaluate(async () => (await window.__controllerQA.controller.listDevices())
    .map((device) => ({ kind: device.kind, deviceId: device.deviceId, label: device.label })));
  expect(devices.length).toBeGreaterThan(0);
  expect(devices.every((device) => device.kind === 'videoinput')).toBe(true);
  expect(devices.some((device) => device.deviceId === first.deviceId)).toBe(true);
  const second = await startController(page, first.deviceId);
  expect(second.deviceId).toBe(first.deviceId);
  let snapshot = await controllerSnapshot(page);
  expect(snapshot.requests).toHaveLength(2);
  expect(snapshot.requests[1].precedingTrackStates).toEqual(['ended']);
  expect(snapshot.tracks).toEqual([['ended'], ['live']]);
  expect(await page.evaluate((id) => window.__controllerQA.controller.start(id), first.deviceId)).toBe(false);
  expect((await controllerSnapshot(page)).requests).toHaveLength(2);
  await page.evaluate(() => window.__controllerQA.controller.dispose());
  snapshot = await controllerSnapshot(page);
  expect(snapshot.tracks).toEqual([['ended'], ['ended']]);
  expect(snapshot.detached).toBe(true);
  expect(snapshot.pending).toEqual([]);
});

for (const action of ['stop', 'dispose']) {
  test(`${action} during deferred real permission ends late tracks without attaching or scheduling`, async ({ page }) => {
    await openController(page, { deferred: true });
    await page.evaluate(() => {
      const probe = window.__controllerQA;
      probe.startResult = probe.controller.start();
    });
    await expect.poll(() => page.evaluate(() => !!window.__controllerQA.release)).toBe(true);
    let snapshot = await controllerSnapshot(page);
    expect(snapshot.states).toEqual(['requesting']);
    expect(snapshot.tracks).toEqual([['live']]);
    expect(snapshot.detached).toBe(true);
    expect(await page.evaluate(() => window.__controllerQA.controller.start('ignored-during-pending'))).toBe(false);
    expect((await controllerSnapshot(page)).requests).toHaveLength(1);
    expect(await page.evaluate(async (method) => {
      const probe = window.__controllerQA;
      probe.controller[method]();
      probe.release();
      return await probe.startResult;
    }, action)).toBe(false);
    snapshot = await controllerSnapshot(page);
    expect(snapshot.tracks).toEqual([['ended']]);
    expect(snapshot.detached).toBe(true);
    expect(snapshot.scheduled).toBe(0);
    expect(snapshot.frameCount).toBe(0);
    expect(snapshot.errors).toEqual([]);
    expect(snapshot.states).toEqual(['requesting', 'idle']);
    await quietInterval(page);
    expect((await controllerSnapshot(page)).frameCount).toBe(0);
    if (action === 'stop') {
      await page.evaluate(() => { window.__controllerQA.deferred = false; });
      await startController(page);
    }
    await page.evaluate(() => window.__controllerQA.controller.dispose());
    expect((await controllerSnapshot(page)).tracks.every((tracks) => tracks.every((state) => state === 'ended'))).toBe(true);
  });
}

test('track revocation cleans up callbacks and all tracks, reports an error, and permits real recovery', async ({ page }) => {
  await openController(page);
  await startController(page);
  const before = await page.evaluate(() => {
    const probe = window.__controllerQA;
    const track = probe.retained[0].tracks[0];
    const pending = [...probe.pending];
    track.stop();
    // stop() intentionally does not fire ended. Model actual device revocation
    // using the retained real track and its production ended listener.
    track.dispatchEvent(new Event('ended'));
    return { frames: probe.frames.length, scheduled: probe.scheduled, pending };
  });
  let snapshot = await controllerSnapshot(page);
  expect(snapshot.tracks).toEqual([['ended']]);
  expect(snapshot.detached).toBe(true);
  expect(snapshot.pending).toEqual([]);
  expect(snapshot.errors).toEqual([{ name: 'NotReadableError', message: 'Camera is unavailable' }]);
  expect(snapshot.states.at(-1)).toBe('idle');
  expect(snapshot.canceled).toContainEqual({ id: before.pending[0], wasPending: true, kind: 'video' });
  await quietInterval(page);
  snapshot = await controllerSnapshot(page);
  expect(snapshot.frameCount).toBe(before.frames);
  expect(snapshot.scheduled).toBe(before.scheduled);
  await startController(page);
  expect((await controllerSnapshot(page)).tracks).toEqual([['ended'], ['live']]);
  await page.evaluate(() => window.__controllerQA.controller.dispose());
  expect((await controllerSnapshot(page)).tracks).toEqual([['ended'], ['ended']]);
});
