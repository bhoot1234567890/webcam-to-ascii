import { mkdir, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {
  test, expect, ROOT, IMAGES, readCommit, uploadFixture, changeSettings, openApp,
} from './fixtures.js';

const DEFAULT_SETTINGS = {
  maxWidthChars: 150, maxHeightLines: 200, sigma1: 1, sigma2: 2, edgeIndexThreshold: 90,
};
const settingsKey = (patch = {}) => JSON.stringify({ ...DEFAULT_SETTINGS, ...patch });
const fixture = (id) => {
  const entry = IMAGES.fixtures.find((image) => image.id === id);
  if (!entry) throw new Error(`Missing canonical image fixture: ${id}`);
  return { ...entry, file: path.join(ROOT, 'qa/fixtures/images', entry.path) };
};

// This wraps delivery, not computation: every successful payload comes unchanged
// from the application's actual module worker. A held reply still occupies the
// app's in-flight slot, making backpressure and stale-result races observable.
async function installWorkerProbe(page) {
  await page.addInitScript(() => {
    const NativeWorker = window.Worker;
    const probe = window.__cameraQA = {
      workers: [], requests: [], computed: [], delivered: [], held: [], commits: [],
      reads: 0, frames: 0, holdNext: false, invalidNextRgba: false, maxOutstanding: 0,
    };
    const readPixels = CanvasRenderingContext2D.prototype.getImageData;
    CanvasRenderingContext2D.prototype.getImageData = function (...args) {
      probe.reads++;
      return readPixels.apply(this, args);
    };
    const frameCallback = HTMLVideoElement.prototype.requestVideoFrameCallback;
    if (frameCallback) {
      HTMLVideoElement.prototype.requestVideoFrameCallback = function (callback) {
        return frameCallback.call(this, (now, metadata) => {
          probe.frames++;
          callback(now, metadata);
        });
      };
    }
    window.Worker = class ObservedWorker extends EventTarget {
      constructor(...args) {
        super();
        this.native = new NativeWorker(...args);
        this.onmessage = null;
        this.onerror = null;
        this.onmessageerror = null;
        this.terminated = false;
        this.index = probe.workers.length;
        probe.workers.push(this);
        this.native.addEventListener('message', (event) => {
          probe.computed.push({ worker: this.index, id: event.data.id, type: event.data.type,
            processingMs: event.data.processingMs });
          const delivery = { owner: this, data: event.data };
          if (probe.holdNext && event.data.type === 'result') {
            probe.holdNext = false;
            probe.held.push(delivery);
          } else this.deliver(delivery);
        });
        this.native.addEventListener('error', (event) => {
          const forwarded = new ErrorEvent('error', { message: event.message,
            filename: event.filename, lineno: event.lineno, colno: event.colno,
            error: event.error, cancelable: true });
          this.dispatchEvent(forwarded);
          this.onerror?.call(this, forwarded);
          if (forwarded.defaultPrevented) event.preventDefault();
        });
        this.native.addEventListener('messageerror', () => {
          const event = new MessageEvent('messageerror');
          this.dispatchEvent(event);
          this.onmessageerror?.call(this, event);
        });
      }
      postMessage(message, transfer) {
        const record = { worker: this.index, id: message.id, width: message.width,
          height: message.height, settings: { ...message.settings },
          bytes: message.rgba?.byteLength, transferred: transfer?.includes(message.rgba) === true };
        probe.requests.push(record);
        const outstanding = probe.requests.filter((request) => request.worker === this.index
          && !probe.delivered.some((reply) => reply.worker === this.index && reply.id === request.id)).length;
        probe.maxOutstanding = Math.max(probe.maxOutstanding, outstanding);
        if (probe.invalidNextRgba) {
          // Error-only adapter: production renderer validates this invalid request
          // and produces its own real error reply. Never invent a result payload.
          probe.invalidNextRgba = false;
          const rgba = new ArrayBuffer(0);
          this.native.postMessage({ ...message, rgba }, [rgba]);
        } else this.native.postMessage(message, transfer);
      }
      deliver(delivery) {
        probe.delivered.push({ worker: this.index, id: delivery.data.id, type: delivery.data.type });
        const event = new MessageEvent('message', { data: delivery.data });
        this.dispatchEvent(event);
        this.onmessage?.call(this, event);
      }
      terminate() {
        this.terminated = true;
        this.native.terminate();
      }
    };
    probe.release = () => {
      const delivery = probe.held.shift();
      if (!delivery) throw new Error('No real computed worker reply is being held');
      delivery.owner.deliver(delivery);
      return delivery.data.id;
    };
    document.addEventListener('DOMContentLoaded', () => {
      const output = document.getElementById('ascii-output');
      if (!output) return;
      new MutationObserver((records) => {
        if (!records.some((record) => record.attributeName === 'data-render-id')) return;
        const label = document.getElementById('performance-label');
        probe.commits.push({ id: Number(output.dataset.renderId),
          generation: Number(output.dataset.sourceGeneration), kind: output.dataset.sourceKind,
          name: output.dataset.sourceName, settingsKey: output.dataset.settingsKey,
          processingMs: Number(label.dataset.processingMs), time: performance.now() });
      }).observe(output, { attributes: true, attributeFilter: ['data-render-id'] });
    });
  });
}

async function renderingTurns(page, milliseconds = 250) {
  await page.evaluate(async (delay) => {
    await new Promise((resolve) => setTimeout(resolve, delay));
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  }, milliseconds);
}

async function retainCamera(page) {
  return page.evaluate(() => {
    const video = document.getElementById('camera-preview');
    if (!(video.srcObject instanceof MediaStream)) throw new Error('No actual camera stream');
    const stream = video.srcObject;
    const tracks = stream.getTracks();
    if (!tracks.length || tracks.some((track) => track.readyState !== 'live')) {
      throw new Error('Camera tracks must be live before retaining them');
    }
    window.__retainedCameras ??= [];
    const index = window.__retainedCameras.push({ stream, tracks }) - 1;
    return { index, width: video.videoWidth, height: video.videoHeight,
      deviceId: stream.getVideoTracks()[0].getSettings().deviceId };
  });
}

async function expectRetainedEnded(page, index, expectDetached = true) {
  await expect.poll(() => page.evaluate((i) => window.__retainedCameras[i].tracks
    .map((track) => track.readyState), index)).toEqual(['ended']);
  if (expectDetached) {
    expect(await page.locator('#camera-preview').evaluate((video) => video.srcObject === null)).toBe(true);
  }
}

async function waitCameraCommit(page, previous = null) {
  await expect.poll(async () => {
    const commit = await readCommit(page);
    return commit.sourceKind === 'camera' && commit.renderId > (previous?.renderId ?? 0)
      && commit.sourceGeneration > (previous?.sourceGeneration ?? 0);
  }).toBe(true);
  const commit = await readCommit(page);
  await expect(page.locator('#status')).toHaveText('Camera live');
  expect(commit.text.split('\n')).toHaveLength(commit.rows);
  expect(commit.text.split('\n').every((row) => row.length === commit.columns)).toBe(true);
  return commit;
}

async function startCamera(page, previous = null) {
  await page.locator('#start-camera').click();
  return waitCameraCommit(page, previous);
}

async function holdRealReply(page) {
  await page.evaluate(() => { window.__cameraQA.holdNext = true; });
  await expect.poll(() => page.evaluate(() => window.__cameraQA.held.length)).toBe(1);
  return page.evaluate(() => {
    const probe = window.__cameraQA;
    const held = probe.held[0];
    return { id: held.data.id, text: held.data.text, reads: probe.reads,
      requests: probe.requests.length, frames: probe.frames, commits: probe.commits.length };
  });
}

async function releaseRealReply(page, id) {
  expect(await page.evaluate(() => window.__cameraQA.release())).toBe(id);
  await expect.poll(() => page.evaluate((heldId) => window.__cameraQA.delivered
    .some((reply) => reply.id === heldId), id)).toBe(true);
  await renderingTurns(page, 0);
}

test.beforeEach(async ({ page }) => {
  await installWorkerProbe(page);
});

test('real camera commits, Stop ends retained tracks, and restart/settings use fresh identities', async ({ page }) => {
  await openApp(page);
  const initial = await startCamera(page);
  const retained = await retainCamera(page);
  expect([retained.width, retained.height]).toEqual([320, 240]);
  expect([initial.columns, initial.rows]).toEqual([300, 112]);
  expect(initial.settingsKey).toBe(settingsKey());
  expect(initial.sourceName).toBe(retained.deviceId);
  await expect(page.locator('#camera-preview')).toBeVisible();
  await page.locator('#stop-camera').click();
  const stopped = await readCommit(page);
  await expectRetainedEnded(page, retained.index);
  await expect(page.locator('#status')).toHaveText('Camera stopped');
  await expect(page.locator('#start-camera')).toBeEnabled();
  await expect(page.locator('#stop-camera')).toBeDisabled();
  await renderingTurns(page);
  expect(await readCommit(page)).toEqual(stopped);
  const restarted = await startCamera(page, stopped);
  const current = await retainCamera(page);
  expect(restarted.sourceGeneration).toBeGreaterThan(initial.sourceGeneration);
  const changed = await changeSettings(page, { maxWidthChars: 60, maxHeightLines: 80, sigma1: 0.5, sigma2: 1.5 });
  expect(changed.sourceGeneration).toBe(restarted.sourceGeneration);
  expect(changed.sourceName).toBe(restarted.sourceName);
  expect(changed.settingsKey).toBe(settingsKey({ maxWidthChars: 60, maxHeightLines: 80, sigma1: 0.5, sigma2: 1.5 }));
  expect([changed.columns, changed.rows]).toEqual([120, 45]);
  await page.locator('#stop-camera').click();
  await expectRetainedEnded(page, current.index);
});

test('uploading a real image releases the live camera and commits the new source', async ({ page }) => {
  await openApp(page);
  const camera = await startCamera(page);
  const retained = await retainCamera(page);
  const image = await uploadFixture(page, 'astronaut-portrait');
  expect(image.renderId).toBeGreaterThan(camera.renderId);
  expect(image.sourceGeneration).toBeGreaterThan(camera.sourceGeneration);
  expect(image.sourceKind).toBe('image');
  await expectRetainedEnded(page, retained.index);
  await expect(page.locator('#camera-preview')).toBeHidden();
  await expect(page.locator('#image-preview')).toBeVisible();
  await renderingTurns(page);
  expect(await readCommit(page)).toEqual(image);
});

test('selecting the enumerated real device replaces the old stream before committing a new generation', async ({ page }) => {
  await openApp(page);
  const first = await startCamera(page);
  const retained = await retainCamera(page);
  await expect.poll(() => page.locator('#camera-select option').count()).toBeGreaterThan(0);
  await expect(page.locator('#camera-select')).toHaveValue(retained.deviceId);
  // Selecting an explicit ID replaces the automatically chosen stream, even
  // on machines with only one fake device.
  await page.locator('#camera-select').dispatchEvent('change');
  const switched = await waitCameraCommit(page, first);
  await expectRetainedEnded(page, retained.index, false);
  expect(await page.locator('#camera-preview').evaluate((video) =>
    video.srcObject !== window.__retainedCameras[0].stream)).toBe(true);
  expect(switched.sourceName).toBe(retained.deviceId);
  const replacement = await retainCamera(page);
  await page.locator('#stop-camera').click();
  await expectRetainedEnded(page, replacement.index);
});

for (const layout of ['studio', 'camera']) {
  test(`${layout} pending permission rejects duplicate starts and ends a stream resolved after cancellation`, async ({ page }) => {
    await page.addInitScript(() => {
      const media = navigator.mediaDevices;
      const real = media.getUserMedia.bind(media);
      window.__pendingCamera = { calls: 0, stream: null, tracks: [], release: null };
      media.getUserMedia = async (constraints) => {
        const probe = window.__pendingCamera;
        probe.calls++;
        const stream = await real(constraints);
        probe.stream = stream;
        probe.tracks = stream.getTracks();
        return new Promise((resolve) => { probe.release = () => resolve(stream); });
      };
    });
    await openApp(page, { layout });
    const startSelector = layout === 'camera' ? '#camera-shutter' : '#start-camera';
    await page.locator(startSelector).click();
    await expect.poll(() => page.evaluate(() => !!window.__pendingCamera.release)).toBe(true);
    await expect(page.locator('#start-camera')).toBeDisabled();
    await expect(page.locator('#camera-select')).toBeDisabled();
    await page.locator('#start-camera').evaluate((button) => { button.click(); button.click(); });
    expect(await page.evaluate(() => window.__pendingCamera.calls)).toBe(1);
    expect(await page.evaluate(() => window.__pendingCamera.tracks.map((track) => track.readyState))).toEqual(['live']);
    await page.locator(layout === 'camera' ? '#camera-shutter' : '#stop-camera').click();
    await page.evaluate(() => window.__pendingCamera.release());
    await expect.poll(() => page.evaluate(() => window.__pendingCamera.tracks
      .map((track) => track.readyState))).toEqual(['ended']);
    expect(await page.locator('#camera-preview').evaluate((video) => video.srcObject === null)).toBe(true);
    await renderingTurns(page);
    expect((await readCommit(page)).renderId || 0).toBe(0);
    await expect(page.locator('#status')).toHaveText('Camera stopped');
    await expect(page.locator('#error')).toBeHidden();
    await expect(page.locator('#start-camera')).toBeEnabled();
    // Remove only this lifecycle adapter; recovery uses real acquisition again.
    await page.evaluate(() => {
      navigator.mediaDevices.getUserMedia = Object.getPrototypeOf(navigator.mediaDevices)
        .getUserMedia.bind(navigator.mediaDevices);
    });
    await page.locator(startSelector).click();
    await waitCameraCommit(page);
    const retained = await retainCamera(page);
    await page.locator(layout === 'camera' ? '#camera-stop' : '#stop-camera').click();
    await expectRetainedEnded(page, retained.index);
  });
}

test('an image accepted during pending real permission remains committed after the late stream resolves', async ({ page }) => {
  await page.addInitScript(() => {
    const media = navigator.mediaDevices;
    const real = media.getUserMedia.bind(media);
    window.__pendingSource = { tracks: [], release: null };
    media.getUserMedia = async (constraints) => {
      const stream = await real(constraints);
      window.__pendingSource.tracks = stream.getTracks();
      return new Promise((resolve) => { window.__pendingSource.release = () => resolve(stream); });
    };
  });
  await openApp(page);
  await page.locator('#start-camera').click();
  await expect.poll(() => page.evaluate(() => !!window.__pendingSource.release)).toBe(true);
  expect(await page.evaluate(() => window.__pendingSource.tracks.map((track) => track.readyState))).toEqual(['live']);
  const accepted = await uploadFixture(page, 'astronaut-portrait');
  await page.evaluate(() => window.__pendingSource.release());
  await expect.poll(() => page.evaluate(() => window.__pendingSource.tracks
    .map((track) => track.readyState))).toEqual(['ended']);
  expect(await page.locator('#camera-preview').evaluate((video) => video.srcObject === null)).toBe(true);
  await renderingTurns(page);
  expect(await readCommit(page)).toEqual(accepted);
  await expect(page.locator('#status')).toHaveText('Image ready');
  await expect(page.locator('#error')).toBeHidden();
});

test('page teardown disposes the real app camera and worker while retained tracks remain inspectable', async ({ page }) => {
  await openApp(page);
  await startCamera(page);
  const retained = await retainCamera(page);
  await page.evaluate(() => window.dispatchEvent(new PageTransitionEvent('pagehide')));
  const disposed = await readCommit(page);
  await expectRetainedEnded(page, retained.index);
  expect(await page.evaluate(() => window.__cameraQA.workers.every((worker) => worker.terminated))).toBe(true);
  await expect(page.locator('#start-camera')).toBeDisabled();
  await renderingTurns(page);
  expect(await readCommit(page)).toEqual(disposed);
});

test('backgrounding ends capture and rejects a late worker result until an explicit restart', async ({ page }) => {
  await openApp(page);
  const initial = await startCamera(page);
  const retained = await retainCamera(page);
  const held = await holdRealReply(page);
  await page.evaluate(() => {
    Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'hidden' });
    document.dispatchEvent(new Event('visibilitychange'));
  });
  const paused = await readCommit(page);
  await expectRetainedEnded(page, retained.index);
  await expect(page.locator('#status')).toContainText('Camera paused');
  await releaseRealReply(page, held.id);
  expect(await readCommit(page)).toEqual(paused);
  await page.evaluate(() => {
    Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'visible' });
    document.dispatchEvent(new Event('visibilitychange'));
  });
  await renderingTurns(page);
  expect(await readCommit(page)).toEqual(paused);
  await expect(page.locator('#start-camera')).toBeEnabled();
  const restarted = await startCamera(page, initial);
  expect(restarted.sourceGeneration).toBeGreaterThan(initial.sourceGeneration);
  const replacement = await retainCamera(page);
  await page.locator('#stop-camera').click();
  await expectRetainedEnded(page, replacement.index);
});

test('a cached page return keeps Start usable but never retains or silently restarts camera capture', async ({ page }) => {
  await openApp(page);
  const initial = await startCamera(page);
  const retained = await retainCamera(page);
  await page.evaluate(() => {
    window.dispatchEvent(new Event('beforeunload'));
    window.dispatchEvent(new PageTransitionEvent('pagehide', { persisted: true }));
  });
  const stopped = await readCommit(page);
  await expectRetainedEnded(page, retained.index);
  await page.evaluate(() => window.dispatchEvent(new PageTransitionEvent('pageshow', { persisted: true })));
  await renderingTurns(page);
  expect(await readCommit(page)).toEqual(stopped);
  await expect(page.locator('#start-camera')).toBeEnabled();
  await startCamera(page, initial);
  const replacement = await retainCamera(page);
  await page.locator('#stop-camera').click();
  await expectRetainedEnded(page, replacement.index);
});

test.describe('document camera policy denial', () => {
  test.use({ allowCameraPolicyError: true });
  test('real Permissions-Policy denial is recoverable and never attaches a stream', async ({ page, context, policyDiagnostics }) => {
    // A fulfilled document has no server IP in Chromium's address-space model.
    // Permit its same-origin dev HMR connection, not camera access or any
    // external traffic; the shared privacy observer remains unchanged.
    await context.grantPermissions(['local-network-access'], { origin: 'http://127.0.0.1:4173' });
    await page.route('**/', async (route) => {
      if (!route.request().isNavigationRequest()) return route.continue();
      const response = await route.fetch();
      await route.fulfill({ response, headers: { ...response.headers(), 'permissions-policy': 'camera=()' } });
    });
    await page.addInitScript(() => {
      const real = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
      window.__policyRejections = [];
      navigator.mediaDevices.getUserMedia = async (constraints) => {
        try { return await real(constraints); }
        catch (error) { window.__policyRejections.push(error.name); throw error; }
      };
    });
    await openApp(page);
    await page.locator('#start-camera').click();
    await expect(page.locator('#error')).toHaveText('Camera access was denied');
    expect(await page.evaluate(() => window.__policyRejections)).toEqual(['NotAllowedError']);
    expect(await page.locator('#camera-preview').evaluate((video) => video.srcObject === null)).toBe(true);
    await expect(page.locator('#start-camera')).toBeEnabled();
    expect((await readCommit(page)).renderId || 0).toBe(0);
    expect(policyDiagnostics.length).toBeGreaterThan(0);
    const image = await uploadFixture(page, 'black');
    expect(image.sourceKind).toBe('image');
    await expect(page.locator('#error')).toBeHidden();
  });
});

for (const [name, message] of [
  ['NotFoundError', 'No camera found'],
  ['NotReadableError', 'Camera is unavailable'],
  ['SecurityError', 'Unable to start camera'],
]) {
  test(`${name} shows its camera error without losing a valid image and a real retry succeeds`, async ({ page }) => {
    await page.addInitScript((errorName) => {
      const media = navigator.mediaDevices;
      const real = media.getUserMedia.bind(media);
      let rejectNext = true;
      media.getUserMedia = (constraints) => {
        if (rejectNext) {
          rejectNext = false;
          return Promise.reject(new DOMException('Scoped acquisition failure', errorName));
        }
        return real(constraints);
      };
    }, name);
    await openApp(page);
    const original = await uploadFixture(page, 'midgray');
    await page.locator('#start-camera').click();
    await expect(page.locator('#error')).toHaveText(message);
    await expect(page.locator('#start-camera')).toBeEnabled();
    expect(await readCommit(page)).toEqual(original);
    expect(await page.locator('#camera-preview').evaluate((video) => video.srcObject === null)).toBe(true);
    await startCamera(page, original);
    await expect(page.locator('#error')).toBeHidden();
    const retained = await retainCamera(page);
    await page.locator('#stop-camera').click();
    await expectRetainedEnded(page, retained.index);
  });
}

for (const boundary of ['missing-mediaDevices', 'insecure-context']) {
  test(`${boundary} displays the secure-context prerequisite`, async ({ page }) => {
    await page.addInitScript((mode) => {
      if (mode === 'missing-mediaDevices') {
        Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: undefined });
      } else Object.defineProperty(window, 'isSecureContext', { configurable: true, value: false });
    }, boundary);
    await openApp(page);
    await page.locator('#start-camera').click();
    await expect(page.locator('#error')).toHaveText('Camera requires HTTPS or localhost');
    await expect(page.locator('#start-camera')).toBeEnabled();
    expect((await readCommit(page)).renderId || 0).toBe(0);
    const image = await uploadFixture(page, 'white');
    expect(image.sourceKind).toBe('image');
  });
}

test('an unchanged real worker reply delivered after Stop cannot commit', async ({ page }) => {
  await openApp(page);
  await startCamera(page);
  const retained = await retainCamera(page);
  const held = await holdRealReply(page);
  const beforeStop = await readCommit(page);
  expect(held.id).toBeGreaterThan(beforeStop.renderId);
  await page.locator('#stop-camera').click({ timeout: 2000 });
  await expectRetainedEnded(page, retained.index);
  await releaseRealReply(page, held.id);
  await renderingTurns(page);
  expect(await readCommit(page)).toEqual(beforeStop);
  expect(await page.evaluate((id) => window.__cameraQA.commits.some((commit) => commit.id === id), held.id)).toBe(false);
});

test('a source switch rejects an unchanged late real camera reply before the image commits', async ({ page }) => {
  await openApp(page);
  const camera = await startCamera(page);
  const retained = await retainCamera(page);
  const held = await holdRealReply(page);
  const image = fixture('white');
  await page.locator('#image-input').setInputFiles(image.file);
  await expect(page.locator('#image-preview')).toHaveAttribute('alt', `Original image: ${image.path}`);
  await expectRetainedEnded(page, retained.index);
  expect(await page.evaluate(() => window.__cameraQA.requests.length)).toBe(held.requests);
  await releaseRealReply(page, held.id);
  await expect.poll(async () => {
    const commit = await readCommit(page);
    return commit.renderId > held.id && commit.sourceKind === 'image' && commit.sourceName === image.path;
  }).toBe(true);
  const result = await readCommit(page);
  expect(result.sourceGeneration).toBeGreaterThan(camera.sourceGeneration);
  expect(result.settingsKey).toBe(settingsKey());
  expect(result.text.replaceAll('\n', '')).toMatch(/^@+$/);
  expect(await page.evaluate((id) => window.__cameraQA.commits.some((commit) => commit.id === id), held.id)).toBe(false);
});

test('backpressure copies no dropped frames and source/settings churn retains only the latest still', async ({ page }) => {
  await openApp(page);
  const camera = await startCamera(page);
  const retained = await retainCamera(page);
  const held = await holdRealReply(page);
  await expect.poll(() => page.evaluate(() => window.__cameraQA.frames)).toBeGreaterThan(held.frames + 2);
  expect(await page.evaluate(() => window.__cameraQA.reads)).toBe(held.reads);
  expect(await page.evaluate(() => window.__cameraQA.requests.length)).toBe(held.requests);
  for (const id of ['black', 'white', 'astronaut-portrait']) {
    const image = fixture(id);
    await page.locator('#image-input').setInputFiles(image.file);
    await expect(page.locator('#image-preview')).toHaveAttribute('alt', `Original image: ${image.path}`);
    await expect(page.locator('#status')).toHaveText('Rendering image…');
  }
  await expectRetainedEnded(page, retained.index);
  for (const width of [70, 90, 60]) {
    await page.locator('#max-width').fill(String(width));
    await page.locator('#max-width').dispatchEvent('change');
  }
  await page.locator('#sigma1').fill('0.5');
  await page.locator('#sigma1').dispatchEvent('change');
  expect(await page.evaluate(() => window.__cameraQA.reads)).toBe(held.reads);
  expect(await page.evaluate(() => window.__cameraQA.requests.length)).toBe(held.requests);
  await releaseRealReply(page, held.id);
  const expectedKey = settingsKey({ maxWidthChars: 60, sigma1: 0.5 });
  await expect.poll(async () => {
    const commit = await readCommit(page);
    return commit.renderId > held.id && commit.sourceKind === 'image'
      && commit.sourceName === fixture('astronaut-portrait').path && commit.settingsKey === expectedKey;
  }).toBe(true);
  const latest = await readCommit(page);
  expect(latest.sourceGeneration).toBe(camera.sourceGeneration + 3);
  await renderingTurns(page);
  const probe = await page.evaluate(() => ({ reads: window.__cameraQA.reads,
    requests: window.__cameraQA.requests, maxOutstanding: window.__cameraQA.maxOutstanding,
    commits: window.__cameraQA.commits }));
  expect(probe.reads).toBe(held.reads + 1);
  expect(probe.requests).toHaveLength(held.requests + 1);
  expect(probe.maxOutstanding).toBe(1);
  expect(probe.requests.every((request) => request.transferred && request.bytes === request.width * request.height * 4)).toBe(true);
  expect(probe.requests.at(-1).settings).toEqual({ ...DEFAULT_SETTINGS, maxWidthChars: 60, sigma1: 0.5 });
  expect(probe.commits.some((commit) => commit.id === held.id)).toBe(false);
  expect(await readCommit(page)).toEqual(latest);
});

for (const recovery of ['camera', 'image']) {
  test(`real worker validation failure stops capture and only an explicit ${recovery} action recovers`, async ({ page }) => {
    await openApp(page);
    await startCamera(page);
    const retained = await retainCamera(page);
    await page.evaluate(() => { window.__cameraQA.invalidNextRgba = true; });
    await expect(page.locator('#error')).toHaveText('Rendering failed');
    await expect(page.locator('#status')).toHaveText('Rendering failed');
    await expectRetainedEnded(page, retained.index);
    const failed = await readCommit(page);
    const workers = await page.evaluate(() => window.__cameraQA.workers.length);
    expect(await page.evaluate(() => window.__cameraQA.computed.some((reply) => reply.type === 'error'))).toBe(true);
    expect(await page.evaluate(() => window.__cameraQA.workers.at(-1).terminated)).toBe(true);
    await page.locator('#max-width').fill('60');
    await page.locator('#max-width').dispatchEvent('change');
    await renderingTurns(page);
    expect(await page.evaluate(() => window.__cameraQA.workers.length)).toBe(workers);
    expect(await readCommit(page)).toEqual(failed);
    await expect(page.locator('#error')).toHaveText('Rendering failed');
    const recovered = recovery === 'camera' ? await startCamera(page, failed) : await uploadFixture(page, 'astronaut');
    expect(recovered.renderId).toBeGreaterThan(failed.renderId);
    expect(recovered.settingsKey).toBe(settingsKey({ maxWidthChars: 60 }));
    expect(await page.evaluate(() => window.__cameraQA.workers.length)).toBe(workers + 1);
    await expect(page.locator('#error')).toBeHidden();
    if (recovery === 'camera') {
      const replacement = await retainCamera(page);
      await page.locator('#stop-camera').click();
      await expectRetainedEnded(page, replacement.index);
    }
  });
}

test('five-second real camera run records warm worker latency and delivered frame rate', async ({ page, browser, browserName }, testInfo) => {
  test.setTimeout(30000);
  await openApp(page);
  const initial = await startCamera(page);
  const retained = await retainCamera(page);
  await expect.poll(() => page.evaluate(() => window.__cameraQA.commits.filter((commit) => commit.kind === 'camera').length)).toBeGreaterThanOrEqual(6);
  const measured = await page.evaluate(async () => {
    const probe = window.__cameraQA;
    const startIndex = probe.commits.length;
    const start = performance.now();
    await new Promise((resolve) => setTimeout(resolve, 5000));
    const end = performance.now();
    return { start, end, durationMs: end - start, commits: probe.commits.slice(startIndex),
      reads: probe.reads, requests: probe.requests.length, maxOutstanding: probe.maxOutstanding,
      userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency };
  });
  expect(measured.durationMs).toBeGreaterThanOrEqual(5000);
  expect(measured.commits.length).toBeGreaterThan(0);
  expect(measured.commits.every((commit) => commit.kind === 'camera'
    && commit.settingsKey === initial.settingsKey && Number.isFinite(commit.processingMs) && commit.processingMs >= 0)).toBe(true);
  expect(measured.maxOutstanding).toBe(1);
  expect(measured.reads).toBe(measured.requests);
  const latencies = measured.commits.map((commit) => commit.processingMs).sort((a, b) => a - b);
  const percentile = (fraction) => latencies[Math.ceil(fraction * latencies.length) - 1];
  const changed = await changeSettings(page, { edgeIndexThreshold: 95 });
  expect(changed.sourceGeneration).toBe(initial.sourceGeneration);
  const stopStart = await page.evaluate(() => performance.now());
  await page.locator('#stop-camera').click({ timeout: 2000 });
  await expect(page.locator('#status')).toHaveText('Camera stopped', { timeout: 2000 });
  await expectRetainedEnded(page, retained.index);
  const stopResponseMs = await page.evaluate((start) => performance.now() - start, stopStart);
  const stopped = await readCommit(page);
  await renderingTurns(page);
  expect(await readCommit(page)).toEqual(stopped);
  const report = {
    schemaVersion: 1, measuredAt: new Date().toISOString(), project: testInfo.project.name,
    browser: { name: browserName, version: browser.version(), userAgent: measured.userAgent },
    machine: { platform: os.platform(), release: os.release(), architecture: os.arch(),
      cpu: os.cpus()[0]?.model, logicalCpuCount: os.cpus().length,
      browserHardwareConcurrency: measured.hardwareConcurrency, totalMemoryBytes: os.totalmem() },
    source: { kind: 'real-getUserMedia-fake-device', width: retained.width, height: retained.height },
    settings: DEFAULT_SETTINGS, columns: initial.columns, rows: initial.rows,
    warmup: { minimumExcludedCommits: 6 }, durationMs: measured.durationMs,
    deliveredFrames: measured.commits.length, deliveredFramesPerSecond: measured.commits.length * 1000 / measured.durationMs,
    processingMs: { p50: percentile(0.5), p95: percentile(0.95), samples: latencies },
    maxOutstandingRenderRequests: measured.maxOutstanding, stopResponseMs,
    physicalWebcam: { status: 'not-run', reason: 'This project uses the deterministic fake device, not physical camera hardware.' },
  };
  const destination = path.join(ROOT, 'qa/artifacts/performance.json');
  await mkdir(path.dirname(destination), { recursive: true });
  await writeFile(destination, `${JSON.stringify(report, null, 2)}\n`);
  await testInfo.attach('camera-performance', { path: destination, contentType: 'application/json' });
});

test('camera shutter freezes a real frame, rejects late replies, retunes the shot and returns to live capture', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 428, height: 746 });
  await openApp(page, { layout: 'camera' });
  await page.locator('#camera-shutter').click();
  const initial = await waitCameraCommit(page);
  const retained = await retainCamera(page);
  const directory = path.join(ROOT, 'qa/artifacts/camera-ui', testInfo.project.name);
  await mkdir(directory, { recursive: true });
  await page.screenshot({ path: path.join(directory, 'live-camera.png') });
  const held = await holdRealReply(page);
  await page.locator('#camera-shutter').click();
  await expectRetainedEnded(page, retained.index);
  await expect(page.locator('#source-label')).toHaveText('Captured frame.png');
  const beforeRelease = await readCommit(page);
  await releaseRealReply(page, held.id);
  await page.waitForFunction((previous) => {
    const data = document.querySelector('#ascii-output').dataset;
    return data.sourceKind === 'image' && data.sourceName === 'Captured frame.png' &&
      Number(data.sourceGeneration) > previous.sourceGeneration && Number(data.renderId) > previous.renderId;
  }, initial);
  const shot = await readCommit(page);
  expect(shot.renderId).toBeGreaterThan(beforeRelease.renderId);
  await expect(page.locator('#status')).toHaveText('Frame captured');
  await renderingTurns(page);
  expect(await readCommit(page)).toEqual(shot);
  await page.screenshot({ path: path.join(directory, 'captured-frame.png') });
  await page.locator('#open-studio').click();
  const tuned = await changeSettings(page, { maxWidthChars: 60, maxHeightLines: 80 });
  expect(tuned.sourceGeneration).toBe(shot.sourceGeneration);
  expect([tuned.columns, tuned.rows]).toEqual([120, 45]);
  await page.locator('#open-camera').click();
  expect(await readCommit(page)).toEqual(tuned);
  await page.locator('#live-mode').click();
  const resumed = await waitCameraCommit(page, tuned);
  const resumeTracks = await retainCamera(page);
  await page.locator('#flip-camera').click();
  await waitCameraCommit(page, resumed);
  await expectRetainedEnded(page, resumeTracks.index, false);
  const flipped = await retainCamera(page);
  await page.locator('#camera-stop').click();
  await expectRetainedEnded(page, flipped.index);
  const stopped = await readCommit(page);
  await renderingTurns(page);
  expect(await readCommit(page)).toEqual(stopped);
});
