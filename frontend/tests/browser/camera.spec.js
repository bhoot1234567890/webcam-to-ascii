import { mkdir, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {
  test, expect, ROOT, IMAGES, readCommit, uploadFixture, changeSettings, openApp, openPanel,
} from './fixtures.js';
import { installVideoCamera, downloadVideo } from './video-fixtures.js';

const DEFAULT_SETTINGS = {
  maxWidthChars: 50, maxHeightLines: 65, sigma1: 1, sigma2: 2, edgeIndexThreshold: 90,
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
          look: { ...message.look },
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
        probe.commits.push({ id: Number(output.dataset.renderId),
          generation: Number(output.dataset.sourceGeneration), kind: output.dataset.sourceKind,
          name: output.dataset.sourceName, settingsKey: output.dataset.settingsKey,
          lookKey: output.dataset.lookKey, framing: output.dataset.framing,
          processingMs: Number(output.dataset.processingMs), time: performance.now() });
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
  await page.locator('#camera-shutter').click();
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
  expect([initial.columns, initial.rows]).toEqual([100, 37]);
  expect(initial.settingsKey).toBe(settingsKey());
  expect(initial.sourceName).toBe(retained.deviceId);
  expect(await page.locator('#camera-preview').evaluate((video) => video.srcObject !== null)).toBe(true);
  await page.locator('#camera-stop').click();
  const stopped = await readCommit(page);
  await expectRetainedEnded(page, retained.index);
  await expect(page.locator('#status')).toHaveText('Camera stopped');
  await expect(page.locator('#camera-shutter')).toBeEnabled();
  await expect(page.locator('#camera-stop')).toBeDisabled();
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
  await page.locator('#camera-stop').click();
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
  await openPanel(page, 'camera');
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
  await page.locator('#camera-stop').click();
  await expectRetainedEnded(page, replacement.index);
});

test('pending permission rejects duplicate starts and ends a stream resolved after cancellation', async ({ page }) => {
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
  await openApp(page);
  await page.locator('#camera-shutter').click();
  await expect.poll(() => page.evaluate(() => !!window.__pendingCamera.release)).toBe(true);
  await expect(page.locator('#camera-shutter')).toHaveAttribute('aria-label', 'Cancel camera request');
  await expect(page.locator('#camera-select')).toBeDisabled();
  await page.locator('#camera-shutter').evaluate((button) => { button.click(); button.click(); });
  expect(await page.evaluate(() => window.__pendingCamera.calls)).toBe(1);
  expect(await page.evaluate(() => window.__pendingCamera.tracks.map((track) => track.readyState))).toEqual(['live']);
  // The double activation above already cancelled (first tap) with the replay swallowed (second tap).
  await page.evaluate(() => window.__pendingCamera.release());
  await expect.poll(() => page.evaluate(() => window.__pendingCamera.tracks
    .map((track) => track.readyState))).toEqual(['ended']);
  expect(await page.locator('#camera-preview').evaluate((video) => video.srcObject === null)).toBe(true);
  await renderingTurns(page);
  expect((await readCommit(page)).renderId || 0).toBe(0);
  await expect(page.locator('#status')).toHaveText('Camera stopped');
  await expect(page.locator('#error')).toBeHidden();
  await expect(page.locator('#camera-shutter')).toBeEnabled();
  // Remove only this lifecycle adapter; recovery uses real acquisition again.
  await page.evaluate(() => {
    navigator.mediaDevices.getUserMedia = Object.getPrototypeOf(navigator.mediaDevices)
      .getUserMedia.bind(navigator.mediaDevices);
  });
  await page.locator('#camera-shutter').click();
  await waitCameraCommit(page);
  const retained = await retainCamera(page);
  await page.locator('#camera-stop').click();
  await expectRetainedEnded(page, retained.index);
});

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
  await page.locator('#camera-shutter').click();
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
  await expect(page.locator('#camera-shutter')).toBeDisabled();
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
  await expect(page.locator('#camera-shutter')).toBeEnabled();
  const restarted = await startCamera(page, initial);
  expect(restarted.sourceGeneration).toBeGreaterThan(initial.sourceGeneration);
  const replacement = await retainCamera(page);
  await page.locator('#camera-stop').click();
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
  await expect(page.locator('#camera-shutter')).toBeEnabled();
  await startCamera(page, initial);
  const replacement = await retainCamera(page);
  await page.locator('#camera-stop').click();
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
    await page.locator('#camera-shutter').click();
    await expect(page.locator('#error-text')).toHaveText('Camera access was denied. Allow camera in your browser site settings, then press CAMERA.');
    expect(await page.evaluate(() => window.__policyRejections)).toEqual(['NotAllowedError']);
    expect(await page.locator('#camera-preview').evaluate((video) => video.srcObject === null)).toBe(true);
    await expect(page.locator('#camera-shutter')).toBeEnabled();
    expect((await readCommit(page)).renderId || 0).toBe(0);
    expect(policyDiagnostics.length).toBeGreaterThan(0);
    const recovery = fixture('black');
    const [picker] = await Promise.all([
      page.waitForEvent('filechooser'),
      page.locator('#error-action').click(),
    ]);
    await picker.setFiles(recovery.file);
    await page.waitForFunction((name) => {
      const data = document.querySelector('#ascii-output').dataset;
      return Number(data.renderId) > 0 && data.sourceKind === 'image' && data.sourceName === name;
    }, path.basename(recovery.file));
    expect((await readCommit(page)).sourceKind).toBe('image');
    await expect(page.locator('#error')).toBeHidden();
  });
});

for (const [name, message] of [
  ['NotFoundError', 'No camera found.'],
  ['NotReadableError', 'Camera is unavailable. Another app may be using it.'],
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
    await page.locator('#camera-shutter').click();
    await expect(page.locator('#error-text')).toHaveText(message);
    await expect(page.locator('#camera-shutter')).toBeEnabled();
    expect(await readCommit(page)).toEqual(original);
    expect(await page.locator('#camera-preview').evaluate((video) => video.srcObject === null)).toBe(true);
    await startCamera(page, original);
    await expect(page.locator('#error')).toBeHidden();
    const retained = await retainCamera(page);
    await page.locator('#camera-stop').click();
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
    await page.locator('#camera-shutter').click();
    await expect(page.locator('#error-text')).toHaveText('Camera requires HTTPS or localhost');
    await expect(page.locator('#camera-shutter')).toBeEnabled();
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
  await page.locator('#camera-stop').click({ timeout: 2000 });
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
    await page.locator('#grid-max-width').evaluate((slider, value) => {
      slider.value = String(value);
      slider.dispatchEvent(new Event('input', { bubbles: true }));
    }, width);
  }
  await openPanel(page, 'tune');
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
    await expect(page.locator('#error-text')).toHaveText('Rendering failed');
    await expect(page.locator('#status')).toHaveText('Camera stopped');
    await expectRetainedEnded(page, retained.index);
    const failed = await readCommit(page);
    const workers = await page.evaluate(() => window.__cameraQA.workers.length);
    expect(await page.evaluate(() => window.__cameraQA.computed.some((reply) => reply.type === 'error'))).toBe(true);
    expect(await page.evaluate(() => window.__cameraQA.workers.at(-1).terminated)).toBe(true);
    await page.locator('#grid-max-width').evaluate((slider) => {
      slider.value = '60';
      slider.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await renderingTurns(page);
    expect(await page.evaluate(() => window.__cameraQA.workers.length)).toBe(workers);
    expect(await readCommit(page)).toEqual(failed);
    await expect(page.locator('#error-text')).toHaveText('Rendering failed');
    const recovered = recovery === 'camera' ? await startCamera(page, failed) : await uploadFixture(page, 'astronaut');
    expect(recovered.renderId).toBeGreaterThan(failed.renderId);
    expect(recovered.settingsKey).toBe(settingsKey({ maxWidthChars: 60 }));
    expect(await page.evaluate(() => window.__cameraQA.workers.length)).toBe(workers + 1);
    await expect(page.locator('#error')).toBeHidden();
    if (recovery === 'camera') {
      const replacement = await retainCamera(page);
      await page.locator('#camera-stop').click();
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
  await page.locator('#camera-stop').click({ timeout: 2000 });
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
  await openApp(page);
  await page.locator('#camera-shutter').click();
  const initial = await waitCameraCommit(page);
  const retained = await retainCamera(page);
  const directory = path.join(ROOT, 'qa/artifacts/camera-features', testInfo.project.name);
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
  const tuned = await changeSettings(page, { maxWidthChars: 60, maxHeightLines: 80 });
  expect(tuned.sourceGeneration).toBe(shot.sourceGeneration);
  expect([tuned.columns, tuned.rows]).toEqual([120, 45]);
  expect(await readCommit(page)).toEqual(tuned);
  await page.locator('#camera-shutter').click();
  const resumed = await waitCameraCommit(page, tuned);
  const resumeTracks = await retainCamera(page);
  await openPanel(page, 'camera');
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

test('look and frame changes reject held replies, gate the shutter, and preserve full captured sources', async ({ page }) => {
  await page.setViewportSize({ width: 428, height: 745 });
  await openApp(page);
  await page.locator('#camera-shutter').click();
  const initial = await waitCameraCommit(page);
  const firstStream = await retainCamera(page);
  const held = await holdRealReply(page);
  await page.locator('#open-looks').click();
  await page.locator('input[name="look"][value="photo"]').check();
  await page.locator('#tab-framing').click();
  await page.locator('input[name="framing"][value="story"]').check();
  await page.locator('#close-look-sheet').click();
  await expect(page.locator('#camera-shutter')).toBeDisabled();
  expect(await page.evaluate(() => window.__cameraQA.reads)).toBe(held.reads);
  expect(await page.evaluate(() => window.__cameraQA.requests.length)).toBe(held.requests);
  await releaseRealReply(page, held.id);
  await page.waitForFunction(() => {
    const data = document.querySelector('#ascii-output').dataset;
    return data.framing === 'story' && JSON.parse(data.lookKey).id === 'photo';
  });
  await expect(page.locator('#camera-shutter')).toBeEnabled();
  const framed = await readCommit(page);
  expect(framed.sourceGeneration).toBe(initial.sourceGeneration);
  expect([framed.columns, framed.rows]).toEqual([72, 65]);
  expect(await page.evaluate((id) => window.__cameraQA.commits.some((commit) => commit.id === id), held.id)).toBe(false);

  const pendingCapture = await holdRealReply(page);
  await page.locator('#camera-shutter').click();
  await expectRetainedEnded(page, firstStream.index);
  await expect(page.locator('#source-label')).toHaveText('Captured frame.png');
  await page.locator('#open-framing').click();
  await page.locator('input[name="framing"][value="wide"]').check();
  await page.locator('#tab-looks').click();
  await page.locator('input[name="look"][value="print"]').check();
  await page.locator('#close-look-sheet').click();
  expect(await page.evaluate(() => window.__cameraQA.reads)).toBe(pendingCapture.reads);
  await releaseRealReply(page, pendingCapture.id);
  await page.waitForFunction(() => {
    const data = document.querySelector('#ascii-output').dataset;
    return data.sourceKind === 'image' && data.framing === 'wide' && JSON.parse(data.lookKey).id === 'print';
  });
  const shot = await readCommit(page);
  expect(shot.sourceGeneration).toBe(initial.sourceGeneration + 1);
  expect([shot.columns, shot.rows]).toEqual([100, 28]);
  const probe = await page.evaluate(() => ({
    maxOutstanding: window.__cameraQA.maxOutstanding,
    commits: window.__cameraQA.commits,
    request: window.__cameraQA.requests.at(-1),
  }));
  expect(probe.maxOutstanding).toBe(1);
  expect(probe.commits.some((commit) => commit.id === pendingCapture.id)).toBe(false);
  expect(probe.request.look.id).toBe('print');

  await page.locator('#open-framing').click();
  await page.locator('input[name="framing"][value="full"]').check();
  await page.locator('#close-look-sheet').click();
  await page.waitForFunction(() => document.querySelector('#ascii-output').dataset.framing === 'full');
  const fullShot = await readCommit(page);
  expect(fullShot.sourceGeneration).toBe(shot.sourceGeneration);
  expect([fullShot.columns, fullShot.rows]).toEqual([100, 37]);
  expect(await page.locator('#image-preview').evaluate((img) => [img.naturalWidth, img.naturalHeight])).toEqual([320, 240]);
  await page.locator('#camera-shutter').click();
  const resumed = await waitCameraCommit(page, fullShot);
  const resumedStream = await retainCamera(page);
  await openPanel(page, 'camera');
  await page.locator('#flip-camera').click();
  await waitCameraCommit(page, resumed);
  await expectRetainedEnded(page, resumedStream.index, false);
  const flipped = await retainCamera(page);
  await page.locator('#camera-stop').click();
  await expectRetainedEnded(page, flipped.index);
  const stopped = await readCommit(page);
  await renderingTurns(page);
  expect(await readCommit(page)).toEqual(stopped);
  await expect(page.locator('#ascii-output')).toHaveAttribute('data-framing', 'full');
  expect(JSON.parse(await page.locator('#ascii-output').getAttribute('data-look-key')).id).toBe('print');
});

async function waitRecording(page, seconds = 1) {
  await expect(page.locator('body')).toHaveAttribute('data-recording-state', 'recording');
  await expect(page.locator('#recording-timer')).toHaveText(`00:0${seconds}`);
}

async function expectVideoReleased(page) {
  expect(await page.evaluate(() => ({
    cameras: window.__videoQA.cameras.every((stream) => stream.getTracks().every((track) => track.readyState === 'ended')),
    canvases: window.__videoQA.captures.every(({ stream }) => stream.getTracks().every((track) => track.readyState === 'ended')),
    encoders: window.__videoQA.recorders.every((recorder) => recorder.state === 'inactive'),
    detached: document.querySelector('#camera-preview').srcObject === null,
  }))).toEqual({ cameras: true, canvases: true, encoders: true, detached: true });
}

test('live ASCII recording saves a silent AVC clip with changing tinted artwork and clean retake', async ({ page }, testInfo) => {
  await installVideoCamera(page);
  await openApp(page);
  await page.locator('#open-looks').click();
  await page.locator('input[name="look"][value="photo"]').check();
  await openPanel(page, 'framing');
  await page.locator('input[name="framing"][value="story"]').check();
  await openPanel(page, 'grid');
  await page.locator('input[name="density"][value="detailed"]').check();
  await openPanel(page, 'style');
  await page.locator('input[name="text-tint"][value="#9be5a2"]').check();
  await page.locator('#soft-glow').check();
  await page.locator('#close-look-sheet').click();
  await page.locator('#video-mode').click();
  await page.locator('#camera-shutter').click();
  await waitRecording(page, 2);
  for (const selector of ['#open-looks', '#open-framing', '#flip-camera', '#camera-import', '#photo-mode', '#studio-video']) {
    await expect(page.locator(selector)).toBeDisabled();
  }
  await expect(page.locator('#camera-stop')).toBeEnabled();
  const configuration = await page.locator('#ascii-output').evaluate((output) => ({
    look: JSON.parse(output.dataset.lookKey).id, frame: output.dataset.framing,
    columns: Number(output.dataset.columns), rows: Number(output.dataset.rows),
  }));
  expect(configuration).toEqual({ look: 'photo', frame: 'story', columns: 224, rows: 200 });
  await page.locator('#camera-shutter').click();
  await expect(page.locator('#video-review')).toBeVisible();
  await expectVideoReleased(page);
  const video = page.locator('#recorded-video');
  await expect.poll(() => video.evaluate((element) => element.readyState)).toBeGreaterThanOrEqual(2);
  const pixels = await video.evaluate(async (element) => {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 64;
    const context = canvas.getContext('2d');
    async function sample(time) {
      await new Promise((resolve) => {
        element.addEventListener('seeked', resolve, { once: true });
        element.currentTime = time;
      });
      context.drawImage(element, 0, 0, 64, 64);
      const pixels = context.getImageData(0, 0, 64, 64).data;
      let hash = 2166136261, red = 0, green = 0, blue = 0, count = 0;
      for (let i = 0; i < pixels.length; i += 4) {
        hash = Math.imul(hash ^ pixels[i] ^ pixels[i + 1] << 8 ^ pixels[i + 2] << 16, 16777619);
        if (Math.max(pixels[i], pixels[i + 1], pixels[i + 2]) > 48) {
          red += pixels[i]; green += pixels[i + 1]; blue += pixels[i + 2]; count++;
        }
      }
      return { hash, red: red / count, green: green / count, blue: blue / count };
    }
    return { first: await sample(0.15), later: await sample(0.9) };
  });
  expect(pixels.first.hash, 'Decoded frames contain actual motion, not a static poster').not.toBe(pixels.later.hash);
  expect(pixels.first.green, 'Exported glyph tint, not the raw camera colors').toBeGreaterThan(pixels.first.red + 8);
  expect(pixels.first.green).toBeGreaterThan(pixels.first.blue + 8);
  const saved = await downloadVideo(page, testInfo);
  expect(saved.filename).toMatch(/^clack-.*\.mp4$/);
  expect(saved.streams).toHaveLength(1);
  expect(saved.streams[0]).toMatchObject({ codec_type: 'video', codec_name: 'h264' });
  expect(Math.max(saved.streams[0].width, saved.streams[0].height)).toBeLessThanOrEqual(1280);
  expect(saved.streams[0].width % 2).toBe(0);
  expect(saved.streams[0].height % 2).toBe(0);
  expect(Number(saved.format.duration)).toBeGreaterThan(1);
  expect(Number(saved.format.duration)).toBeLessThan(5);
  expect(await page.evaluate(() => window.__videoQA.requests.every((request) => request.audio === false))).toBe(true);
  const previousURL = await video.getAttribute('src');
  await page.locator('#video-retake').click();
  await expect(page.locator('#video-review')).toBeHidden();
  await expect.poll(() => page.evaluate(() => window.__videoQA.revoked.includes(window.__videoQA.urls[0].url))).toBe(true);
  await expect(page.locator('#status')).toHaveText('Camera live');
  await expect(page.locator('body')).toHaveAttribute('data-recording-state', 'idle');
  await page.locator('#camera-shutter').click();
  await waitRecording(page);
  await page.locator('#camera-stop').click();
  await expect(page.locator('#video-review')).toBeVisible();
  expect(await video.getAttribute('src')).not.toBe(previousURL);
  await expectVideoReleased(page);
  // A saved video can also be opened after starting a later Photo preview.
  await page.locator('#close-video-review').click();
  await page.locator('#photo-mode').click();
  await page.locator('#camera-shutter').click();
  await expect(page.locator('#status')).toHaveText('Camera live');
  await page.locator('#review-video').click();
  await expect(page.locator('#video-review')).toBeVisible();
  await expectVideoReleased(page);
});

test('canceling Video before its first committed reply never records stale artwork', async ({ page }) => {
  await installVideoCamera(page);
  await openApp(page);
  const prior = await uploadFixture(page, 'astronaut');
  await page.evaluate(() => { window.__cameraQA.holdNext = true; });
  await page.locator('#video-mode').click();
  await page.locator('#camera-shutter').click();
  await expect.poll(() => page.evaluate(() => window.__cameraQA.held.length)).toBe(1);
  const heldId = await page.evaluate(() => window.__cameraQA.held[0].data.id);
  await expect(page.locator('body')).toHaveAttribute('data-recording-state', 'pending');
  expect(await page.evaluate(() => window.__videoQA.recorders.length)).toBe(0);
  await page.locator('#camera-shutter').click();
  await releaseRealReply(page, heldId);
  await expect(page.locator('body')).toHaveAttribute('data-recording-state', 'idle');
  expect(await readCommit(page)).toEqual(prior);
  expect(await page.evaluate(() => window.__videoQA.recorders.length)).toBe(0);
  await expect(page.locator('#video-review')).toBeHidden();
  await expectVideoReleased(page);
});

for (const failure of ['startup', 'encoder']) {
  test(`recording ${failure} failure releases real streams and a new explicit recording recovers`, async ({ page }) => {
    await installVideoCamera(page);
    await openApp(page);
    await page.locator('#video-mode').click();
    if (failure === 'startup') await page.evaluate(() => { window.__videoQA.failNextEncoder = true; });
    await page.locator('#camera-shutter').click();
    if (failure === 'encoder') {
      await waitRecording(page);
      // Error-only adapter: no successful encoded payload is invented.
      await page.evaluate(() => window.__videoQA.recorders[0].dispatchEvent(new Event('error')));
    }
    await expect(page.locator('#error')).toBeVisible();
    await expect(page.locator('body')).toHaveAttribute('data-recording-state', 'idle');
    await expect(page.locator('#video-review')).toBeHidden();
    await expectVideoReleased(page);
    await expect(page.locator('#camera-shutter')).toBeEnabled();
    await page.locator('#camera-shutter').click();
    await waitRecording(page);
    await page.locator('#camera-shutter').click();
    await expect(page.locator('#video-review')).toBeVisible();
    await expectVideoReleased(page);
  });
}

test('background release finishes the existing video and opens review only after returning', async ({ page }) => {
  await installVideoCamera(page);
  await openApp(page);
  await page.locator('#video-mode').click();
  await page.locator('#camera-shutter').click();
  await waitRecording(page);
  await page.evaluate(() => {
    Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'hidden' });
    document.dispatchEvent(new Event('visibilitychange'));
  });
  await expect(page.locator('body')).toHaveAttribute('data-recording-state', 'ready');
  await expectVideoReleased(page);
  await expect(page.locator('#video-review')).toBeHidden();
  const stopped = await readCommit(page);
  await page.evaluate(() => {
    Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'visible' });
    document.dispatchEvent(new Event('visibilitychange'));
  });
  await expect(page.locator('#video-review')).toBeVisible();
  expect(await readCommit(page)).toEqual(stopped);
});

test('a real source rotation finishes a fixed-size clip instead of stretching new frames', async ({ page }) => {
  await installVideoCamera(page);
  await openApp(page);
  await page.locator('#video-mode').click();
  await page.locator('#camera-shutter').click();
  await waitRecording(page);
  const dimensions = await page.evaluate(() => {
    const canvas = window.__videoQA.captures[0].canvas;
    const dimensions = [canvas.width, canvas.height];
    window.__videoQA.sources[0].width = 240;
    window.__videoQA.sources[0].height = 320;
    return dimensions;
  });
  await expect(page.locator('#video-review')).toBeVisible();
  await expect.poll(() => page.locator('#recorded-video').evaluate((video) => [video.videoWidth, video.videoHeight])).toEqual(dimensions);
  await expectVideoReleased(page);
});

test('the actual recording duration cap finalizes a playable clip and releases capture', async ({ page }) => {
  test.setTimeout(45000);
  await installVideoCamera(page);
  await openApp(page);
  await page.locator('#video-mode').click();
  await page.locator('#camera-shutter').click();
  await expect(page.locator('body')).toHaveAttribute('data-recording-state', 'recording');
  await expect(page.locator('#video-review')).toBeVisible({ timeout: 35000 });
  await expect.poll(() => page.locator('#recorded-video').evaluate((video) => video.duration)).toBeGreaterThan(29);
  expect(await page.locator('#recorded-video').evaluate((video) => video.duration)).toBeLessThan(32);
  await expectVideoReleased(page);
});

test('WebM fallback encodes and downloads its actual format without disguising it as MP4', async ({ page }, testInfo) => {
  await installVideoCamera(page);
  await page.addInitScript(() => {
    const supported = window.MediaRecorder.isTypeSupported.bind(window.MediaRecorder);
    window.MediaRecorder.isTypeSupported = (type) => !type.startsWith('video/mp4') && supported(type);
  });
  await openApp(page);
  await page.locator('#video-mode').click();
  await page.locator('#camera-shutter').click();
  await waitRecording(page);
  await page.locator('#camera-shutter').click();
  await expect(page.locator('#video-review')).toBeVisible();
  await page.locator('#recorded-video').evaluate((video) => video.play());
  await expect.poll(() => page.locator('#recorded-video').evaluate((video) => video.currentTime)).toBeGreaterThan(0);
  const saved = await downloadVideo(page, testInfo);
  expect(saved.filename).toMatch(/\.webm$/);
  expect(saved.streams).toHaveLength(1);
  expect(saved.streams[0]).toMatchObject({ codec_type: 'video', codec_name: 'vp8' });
  expect(saved.format.format_name).toContain('webm');
  await expectVideoReleased(page);
});

test('canceling the native-share request preserves the playable clip and re-enables actions', async ({ page }) => {
  await installVideoCamera(page);
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'canShare', { configurable: true, value: () => true });
    Object.defineProperty(navigator, 'share', { configurable: true, value: () => Promise.reject(new DOMException('Canceled', 'AbortError')) });
  });
  await openApp(page);
  await page.locator('#video-mode').click();
  await page.locator('#camera-shutter').click();
  await waitRecording(page);
  await page.locator('#camera-shutter').click();
  await expect(page.locator('#video-review')).toBeVisible();
  const source = await page.locator('#recorded-video').getAttribute('src');
  await page.locator('#video-share').click();
  await expect(page.locator('#video-share')).toBeEnabled();
  await expect(page.locator('#video-retake')).toBeEnabled();
  await expect(page.locator('#recorded-video')).toHaveAttribute('src', source);
  await page.locator('#recorded-video').evaluate((video) => video.play());
  await expect.poll(() => page.locator('#recorded-video').evaluate((video) => video.currentTime)).toBeGreaterThan(0);
});

test('recording waits for the selected configuration without reacquiring an already live camera', async ({ page }) => {
  await installVideoCamera(page);
  await openApp(page);
  await page.locator('#camera-shutter').click();
  const initial = await waitCameraCommit(page);
  const held = await holdRealReply(page);
  await page.locator('#open-looks').click();
  await page.locator('input[name="look"][value="print"]').check();
  await openPanel(page, 'framing');
  await page.locator('input[name="framing"][value="portrait"]').check();
  await page.locator('#close-look-sheet').click();
  await page.locator('#video-mode').click();
  await page.locator('#camera-shutter').click();
  await expect(page.locator('body')).toHaveAttribute('data-recording-state', 'pending');
  expect(await page.evaluate(() => window.__videoQA.requests.length)).toBe(1);
  expect(await page.evaluate(() => window.__videoQA.cameras[0].getVideoTracks()[0].readyState)).toBe('live');
  expect(await page.evaluate(() => window.__videoQA.recorders.length)).toBe(0);
  await releaseRealReply(page, held.id);
  await waitRecording(page);
  const committed = await page.locator('#ascii-output').evaluate((output) => ({
    generation: Number(output.dataset.sourceGeneration),
    look: JSON.parse(output.dataset.lookKey).id, framing: output.dataset.framing,
  }));
  expect(committed).toEqual({ generation: initial.sourceGeneration, look: 'print', framing: 'portrait' });
  await page.locator('#camera-shutter').click();
  await expect(page.locator('#video-review')).toBeVisible();
  await expectVideoReleased(page);
});

test('restored density and preset configuration is recorded while pending and active controls stay locked', async ({ page }) => {
  await installVideoCamera(page);
  await openApp(page);
  await uploadFixture(page, 'astronaut');
  await openPanel(page, 'presets');
  await page.locator('#preset-name').fill('Other');
  await page.locator('#preset-save').click();
  await expect(page.locator('button[data-preset-apply="Other"]')).toBeVisible();

  await changeSettings(page, { sigma1: 1.5, sigma2: 2.5, edgeIndexThreshold: 70 });
  await openPanel(page, 'grid');
  await page.locator('input[name="density"][value="fine"]').check();
  await openPanel(page, 'looks');
  await page.locator('input[name="look"][value="photo"]').check();
  await openPanel(page, 'framing');
  await page.locator('input[name="framing"][value="portrait"]').check();
  await openPanel(page, 'style');
  await page.locator('input[name="text-tint"][value="#ffd663"]').check();
  await page.locator('#soft-glow').check();
  await page.locator('#close-look-sheet').click();
  const expectedSettings = settingsKey({
    maxWidthChars: 75, maxHeightLines: 100, sigma1: 1.5, sigma2: 2.5, edgeIndexThreshold: 70,
  });
  await expect.poll(async () => (await readCommit(page)).settingsKey).toBe(expectedSettings);

  await page.reload();
  await expect(page.locator('#status')).toContainText('No source');
  await openPanel(page, 'grid');
  await expect(page.locator('input[name="density"][value="fine"]')).toBeChecked();
  await expect(page.locator('#grid-max-width')).toHaveValue('75');
  await expect(page.locator('#grid-max-height')).toHaveValue('100');
  await openPanel(page, 'presets');
  await expect(page.locator('button[data-preset-apply="Other"]')).toBeVisible();
  await page.locator('#close-look-sheet').click();

  await page.evaluate(() => { window.__cameraQA.holdNext = true; });
  await page.locator('#video-mode').click();
  await page.locator('#camera-shutter').click();
  await expect.poll(() => page.evaluate(() => window.__cameraQA.held.length)).toBe(1);
  const heldId = await page.evaluate(() => window.__cameraQA.held[0].data.id);
  const lockedControls = [
    'input[name="density"][value="chunky"]', '#grid-max-width', '#grid-max-height',
    '#preset-name', '#preset-save', 'button[data-preset-apply="Other"]',
    'button[data-preset-delete="Other"]', '#copy-output', '#save-output',
  ];
  await expect(page.locator('body')).toHaveAttribute('data-recording-state', 'pending');
  for (const selector of lockedControls) await expect(page.locator(selector)).toBeDisabled();
  await expect(page.locator('#camera-shutter')).toBeEnabled();
  await expect(page.locator('#camera-stop')).toBeEnabled();
  expect(await page.evaluate(() => window.__videoQA.recorders.length)).toBe(0);
  await releaseRealReply(page, heldId);
  await waitRecording(page);
  for (const selector of lockedControls) await expect(page.locator(selector)).toBeDisabled();
  const commit = await readCommit(page);
  expect(commit.settingsKey).toBe(expectedSettings);
  expect([commit.columns, commit.rows]).toEqual([150, 100]);
  await expect(page.locator('#ascii-output')).toHaveAttribute('data-framing', 'portrait');
  expect(await page.locator('#ascii-output').evaluate((output) => JSON.parse(output.dataset.lookKey).id)).toBe('photo');
  await expect(page.locator('#ascii-output')).toHaveCSS('color', 'rgb(255, 214, 99)');
  await expect(page.locator('body')).toHaveAttribute('data-glow', 'on');
  const dimensions = await page.evaluate(() => {
    const { canvas } = window.__videoQA.captures[0];
    return [canvas.width, canvas.height];
  });
  expect(dimensions[1]).toBeGreaterThan(dimensions[0]);
  await page.locator('#camera-shutter').click();
  await expect(page.locator('#video-review')).toBeVisible();
  await expect.poll(() => page.locator('#recorded-video').evaluate((video) => [video.videoWidth, video.videoHeight])).toEqual(dimensions);
  const ink = await page.locator('#recorded-video').evaluate(async (video) => {
    await video.play();
    await new Promise((resolve) => video.requestVideoFrameCallback(resolve));
    video.pause();
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let red = 0, green = 0, blue = 0, count = 0;
    for (let i = 0; i < pixels.length; i += 4) {
      if (pixels[i] > 100) {
        red += pixels[i]; green += pixels[i + 1]; blue += pixels[i + 2]; count++;
      }
    }
    return { count, red: red / count, green: green / count, blue: blue / count };
  });
  expect(ink.count).toBeGreaterThan(0);
  expect(ink.red).toBeGreaterThan(ink.green + 8);
  expect(ink.green).toBeGreaterThan(ink.blue + 20);
  await expectVideoReleased(page);
  await page.locator('#close-video-review').click();
  await openPanel(page, 'grid');
  await expect(page.locator('input[name="density"][value="fine"]')).toBeChecked();
  await expect(page.locator('input[name="density"][value="chunky"]')).toBeEnabled();
  await openPanel(page, 'presets');
  await expect(page.locator('button[data-preset-apply="Other"]')).toBeEnabled();
  await expect(page.locator('button[data-preset-delete="Other"]')).toBeEnabled();
});

test('empty state starts the camera and retires once a frame commits', async ({ page }) => {
  await openApp(page);
  await expect(page.locator('#empty-state')).toBeVisible();
  await page.locator('#camera-shutter').click();
  await waitCameraCommit(page);
  await expect(page.locator('#empty-state')).toBeHidden();
  await expect(page.locator('#camera-stop')).toBeVisible();
  const retained = await retainCamera(page);
  await page.locator('#camera-stop').click();
  await expectRetainedEnded(page, retained.index);
  await expect(page.locator('#empty-state')).toBeHidden();
});

test('flip stays hidden until a camera runs', async ({ page }) => {
  await openApp(page);
  const flip = page.locator('#flip-camera');
  await expect(flip).toBeHidden();
  await expect(page.locator('#camera-stop')).toBeHidden();
  expect(await page.locator('#look-sheet').evaluate((sheet) => sheet.hidden), 'Flip is reachable without opening the drawer').toBe(true);

  await page.locator('#camera-shutter').click();
  const first = await waitCameraCommit(page);
  await expect(flip).toBeVisible();
  await expect(flip).toBeEnabled();
  await expect(flip).toHaveAttribute('aria-label', 'Switch front and back camera');
  const firstStream = await retainCamera(page);

  await flip.click();
  await expect(page.locator('#camera-facing')).toHaveValue('environment');
  const second = await waitCameraCommit(page, first);
  expect(second.sourceGeneration).toBeGreaterThan(first.sourceGeneration);
  await expectRetainedEnded(page, firstStream.index, false);

  await flip.click();
  await expect(page.locator('#camera-facing')).toHaveValue('user');
  const third = await waitCameraCommit(page, second);
  expect(third.sourceGeneration).toBeGreaterThan(second.sourceGeneration);

  await page.locator('#camera-stop').click();
  await expect(page.locator('#status')).toHaveText('Camera stopped');
  await expect(flip).toBeHidden();
});

test('the camera panel keeps its record shortcut in view while the monitor is live', async ({ page }) => {
  await openApp(page);
  await page.setViewportSize({ width: 375, height: 667 });
  await page.locator('#camera-shutter').click();
  await waitCameraCommit(page);
  await openPanel(page, 'camera');

  const monitorHeight = await page.locator('#panel-camera .monitor video')
    .evaluate((video) => video.getBoundingClientRect().height);
  expect(monitorHeight).toBeGreaterThan(0);
  expect(monitorHeight).toBeLessThanOrEqual(120);

  const layout = await page.evaluate(() => {
    const body = document.querySelector('.drawer-body').getBoundingClientRect();
    const shortcut = document.getElementById('studio-video').getBoundingClientRect();
    const flip = document.getElementById('flip-camera').getBoundingClientRect();
    return {
      shortcutInsideDrawer: shortcut.top >= body.top && shortcut.bottom <= body.bottom,
      flipInsideViewport: flip.top >= 0 && flip.bottom <= innerHeight,
      flipInDrawer: document.getElementById('look-sheet').contains(document.getElementById('flip-camera')),
      pageOverflowX: document.documentElement.scrollWidth - innerWidth,
    };
  });
  expect(layout).toEqual({
    shortcutInsideDrawer: true, flipInsideViewport: true, flipInDrawer: false, pageOverflowX: 0,
  });

  await page.locator('#studio-video').click();
  await expect(page.locator('#look-sheet')).toBeHidden();
  await expect(page.locator('#video-mode')).toHaveAttribute('aria-pressed', 'true');
});

test('the rail keeps deliberate rows and a stable topbar across camera state', async ({ page }) => {
  await page.setViewportSize({ width: 356, height: 784 });
  await openApp(page);
  await expect(page.locator('#status')).toHaveText('No source');
  await expect(page.locator('#flip-camera')).toBeHidden();
  await expect(page.locator('#camera-stop')).toBeHidden();
  await expect(page.locator('#fit-output')).toBeDisabled();
  await expect(page.locator('#detail-output')).toBeDisabled();

  const guidesX = (await page.locator('#toggle-guides').boundingBox()).x;
  const rows = await page.evaluate(() => ({
    modes: document.querySelector('.rail-modes').getBoundingClientRect().top,
    capture: document.querySelector('.rail-capture').getBoundingClientRect().top,
    output: document.querySelector('.rail-output').getBoundingClientRect().top,
    display: document.querySelector('.rail-display').getBoundingClientRect().top,
  }));
  expect(rows.modes, 'modes row leads the rail').toBeLessThan(rows.capture);
  expect(rows.capture, 'capture row follows modes').toBeLessThan(rows.output);
  expect(rows.output, 'output row follows capture').toBeLessThan(rows.display);
  expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBe(0);

  const seams = await page.evaluate(() => {
    const right = (selector) => document.querySelector(selector).getBoundingClientRect().right;
    return { output: right('.rail-output'), display: right('.view-controls'), tiles: right('.readout-tiles') };
  });
  expect(seams.output, 'output row shares the tiles right edge').toBeCloseTo(seams.tiles, 1);
  expect(seams.display, 'view row shares the tiles right edge').toBeCloseTo(seams.tiles, 1);

  await page.locator('#camera-shutter').click();
  await waitCameraCommit(page);
  const flip = page.locator('#flip-camera');
  await expect(flip).toBeVisible();
  await expect(flip).toBeEnabled();
  expect(await flip.evaluate((button) => getComputedStyle(button).color), 'flip reads as an action, not a mode chip')
    .toBe(await page.locator('#camera-import').evaluate((button) => getComputedStyle(button).color));
  await expect(page.locator('#camera-stop')).toBeVisible();
  await expect(page.locator('#fit-output')).toBeEnabled();
  await expect(page.locator('#detail-output')).toBeEnabled();
  expect((await page.locator('#toggle-guides').boundingBox()).x, 'starting the camera never shifts GUIDES').toBe(guidesX);

  const liveSeams = await page.evaluate(() => {
    const rect = (id) => document.getElementById(id).getBoundingClientRect();
    const modes = document.querySelector('.rail-modes').getBoundingClientRect();
    return {
      shutterRight: rect('camera-shutter').right,
      importRight: rect('camera-import').right,
      flipLeft: rect('flip-camera').left,
      copyLeft: rect('copy-output').left,
      chipSeam: rect('video-mode').left,
      modesCenter: (modes.left + modes.right) / 2,
      modesRight: modes.right,
      tilesRight: document.querySelector('.readout-tiles').getBoundingClientRect().right,
    };
  });
  expect(liveSeams.shutterRight, 'FREEZE shares IMPORT right edge').toBeCloseTo(liveSeams.importRight, 1);
  expect(liveSeams.flipLeft, 'FLIP sits on COPY left edge').toBeCloseTo(liveSeams.copyLeft, 1);
  expect(liveSeams.modesRight, 'chips fill row 1 at steady live').toBeCloseTo(liveSeams.tilesRight, 1);
  // The 4px chip gap shifts the VIDEO border half a gap right of center; its
  // hole is centered on the row like every other seam.
  expect(liveSeams.chipSeam, 'chips split row 1 half and half').toBeCloseTo(liveSeams.modesCenter + 2, 0);

  await expect(page.locator('#copy-output'), 'live grid exports the committed frame').toBeEnabled();
  await expect(page.locator('#save-output')).toBeEnabled();
  await expect(page.locator('#shutter-help')).toHaveText('');

  await page.locator('#camera-stop').click();
  await expect(page.locator('#status')).toHaveText('Camera stopped');
  await expect(flip).toBeHidden();
  await expect(page.locator('#camera-stop')).toBeHidden();
  expect((await page.locator('#toggle-guides').boundingBox()).x, 'stopping the camera never shifts GUIDES').toBe(guidesX);

  await page.locator('#camera-shutter').click();
  await waitCameraCommit(page);
  await page.locator('#camera-shutter').click();
  await expect(page.locator('#status')).toHaveText('Frame captured');
  await expect(page.locator('#shutter-help')).toHaveText('COPY or SAVE the frame');
  await expect(page.locator('#copy-output')).toBeEnabled();
  await expect(page.locator('#save-output')).toBeEnabled();
  expect((await page.locator('#toggle-guides').boundingBox()).x, 'stopping the camera never shifts GUIDES').toBe(guidesX);
});

test('the readout separates settings tiles from grid telemetry', async ({ page }) => {
  await openApp(page);
  await uploadFixture(page, 'astronaut');
  await expect(page.locator('#grid-label')).toHaveText(/^GRID \d+ × \d+$/);
  const separation = await page.evaluate(() => {
    const meta = document.querySelector('.readout-meta');
    const tiles = document.querySelector('.readout-tiles');
    const buttons = [...tiles.querySelectorAll('button')];
    return {
      gridInMeta: meta.contains(document.getElementById('grid-label')),
      gridInTiles: tiles.contains(document.getElementById('grid-label')),
      privacyInMeta: meta.contains(document.querySelector('.readout-static')),
      tileCount: buttons.length,
      tileHeights: [...new Set(buttons.map((button) => button.getBoundingClientRect().height))],
      tileBorders: [...new Set(buttons.map((button) => getComputedStyle(button).borderStyle))],
    };
  });
  expect(separation.gridInMeta).toBe(true);
  expect(separation.gridInTiles).toBe(false);
  expect(separation.privacyInMeta).toBe(true);
  expect(separation.tileCount).toBe(4);
  expect(separation.tileHeights).toHaveLength(1);
  expect(separation.tileBorders).toEqual(['solid']);
});

test('100% view marks the art as pannable', async ({ page }) => {
  await page.setViewportSize({ width: 356, height: 784 });
  await openApp(page);
  await page.locator('#camera-shutter').click();
  await waitCameraCommit(page);
  const viewport = page.locator('.ascii-viewport');
  await page.locator('#detail-output').click();
  await expect(viewport).toHaveAttribute('data-pannable', 'true');
  expect(await viewport.evaluate((element) => element.scrollWidth > element.clientWidth)).toBe(true);
  await page.locator('#fit-output').click();
  await expect(viewport).toHaveAttribute('data-pannable', 'false');
});
