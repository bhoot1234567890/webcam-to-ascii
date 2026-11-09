import { readFileSync } from 'node:fs';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { test, expect, openApp, openPanel, readCommit } from './fixtures.js';
import { installVideoCamera, downloadVideo } from './video-fixtures.js';

const run = promisify(execFile);

async function captureState(page) {
  return page.evaluate(() => {
    const probe = window.__videoQA;
    const capture = probe.captures.find(({ stream }) => stream === probe.recorders[0].stream);
    return {
      rate: capture.frameRate,
      manual: typeof capture.stream.getVideoTracks()[0].requestFrame === 'function',
      requests: capture.frameRequests.length,
      text: document.getElementById('ascii-output').textContent,
    };
  });
}

async function expectReleased(page) {
  expect(await page.evaluate(() => ({
    cameras: window.__videoQA.cameras.every(stream => stream.getTracks().every(track => track.readyState === 'ended')),
    captures: window.__videoQA.captures.every(({ stream }) => stream.getTracks().every(track => track.readyState === 'ended')),
    encoders: window.__videoQA.recorders.every(recorder => recorder.state === 'inactive'),
    detached: document.getElementById('camera-preview').srcObject === null,
  }))).toEqual({ cameras: true, captures: true, encoders: true, detached: true });
}

async function expectPlayableTimeline(page, saved) {
  expect(saved.streams).toHaveLength(1);
  const stream = saved.streams[0];
  expect(stream.codec_type).toBe('video');
  if (saved.filename.endsWith('.mp4')) expect(stream.codec_name).toBe('h264');
  else {
    expect(saved.filename).toMatch(/\.webm$/);
    expect(stream.codec_name).toMatch(/^vp[89]$/);
    expect(saved.format.format_name).toContain('webm');
  }
  expect(Math.max(stream.width, stream.height)).toBeLessThanOrEqual(1280);
  expect(stream.width % 2).toBe(0);
  expect(stream.height % 2).toBe(0);
  const { stdout } = await run('ffprobe', [
    '-v', 'error', '-show_entries', 'packet=pts_time,duration_time', '-of', 'json', saved.path,
  ]);
  const packets = JSON.parse(stdout).packets;
  const timestamps = packets.map(packet => Number(packet.pts_time)).filter(Number.isFinite);
  // Firefox's native WebM may omit Duration. Packet time span and real playback
  // prove the held interval without inventing or remuxing metadata.
  const span = Math.max(...timestamps) - Math.min(...timestamps);
  expect(span, 'Saved artwork must retain the recording interval, not become a single 33ms frame').toBeGreaterThan(0.75);
  expect(span).toBeLessThan(3);
  const player = page.locator('#recorded-video');
  await player.evaluate(video => video.play());
  await expect.poll(() => player.evaluate(video => video.currentTime)).toBeGreaterThan(0.5);
  return packets;
}

// Runs inside the browser: summarize actual canvas pixels or a decoded, serialized PNG.
async function artworkRaster({ png } = {}) {
  let source;
  if (png) source = await createImageBitmap(new Blob([new Uint8Array(png)], { type: 'image/png' }));
  else {
    const probe = window.__videoQA;
    source = probe.captures.find(({ stream }) => stream === probe.recorders[0].stream).canvas;
  }
  const canvas = document.createElement('canvas');
  canvas.width = source.width;
  canvas.height = source.height;
  const context = canvas.getContext('2d');
  context.drawImage(source, 0, 0);
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
  const background = [...pixels.subarray(0, 4)];
  let left = canvas.width, right = -1, top = canvas.height, bottom = -1;
  let red = 0, green = 0, blue = 0, ink = 0, minAlpha = 255;
  let pink = 0, purple = 0;
  for (let index = 0; index < pixels.length; index += 4) {
    minAlpha = Math.min(minAlpha, pixels[index + 3]);
    if (pixels[index] === 255 && pixels[index + 1] === 79 && pixels[index + 2] === 216) pink += 1;
    if (pixels[index] === 168 && pixels[index + 1] === 85 && pixels[index + 2] === 247) purple += 1;
    if (Math.max(pixels[index], pixels[index + 1], pixels[index + 2]) <= 48) continue;
    const x = (index / 4) % canvas.width;
    const y = Math.floor(index / 4 / canvas.width);
    left = Math.min(left, x); right = Math.max(right, x);
    top = Math.min(top, y); bottom = Math.max(bottom, y);
    red += pixels[index]; green += pixels[index + 1]; blue += pixels[index + 2]; ink++;
  }
  const reduced = document.createElement('canvas');
  reduced.width = reduced.height = 64;
  const small = reduced.getContext('2d');
  small.drawImage(canvas, 0, 0, 64, 64);
  const samples = [...small.getImageData(0, 0, 64, 64).data];
  const result = {
    width: canvas.width, height: canvas.height, background, minAlpha, samples,
    pink, purple,
    bounds: [left / canvas.width, right / canvas.width, top / canvas.height, bottom / canvas.height],
    tint: [red / ink, green / ink, blue / ink],
  };
  if (png) source.close();
  canvas.width = canvas.height = reduced.width = reduced.height = 0;
  return result;
}

test('static committed artwork records a playable native interval without periodic manual requests', async ({ page }, testInfo) => {
  await installVideoCamera(page, { staticScene: true });
  await openApp(page);
  await page.locator('#video-mode').click();
  await page.locator('#camera-shutter').click();
  await expect(page.locator('body')).toHaveAttribute('data-recording-state', 'recording');
  const first = await readCommit(page);
  await expect(page.locator('#recording-timer')).toHaveText('00:01');
  const later = await readCommit(page);
  expect(later.renderId, 'The preview keeps committing while static video capture is deduplicated').toBeGreaterThan(first.renderId);
  const raster = await page.evaluate(artworkRaster);
  expect(raster.pink, 'Recorded neon keeps its pink highlight ink').toBeGreaterThan(50);
  expect(raster.purple, 'Recorded neon keeps its purple midtone ink').toBeGreaterThan(50);
  const capture = await captureState(page);
  expect(capture.text).toBe(first.text);
  if (capture.manual) {
    expect(capture.rate).toBe(0);
    expect(capture.requests).toBe(1);
  } else expect(capture.rate).toBe(15);
  await page.locator('#camera-shutter').click();
  await expect(page.locator('#video-review')).toBeVisible();
  const final = await captureState(page);
  if (final.manual) expect(final.requests).toBe(2); // Initial picture plus explicit Stop hold boundary.
  const saved = await downloadVideo(page, testInfo);
  await expectPlayableTimeline(page, saved);
  await expectReleased(page);
  expect(await page.evaluate(() => window.__videoQA.requests.every(request => request.audio === false))).toBe(true);
});

test('PNG and recording retain the same committed artwork, background, tint and glow', async ({ page }) => {
  await installVideoCamera(page, { staticScene: true });
  await openApp(page);
  // The desktop save frame is a still-image finish; compare the raw artwork here.
  await openPanel(page, 'window');
  await page.locator('input[name="export-frame"][value="none"]').check();
  await page.locator('#close-look-sheet').click();
  await openPanel(page, 'looks');
  await page.locator('input[name="look"][value="photo"]').check();
  await openPanel(page, 'style');
  await page.locator('input[name="text-tint"][value="#9be5a2"]').check();
  await page.locator('#soft-glow').check();
  await page.locator('#close-look-sheet').click();
  await page.locator('#video-mode').click();
  await page.locator('#camera-shutter').click();
  await expect(page.locator('#recording-timer')).toHaveText('00:01');
  const committed = await readCommit(page);
  const recorded = await page.evaluate(artworkRaster);
  await page.locator('#camera-shutter').click();
  await expect(page.locator('#video-review')).toBeVisible();
  await page.locator('#close-video-review').click();
  const stopped = await readCommit(page);
  expect(stopped.text).toBe(committed.text);
  expect(stopped.settingsKey).toBe(committed.settingsKey);
  const [download] = await Promise.all([
    page.waitForEvent('download'), page.locator('#save-output').click(),
  ]);
  const png = [...readFileSync(await download.path())];
  const exported = await page.evaluate(artworkRaster, { png });
  expect(recorded.background).toEqual([9, 9, 11, 255]);
  expect(exported.background).toEqual(recorded.background);
  expect(recorded.minAlpha).toBe(255);
  expect(exported.minAlpha).toBe(255);
  // PNG retains its intentional 2x still-image resolution. Compare normalized
  // artwork geometry and raster rather than equating lossy video bytes with PNG.
  // The video canvas rounds to even dimensions, so a few output pixels of width
  // difference are expected; normalized bounds and samples carry the real check.
  expect(Math.abs(exported.width - recorded.width * 2)).toBeLessThanOrEqual(4);
  expect(Math.abs(exported.height - recorded.height * 2)).toBeLessThanOrEqual(4);
  for (let index = 0; index < recorded.bounds.length; index++) {
    expect(Math.abs(recorded.bounds[index] - exported.bounds[index])).toBeLessThan(0.015);
  }
  for (const raster of [recorded, exported]) {
    expect(raster.tint[1]).toBeGreaterThan(raster.tint[0] + 30);
    expect(raster.tint[1]).toBeGreaterThan(raster.tint[2] + 30);
  }
  const meanError = recorded.samples.reduce((sum, value, index) => sum + Math.abs(value - exported.samples[index]), 0) /
    (recorded.samples.length * 255);
  expect(meanError, 'Normalized PNG and recording artwork should retain the same glyph picture').toBeLessThan(0.04);
  await expectReleased(page);
});

test('hidden-visibility finalization retains static time without depending on RAF', async ({ page }, testInfo) => {
  await installVideoCamera(page, { staticScene: true });
  await openApp(page);
  await page.locator('#video-mode').click();
  await page.locator('#camera-shutter').click();
  await expect(page.locator('#recording-timer')).toHaveText('00:01');
  await page.evaluate(() => {
    Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => 'hidden' });
    document.dispatchEvent(new Event('visibilitychange'));
  });
  await expect(page.locator('body')).toHaveAttribute('data-recording-state', 'ready');
  await expect(page.locator('#video-review')).toBeHidden();
  await expectReleased(page);
  await page.evaluate(() => {
    Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => 'visible' });
    document.dispatchEvent(new Event('visibilitychange'));
  });
  await expect(page.locator('#video-review')).toBeVisible();
  await expectPlayableTimeline(page, await downloadVideo(page, testInfo));
});

test('Firefox encodes changing ASCII artwork and the selected tint @firefox-motion', async ({ page }, testInfo) => {
  await installVideoCamera(page);
  await openApp(page);
  await openPanel(page, 'looks');
  await page.locator('input[name="look"][value="photo"]').check();
  await openPanel(page, 'style');
  await page.locator('input[name="text-tint"][value="#9be5a2"]').check();
  await page.locator('#soft-glow').uncheck();
  await page.locator('#close-look-sheet').click();
  await page.locator('#video-mode').click();
  await page.locator('#camera-shutter').click();
  await expect(page.locator('#recording-timer')).toHaveText('00:02');
  await page.locator('#camera-shutter').click();
  await expect(page.locator('#video-review')).toBeVisible();
  const pictures = await page.locator('#recorded-video').evaluate(async video => {
    const canvas = document.createElement('canvas');
    canvas.width = 320;
    canvas.height = 240;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    const sample = async after => {
      while (video.currentTime < after) await new Promise(resolve => video.requestVideoFrameCallback(resolve));
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
      let hash = 2166136261, ink = 0, red = 0, green = 0;
      for (let i = 0; i < pixels.length; i += 4) {
        for (let channel = 0; channel < 3; channel++) hash = Math.imul(hash ^ pixels[i + channel], 16777619);
        if (Math.max(pixels[i], pixels[i + 1], pixels[i + 2]) > 80) {
          ink++;
          red += pixels[i];
          green += pixels[i + 1];
        }
      }
      return { hash: hash >>> 0, red: red / ink, green: green / ink };
    };
    await video.play();
    return [await sample(0.2), await sample(0.95)];
  });
  expect(pictures[1].hash, 'Decoded playback must change, not repeat a frozen poster').not.toBe(pictures[0].hash);
  for (const picture of pictures) expect(picture.green).toBeGreaterThan(picture.red + 20);
  await expectPlayableTimeline(page, await downloadVideo(page, testInfo));
  await expectReleased(page);
});
