import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { basename, resolve } from 'node:path';
import {
  test, expect, ROOT, IMAGES, REFERENCE, readCommit, uploadFixture,
  changeSettings, openApp, openPanel, captureApp,
} from './fixtures.js';
import { installVideoCamera, downloadVideo } from './video-fixtures.js';

const viewports = [
  { name: 'desktop', width: 1440, height: 1000 },
  { name: 'mobile', width: 390, height: 844 },
];

function assertGrid(commit, id, profile) {
  const reference = REFERENCE.fullImages.find((entry) => entry.fixtureId === id && entry.profile === profile);
  expect(reference, `Disk reference ${profile}:${id}`).toBeTruthy();
  expect(commit.columns).toBe(reference.width);
  expect(commit.rows).toBe(reference.height);
  const rows = commit.text.split('\n');
  expect(rows).toHaveLength(reference.height);
  expect(rows.every((row) => row.length === reference.width)).toBe(true);
  expect(commit.text.endsWith('\n')).toBe(false);
  expect(commit.sourceKind).toBe('image');
  return reference;
}

async function readPreview(page, fixture) {
  await openPanel(page, 'camera');
  const image = page.locator('#image-preview');
  await expect(image).toBeVisible();
  const preview = await image.evaluate((element) => ({
    src: element.getAttribute('src'), width: element.naturalWidth, height: element.naturalHeight,
  }));
  expect(preview.src).toMatch(/^blob:/);
  expect(preview.width).toBe(fixture.width);
  expect(preview.height).toBe(fixture.height);
  return preview;
}

for (const viewport of viewports) {
  test.describe(`static production ${viewport.name}`, () => {
    test.use({ viewport: { width: viewport.width, height: viewport.height } });

    test('built UI uploads photographs and applies settings through its bundled worker', async ({ page, context }, testInfo) => {
      test.setTimeout(90_000);
      expect(process.env.QA_PRODUCTION, 'This suite must use freshly built Vite preview, not development').toBe('1');
      const requests = [];
      // Installed before navigation; the shared automatic fixture additionally enforces
      // the complete context/worker privacy, no-WebSocket and console/pageerror policy.
      context.on('request', (request) => requests.push({ url: request.url(), method: request.method() }));
      await page.addInitScript(() => {
        window.__presentationRenderRequests = 0;
        const postMessage = Worker.prototype.postMessage;
        Worker.prototype.postMessage = function (message, ...options) {
          if (message?.type === 'render') window.__presentationRenderRequests++;
          return postMessage.call(this, message, ...options);
        };
      });
      await openApp(page);
      const scripts = await page.locator('script[src]').evaluateAll((elements) => elements.map((element) => element.getAttribute('src')));
      expect(scripts.length).toBeGreaterThan(0);
      expect(scripts.every((src) => /^\.?\/assets\/[^/]+\.js$/.test(src))).toBe(true);
      expect(await readCommit(page)).toMatchObject({ renderId: 0, sourceGeneration: 0, text: '' });

      const commits = [];
      for (const id of ['astronaut', 'astronaut-portrait']) {
        const fixture = IMAGES.fixtures.find((entry) => entry.id === id);
        expect(fixture).toBeTruthy();
        const before = await readCommit(page);
        const commit = await uploadFixture(page, id);
        expect(commit.renderId).toBeGreaterThan(before.renderId);
        expect(commit.sourceGeneration).toBeGreaterThan(before.sourceGeneration);
        expect(commit.sourceName).toBe(basename(fixture.path));
        expect(commit.settingsKey).toBe(JSON.stringify(REFERENCE.profiles.default));
        const reference = assertGrid(commit, id, 'default');
        const preview = await readPreview(page, fixture);
        const hash = createHash('sha256').update(readFileSync(resolve(ROOT, 'qa/fixtures/images', fixture.path))).digest('hex');
        expect(hash).toBe(fixture.sha256);
        expect(hash).toBe(reference.inputSha256);
        const { text, ...identity } = commit;
        commits.push({ fixtureId: id, inputSha256: hash, identity,
          textSha256: createHash('sha256').update(text).digest('hex'), preview });
        if (id === 'astronaut-portrait') expect(commit.rows).toBeGreaterThan(commit.columns / 2);
        await captureApp(page, testInfo, `production-${viewport.name}-${id}`);
      }

      const portrait = IMAGES.fixtures.find((entry) => entry.id === 'astronaut-portrait');
      const before = await readCommit(page);
      const originalPreview = await readPreview(page, portrait);
      const patch = { maxWidthChars: 60, maxHeightLines: 80, sigma1: 0.5, sigma2: 1.5, edgeIndexThreshold: 95 };
      const tuned = await changeSettings(page, patch);
      expect(tuned.renderId).toBeGreaterThan(before.renderId);
      expect(tuned.sourceGeneration).toBe(before.sourceGeneration);
      expect(tuned.sourceKind).toBe(before.sourceKind);
      expect(tuned.sourceName).toBe(before.sourceName);
      expect(tuned.settingsKey).toBe(JSON.stringify(patch));
      assertGrid(tuned, 'astronaut-portrait', 'comparison');
      expect(await readPreview(page, portrait)).toEqual(originalPreview);
      await expect(page.locator('#error')).toBeHidden();
      await expect(page.locator('#status')).toHaveText('Image ready');
      const panelControls = [
        ['camera', ['camera-select', 'camera-facing']],
        ['tune', ['sigma1', 'sigma2', 'edge-threshold']],
        ['grid', ['grid-max-width', 'grid-max-height']],
      ];
      for (const [tab, ids] of panelControls) {
        await openPanel(page, tab);
        for (const id of ids) {
          const control = page.locator(`#${id}`);
          await control.scrollIntoViewIfNeeded();
          await expect(control).toBeVisible();
          const bounds = await control.boundingBox();
          expect(bounds.x).toBeGreaterThanOrEqual(0);
          expect(bounds.x + bounds.width).toBeLessThanOrEqual(viewport.width + 1);
        }
      }
      await expect(page.locator('#image-input')).toBeEnabled();
      await page.locator('#close-look-sheet').click();
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
      await captureApp(page, testInfo, `production-${viewport.name}-portrait-tuned`);
      expect(await readCommit(page)).toEqual(tuned);
      const { text, ...identity } = tuned;
      commits.push({ fixtureId: 'astronaut-portrait', identity,
        textSha256: createHash('sha256').update(text).digest('hex'), preview: originalPreview });

      await page.locator('#open-looks').click();
      for (const id of ['photo', 'print', 'outline', 'original']) {
        const previous = await readCommit(page);
        await page.locator(`input[name="look"][value="${id}"]`).check();
        await expect.poll(async () => (await readCommit(page)).renderId).toBeGreaterThan(previous.renderId);
        await expect.poll(() => page.locator('#ascii-output').evaluate((output) => JSON.parse(output.dataset.lookKey).id)).toBe(id);
        const selected = await readCommit(page);
        expect(selected.sourceGeneration).toBe(tuned.sourceGeneration);
        expect(selected.settingsKey).toBe(tuned.settingsKey);
        if (id === 'original') expect(selected.text).toBe(tuned.text);
        else expect(selected.text).not.toBe(tuned.text);
      }
      await page.locator('#tab-framing').click();
      for (const id of ['square', 'portrait', 'story', 'classic', 'wide', 'full']) {
        const previous = await readCommit(page);
        await page.locator(`input[name="framing"][value="${id}"]`).check();
        await expect.poll(async () => (await readCommit(page)).renderId).toBeGreaterThan(previous.renderId);
        await expect(page.locator('#ascii-output')).toHaveAttribute('data-framing', id);
        expect((await readCommit(page)).sourceGeneration).toBe(tuned.sourceGeneration);
      }
      expect((await readCommit(page)).text).toBe(tuned.text);
      const presentationBaseline = await readCommit(page);
      const renderRequests = await page.evaluate(() => window.__presentationRenderRequests);
      await page.locator('#tab-style').click();
      for (const [value, color] of [
        ['#ffffff', 'rgb(255, 255, 255)'], ['#ffd663', 'rgb(255, 214, 99)'], ['#9be5a2', 'rgb(155, 229, 162)'],
        ['#3aff81', 'rgb(58, 255, 129)'], ['#6ce4ff', 'rgb(108, 228, 255)'], ['#ffe14d', 'rgb(255, 225, 77)'],
        ['#ff5c5c', 'rgb(255, 92, 92)'], ['#ff6bd6', 'rgb(255, 107, 214)'],
      ]) {
        await page.locator(`input[name="text-tint"][value="${value}"]`).check();
        await expect(page.locator('#ascii-output')).toHaveCSS('color', color);
        expect(await readCommit(page)).toEqual(presentationBaseline);
      }
      // Native color dialogs are OS UI; set the native input's value and exercise its real input event.
      await page.locator('#text-color').evaluate((picker) => {
        picker.value = '#87b7ff';
        picker.dispatchEvent(new Event('input', { bubbles: true }));
      });
      await expect(page.locator('#ascii-output')).toHaveCSS('color', 'rgb(135, 183, 255)');
      await page.locator('#soft-glow').check();
      await expect(page.locator('body')).toHaveAttribute('data-glow', 'on');
      await expect(page.locator('#ascii-output')).not.toHaveCSS('text-shadow', 'none');
      await page.locator('#close-look-sheet').click();
      await expect(page.locator('#open-looks')).toBeFocused();
      expect(await readCommit(page)).toEqual(presentationBaseline);
      expect(await page.evaluate(() => window.__presentationRenderRequests), 'Tint and glow never request processing').toBe(renderRequests);
      await captureApp(page, testInfo, `production-${viewport.name}-custom-glow`);
      await page.locator('#open-looks').click();
      await page.locator('#tab-style').click();
      await expect(page.locator('#text-color')).toHaveValue('#87b7ff');
      await page.locator('#soft-glow').uncheck();
      await page.locator('input[name="text-tint"][value="default"]').check();
      await expect(page.locator('#ascii-output')).toHaveCSS('text-shadow', 'none');
      await expect(page.locator('#ascii-output')).toHaveCSS('color', 'rgb(226, 233, 231)');
      await page.locator('#close-look-sheet').click();
      expect(await readCommit(page)).toEqual(presentationBaseline);
      expect(await page.evaluate(() => window.__presentationRenderRequests)).toBe(renderRequests);

      expect(requests.length).toBeGreaterThan(0);
      for (const request of requests) {
        const url = new URL(request.url);
        expect(request.method).toBe('GET');
        if (url.protocol === 'blob:') continue;
        expect(url.origin).toBe('http://127.0.0.1:4173');
        expect(url.pathname, 'No development modules or QA HTTP endpoints in production').not.toMatch(/^\/(?:src|tests|__qa|@vite|@fs|node_modules)(?:\/|$)/);
      }
      await testInfo.attach('production-ui-evidence', {
        body: JSON.stringify({ viewport, commits, requests }, null, 2), contentType: 'application/json',
      });
    });
  });
}

test('bundled production records, decodes and saves ASCII video in desktop and phone camera views', async ({ page }, testInfo) => {
  test.setTimeout(45000);
  await installVideoCamera(page);
  for (const viewport of viewports) {
    await page.setViewportSize({ width: viewport.width, height: viewport.height });
    await openApp(page);
    await page.locator('#open-looks').click();
    await page.locator('input[name="look"][value="print"]').check();
    await openPanel(page, 'framing');
    await page.locator('input[name="framing"][value="story"]').check();
    await openPanel(page, 'style');
    await page.locator('input[name="text-tint"][value="#ffd663"]').check();
    await page.locator('#close-look-sheet').click();
    await page.locator('#video-mode').click();
    await page.locator('#camera-shutter').click();
    await expect(page.locator('body')).toHaveAttribute('data-recording-state', 'recording');
    await expect(page.locator('#recording-timer')).toHaveText('00:01');
    await page.locator('#camera-shutter').click();
    await expect(page.locator('#video-review')).toBeVisible();
    const video = page.locator('#recorded-video');
    await video.evaluate((element) => element.play());
    await expect.poll(() => video.evaluate((element) => element.currentTime)).toBeGreaterThan(0);
    const saved = await downloadVideo(page, testInfo, viewport.name);
    expect(saved.streams).toHaveLength(1);
    expect(saved.streams[0]).toMatchObject({ codec_type: 'video', codec_name: 'h264' });
    expect(saved.streams[0].height).toBeGreaterThan(saved.streams[0].width);
    expect(Math.max(saved.streams[0].width, saved.streams[0].height)).toBeLessThanOrEqual(1280);
    expect(Number(saved.format.duration)).toBeGreaterThan(0.5);
    const url = await video.getAttribute('src');
    await page.locator('#close-video-review').click();
    await openPanel(page, 'camera');
    await page.locator('#studio-video').click();
    await expect(page.locator('#look-sheet')).toBeHidden();
    await page.locator('#review-video').click();
    await expect(video).toHaveAttribute('src', url);
    expect(await page.evaluate(() => ({
      microphoneRequested: window.__videoQA.requests.some((request) => request.audio !== false),
      cameraEnded: window.__videoQA.cameras.every((stream) => stream.getTracks().every((track) => track.readyState === 'ended')),
      captureEnded: window.__videoQA.captures.every(({ stream }) => stream.getTracks().every((track) => track.readyState === 'ended')),
    }))).toEqual({ microphoneRequested: false, cameraEnded: true, captureEnded: true });
  }
});
