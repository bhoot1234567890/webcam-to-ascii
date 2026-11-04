import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { basename, resolve } from 'node:path';
import {
  test, expect, ROOT, IMAGES, REFERENCE, readCommit, uploadFixture,
  changeSettings, openApp, captureApp,
} from './fixtures.js';

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
  const image = page.locator('#image-preview');
  await expect(image).toBeVisible();
  await expect.poll(() => image.evaluate((element) => element.complete && element.naturalWidth)).toBe(fixture.width);
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
      await openApp(page);
      const scripts = await page.locator('script[src]').evaluateAll((elements) => elements.map((element) => element.getAttribute('src')));
      expect(scripts.length).toBeGreaterThan(0);
      expect(scripts.every((src) => src.startsWith('/assets/') && src.endsWith('.js'))).toBe(true);
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
      if (!await page.locator('#camera-options').evaluate((panel) => panel.open)) {
        await page.locator('#camera-options > summary').click();
      }
      for (const id of ['image-input', 'start-camera', 'stop-camera', 'camera-select',
        'max-width', 'max-height', 'sigma1', 'sigma2', 'edge-threshold']) {
        const control = page.locator(`#${id}`);
        await control.scrollIntoViewIfNeeded();
        await expect(control).toBeVisible();
        const bounds = await control.boundingBox();
        expect(bounds.x).toBeGreaterThanOrEqual(0);
        expect(bounds.x + bounds.width).toBeLessThanOrEqual(viewport.width + 1);
      }
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
      await captureApp(page, testInfo, `production-${viewport.name}-portrait-tuned`);
      expect(await readCommit(page)).toEqual(tuned);
      const { text, ...identity } = tuned;
      commits.push({ fixtureId: 'astronaut-portrait', identity,
        textSha256: createHash('sha256').update(text).digest('hex'), preview: originalPreview });

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
