import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { test, expect, ROOT, IMAGES, openApp, openPanel, readCommit, changeSettings, uploadFixture } from './fixtures.js';
import { installVideoCamera } from './video-fixtures.js';

async function geometry(page) {
  return page.locator('.ascii-viewport').evaluate((viewport) => {
    const output = viewport.querySelector('pre');
    const box = viewport.getBoundingClientRect();
    const art = output.getBoundingClientRect();
    return {
      viewport: { width: innerWidth, height: innerHeight, dpr: devicePixelRatio },
      region: { left: box.left, top: box.top, right: box.right, bottom: box.bottom },
      art: { left: art.left, top: art.top, right: art.right, bottom: art.bottom },
      overflowX: viewport.scrollWidth - viewport.clientWidth,
      overflowY: viewport.scrollHeight - viewport.clientHeight,
      pageOverflow: document.documentElement.scrollWidth - innerWidth,
    };
  });
}

async function completeOverview(page) {
  await expect.poll(async () => {
    const { art, region, overflowX, overflowY, pageOverflow } = await geometry(page);
    return art.left >= region.left && art.right <= region.right &&
      art.top >= region.top && art.bottom <= region.bottom &&
      overflowX <= 1 && overflowY <= 1 && pageOverflow <= 1;
  }, { message: 'Fit contains the complete proportional grid without page or region overflow' }).toBe(true);
}

async function choosePhoto(page, id) {
  const fixture = IMAGES.fixtures.find((entry) => entry.id === id);
  const before = await readCommit(page);
  const [picker] = await Promise.all([
    page.waitForEvent('filechooser'),
    page.locator('#camera-import').tap(),
  ]);
  await picker.setFiles(path.join(ROOT, 'qa/fixtures/images', fixture.path));
  await page.waitForFunction(({ generation, name }) => {
    const data = document.querySelector('#ascii-output').dataset;
    return Number(data.sourceGeneration) > generation && data.sourceKind === 'image' && data.sourceName === name;
  }, { generation: before.sourceGeneration, name: path.basename(fixture.path) });
  return readCommit(page);
}

async function capture(page, testInfo, name, evidence) {
  const dir = path.join(ROOT, 'qa/artifacts/camera-features', testInfo.project.name);
  await mkdir(dir, { recursive: true });
  await page.evaluate(() => scrollTo(0, 0));
  const file = path.join(dir, `${name}.png`);
  await page.screenshot({ path: file });
  await testInfo.attach(name, { path: file, contentType: 'image/png' });
  const commit = await readCommit(page);
  const { text, ...identity } = commit;
  await writeFile(path.join(dir, `${name}.json`), JSON.stringify({
    ...evidence, identity, geometry: await geometry(page),
    runtime: await page.evaluate(() => ({ userAgent: navigator.userAgent, touchPoints: navigator.maxTouchPoints })),
    limitation: 'Desktop WebKit iPhone emulation, not physical iPhone hardware or Safari camera validation',
  }, null, 2));
}

test('camera view imports complete photographs without permission, with touch controls and rotation', async ({ page }, testInfo) => {
  await page.route('**/*.css', async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 250));
    await route.fallback();
  });
  await openApp(page);
  await capture(page, testInfo, 'iphone-14-plus-empty', {});
  await expect(page.locator('.rail')).toBeVisible();
  for (const id of ['astronaut', 'astronaut-portrait', 'astronaut-landscape']) {
    const commit = await choosePhoto(page, id);
    await completeOverview(page);
    const shape = await geometry(page);
    expect(shape.art.bottom, 'Entire artwork appears in the initial phone viewport').toBeLessThanOrEqual(shape.viewport.height);
    expect(shape.region.bottom - shape.region.top, 'Viewfinder is the primary surface, not a collapsed card').toBeGreaterThan(shape.viewport.height / 2);
    for (const selector of ['#camera-shutter', '#camera-import', '#toggle-guides',
      '#fit-output', '#photo-mode', '#video-mode', '#open-looks', '#open-grid', '#open-chars']) {
      const bounds = await page.locator(selector).boundingBox();
      expect(bounds.width, `${selector} touch width`).toBeGreaterThanOrEqual(44);
      expect(bounds.height, `${selector} touch height`).toBeGreaterThanOrEqual(44);
    }
    await expect(page.locator('#camera-stop')).toBeHidden();
    await capture(page, testInfo, `iphone-14-plus-${id}`, {});
    expect(await readCommit(page)).toEqual(commit);
  }
  const before = await readCommit(page);
  await page.locator('#toggle-guides').tap();
  await expect(page.locator('.composition-grid')).toBeVisible();
  await capture(page, testInfo, 'iphone-14-plus-guides', {});
  expect(await readCommit(page)).toEqual(before);
  await page.locator('#toggle-guides').tap();
  await expect(page.locator('.composition-grid')).toBeHidden();
  for (const [name, viewport] of [
    ['landscape', { width: 832, height: 378 }],
    ['full-height', { width: 428, height: 926 }],
    ['narrow', { width: 320, height: 568 }],
    ['minimum-width', { width: 280, height: 568 }],
  ]) {
    await page.setViewportSize(viewport);
    await completeOverview(page);
    const brand = await page.locator('.wordmark').boundingBox();
    const guides = await page.locator('#toggle-guides').boundingBox();
    expect(brand.x + brand.width, 'Brand stays clear of the Guides control').toBeLessThanOrEqual(guides.x);
    expect(await readCommit(page)).toEqual(before);
    await capture(page, testInfo, `iphone-14-plus-${name}`, { emulatedViewport: viewport });
  }
});

test('Fit keeps the whole grid contained and tuning refits without changing source', async ({ page }, testInfo) => {
  await openApp(page);
  const initial = await choosePhoto(page, 'astronaut-portrait');
  await completeOverview(page);
  const geometryBefore = await geometry(page);
  expect(geometryBefore.overflowX, 'Fit never scrolls horizontally').toBeLessThanOrEqual(1);
  expect(geometryBefore.overflowY, 'Fit never scrolls vertically').toBeLessThanOrEqual(1);
  const chrome = await page.evaluate(() => {
    const viewport = document.querySelector('.ascii-viewport').getBoundingClientRect();
    const status = document.querySelector('.statusline').getBoundingClientRect();
    const rail = document.querySelector('.rail').getBoundingClientRect();
    return { viewportTop: viewport.top, viewportBottom: viewport.bottom, statusBottom: status.bottom, railTop: rail.top };
  });
  expect(chrome.statusBottom, 'Status never obscures the glyphs').toBeLessThanOrEqual(chrome.viewportTop);
  expect(chrome.railTop, 'Controls never obscure the glyphs').toBeGreaterThanOrEqual(chrome.viewportBottom);
  await capture(page, testInfo, 'iphone-14-plus-fit', {});
  const tuned = await changeSettings(page, { maxHeightLines: 80 });
  expect(tuned.sourceGeneration).toBe(initial.sourceGeneration);
  expect(tuned.rows).toBe(80);
  expect(tuned.columns).toBe(80);
  await completeOverview(page);
  await page.locator('#close-look-sheet').tap();
  await capture(page, testInfo, 'iphone-14-plus-tuned', {});
  await openPanel(page, 'camera');
  for (const selector of ['#camera-facing', '#camera-select']) {
    const bounds = await page.locator(selector).boundingBox();
    expect(bounds.x).toBeGreaterThanOrEqual(0);
    expect(bounds.x + bounds.width).toBeLessThanOrEqual(428);
    expect(bounds.height).toBeGreaterThanOrEqual(44);
  }
  expect(await readCommit(page)).toEqual(tuned);
  await page.locator('#close-look-sheet').tap();
});

async function selectProcessing(page, name, value) {
  const before = await readCommit(page);
  await page.locator(`input[name="${name}"][value="${value}"]`).check();
  await expect.poll(async () => (await readCommit(page)).renderId).toBeGreaterThan(before.renderId);
  return readCommit(page);
}

async function haloGeometry(page) {
  return page.locator('.ascii-stage').evaluate((stage) => {
    const art = stage.querySelector('pre').getBoundingClientRect();
    const box = stage.getBoundingClientRect();
    const guides = stage.querySelector('.composition-grid').getBoundingClientRect();
    const region = stage.parentElement.getBoundingClientRect();
    return {
      left: art.left - box.left, right: box.right - art.right,
      top: art.top - box.top, bottom: box.bottom - art.bottom,
      guides: { left: guides.left - art.left, top: guides.top - art.top,
        width: guides.width - art.width, height: guides.height - art.height },
      contained: box.left >= region.left && box.right <= region.right &&
        box.top >= region.top && box.bottom <= region.bottom,
    };
  });
}

test('product defaults open on Photo with Neon tritone, tone ceilings and fit-only view', async ({ page }) => {
  await openApp(page);
  await expect(page.locator('#look-label')).toHaveText('Photo');
  await expect(page.locator('#chars-label')).toHaveText('Dense');
  await expect(page.locator('#open-ink')).toContainText('Neon');
  await expect(page.locator('body')).toHaveAttribute('data-glow', 'on');
  await expect(page.locator('#fit-output')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('#detail-output')).toHaveCount(0);
  await page.locator('#open-looks').tap();
  await expect(page.locator('input[name="look"][value="photo"]')).toBeChecked();
  await expect(page.locator('#look-auto-contrast')).toBeChecked();
  await expect(page.locator('#look-midtone')).toHaveValue('2');
  await expect(page.locator('#look-sharpness')).toHaveValue('1.5');
  await expect(page.locator('#look-adjusted')).toBeVisible();
  await page.locator('#tab-style').tap();
  await expect(page.locator('input[name="char-set"][value="dense"]')).toBeChecked();
  await expect(page.locator('input[name="text-tint"][value="neon"]')).toBeChecked();
  await expect(page.locator('#soft-glow')).toBeChecked();
  await expect(page.locator('#text-color-value')).toHaveText('#A855F7');
  await page.locator('#tab-window').tap();
  await expect(page.locator('input[name="export-frame"][value="win95"]')).toBeChecked();
  await page.locator('#close-look-sheet').tap();
});

test('character set menu changes the committed ramp and survives reload', async ({ page }) => {
  await openApp(page);
  await choosePhoto(page, 'astronaut');
  const before = await readCommit(page);
  await page.locator('#open-chars').tap();
  await expect(page.locator('#panel-style')).toBeVisible();
  await expect(page.locator('input[name="char-set"][value="dense"]')).toBeChecked();
  await page.locator('input[name="char-set"][value="minimal"]').check();
  await expect.poll(async () => (await readCommit(page)).text).not.toBe(before.text);
  const commit = await readCommit(page);
  expect([...commit.text.replaceAll('\n', '')].every((glyph) => ' .-+*#'.includes(glyph))).toBe(true);
  await expect(page.locator('#chars-label')).toHaveText('Minimal');
  await page.reload();
  await expect(page.locator('#chars-label')).toHaveText('Minimal');
  await expect(page.locator('#look-label')).toHaveText('Photo');
});

test('tapping the artwork cycles crops and announces the frame', async ({ page }) => {
  await openApp(page);
  await choosePhoto(page, 'astronaut');
  await expect(page.locator('#open-framing')).toHaveCount(0);
  const framing = () => page.locator('#ascii-output').evaluate((output) => output.dataset.framing);
  await expect.poll(framing).toBe('full');
  await page.locator('.ascii-viewport').click();
  await expect.poll(framing).toBe('square');
  await expect(page.locator('#status')).toHaveText('Crop 1:1');
  await expect(page.locator('#grid-label')).toHaveText(/^GRID \d+ × \d+$/);
  await page.locator('.ascii-viewport').click();
  await expect.poll(framing).toBe('portrait');
  await expect(page.locator('#status')).toHaveText('Crop 3:4');
});

test('video mode keeps consistent mode chips and fits the rail at 368x784', async ({ page }) => {
  await page.setViewportSize({ width: 368, height: 784 });
  await openApp(page);
  await choosePhoto(page, 'astronaut');
  await page.locator('#video-mode').tap();
  await expect(page.locator('#shutter-help')).toHaveText('');
  await expect(page.locator('#shutter-help')).toBeHidden();
  const layout = await page.evaluate(() => {
    const rect = (selector) => {
      const box = document.querySelector(selector).getBoundingClientRect();
      return { width: box.width, top: box.top, bottom: box.bottom };
    };
    return {
      modes: rect('.capture-modes'),
      photo: rect('#photo-mode'),
      video: rect('#video-mode'),
      rail: rect('.rail'),
      meta: rect('.readout-meta'),
      doc: document.documentElement.scrollHeight,
      inner: innerHeight,
    };
  });
  expect(layout.doc).toBe(layout.inner);
  expect(layout.rail.bottom).toBeLessThanOrEqual(layout.inner);
  expect(layout.meta.bottom).toBeLessThanOrEqual(layout.inner);
  expect(layout.modes.width).toBeGreaterThan(300);
  expect(layout.photo.width).toBeCloseTo(layout.video.width, 0);
  await expect(page.locator('#status')).toHaveText('Video mode — press RECORD to begin.');
});

test('shared sheet applies complete recipes, restores focus, and keeps edge settings for shade-only looks', async ({ page }) => {
  await openApp(page);
  const original = await uploadFixture(page, 'astronaut');
  await page.locator('#open-looks').tap();
  await expect(page.locator('#look-sheet')).toBeVisible();
  await expect(page.locator('input[name="look"][value="original"]')).toBeFocused();
  const photo = await selectProcessing(page, 'look', 'photo');
  expect(photo.text).not.toBe(original.text);
  await page.locator('#tab-looks').tap();
  let before = await readCommit(page);
  await page.locator('#look-auto-contrast').check();
  await expect.poll(async () => (await readCommit(page)).renderId).toBeGreaterThan(before.renderId);
  before = await readCommit(page);
  await page.locator('#look-midtone').focus();
  await page.locator('#look-midtone').press('Home');
  await expect.poll(async () => (await readCommit(page)).renderId).toBeGreaterThan(before.renderId);
  before = await readCommit(page);
  await page.locator('#look-sharpness').focus();
  await page.locator('#look-sharpness').press('End');
  await expect.poll(async () => (await readCommit(page)).renderId).toBeGreaterThan(before.renderId);
  before = await readCommit(page);
  await page.locator('#look-invert').check();
  await expect.poll(async () => (await readCommit(page)).renderId).toBeGreaterThan(before.renderId);
  await page.locator('#tab-looks').tap();
  await expect(page.locator('#look-adjusted')).toBeVisible();
  // Reapplying the same selected recipe must reset, not leave an adjusted Photo selected.
  before = await readCommit(page);
  await page.locator('input[name="look"][value="photo"]').click();
  await expect.poll(async () => (await readCommit(page)).renderId).toBeGreaterThan(before.renderId);
  expect((await readCommit(page)).text).toBe(photo.text);
  await expect(page.locator('#look-auto-contrast')).not.toBeChecked();
  await expect(page.locator('#look-midtone')).toHaveValue('1');
  await expect(page.locator('#look-sharpness')).toHaveValue('0');
  await expect(page.locator('#look-invert')).not.toBeChecked();
  await expect(page.locator('#look-adjusted')).toBeHidden();
  await page.locator('#close-look-sheet').tap();
  await expect(page.locator('#open-looks')).toBeFocused();
  await page.locator('#open-looks').tap();
  await page.locator('#tab-tune').tap();
  for (const [selector, value] of [['#sigma1', '1'], ['#sigma2', '2'], ['#edge-threshold', '90']]) {
    await expect(page.locator(selector)).toBeDisabled();
    await expect(page.locator(selector)).toHaveValue(value);
  }
  await expect(page.locator('#edge-controls-help')).toBeVisible();
  await page.locator('#tab-looks').tap();
  await expect(page.locator('input[name="look"][value="photo"]')).toBeChecked();
  const reset = await selectProcessing(page, 'look', 'original');
  expect(reset.text).toBe(original.text);
  await page.keyboard.press('Escape');
  await expect(page.locator('#look-sheet')).toBeHidden();
  await expect(page.locator('#open-looks')).toBeFocused();
  for (const selector of ['#sigma1', '#sigma2', '#edge-threshold']) await expect(page.locator(selector)).toBeEnabled();
  await expect(page.locator('#edge-controls-help')).toBeHidden();
});

test('framed artwork, guide bounds and glow survive Fit and narrow rotation', async ({ page }, testInfo) => {
  await openApp(page);
  await choosePhoto(page, 'astronaut-landscape');
  await openPanel(page, 'framing');
  await expect(page.locator('input[name="framing"][value="full"]')).toBeChecked();
  const framed = await selectProcessing(page, 'framing', 'story');
  expect(framed.columns / (framed.rows * 2)).toBeCloseTo(9 / 16, 1);
  await expect(page.locator('#ascii-output')).toHaveAttribute('data-framing', 'story');
  const originalViewport = page.viewportSize();
  await page.locator('#tab-style').tap();
  const before = await readCommit(page);
  await page.locator('input[name="text-tint"][value="#ffd663"]').check();
  await page.locator('#soft-glow').check();
  await page.locator('#close-look-sheet').tap();
  await expect(page.locator('#ascii-output')).toHaveCSS('color', 'rgb(255, 214, 99)');
  await page.locator('#toggle-guides').tap();
  for (const [name, viewport] of [
    ['minimum', { width: 280, height: 568 }], ['narrow', { width: 320, height: 568 }],
    ['phone', { width: 428, height: 926 }], ['landscape', { width: 832, height: 378 }],
  ]) {
    await page.setViewportSize(viewport);
    await completeOverview(page);
    await expect.poll(async () => (await haloGeometry(page)).contained).toBe(true);
    const halo = await haloGeometry(page);
    for (const side of ['left', 'right', 'top', 'bottom']) expect(halo[side], `${name} ${side} halo`).toBeGreaterThanOrEqual(7.9);
    for (const delta of Object.values(halo.guides)) expect(Math.abs(delta), 'Guides follow artwork, not viewport letterboxing').toBeLessThan(0.1);
    for (const selector of ['#open-chars', '#open-looks', '#fit-output']) {
      const bounds = await page.locator(selector).boundingBox();
      expect(bounds.width).toBeGreaterThanOrEqual(44);
      expect(bounds.height).toBeGreaterThanOrEqual(44);
      expect(bounds.x).toBeGreaterThanOrEqual(0);
      expect(bounds.x + bounds.width).toBeLessThanOrEqual(viewport.width);
    }
    await openPanel(page, 'framing');
    await expect(page.locator('input[name="framing"][value="story"]')).toBeChecked();
    await expect.poll(() => page.locator('.ascii-viewport').evaluate((viewport) => viewport.clientHeight),
      { message: 'Artwork pane keeps a usable floor with the drawer open at every rotation' }).toBeGreaterThanOrEqual(140);
    for (const selector of ['#panel-framing .choice', '.tab']) {
      for (const target of await page.locator(selector).all()) {
        const bounds = await target.boundingBox();
        expect(bounds.width, `${selector} target width`).toBeGreaterThanOrEqual(44);
        expect(bounds.height, `${selector} target height`).toBeGreaterThanOrEqual(44);
      }
    }
    await page.locator('#tab-looks').tap();
    for (const selector of ['#panel-looks .check', '#panel-looks input[type="range"]']) {
      for (const target of await page.locator(selector).all()) {
        const bounds = await target.boundingBox();
        expect(bounds.height, `${selector} target height`).toBeGreaterThanOrEqual(44);
      }
    }
    await page.locator('#tab-tune').tap();
    for (const selector of ['#panel-tune input[type="number"]']) {
      for (const target of await page.locator(selector).all()) {
        const bounds = await target.boundingBox();
        expect(bounds.height, `${selector} target height`).toBeGreaterThanOrEqual(44);
      }
    }
    await page.locator('#tab-style').tap();
    for (const selector of ['#panel-style .choice', '#text-color', '#panel-style .check']) {
      for (const target of await page.locator(selector).all()) {
        const bounds = await target.boundingBox();
        expect(bounds.height, `${selector} target height`).toBeGreaterThanOrEqual(44);
      }
    }
    expect(await page.locator('#look-sheet').evaluate((sheet) => sheet.scrollWidth <= sheet.clientWidth)).toBe(true);
    await page.locator('#close-look-sheet').tap();
    await completeOverview(page);
    expect(await readCommit(page)).toEqual(before);
    await capture(page, testInfo, `looks-glow-${name}`, { emulatedViewport: viewport });
  }
  await page.setViewportSize(originalViewport);
  await completeOverview(page);
  // Baseline with glow ON at the settled viewport; emulated WebKit can re-resolve
  // svh across viewport round-trips, so wait for the layout to stop moving first.
  await expect.poll(async () => {
    const first = await geometry(page);
    await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const second = await geometry(page);
    return Math.abs(second.art.top - first.art.top) < 0.5 && Math.abs(second.art.left - first.art.left) < 0.5;
  }, { message: 'Viewport geometry settles after the rotation round-trip' }).toBe(true);
  const glowOnGeometry = await geometry(page);
  await page.locator('#open-looks').tap();
  await page.locator('#tab-style').tap();
  await page.locator('#soft-glow').uncheck();
  await page.locator('#close-look-sheet').tap();
  await completeOverview(page);
  expect((await haloGeometry(page)).left).toBe(0);
  expect((await haloGeometry(page)).top).toBe(0);
  const restoredGeometry = await geometry(page);
  for (const edge of ['left', 'right', 'top', 'bottom']) {
    expect(Math.abs(restoredGeometry.art[edge] - glowOnGeometry.art[edge]), 'Disabling glow changes only the 8px halo gutter').toBeLessThanOrEqual(9);
  }
  expect(await readCommit(page)).toEqual(before);
});

const VIDEO_VIEWPORTS = [
  { width: 280, height: 568 }, { width: 320, height: 568 },
  { width: 428, height: 926 }, { width: 832, height: 378 },
];

async function trackCameraRequests(page) {
  await page.addInitScript(() => {
    window.__cameraRequests = 0;
    const devices = navigator.mediaDevices;
    if (!devices?.getUserMedia) return;
    const getUserMedia = devices.getUserMedia.bind(devices);
    devices.getUserMedia = (...args) => {
      window.__cameraRequests += 1;
      return getUserMedia(...args);
    };
  });
}

async function containedTouchTargets(page, selectors, viewport) {
  for (const selector of selectors) {
    const box = await page.locator(selector).boundingBox();
    expect(box, `${selector} is visible`).not.toBeNull();
    expect(box.width, `${selector} touch width`).toBeGreaterThanOrEqual(44);
    expect(box.height, `${selector} touch height`).toBeGreaterThanOrEqual(44);
    expect(box.x, `${selector} left edge`).toBeGreaterThanOrEqual(0);
    expect(box.x + box.width, `${selector} right edge`).toBeLessThanOrEqual(viewport.width);
    expect(box.y, `${selector} top edge`).toBeGreaterThanOrEqual(0);
    expect(box.y + box.height, `${selector} bottom edge`).toBeLessThanOrEqual(viewport.height);
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
}

test('Photo and Video modes keep the mobile camera geometry and do not request permission until the shutter', async ({ page }) => {
  await trackCameraRequests(page);
  await openApp(page);
  await expect(page.locator('#photo-mode')).toHaveAttribute('aria-pressed', 'true');
  for (const viewport of VIDEO_VIEWPORTS) {
    await page.setViewportSize(viewport);
    await page.locator('#photo-mode').tap();
    const photoBounds = await page.locator('.ascii-viewport').boundingBox();
    const photoColor = await page.locator('#camera-shutter').evaluate((button) => getComputedStyle(button).color);
    await page.locator('#video-mode').tap();
    await expect(page.locator('#video-mode')).toHaveAttribute('aria-pressed', 'true');
    await expect(page.locator('#photo-mode')).toHaveAttribute('aria-pressed', 'false');
    await expect(page.locator('#camera-shutter')).toHaveAttribute('aria-label', 'Start ASCII recording');
    await expect(page.locator('#camera-shutter')).not.toHaveCSS('color', photoColor);
    await expect(page.locator('#recording-indicator')).toBeHidden();
    await expect(page.locator('#review-video')).toBeHidden();
    await containedTouchTargets(page, ['#photo-mode', '#video-mode', '#camera-shutter', '#camera-import', '#copy-output', '#save-output', '#fit-output'], viewport);
    const videoBounds = await page.locator('.ascii-viewport').boundingBox();
    expect(videoBounds, 'Video uses the existing viewfinder, not a smaller surface').toEqual(photoBounds);
    await expect(page.locator('#flip-camera')).toBeHidden();
    const tileOverflow = await page.evaluate(() => {
      const offenders = [];
      for (const tile of document.querySelectorAll('.readout-tiles button')) {
        const value = tile.querySelector('span[id$="-label"]');
        if (!value) continue;
        const tileBox = tile.getBoundingClientRect();
        const valueBox = value.getBoundingClientRect();
        if (valueBox.right > tileBox.right + 0.5 || valueBox.left < tileBox.left - 0.5) offenders.push(tile.id);
      }
      return offenders;
    });
    expect(tileOverflow, 'readout tile values stay inside their tiles').toEqual([]);
    await openPanel(page, 'camera');
    await page.locator('#studio-video').scrollIntoViewIfNeeded();
    await containedTouchTargets(page, ['#studio-video'], viewport);
    await page.locator('#close-look-sheet').tap();
    await page.locator('#photo-mode').tap();
    await expect(page.locator('#camera-shutter')).toBeEnabled();
  }
  await openPanel(page, 'camera');
  await page.locator('#studio-video').tap();
  await expect(page.locator('#look-sheet')).toBeHidden();
  const shutter = page.locator('#camera-shutter');
  await expect(await shutter.isEnabled() ? shutter : page.locator('#photo-mode')).toBeFocused();
  await expect(page.locator('#video-mode')).toHaveAttribute('aria-pressed', 'true');
  expect(await page.evaluate(() => window.__cameraRequests)).toBe(0);
});

test('missing recording capability explains Video while Photo and local imports remain usable at every mobile size', async ({ page }) => {
  await trackCameraRequests(page);
  await page.addInitScript(() => {
    // Remove the actual encoder capability; do not simulate a successful recording.
    Object.defineProperty(window, 'MediaRecorder', { configurable: true, value: undefined });
  });
  await openApp(page);
  for (const viewport of VIDEO_VIEWPORTS) {
    await page.setViewportSize(viewport);
    await page.locator('#video-mode').tap();
    await expect(page.locator('#camera-shutter')).toBeDisabled();
    await expect(page.locator('#photo-mode')).toBeEnabled();
    await expect(page.locator('#camera-shutter')).toHaveAttribute('aria-describedby', 'shutter-help');
    await expect(page.locator('#shutter-help')).toBeVisible();
    await containedTouchTargets(page, ['#photo-mode', '#video-mode', '#camera-shutter', '#camera-import'], viewport);
    await page.locator('#photo-mode').tap();
    await expect(page.locator('#camera-shutter')).toBeEnabled();
    await expect(page.locator('#camera-import')).toBeEnabled();
  }
  const photo = await choosePhoto(page, 'astronaut');
  await completeOverview(page);
  await page.locator('#video-mode').tap();
  await expect(page.locator('#camera-shutter')).toBeDisabled();
  expect(await readCommit(page)).toEqual(photo);
  await page.locator('#photo-mode').tap();
  await expect(page.locator('#camera-shutter')).toBeEnabled();
  expect(await page.evaluate(() => window.__cameraRequests)).toBe(0);
});

test('a real locally encoded ASCII clip plays and reviews without overflow across mobile sizes', { tag: '@recording' }, async ({ page }) => {
  await installVideoCamera(page);
  await openApp(page);
  await page.locator('#video-mode').tap();
  await page.locator('#camera-shutter').tap();
  await expect(page.locator('body')).toHaveAttribute('data-recording-state', 'recording');
  await expect(page.locator('#camera-shutter')).toHaveAttribute('aria-label', 'Stop recording');
  await expect(page.locator('#recording-indicator')).toBeVisible();
  await expect(page.locator('#recording-timer')).toHaveText(/00:0[1-9]/);
  await page.locator('#camera-shutter').tap();
  await expect(page.locator('#video-review')).toBeVisible();
  await expect.poll(() => page.locator('#recorded-video').evaluate((video) => ({
    width: video.videoWidth, height: video.videoHeight,
    playable: video.readyState >= HTMLMediaElement.HAVE_METADATA,
    duration: Number.isFinite(video.duration) && video.duration > 0,
  }))).toMatchObject({ playable: true, duration: true });
  const dimensions = await page.locator('#recorded-video').evaluate((video) => [video.videoWidth, video.videoHeight]);
  expect(Math.min(...dimensions)).toBeGreaterThan(0);
  await page.locator('#recorded-video').evaluate((video) => video.play());
  await expect.poll(() => page.locator('#recorded-video').evaluate((video) => video.currentTime)).toBeGreaterThan(0);
  await expect(page.locator('#video-save')).toBeEnabled();
  for (const viewport of VIDEO_VIEWPORTS) {
    await page.setViewportSize(viewport);
    expect(await page.locator('#video-review').evaluate((dialog) => dialog.scrollWidth <= dialog.clientWidth)).toBe(true);
    expect(await page.locator('.video-review-body').evaluate((body) => body.scrollWidth <= body.clientWidth)).toBe(true);
    await containedTouchTargets(page, ['#close-video-review'], viewport);
    for (const selector of ['#video-save', '#video-share', '#video-retake']) {
      await page.locator(selector).scrollIntoViewIfNeeded();
      await containedTouchTargets(page, [selector], viewport);
    }
  }
  await page.locator('#close-video-review').tap();
  await expect(page.locator('#video-review')).toBeHidden();
  await expect(page.locator('#camera-shutter')).toBeFocused();
  expect(await page.locator('#recorded-video').evaluate((video) => video.paused)).toBe(true);
  const source = await page.locator('#recorded-video').getAttribute('src');
  await page.locator('#review-video').tap();
  await expect(page.locator('#video-review')).toBeVisible();
  await expect(page.locator('#recorded-video')).toHaveAttribute('src', source);
  await page.locator('#close-video-review').tap();
  await expect(page.locator('#review-video')).toBeFocused();
  await page.locator('#review-video').tap();
  await page.locator('#video-retake').tap();
  await expect(page.locator('#video-review')).toBeHidden();
  await expect(page.locator('#review-video')).toBeHidden();
});
