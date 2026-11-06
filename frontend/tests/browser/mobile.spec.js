import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { test, expect, ROOT, IMAGES, openApp, openPanel, readCommit, changeSettings } from './fixtures.js';

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
      '#fit-output', '#detail-output', '#open-looks', '#open-grid', '#open-framing']) {
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

test('detail reaches the last glyph, Fit restores the whole result, and tuning refits without changing source', async ({ page }, testInfo) => {
  await openApp(page);
  const initial = await choosePhoto(page, 'astronaut-portrait');
  await page.locator('#detail-output').tap();
  await expect(page.locator('#detail-output')).toHaveAttribute('aria-pressed', 'true');
  const chrome = await page.evaluate(() => {
    const viewport = document.querySelector('.ascii-viewport').getBoundingClientRect();
    const status = document.querySelector('.statusline').getBoundingClientRect();
    const rail = document.querySelector('.rail').getBoundingClientRect();
    return { viewportTop: viewport.top, viewportBottom: viewport.bottom, statusBottom: status.bottom, railTop: rail.top };
  });
  expect(chrome.statusBottom, 'Status never obscures the scrolling glyphs').toBeLessThanOrEqual(chrome.viewportTop);
  expect(chrome.railTop, 'Controls never obscure the scrolling glyphs').toBeGreaterThanOrEqual(chrome.viewportBottom);
  const last = await page.locator('.ascii-viewport').evaluate((viewport) => {
    viewport.scrollLeft = viewport.scrollWidth;
    viewport.scrollTop = viewport.scrollHeight;
    const text = viewport.querySelector('pre').firstChild;
    const range = document.createRange();
    range.setStart(text, text.length - 1);
    range.setEnd(text, text.length);
    const glyph = range.getBoundingClientRect();
    const box = viewport.getBoundingClientRect();
    return { left: glyph.left, right: glyph.right, top: glyph.top, bottom: glyph.bottom,
      regionLeft: box.left, regionRight: box.right, regionTop: box.top, regionBottom: box.bottom,
      scrolledX: viewport.scrollLeft, scrolledY: viewport.scrollTop };
  });
  expect(last.scrolledX).toBeGreaterThan(0);
  expect(last.scrolledY).toBeGreaterThan(0);
  expect(last.left).toBeGreaterThanOrEqual(last.regionLeft);
  expect(last.right).toBeLessThanOrEqual(last.regionRight);
  expect(last.top).toBeGreaterThanOrEqual(last.regionTop);
  expect(last.bottom).toBeLessThanOrEqual(last.regionBottom);
  await page.locator('.ascii-viewport').evaluate((viewport) => {
    viewport.scrollLeft = (viewport.scrollWidth - viewport.clientWidth) / 2;
    viewport.scrollTop = (viewport.scrollHeight - viewport.clientHeight) / 2;
  });
  await capture(page, testInfo, 'iphone-14-plus-detail', {});
  await page.locator('#fit-output').tap();
  await completeOverview(page);
  expect(await readCommit(page)).toEqual(initial);
  const tuned = await changeSettings(page, { maxHeightLines: 80 });
  expect(tuned.sourceGeneration).toBe(initial.sourceGeneration);
  expect(tuned.rows).toBe(80);
  expect(tuned.columns).toBe(80);
  await completeOverview(page);
  await page.locator('#close-look-sheet').tap();
  await capture(page, testInfo, 'iphone-14-plus-tuned', {});
  await openPanel(page, 'camera');
  for (const selector of ['#camera-facing', '#camera-select', '#flip-camera']) {
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

test('shared sheet applies complete recipes, restores focus, and keeps edge settings for shade-only looks', async ({ page }) => {
  await openApp(page);
  const original = await choosePhoto(page, 'astronaut');
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

test('framed artwork, guide bounds and glow survive Fit, last-glyph Detail and narrow rotation', async ({ page }, testInfo) => {
  await openApp(page);
  await choosePhoto(page, 'astronaut-landscape');
  await page.locator('#open-framing').tap();
  await expect(page.locator('input[name="framing"][value="full"]')).toBeFocused();
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
    for (const selector of ['#open-framing', '#open-looks', '#fit-output', '#detail-output']) {
      const bounds = await page.locator(selector).boundingBox();
      expect(bounds.width).toBeGreaterThanOrEqual(44);
      expect(bounds.height).toBeGreaterThanOrEqual(44);
      expect(bounds.x).toBeGreaterThanOrEqual(0);
      expect(bounds.x + bounds.width).toBeLessThanOrEqual(viewport.width);
    }
    await page.locator('#open-framing').tap();
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
    await page.locator('#detail-output').tap();
    const last = await page.locator('.ascii-viewport').evaluate((viewport) => {
      viewport.scrollLeft = viewport.scrollWidth;
      viewport.scrollTop = viewport.scrollHeight;
      const text = viewport.querySelector('pre').firstChild;
      const range = document.createRange();
      range.setStart(text, text.length - 1);
      range.setEnd(text, text.length);
      const glyph = range.getBoundingClientRect();
      const region = viewport.getBoundingClientRect();
      const stage = viewport.querySelector('.ascii-stage').getBoundingClientRect();
      return { visible: glyph.left >= region.left && glyph.right <= region.right &&
        glyph.top >= region.top && glyph.bottom <= region.bottom,
        rightHalo: stage.right - glyph.right, bottomHalo: stage.bottom - glyph.bottom };
    });
    expect(last.visible, `${name} Detail reaches last glyph`).toBe(true);
    expect(last.rightHalo).toBeGreaterThanOrEqual(7.9);
    expect(last.bottomHalo).toBeGreaterThanOrEqual(7.9);
    await page.locator('#fit-output').tap();
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
