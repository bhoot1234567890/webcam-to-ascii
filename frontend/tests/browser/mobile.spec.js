import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { test, expect, ROOT, IMAGES, openApp, readCommit, changeSettings } from './fixtures.js';

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
  const dir = path.join(ROOT, 'qa/artifacts/clack', testInfo.project.name);
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
  await openApp(page, { layout: 'camera' });
  await capture(page, testInfo, 'iphone-14-plus-empty', {});
  await expect(page.locator('.source-card')).toBeHidden();
  for (const id of ['astronaut', 'astronaut-portrait', 'astronaut-landscape']) {
    const commit = await choosePhoto(page, id);
    await completeOverview(page);
    const shape = await geometry(page);
    expect(shape.art.bottom, 'Entire artwork appears in the initial phone viewport').toBeLessThanOrEqual(shape.viewport.height);
    expect(shape.region.bottom - shape.region.top, 'Viewfinder is the primary surface, not a collapsed card').toBeGreaterThan(shape.viewport.height / 2);
    for (const selector of ['#camera-shutter', '#camera-import', '#flip-camera', '#open-studio',
      '#camera-stop', '#toggle-guides', '#fit-output', '#detail-output', '#live-mode', '#still-mode']) {
      const bounds = await page.locator(selector).boundingBox();
      expect(bounds.width, `${selector} touch width`).toBeGreaterThanOrEqual(44);
      expect(bounds.height, `${selector} touch height`).toBeGreaterThanOrEqual(44);
    }
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
    const brand = await page.locator('.camera-brand').boundingBox();
    const guides = await page.locator('#toggle-guides').boundingBox();
    const actions = await page.locator('.camera-toolbar-actions').boundingBox();
    expect(brand.x, 'Brand stays clear of the grid control').toBeGreaterThanOrEqual(guides.x + guides.width);
    expect(brand.x + brand.width, 'Brand stays clear of Stop and Studio controls').toBeLessThanOrEqual(actions.x);
    expect(await readCommit(page)).toEqual(before);
    await capture(page, testInfo, `iphone-14-plus-${name}`, { emulatedViewport: viewport });
  }
});

test('detail reaches the last glyph, Fit restores the whole result, and tuning refits without changing source', async ({ page }, testInfo) => {
  await openApp(page, { layout: 'camera' });
  const initial = await choosePhoto(page, 'astronaut-portrait');
  await page.locator('#detail-output').tap();
  await expect(page.locator('#detail-output')).toHaveAttribute('aria-pressed', 'true');
  const chrome = await page.evaluate(() => {
    const viewport = document.querySelector('.ascii-viewport').getBoundingClientRect();
    const status = document.querySelector('.output-status').getBoundingClientRect();
    const controls = document.querySelector('.view-controls').getBoundingClientRect();
    return { viewportTop: viewport.top, viewportBottom: viewport.bottom, statusBottom: status.bottom, controlsTop: controls.top };
  });
  expect(chrome.statusBottom, 'Status never obscures the scrolling glyphs').toBeLessThanOrEqual(chrome.viewportTop);
  expect(chrome.controlsTop, 'Zoom controls never obscure the scrolling glyphs').toBeGreaterThanOrEqual(chrome.viewportBottom);
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
  await page.locator('#open-studio').tap();
  const tuned = await changeSettings(page, { maxHeightLines: 80 });
  expect(tuned.sourceGeneration).toBe(initial.sourceGeneration);
  expect(tuned.rows).toBe(80);
  expect(tuned.columns).toBe(80);
  await completeOverview(page);
  await page.locator('#settings-panel > summary').tap();
  await page.locator('#open-camera').tap();
  await capture(page, testInfo, 'iphone-14-plus-tuned', {});
  await page.locator('#open-studio').tap();
  await page.locator('#camera-options > summary').tap();
  for (const selector of ['#camera-facing', '#camera-select']) {
    const bounds = await page.locator(selector).boundingBox();
    expect(bounds.x).toBeGreaterThanOrEqual(0);
    expect(bounds.x + bounds.width).toBeLessThanOrEqual(428);
    expect(bounds.height).toBeGreaterThanOrEqual(44);
  }
  expect(await readCommit(page)).toEqual(tuned);
  await page.locator('#open-camera').tap();
});
