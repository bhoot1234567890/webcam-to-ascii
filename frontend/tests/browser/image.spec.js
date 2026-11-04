import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { basename, resolve } from 'node:path';
import { promisify } from 'node:util';
import {
  test, expect, ROOT, IMAGES, REFERENCE, readCommit, uploadFixture,
  changeSettings, openApp, captureApp, readReferenceText,
} from './fixtures.js';

const runFile = promisify(execFile);
const ink = new Map(JSON.parse(readFileSync(resolve(ROOT, 'ink-usage.json'), 'utf8')));
const viewports = [
  { name: 'desktop', width: 1440, height: 1000 },
  { name: 'mobile', width: 390, height: 844 },
];
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

function referenceFor(id, profile = 'default') {
  const entry = REFERENCE.fullImages.find((item) => item.fixtureId === id && item.profile === profile);
  expect(entry, `Required Python reference ${profile}:${id}`).toBeTruthy();
  return entry;
}

function completeGrid(commit, reference) {
  expect(commit.columns).toBe(reference.width);
  expect(commit.rows).toBe(reference.height);
  const rows = commit.text.split('\n');
  expect(rows).toHaveLength(reference.height);
  for (const [row, text] of rows.entries()) {
    expect(text.length, `Complete unwrapped row ${row}`).toBe(reference.width);
  }
  expect(commit.text.endsWith('\n')).toBe(false);
  return rows;
}

async function previewState(page, fixture) {
  const preview = page.locator('#image-preview');
  await expect(preview).toBeVisible();
  await expect.poll(() => preview.evaluate((image) => image.complete && image.naturalWidth)).toBe(fixture.width);
  const state = await preview.evaluate((image) => ({
    src: image.getAttribute('src'), width: image.naturalWidth, height: image.naturalHeight,
    objectFit: getComputedStyle(image).objectFit,
  }));
  expect(state.src).toMatch(/^blob:/);
  expect(state.width).toBe(fixture.width);
  expect(state.height).toBe(fixture.height);
  expect(state.objectFit).toBe('contain');
  return state;
}

// Numerical parity has its own equal-stage-input gates. Here sampling differences are
// recorded, not mislabeled as byte-identical Pillow/Canvas output.
function observation(actual, expected) {
  let mismatches = 0;
  const centers = (text) => {
    const rows = text.split('\n');
    let mass = 0;
    let xMass = 0;
    let yMass = 0;
    for (let y = 0; y < rows.length; y++) {
      for (let x = 0; x < rows[y].length; x++) {
        const value = ink.get(rows[y][x]);
        if (value === undefined) throw new Error(`Unknown rendered ink glyph ${JSON.stringify(rows[y][x])}`);
        mass += value;
        xMass += value * (x + 0.5) / rows[y].length;
        yMass += value * (y + 0.5) / rows.length;
      }
    }
    return mass ? { x: xMass / mass, y: yMass / mass } : null;
  };
  for (let i = 0; i < actual.length; i++) if (actual[i] !== expected[i]) mismatches++;
  return { differingCharacters: mismatches, comparedCharacters: actual.length,
    mismatchFraction: mismatches / actual.length,
    actualInkCenter: centers(actual), referenceInkCenter: centers(expected) };
}

async function renderingTurn(page) {
  await page.evaluate(() => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done))));
}

async function accessibleControls(page) {
  for (const id of ['settings-panel', 'camera-options']) {
    if (!await page.locator(`#${id}`).evaluate((panel) => panel.open)) {
      await page.locator(`#${id} > summary`).click();
    }
  }
  for (const id of ['start-camera', 'stop-camera', 'camera-select', 'image-input',
    'max-width', 'max-height', 'sigma1', 'sigma2', 'edge-threshold']) {
    const control = page.locator(`#${id}`);
    await control.scrollIntoViewIfNeeded();
    await expect(control).toBeVisible();
    const state = await control.evaluate((element) => {
      const rect = element.getBoundingClientRect();
      return { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom,
        width: innerWidth, height: innerHeight,
        named: Boolean(element.textContent.trim() || element.labels?.length || element.getAttribute('aria-label')) };
    });
    expect(state.named, `${id} has an accessible label`).toBe(true);
    expect(state.left, `${id} left edge`).toBeGreaterThanOrEqual(0);
    expect(state.right, `${id} right edge`).toBeLessThanOrEqual(state.width + 1);
    expect(state.bottom).toBeGreaterThan(0);
    expect(state.top).toBeLessThan(state.height);
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1),
    'Only the ASCII region, not the document, scrolls horizontally').toBe(true);
}

async function inspectScrollableGrid(page, commit) {
  await page.locator('#detail-output').click();
  const region = page.locator('.ascii-viewport');
  await region.scrollIntoViewIfNeeded();
  await region.focus();
  const geometry = await region.evaluate((viewport) => {
    const pre = viewport.querySelector('pre');
    const style = getComputedStyle(pre);
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    // CSSOM's font shorthand may be empty or include line-height syntax that
    // Canvas rejects, silently leaving its default 10px sans-serif font.
    context.font = `${style.fontStyle} ${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
    const characterWidth = context.measureText('M').width;
    viewport.scrollLeft = viewport.scrollWidth;
    viewport.scrollTop = viewport.scrollHeight;
    const node = pre.firstChild;
    const range = document.createRange();
    range.setStart(node, node.length - 1);
    range.setEnd(node, node.length);
    const last = range.getBoundingClientRect();
    const box = viewport.getBoundingClientRect();
    return { whiteSpace: style.whiteSpace, wrap: style.overflowWrap,
      preWidth: pre.getBoundingClientRect().width, preHeight: pre.getBoundingClientRect().height,
      characterWidth, lineHeight: Number.parseFloat(style.lineHeight),
      scrollLeft: viewport.scrollLeft, scrollTop: viewport.scrollTop,
      maxLeft: viewport.scrollWidth - viewport.clientWidth, maxTop: viewport.scrollHeight - viewport.clientHeight,
      lastLeft: last.left, lastRight: last.right, lastTop: last.top, lastBottom: last.bottom,
      left: box.left + viewport.clientLeft, right: box.left + viewport.clientLeft + viewport.clientWidth,
      top: box.top + viewport.clientTop, bottom: box.top + viewport.clientTop + viewport.clientHeight };
  });
  expect(geometry.whiteSpace).toBe('pre');
  expect(geometry.wrap).toBe('normal');
  expect(Math.abs(geometry.preWidth - commit.columns * geometry.characterWidth)).toBeLessThan(2);
  expect(Math.abs(geometry.preHeight - commit.rows * geometry.lineHeight)).toBeLessThan(2);
  expect(geometry.maxLeft).toBeGreaterThan(0);
  expect(geometry.maxTop).toBeGreaterThan(0);
  expect(Math.abs(geometry.scrollLeft - geometry.maxLeft)).toBeLessThanOrEqual(1);
  expect(Math.abs(geometry.scrollTop - geometry.maxTop)).toBeLessThanOrEqual(1);
  expect(geometry.lastLeft).toBeGreaterThanOrEqual(geometry.left - 1);
  expect(geometry.lastRight).toBeLessThanOrEqual(geometry.right + 1);
  expect(geometry.lastTop).toBeGreaterThanOrEqual(geometry.top - 1);
  expect(geometry.lastBottom).toBeLessThanOrEqual(geometry.bottom + 1);
  expect(await readCommit(page)).toEqual(commit);
  await region.evaluate((viewport) => { viewport.scrollTop = 0; viewport.scrollLeft = 0; });
  await page.locator('#fit-output').click();
}

for (const viewport of viewports) {
  test.describe(`${viewport.name} local images`, () => {
    test.use({ viewport: { width: viewport.width, height: viewport.height } });

    test('all canonical PNGs commit real grids and preserve original previews', async ({ page }, testInfo) => {
      test.setTimeout(180_000);
      await openApp(page);
      expect(IMAGES.fixtures).toHaveLength(16);
      const evidence = [];
      for (const fixture of IMAGES.fixtures) {
        await test.step(fixture.id, async () => {
          const before = await readCommit(page);
          const commit = await uploadFixture(page, fixture.id);
          expect(commit.renderId).toBeGreaterThan(before.renderId);
          expect(commit.sourceGeneration).toBeGreaterThan(before.sourceGeneration);
          expect(commit.sourceKind).toBe('image');
          expect(commit.sourceName).toBe(basename(fixture.path));
          expect(commit.settingsKey).toBe(JSON.stringify(REFERENCE.profiles.default));
          const reference = referenceFor(fixture.id);
          const rows = completeGrid(commit, reference);
          const preview = await previewState(page, fixture);
          const actualHash = sha256(readFileSync(resolve(ROOT, 'qa/fixtures/images', fixture.path)));
          expect(actualHash).toBe(fixture.sha256);
          expect(actualHash).toBe(reference.inputSha256);
          const referenceText = await readReferenceText(reference);
          const shape = observation(commit.text, referenceText);
          if (fixture.id === 'black') expect(rows.every((row) => /^ +$/.test(row))).toBe(true);
          if (fixture.id === 'white') expect(rows.every((row) => /^@+$/.test(row))).toBe(true);
          if (fixture.id.startsWith('step-')) {
            // A gross flip/rotation is not a permitted resampling difference.
            expect(shape.actualInkCenter).not.toBeNull();
            expect(Math.abs(shape.actualInkCenter.x - shape.referenceInkCenter.x)).toBeLessThan(0.04);
            expect(Math.abs(shape.actualInkCenter.y - shape.referenceInkCenter.y)).toBeLessThan(0.04);
          }
          if (fixture.id === 'ramp') {
            const shading = await changeSettings(page, { edgeIndexThreshold: 0 });
            const shadingRows = completeGrid(shading, reference);
            // Canvas resampling can round a transition differently across rows;
            // every complete row must still preserve dark-to-light density order.
            shape.rampShading = { rowCount: shadingRows.length, distinctRows: new Set(shadingRows).size };
            for (const [y, row] of shadingRows.entries()) {
              const densities = [...row].map((glyph) => ink.get(glyph));
              expect(densities[0], `Ramp row ${y} starts dark`).toBe(ink.get(' '));
              expect(densities.at(-1), `Ramp row ${y} ends light`).toBe(ink.get('@'));
              for (let x = 1; x < densities.length; x++) {
                expect(densities[x], `Ramp density row ${y}, column ${x}`).toBeGreaterThanOrEqual(densities[x - 1]);
              }
            }
            await changeSettings(page, { edgeIndexThreshold: 90 });
            expect(await previewState(page, fixture)).toEqual(preview);
          }
          if (fixture.id === 'astronaut-portrait') {
            expect(commit.rows).toBeGreaterThan(commit.columns / 2);
            await inspectScrollableGrid(page, commit);
            await accessibleControls(page);
            await page.evaluate(() => scrollTo(0, 0));
            await captureApp(page, testInfo, `image-${viewport.name}-portrait`);
          }
          if (fixture.id === 'astronaut') {
            await page.evaluate(() => scrollTo(0, 0));
            await captureApp(page, testInfo, `image-${viewport.name}-astronaut`);
          }
          const { text, ...identity } = commit;
          evidence.push({ fixtureId: fixture.id, inputSha256: actualHash, identity,
            outputSha256: sha256(text), preview, referenceId: reference.id, observation: shape });
        });
      }
      await testInfo.attach('canonical-image-commits', { body: JSON.stringify(evidence, null, 2), contentType: 'application/json' });
    });

    test('invalid controls preserve the last valid commit and valid retries include equal/reversed sigmas', async ({ page }) => {
      await openApp(page);
      await uploadFixture(page, 'astronaut');
      if (!await page.locator('#settings-panel').evaluate((panel) => panel.open)) {
        await page.locator('#settings-panel > summary').click();
      }
      for (const invalid of [
        { id: 'max-width', field: 'maxWidthChars', bad: '0', good: 60 },
        { id: 'max-height', field: 'maxHeightLines', bad: '201', good: 80 },
        { id: 'sigma1', field: 'sigma1', bad: '0', good: 1.5 },
        { id: 'sigma2', field: 'sigma2', bad: '5.1', good: 2.5 },
        { id: 'edge-threshold', field: 'edgeIndexThreshold', bad: '90.5', good: 95 },
        { id: 'edge-threshold', field: 'edgeIndexThreshold', bad: '', good: 90 },
      ]) {
        const before = await readCommit(page);
        const preview = await page.locator('#image-preview').getAttribute('src');
        await page.locator(`#${invalid.id}`).fill(invalid.bad);
        await page.locator(`#${invalid.id}`).dispatchEvent('change');
        await expect(page.locator('#error')).toContainText('Invalid settings');
        await expect(page.locator(`#${invalid.id}`)).toHaveAttribute('aria-invalid', 'true');
        await renderingTurn(page);
        expect(await readCommit(page)).toEqual(before);
        const retry = await changeSettings(page, { [invalid.field]: invalid.good });
        expect(retry.renderId).toBeGreaterThan(before.renderId);
        expect(retry.sourceGeneration).toBe(before.sourceGeneration);
        expect(retry.settingsKey).not.toBe(before.settingsKey);
        expect(await page.locator('#image-preview').getAttribute('src')).toBe(preview);
        await expect(page.locator('#error')).toBeHidden();
        await expect(page.locator(`#${invalid.id}`)).toHaveAttribute('aria-invalid', 'false');
      }
      const equal = await changeSettings(page, { sigma1: 2, sigma2: 2 });
      const withoutEdges = await changeSettings(page, { edgeIndexThreshold: 0 });
      expect(withoutEdges.text).toBe(equal.text);
      const reversed = await changeSettings(page, { sigma1: 3, sigma2: 1, edgeIndexThreshold: 90 });
      expect(JSON.parse(reversed.settingsKey)).toMatchObject({ sigma1: 3, sigma2: 1 });
      expect(reversed.sourceGeneration).toBe(equal.sourceGeneration);
      completeGrid(reversed, referenceFor('astronaut', 'comparison'));
    });

    test('corrupt, zero-grid and empty uploads cannot replace a valid source', async ({ page }, testInfo) => {
      await openApp(page);
      const fixture = IMAGES.fixtures.find((item) => item.id === 'astronaut-portrait');
      const before = await uploadFixture(page, fixture.id);
      const preview = await previewState(page, fixture);
      await page.locator('#image-input').setInputFiles({
        name: 'corrupt.png', mimeType: 'image/png', buffer: Buffer.from('not a PNG'),
      });
      await expect(page.locator('#error')).toHaveText('Unable to read this image');
      await renderingTurn(page);
      expect(await readCommit(page)).toEqual(before);
      expect(await previewState(page, fixture)).toEqual(preview);

      expect(process.env.QA_PYTHON, 'QA_PYTHON must select the locked Pillow environment').toBeTruthy();
      const extremePath = testInfo.outputPath('extreme-aspect.png');
      await runFile(process.env.QA_PYTHON, ['-c',
        'from PIL import Image; import sys; Image.new("RGB", (1, 4096), "white").save(sys.argv[1])', extremePath]);
      await page.locator('#image-input').setInputFiles(extremePath);
      await expect(page.locator('#error')).toHaveText('Image aspect ratio is too extreme for this resolution');
      await renderingTurn(page);
      expect(await readCommit(page)).toEqual(before);
      expect(await previewState(page, fixture)).toEqual(preview);

      await page.locator('#image-input').setInputFiles([]);
      await renderingTurn(page);
      expect(await readCommit(page)).toEqual(before);
      expect(await previewState(page, fixture)).toEqual(preview);
      const retry = await uploadFixture(page, 'white');
      expect(retry.renderId).toBeGreaterThan(before.renderId);
      expect(retry.sourceGeneration).toBeGreaterThan(before.sourceGeneration);
      await expect(page.locator('#error')).toBeHidden();
      completeGrid(retry, referenceFor('white'));
    });
  });
}
