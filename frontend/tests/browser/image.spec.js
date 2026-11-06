import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { basename, resolve } from 'node:path';
import { promisify } from 'node:util';
import {
  test, expect, ROOT, IMAGES, REFERENCE, readCommit, uploadFixture,
  changeSettings, openApp, openPanel, captureApp, readReferenceText,
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
  await openPanel(page, 'camera');
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
  const panels = {
    camera: ['camera-select', 'camera-facing'],
    grid: ['grid-max-width', 'grid-max-height'],
    tune: ['sigma1', 'sigma2', 'edge-threshold'],
  };
  const check = async (id) => {
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
  };
  for (const id of ['camera-shutter', 'camera-import']) await check(id);
  for (const [tab, ids] of Object.entries(panels)) {
    await openPanel(page, tab);
    for (const id of ids) await check(id);
  }
  await expect(page.locator('#image-input')).toBeEnabled();
  await page.locator('#close-look-sheet').click();
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
            const detailed = await changeSettings(page, { maxWidthChars: 150, maxHeightLines: 200 });
            await inspectScrollableGrid(page, detailed);
            await changeSettings(page, { maxWidthChars: 50, maxHeightLines: 65 });
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
      await openPanel(page, 'tune');
      // Grid sliders are range-bounded; only the numeric tune fields can go invalid.
      for (const invalid of [
        { id: 'sigma1', field: 'sigma1', bad: '0', good: 1.5 },
        { id: 'sigma2', field: 'sigma2', bad: '5.1', good: 2.5 },
        { id: 'edge-threshold', field: 'edgeIndexThreshold', bad: '90.5', good: 95 },
        { id: 'edge-threshold', field: 'edgeIndexThreshold', bad: '', good: 90 },
      ]) {
        const before = await readCommit(page);
        const preview = await page.locator('#image-preview').getAttribute('src');
        await page.locator(`#${invalid.id}`).fill(invalid.bad);
        await page.locator(`#${invalid.id}`).dispatchEvent('change');
        await expect(page.locator(`#${invalid.id}-error`)).toBeVisible();
        await expect(page.locator('#error')).toBeHidden();
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
        await expect(page.locator(`#${invalid.id}-error`)).toBeHidden();
      }
      const equal = await changeSettings(page, { maxWidthChars: 60, maxHeightLines: 80, sigma1: 2, sigma2: 2 });
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
      await expect(page.locator('#error-text')).toHaveText('Unable to read this image');
      await renderingTurn(page);
      expect(await readCommit(page)).toEqual(before);
      expect(await previewState(page, fixture)).toEqual(preview);

      expect(process.env.QA_PYTHON, 'QA_PYTHON must select the locked Pillow environment').toBeTruthy();
      const extremePath = testInfo.outputPath('extreme-aspect.png');
      await runFile(process.env.QA_PYTHON, ['-c',
        'from PIL import Image; import sys; Image.new("RGB", (1, 4096), "white").save(sys.argv[1])', extremePath]);
      await page.locator('#image-input').setInputFiles(extremePath);
      await expect(page.locator('#error-text')).toHaveText('Image aspect ratio is too extreme for this resolution');
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

test('curated looks preserve source identity and Original restores the exact prior text', async ({ page }) => {
  await openApp(page);
  const original = await uploadFixture(page, 'astronaut');
  const preview = await page.locator('#image-preview').getAttribute('src');
  for (const id of ['photo', 'print', 'outline', 'original']) {
    const before = await readCommit(page);
    await page.locator('#open-looks').click();
    await page.locator(`input[name="look"][value="${id}"]`).check();
    await page.locator('#close-look-sheet').click();
    await page.waitForFunction(({ id, before }) => {
      const data = document.querySelector('#ascii-output').dataset;
      return Number(data.renderId) > before && JSON.parse(data.lookKey).id === id;
    }, { id, before: before.renderId });
    const commit = await readCommit(page);
    expect(commit.sourceGeneration).toBe(original.sourceGeneration);
    expect(commit.settingsKey).toBe(original.settingsKey);
    expect([commit.columns, commit.rows]).toEqual([original.columns, original.rows]);
    expect(await page.locator('#image-preview').getAttribute('src')).toBe(preview);
    if (id === 'original') expect(commit.text).toBe(original.text);
    else expect(commit.text).not.toBe(original.text);
    if (id === 'outline') expect(commit.text.replace(/[ /\\|\n-]/g, '')).toBe('');
  }
  // Outline must remove a uniform interior, not quietly reuse photographic shading.
  await uploadFixture(page, 'midgray');
  await page.locator('#open-looks').click();
  await page.locator('input[name="look"][value="outline"]').check();
  await page.locator('#close-look-sheet').click();
  await page.waitForFunction(() => JSON.parse(document.querySelector('#ascii-output').dataset.lookKey).id === 'outline');
  expect((await readCommit(page)).text.replace(/[ \n]/g, '')).toBe('');
});

test('framing crops native center pixels once and Full restores the unmodified source', async ({ page }) => {
  await openApp(page);
  const image = await page.evaluate(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 320;
    canvas.height = 160;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, 320, 160);
    ctx.fillStyle = 'white';
    ctx.fillRect(80, 0, 160, 160);
    return canvas.toDataURL('image/png').split(',')[1];
  });
  await page.locator('#image-input').setInputFiles({ name: 'center-landmark.png', mimeType: 'image/png', buffer: Buffer.from(image, 'base64') });
  await page.waitForFunction(() => document.querySelector('#ascii-output').dataset.sourceName === 'center-landmark.png');
  const original = await readCommit(page);
  const preview = await page.locator('#image-preview').getAttribute('src');
  expect(original.text.split('\n')[0].startsWith(' ')).toBe(true);
  for (const frame of ['square', 'portrait', 'story', 'full']) {
    const before = await readCommit(page);
    await page.locator('#open-framing').click();
    await page.locator(`input[name="framing"][value="${frame}"]`).check();
    await page.locator('#close-look-sheet').click();
    await page.waitForFunction(({ frame, before }) => {
      const data = document.querySelector('#ascii-output').dataset;
      return data.framing === frame && Number(data.renderId) > before;
    }, { frame, before: before.renderId });
    const cropped = await readCommit(page);
    expect(cropped.sourceGeneration).toBe(original.sourceGeneration);
    if (frame !== 'full') {
      expect(cropped.text.split('\n').every((row) => /^@+$/.test(row))).toBe(true);
    } else expect(cropped.text).toBe(original.text);
    expect(await page.locator('#image-preview').getAttribute('src')).toBe(preview);
    expect(await page.locator('#image-preview').evaluate((img) => [img.naturalWidth, img.naturalHeight])).toEqual([320, 160]);
  }
});

test('a tiny source rejects unrepresentable framing and keeps its last valid configuration', async ({ page }) => {
  await openApp(page);
  const image = await page.evaluate(() => {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 2;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, 2, 2);
    return canvas.toDataURL('image/png').split(',')[1];
  });
  await page.locator('#image-input').setInputFiles({ name: 'tiny.png', mimeType: 'image/png', buffer: Buffer.from(image, 'base64') });
  await page.waitForFunction(() => document.querySelector('#ascii-output').dataset.sourceName === 'tiny.png');
  const before = await readCommit(page);
  await page.locator('#open-framing').click();
  await page.locator('input[name="framing"][value="story"]').click();
  await expect(page.locator('input[name="framing"][value="full"]')).toBeChecked();
  await page.locator('#close-look-sheet').click();
  await expect(page.locator('#error-text')).toContainText('too small');
  await renderingTurn(page);
  expect(await readCommit(page)).toEqual(before);
  await expect(page.locator('#ascii-output')).toHaveAttribute('data-framing', 'full');
});

test('empty state offers both start paths and retires after the first commit', async ({ page }) => {
  await openApp(page);
  await expect(page.locator('#empty-state')).toBeVisible();
  await expect(page.locator('#empty-start')).toBeVisible();
  await expect(page.locator('#empty-photo')).toBeVisible();
  await expect(page.locator('#empty-hint')).toBeHidden();
  await expect(page.locator('#grid-label')).toBeHidden();
  const entry = IMAGES.fixtures.find((item) => item.id === 'astronaut');
  const [picker] = await Promise.all([
    page.waitForEvent('filechooser'),
    page.locator('#empty-photo').click(),
  ]);
  await picker.setFiles(resolve(ROOT, 'qa/fixtures/images', entry.path));
  await page.waitForFunction((name) => {
    const data = document.querySelector('#ascii-output').dataset;
    return Number(data.renderId) > 0 && data.sourceKind === 'image' && data.sourceName === name;
  }, basename(entry.path));
  await expect(page.locator('#empty-state')).toBeHidden();
  await expect(page.locator('#ascii-output')).toHaveAttribute('role', 'img');
  await expect(page.locator('#ascii-output')).toHaveAttribute('aria-label', /ASCII art, \d+ by \d+ characters/);
});

test('COPY places the committed ASCII text on the clipboard', async ({ page, context }) => {
  await openApp(page);
  const commit = await uploadFixture(page, 'astronaut');
  const chromium = context.browser().browserType().name() === 'chromium';
  if (chromium) await context.grantPermissions(['clipboard-read', 'clipboard-write'], { origin: 'http://127.0.0.1:4173' });
  await page.locator('#copy-output').click();
  await expect(page.locator('#status')).toHaveText('Copied. Paste it anywhere.');
  if (chromium) expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(commit.text);
});

test('SAVE downloads a PNG of the committed frame', async ({ page }) => {
  await openApp(page);
  await uploadFixture(page, 'astronaut');
  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.locator('#save-output').click(),
  ]);
  expect(download.suggestedFilename()).toMatch(/^clack-\d{14}\.png$/);
  const bytes = readFileSync(await download.path());
  expect(bytes.byteLength).toBeGreaterThan(0);
  expect(bytes.subarray(0, 8)).toEqual(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]));
});

test('SOURCE holds the original image over the ASCII while pressed', async ({ page }) => {
  await openApp(page);
  await uploadFixture(page, 'astronaut-portrait');
  const peek = page.locator('#source-peek');
  await expect(peek).toBeHidden();
  await page.locator('#compare-output').focus();
  await page.keyboard.down(' ');
  await expect(peek).toBeVisible();
  await expect(page.locator('#compare-output')).toHaveAttribute('aria-pressed', 'true');
  const [width, height] = await peek.evaluate((canvas) => [canvas.width, canvas.height]);
  expect(width).toBeGreaterThan(0);
  expect(height).toBeGreaterThan(0);
  await page.keyboard.up(' ');
  await expect(peek).toBeHidden();
  await expect(page.locator('#compare-output')).toHaveAttribute('aria-pressed', 'false');
});

test('look, ink and glow persist across a reload', async ({ page }) => {
  await openApp(page);
  await uploadFixture(page, 'astronaut');
  await page.locator('#open-looks').click();
  await page.locator('input[name="look"][value="photo"]').check();
  await page.locator('#tab-style').click();
  await page.locator('input[name="text-tint"][value="#ffd663"]').check();
  await page.locator('#soft-glow').check();
  await page.locator('#close-look-sheet').click();
  await page.waitForFunction(() => JSON.parse(document.querySelector('#ascii-output').dataset.lookKey).id === 'photo');
  await page.reload();
  await expect(page.locator('#status')).toContainText('Ready');
  await expect(page.locator('#open-looks')).toContainText('Photo');
  await expect(page.locator('#open-ink')).toContainText('Amber');
  await expect(page.locator('#ascii-output')).toHaveCSS('color', 'rgb(255, 214, 99)');
  await page.locator('#open-looks').click();
  await expect(page.locator('input[name="look"][value="photo"]')).toBeChecked();
  await expect(page.locator('input[name="text-tint"][value="#ffd663"]')).toBeChecked();
  await expect(page.locator('#soft-glow')).toBeChecked();
  await page.locator('#close-look-sheet').click();
  const restored = await uploadFixture(page, 'astronaut');
  expect(restored.sourceGeneration).toBeGreaterThan(0);
  await page.waitForFunction(() => JSON.parse(document.querySelector('#ascii-output').dataset.lookKey).id === 'photo');
});

test('edge settings survive a reload', async ({ page }) => {
  await openApp(page);
  await uploadFixture(page, 'astronaut');
  await openPanel(page, 'tune');
  await page.locator('#sigma1').fill('1.5');
  await page.locator('#sigma2').fill('2.5');
  await page.locator('#edge-threshold').fill('70');
  await page.waitForFunction(() => {
    const key = JSON.parse(document.querySelector('#ascii-output').dataset.settingsKey);
    return key.sigma1 === 1.5 && key.sigma2 === 2.5 && key.edgeIndexThreshold === 70;
  });
  await page.reload();
  await expect(page.locator('#status')).toContainText('Ready');
  await openPanel(page, 'tune');
  await expect(page.locator('#sigma1')).toHaveValue('1.5');
  await expect(page.locator('#sigma2')).toHaveValue('2.5');
  await expect(page.locator('#edge-threshold')).toHaveValue('70');
});

test('named presets save, apply, persist, and delete', async ({ page }) => {
  await openApp(page);
  await uploadFixture(page, 'astronaut');
  await openPanel(page, 'looks');
  await page.locator('input[name="look"][value="photo"]').check();
  await page.locator('#look-midtone').evaluate((slider) => {
    slider.value = '1.2';
    slider.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await openPanel(page, 'framing');
  await page.locator('input[name="framing"][value="square"]').check();
  await openPanel(page, 'grid');
  await page.locator('input[name="density"][value="fine"]').check();
  await openPanel(page, 'style');
  await page.locator('input[name="text-tint"][value="#ffd663"]').check();

  await openPanel(page, 'presets');
  await page.locator('#preset-name').fill('Studio');
  await page.locator('#preset-save').click();
  await expect(page.locator('#preset-list')).toContainText('Studio');
  await expect(page.locator('#preset-empty')).toBeHidden();

  await openPanel(page, 'looks');
  await page.locator('input[name="look"][value="print"]').check();
  await openPanel(page, 'framing');
  await page.locator('input[name="framing"][value="wide"]').check();
  await page.waitForFunction(() => JSON.parse(document.querySelector('#ascii-output').dataset.lookKey).id === 'print');

  await openPanel(page, 'presets');
  await page.locator('button[data-preset-apply="Studio"]').click();
  await page.waitForFunction(() => {
    const output = document.querySelector('#ascii-output');
    return JSON.parse(output.dataset.lookKey).id === 'photo' && output.dataset.framing === 'square';
  });
  await expect(page.locator('#open-ink')).toContainText('Amber');
  await expect(page.locator('#ascii-output')).toHaveCSS('color', 'rgb(255, 214, 99)');

  await page.reload();
  await expect(page.locator('#status')).toContainText('Ready');
  await openPanel(page, 'looks');
  await page.locator('input[name="look"][value="print"]').check();
  await openPanel(page, 'presets');
  await expect(page.locator('#preset-list')).toContainText('Studio');
  await page.locator('button[data-preset-apply="Studio"]').click();
  await openPanel(page, 'looks');
  await expect(page.locator('input[name="look"][value="photo"]')).toBeChecked();
  await expect(page.locator('#look-midtone')).toHaveValue('1.2');
  await openPanel(page, 'framing');
  await expect(page.locator('input[name="framing"][value="square"]')).toBeChecked();
  await openPanel(page, 'style');
  await expect(page.locator('input[name="text-tint"][value="#ffd663"]')).toBeChecked();
  await expect(page.locator('#ascii-output')).toHaveCSS('color', 'rgb(255, 214, 99)');
  await page.locator('#close-look-sheet').click();
  const restoredPreset = await uploadFixture(page, 'astronaut');
  expect(JSON.parse(restoredPreset.settingsKey)).toMatchObject({ maxWidthChars: 75, maxHeightLines: 100 });
  await page.waitForFunction(() => {
    const output = document.querySelector('#ascii-output');
    return JSON.parse(output.dataset.lookKey).id === 'photo' && output.dataset.framing === 'square';
  });

  await openPanel(page, 'presets');
  await page.locator('button[data-preset-delete="Studio"]').click();
  await expect(page.locator('#preset-list')).toBeEmpty();
  await expect(page.locator('#preset-empty')).toBeVisible();

  await page.reload();
  await expect(page.locator('#status')).toContainText('Ready');
  await openPanel(page, 'presets');
  await expect(page.locator('#preset-empty')).toBeVisible();
});

test('tabs follow the automatic-activation keyboard pattern', async ({ page }) => {
  await openApp(page);
  await uploadFixture(page, 'astronaut');
  await page.locator('#open-looks').click();
  await page.locator('#tab-looks').focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.locator('#tab-grid')).toHaveAttribute('aria-selected', 'true');
  await expect(page.locator('#tab-grid')).toBeFocused();
  await expect(page.locator('#panel-grid')).toBeVisible();
  await page.keyboard.press('End');
  await expect(page.locator('#tab-presets')).toHaveAttribute('aria-selected', 'true');
  await expect(page.locator('#tab-presets')).toBeFocused();
  await expect(page.locator('#panel-presets')).toBeVisible();
  await page.keyboard.press('Home');
  await expect(page.locator('#tab-camera')).toHaveAttribute('aria-selected', 'true');
  await page.keyboard.press('ArrowLeft');
  await expect(page.locator('#tab-presets')).toHaveAttribute('aria-selected', 'true');
  await expect(page.locator('#tab-presets')).toBeFocused();
});

test('the sheet traps focus, closes on Escape, and restores its opener', async ({ page }) => {
  await openApp(page);
  await uploadFixture(page, 'astronaut');
  await expect(page.locator('#look-sheet')).toHaveAttribute('role', 'dialog');
  await expect(page.locator('#look-sheet')).toHaveAttribute('aria-modal', 'true');
  await page.locator('#open-looks').click();
  for (let index = 0; index < 12; index += 1) await page.keyboard.press('Tab');
  expect(await page.evaluate(() => document.getElementById('look-sheet').contains(document.activeElement))).toBe(true);
  await page.keyboard.press('Escape');
  await expect(page.locator('#look-sheet')).toBeHidden();
  await expect(page.locator('#open-looks')).toBeFocused();
});
