import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { basename, join, resolve, relative, isAbsolute } from 'node:path';
import {
  test, expect, ROOT, IMAGES, REFERENCE, openApp, uploadFixture,
  changeSettings, readCommit, readReferenceText,
} from './fixtures.js';

const canonicalIds = [
  'black', 'midgray', 'white', 'ramp', 'step-horizontal', 'step-vertical',
  'step-diagonal-down', 'step-diagonal-up', 'impulse-center', 'impulse-corner',
  'checker4', 'noise', 'rgb-patches', 'astronaut', 'astronaut-portrait', 'astronaut-landscape',
];
const matrix = [
  ...canonicalIds.map((id) => ['comparison', id]),
  ...['astronaut', 'astronaut-portrait', 'step-diagonal-down', 'step-diagonal-up'].map((id) => ['default', id]),
];
const centers = [[0.5, 0.25], [0.5, 0.5], [0.5, 0.75], [0.1, 0.5], [0.9, 0.5]];
const sha256 = (data) => createHash('sha256').update(data).digest('hex');
const imageRoot = resolve(ROOT, 'qa/fixtures/images');
const json = (value) => `${JSON.stringify(value, null, 2)}\n`;

async function checkedImage(fixture) {
  if (!fixture || !canonicalIds.includes(fixture.id) || typeof fixture.path !== 'string' || isAbsolute(fixture.path)) {
    throw new Error('Unknown or malformed canonical image');
  }
  const path = resolve(imageRoot, fixture.path);
  const inside = relative(imageRoot, path);
  if (inside.startsWith('..') || isAbsolute(inside)) throw new Error(`Fixture escapes image root: ${fixture.id}`);
  const bytes = await readFile(path);
  expect(sha256(bytes), `${fixture.id} source SHA-256`).toBe(fixture.sha256);
  expect(bytes.length).toBe(fixture.bytes);
  expect(bytes.subarray(0, 8).toString('hex')).toBe('89504e470d0a1a0a');
  expect(bytes.readUInt32BE(16)).toBe(fixture.width);
  expect(bytes.readUInt32BE(20)).toBe(fixture.height);
  return bytes;
}

function metricsFor(left, right, columns, rows) {
  const a = left.split('\n');
  const b = right.split('\n');
  const perRow = Array(rows).fill(0);
  const perColumn = Array(columns).fill(0);
  let mismatchCount = 0;
  let firstMismatch = null;
  const leftHistogram = {};
  const rightHistogram = {};
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < columns; x++) {
      const l = a[y][x];
      const r = b[y][x];
      leftHistogram[l] = (leftHistogram[l] ?? 0) + 1;
      rightHistogram[r] = (rightHistogram[r] ?? 0) + 1;
      if (l !== r) {
        mismatchCount++;
        perRow[y]++;
        perColumn[x]++;
        firstMismatch ??= { x, y, leftCode: l.charCodeAt(0), rightCode: r.charCodeAt(0) };
      }
    }
  }
  return {
    schemaVersion: 1, columns, rows, cells: columns * rows, mismatchCount,
    mismatchFraction: mismatchCount / (columns * rows),
    equalFraction: 1 - mismatchCount / (columns * rows), firstMismatch,
    perRow, perColumn, leftHistogram, rightHistogram,
    interpretation: 'Symmetric raw character comparison. Canvas and Pillow resampling differ; mismatch is reported, not a quality golden or exact-parity assertion.',
  };
}

for (const [profile, fixtureId] of matrix) {
  test(`${profile} / ${fixtureId} complete blinded evidence`, async ({ page, browser }, testInfo) => {
    test.setTimeout(90_000);
    const fixture = IMAGES.fixtures.find((entry) => entry.id === fixtureId);
    const source = await checkedImage(fixture);
    const reference = REFERENCE.fullImages.find((entry) => entry.id === `${profile}:${fixtureId}`);
    expect(reference, 'Complete independent Python reference entry').toBeTruthy();
    const pythonText = await readReferenceText(reference);
    const settings = REFERENCE.profiles[profile];
    expect(reference.settings).toEqual(settings);
    await openApp(page);
    let commit = await uploadFixture(page, fixtureId);
    if (profile === 'comparison') commit = await changeSettings(page, settings);
    // Read all observable fields atomically from the actual worker/UI commit, never inject an answer into the app.
    const stableCommit = await readCommit(page);
    expect(stableCommit).toEqual(commit);
    expect(commit.sourceKind).toBe('image');
    expect(commit.sourceName).toBe(basename(fixture.path));
    expect(commit.settingsKey).toBe(JSON.stringify(settings));
    expect(commit.columns).toBe(reference.width);
    expect(commit.rows).toBe(reference.height);
    const candidateText = commit.text;
    for (const text of [pythonText, candidateText]) {
      const rows = text.split('\n');
      expect(rows).toHaveLength(reference.height);
      expect(rows.every((row) => row.length === reference.width && !/[^\x20-\x7e]/.test(row))).toBe(true);
    }

    const permutationHash = sha256(`20261104:${profile}:${fixtureId}`);
    const pythonSide = (parseInt(permutationHash.slice(0, 2), 16) & 1) === 0 ? 'left' : 'right';
    const leftText = pythonSide === 'left' ? pythonText : candidateText;
    const rightText = pythonSide === 'right' ? pythonText : candidateText;
    const directory = join(ROOT, 'qa/artifacts/ab', testInfo.project.name, profile, fixtureId);
    await mkdir(directory, { recursive: true });
    const files = {};
    const save = async (name, data) => {
      const bytes = Buffer.isBuffer(data) ? data : Buffer.from(data);
      await writeFile(join(directory, name), bytes);
      files[name] = { path: name, bytes: bytes.length, sha256: sha256(bytes) };
      if (name.endsWith('.png')) {
        expect(bytes.subarray(0, 8).toString('hex')).toBe('89504e470d0a1a0a');
        files[name].width = bytes.readUInt32BE(16);
        files[name].height = bytes.readUInt32BE(20);
      }
      return bytes;
    };
    await save('source.png', source);
    await save('left.txt', leftText);
    await save('right.txt', rightText);
    await save('answer-key.json', json({
      schemaVersion: 1, profile, fixtureId, seed: '20261104', permutationSha256: permutationHash,
      python: pythonSide, candidate: pythonSide === 'left' ? 'right' : 'left',
    }));
    const metrics = metricsFor(leftText, rightText, reference.width, reference.height);
    await save('metrics.json', json(metrics));

    await page.route('**/__qa/images/**', async (route) => {
      const url = new URL(route.request().url());
      const match = /^\/__qa\/images\/([a-z0-9-]+)\.png$/.exec(url.pathname);
      const requested = match && IMAGES.fixtures.find((entry) => entry.id === match[1]);
      if (route.request().method() !== 'GET' || url.search || !requested || !canonicalIds.includes(requested.id)) {
        await route.abort('blockedbyclient');
        return;
      }
      await route.fulfill({ status: 200, contentType: 'image/png', body: await checkedImage(requested) });
    });
    await page.goto('/tests/ab.html');
    await page.waitForFunction(() => typeof window.setComparison === 'function');
    await page.evaluate(async ({ fixtureId, leftText, rightText }) => {
      await window.setComparison({ sourceUrl: `/__qa/images/${fixtureId}.png`, leftText, rightText });
    }, { fixtureId, leftText, rightText });
    const geometry = await page.evaluate(() => {
      const measure = (id) => {
        const element = document.querySelector(id);
        const box = element.getBoundingClientRect();
        const style = getComputedStyle(element);
        return {
          width: box.width, height: box.height, scrollWidth: element.scrollWidth, scrollHeight: element.scrollHeight,
          fontFamily: style.fontFamily, fontSize: style.fontSize, lineHeight: style.lineHeight,
          bottomGutterPx: parseFloat(style.paddingBottom),
          whiteSpace: style.whiteSpace, overflowWrap: style.overflowWrap, color: style.color,
          background: style.backgroundColor, transform: style.transform, zoom: style.zoom,
        };
      };
      return { left: measure('#left-grid'), right: measure('#right-grid'), pair: measure('#comparison'), devicePixelRatio };
    });
    expect(geometry.left).toEqual(geometry.right);
    expect(geometry.left.fontFamily).toContain('Roboto Mono');
    expect(geometry.left.fontSize).toBe('12px');
    expect(geometry.left.whiteSpace).toBe('pre');
    expect(geometry.left.transform).toBe('none');
    expect(geometry.devicePixelRatio).toBe(1);
    expect(geometry.left.bottomGutterPx).toBe(2);
    expect(Math.abs(geometry.left.height - geometry.left.bottomGutterPx - parseFloat(geometry.left.lineHeight) * reference.height)).toBeLessThan(1);
    expect(geometry.left.scrollWidth).toBeLessThanOrEqual(Math.ceil(geometry.left.width));
    expect(geometry.left.scrollHeight).toBeLessThanOrEqual(Math.ceil(geometry.left.height));
    await page.setViewportSize({ width: Math.ceil(geometry.pair.width + 48), height: 1000 });
    const leftPng = await save('left.png', await page.locator('#left-grid').screenshot({ animations: 'disabled', scale: 'css' }));
    const rightPng = await save('right.png', await page.locator('#right-grid').screenshot({ animations: 'disabled', scale: 'css' }));
    await save('pair.png', await page.locator('#comparison').screenshot({ animations: 'disabled', scale: 'css' }));
    expect(files['left.png'].width).toBe(files['right.png'].width);
    expect(files['left.png'].height).toBe(files['right.png'].height);
    for (const name of ['left.png', 'right.png', 'pair.png']) {
      await testInfo.attach(name, { path: join(directory, name), contentType: 'image/png' });
    }

    const visualizations = await page.evaluate(async ({ left, right, centers, leftText, rightText, columns, rows }) => {
      const image = async (base64) => {
        const value = new Image();
        const bytes = Uint8Array.from(atob(base64), (character) => character.charCodeAt(0));
        const url = URL.createObjectURL(new Blob([bytes], { type: 'image/png' }));
        value.src = url;
        try { await value.decode(); } finally { URL.revokeObjectURL(url); }
        return value;
      };
      const l = await image(left);
      const r = await image(right);
      const png = (canvas) => canvas.toDataURL('image/png').split(',')[1];
      const canvas = (width, height) => {
        const value = document.createElement('canvas');
        value.width = width;
        value.height = height;
        return value;
      };
      const crops = centers.map(([cx, cy], index) => {
        const width = Math.min(512, l.width);
        const height = Math.min(512, l.height);
        const x = Math.max(0, Math.min(l.width - width, Math.round(cx * l.width - width / 2)));
        const y = Math.max(0, Math.min(l.height - height, Math.round(cy * l.height - height / 2)));
        const leftCrop = canvas(width, height);
        const rightCrop = canvas(width, height);
        leftCrop.getContext('2d').drawImage(l, x, y, width, height, 0, 0, width, height);
        rightCrop.getContext('2d').drawImage(r, x, y, width, height, 0, 0, width, height);
        const pair = canvas(width * 2 + 16, height);
        const context = pair.getContext('2d');
        context.fillStyle = '#10171d';
        context.fillRect(0, 0, pair.width, pair.height);
        context.drawImage(leftCrop, 0, 0);
        context.drawImage(rightCrop, width + 16, 0);
        return { index: index + 1, center: [cx, cy], box: { x, y, width, height }, left: png(leftCrop), right: png(rightCrop), pair: png(pair) };
      });
      const heatmap = canvas(columns * 4, rows * 4);
      const context = heatmap.getContext('2d');
      const a = leftText.split('\n');
      const b = rightText.split('\n');
      for (let y = 0; y < rows; y++) for (let x = 0; x < columns; x++) {
        context.fillStyle = a[y][x] === b[y][x] ? '#10171d' : '#ff8666';
        context.fillRect(x * 4, y * 4, 4, 4);
      }
      return { crops, heatmap: png(heatmap) };
    }, {
      left: leftPng.toString('base64'), right: rightPng.toString('base64'),
      centers: profile === 'default' ? centers : [], leftText, rightText,
      columns: reference.width, rows: reference.height,
    });
    await save('heatmap.png', Buffer.from(visualizations.heatmap, 'base64'));
    const crops = [];
    for (const crop of visualizations.crops) {
      const paths = {};
      for (const side of ['left', 'right', 'pair']) {
        const name = `crop-${crop.index}-${side}.png`;
        await save(name, Buffer.from(crop[side], 'base64'));
        paths[side] = name;
      }
      crops.push({ index: crop.index, center: crop.center, box: crop.box, files: paths });
      await testInfo.attach(`crop-${crop.index}-pair.png`, { path: join(directory, paths.pair), contentType: 'image/png' });
    }
    const { text: _text, ...identity } = commit;
    await writeFile(join(directory, 'metadata.json'), json({
      schemaVersion: 1, project: testInfo.project.name, profile, fixtureId,
      createdAt: new Date().toISOString(), browserVersion: browser.version(),
      source: { file: 'source.png', fixturePath: fixture.path, sha256: fixture.sha256, width: fixture.width, height: fixture.height },
      settings, columns: reference.width, rows: reference.height,
      commit: { ...identity, textSha256: sha256(candidateText) },
      reference: { id: reference.id, textSha256: sha256(pythonText), manifestSha256: sha256(await readFile(join(ROOT, 'qa/artifacts/reference/manifest.json'))) },
      presentation: geometry, crops, files,
      preprocessing: 'Independent Pillow reference preprocessing versus browser Canvas acquisition with verified Pillow-compatible JavaScript Lanczos sampling and the dedicated renderer worker. Raw character differences are descriptive, not a quality assertion.',
    }));
  });
}
