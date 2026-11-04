import { afterAll, describe, expect, test } from 'bun:test';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import {
  computeGrid, createRenderer, rgbaToGray, makeGaussianKernel,
  blurNormalized, dogFromBlurs, sobelWithAngle,
} from '../../src/renderer.js';
import {
  buildRampLookup, angleToChar, composeAscii, fillCodes, codesToText,
} from '../../src/ascii.js';
import { array, manifest, manifestSha256, repositoryRoot, sha256, text } from './reference.js';

const artifactRoot = resolve(repositoryRoot, 'qa/artifacts/unit');
mkdirSync(artifactRoot, { recursive: true });
const report = {
  schemaVersion: 1,
  reference: 'qa/artifacts/reference/manifest.json',
  referenceSha256: manifestSha256,
  tolerances: { gaussianAtol: 1e-12, gaussianRtol: 1e-10, angleDegrees: 1e-8, dogRawScaled: 1e-8, minimumTextAgreement: 0.995 },
  counts: { components: manifest.components.length, fullImages: manifest.fullImages.length, independentSobel: manifest.sobelCases.length, rgb: manifest.rgbCases.length },
  comparisons: [], quantizationExceptions: [],
};
afterAll(() => {
  report.summary = {
    comparisons: report.comparisons.length,
    failedComparisons: report.comparisons.filter((entry) => entry.violations > 0).length,
    quantizationExceptions: report.quantizationExceptions.length,
    maximumGaussianError: Math.max(0, ...report.comparisons.filter((entry) => /blur[12]$/.test(entry.stage)).map((entry) => entry.maxError)),
    minimumWholePipelineTextAgreement: Math.min(1, ...report.comparisons.filter((entry) => entry.stage === 'pipeline/text').map((entry) => 1 - entry.mismatchFraction)),
  };
  writeFileSync(resolve(artifactRoot, 'numerical-report.json'), `${JSON.stringify(report, null, 2)}\n`);
});

const circularDistance = (a, b) => Math.abs((((a - b) % 360) + 540) % 360 - 180);
function neighborhood(values, index, width) {
  const height = Math.ceil(values.length / width), x = index % width, y = Math.floor(index / width);
  const rows = [];
  for (let yy = Math.max(0, y - 1); yy <= Math.min(height - 1, y + 1); yy++) {
    rows.push(Array.from(values.slice(yy * width + Math.max(0, x - 1), yy * width + Math.min(width, x + 2))));
  }
  return rows;
}
function compare(stage, fixture, actual, expected, width, options = {}) {
  if (actual.length !== expected.length) throw new Error(`${stage} ${fixture.id}: length ${actual.length} != ${expected.length}`);
  const errors = new Float64Array(actual.length);
  let maxError = 0, mismatches = 0, violations = 0, first = -1;
  for (let i = 0; i < actual.length; i++) {
    const error = options.circular ? circularDistance(actual[i], expected[i]) : Math.abs(actual[i] - expected[i]);
    errors[i] = Number.isFinite(error) ? error : Infinity;
    maxError = Math.max(maxError, errors[i]);
    if (actual[i] !== expected[i]) mismatches++;
    const limit = (options.atol ?? 0) + (options.rtol ?? 0) * Math.abs(expected[i]);
    if ((!Number.isFinite(error) || error > limit) && !options.allowed?.[i]) {
      violations++;
      if (first === -1) first = i;
    }
  }
  if (options.minimumAgreement !== undefined && 1 - mismatches / actual.length < options.minimumAgreement) {
    violations++;
    if (first === -1) first = actual.findIndex((value, i) => value !== expected[i]);
  }
  const result = { stage, fixture: fixture.id, settings: fixture.settings ?? null, size: actual.length, maxError, mismatchFraction: mismatches / actual.length, violations };
  report.comparisons.push(result);
  if (!violations) return result;
  result.firstDifference = {
    x: first % width, y: Math.floor(first / width), actual: actual[first], expected: expected[first],
    actualNeighborhood: neighborhood(actual, first, width), expectedNeighborhood: neighborhood(expected, first, width),
  };
  const filename = `${fixture.id.replace(/[^a-zA-Z0-9_-]/g, '_')}-${stage.replace(/[^a-zA-Z0-9_-]/g, '_')}-${sha256(Buffer.from(`${fixture.id}:${stage}`)).slice(0, 8)}.pgm`;
  const pixels = Buffer.alloc(Math.ceil(actual.length / width) * width);
  for (let i = 0; i < errors.length; i++) {
    pixels[i] = !Number.isFinite(errors[i]) ? 255 : errors[i] === 0 ? 0
      : Math.max(1, Math.round(255 * errors[i] / (Number.isFinite(maxError) && maxError > 0 ? maxError : 1)));
  }
  writeFileSync(resolve(artifactRoot, filename), Buffer.concat([Buffer.from(`P5\n${width} ${pixels.length / width}\n255\n`), pixels]));
  result.heatmap = `qa/artifacts/unit/${filename}`;
  throw new Error(JSON.stringify(result, null, 2));
}
function asciiBytes(value, width, height, label) {
  const rows = value.split('\n');
  if (rows.length !== height || rows.some((row) => row.length !== width || /[^\x20-\x7e]/.test(row))) throw new Error(`${label}: incomplete rows or non-ASCII output (${width}x${height})`);
  return Uint8Array.from(Buffer.from(rows.join(''), 'ascii'));
}
function sobelOutput(size) {
  return { gx: new Float64Array(size), gy: new Float64Array(size), magnitude: new Uint8Array(size), angles: new Float64Array(size) };
}
function checkSobel(fixture, dog, expected, width, height, prefix = 'sobel', allowed) {
  const output = sobelOutput(width * height);
  expect(sobelWithAngle(dog, width, height, output)).toBe(output);
  for (const key of ['gx', 'gy', 'magnitude', 'angles']) compare(`${prefix}/${key}`, fixture, output[key], expected[key], width, key === 'angles' ? { atol: 1e-8, circular: true, allowed } : { allowed });
  return output;
}
function checkDog(fixture, blur1, blur2, dog, expected, width, prefix) {
  const own = Uint8Array.from(blur1, (value, i) => Math.floor(Math.min(1, Math.max(0, value - blur2[i])) * 255));
  compare(`${prefix}/own-quantization`, fixture, dog, own, width);
  const allowed = new Uint8Array(dog.length);
  for (let i = 0; i < dog.length; i++) {
    const pythonRaw = Math.min(1, Math.max(0, expected.blur1[i] - expected.blur2[i])) * 255;
    if (Math.floor(pythonRaw) !== expected.dog[i]) throw new Error(`${fixture.id}: Python DoG does not quantize its own blur planes at (${i % width},${Math.floor(i / width)})`);
    if (dog[i] === expected.dog[i]) continue;
    const jsRaw = Math.min(1, Math.max(0, blur1[i] - blur2[i])) * 255;
    const lower = Math.min(jsRaw, pythonRaw), upper = Math.max(jsRaw, pythonRaw);
    const boundary = Math.max(dog[i], expected.dog[i]);
    const valid = Math.abs(dog[i] - expected.dog[i]) === 1 && Math.abs(jsRaw - pythonRaw) <= 1e-8 &&
      Math.floor(lower) + 1 === boundary && Math.floor(upper) === boundary && lower < boundary && upper >= boundary && boundary > 0;
    if (valid) {
      allowed[i] = 1;
      const exception = { stage: prefix, fixture: fixture.id, settings: fixture.settings, x: i % width, y: Math.floor(i / width), jsRaw, pythonRaw, jsByte: dog[i], pythonByte: expected.dog[i], boundary };
      report.quantizationExceptions.push(exception);
      console.info(`Permitted adjacent quantization bins: ${JSON.stringify(exception)}`);
    }
  }
  compare(`${prefix}/python-quantization`, fixture, dog, expected.dog, width, { allowed });
  if (fixture.settings.sigma1 === fixture.settings.sigma2) compare(`${prefix}/equal-sigma-zero`, fixture, dog, new Uint8Array(dog.length), width);
  return allowed;
}
function sobelInfluence(mask, width, height) {
  const affected = new Uint8Array(mask.length);
  for (let i = 0; i < mask.length; i++) if (mask[i]) {
    const x = i % width, y = Math.floor(i / width);
    for (let yy = Math.max(0, y - 1); yy <= Math.min(height - 1, y + 1); yy++) {
      for (let xx = Math.max(0, x - 1); xx <= Math.min(width - 1, x + 1); xx++) affected[yy * width + xx] = 1;
    }
  }
  return affected;
}

const cases = [...manifest.components, ...manifest.fullImages];
describe('Python independently supplied stage inputs and complete gray pipeline', () => {
  for (const fixture of cases) test(fixture.id, () => {
    const { width, height } = fixture, size = width * height;
    const expected = Object.fromEntries(Object.entries(fixture.outputs).map(([key, spec]) => [key, array(spec)]));
    const gray = array(fixture.input ?? fixture.outputs.gray);
    const normalized = Float64Array.from(gray, (value) => value / 255);
    const blur1 = new Float64Array(size), blur2 = new Float64Array(size), scratch = new Float64Array(size);
    for (const [key, out, sigma] of [['blur1', blur1, fixture.settings.sigma1], ['blur2', blur2, fixture.settings.sigma2]]) {
      const kernel = makeGaussianKernel(sigma);
      expect(kernel).toBeInstanceOf(Float64Array);
      expect(kernel.length).toBe(2 * Math.floor(4 * sigma + 0.5) + 1);
      expect(blurNormalized(normalized, width, height, kernel, out, scratch)).toBe(out);
      compare(`independent/${key}`, fixture, out, expected[key], width, { atol: 1e-12, rtol: 1e-10 });
    }
    const dog = new Uint8Array(size);
    expect(dogFromBlurs(blur1, blur2, dog)).toBe(dog);
    checkDog(fixture, blur1, blur2, dog, expected, width, 'independent/dog');
    checkSobel(fixture, expected.dog, expected, width, height, 'independent/sobel');
    const shade = new Uint8Array(size), edge = new Uint8Array(size);
    fillCodes(expected.gray, expected.magnitude, expected.angles, shade, edge, fixture.settings.edgeIndexThreshold);
    compare('independent/shadeCodes', fixture, shade, expected.shadeCodes, width);
    compare('independent/edgeCodes', fixture, edge, expected.edgeCodes, width);
    const expectedText = asciiBytes(text(fixture.text), width, height, fixture.id);
    compare('independent/composeAscii', fixture, asciiBytes(composeAscii(expected.gray, expected.magnitude, expected.angles, width, height, fixture.settings.edgeIndexThreshold), width, height, fixture.id), expectedText, width);
    compare('independent/codesToText', fixture, asciiBytes(codesToText(shade, edge, width, height, new Array(height)), width, height, fixture.id), expectedText, width);

    // No expected intermediate plane is passed into the real production workspace.
    const renderer = createRenderer(width, height, fixture.settings);
    const outputText = renderer.renderGray(gray);
    const snapshot = renderer.snapshot();
    for (const key of ['gray', 'blur1', 'blur2']) compare(`pipeline/${key}`, fixture, snapshot[key], expected[key], width, key === 'gray' ? {} : { atol: 1e-12, rtol: 1e-10 });
    const quantized = checkDog(fixture, snapshot.blur1, snapshot.blur2, snapshot.dog, expected, width, 'pipeline/dog');
    const downstream = sobelInfluence(quantized, width, height);
    for (const key of ['gx', 'gy', 'magnitude', 'angles', 'edgeCodes']) compare(`pipeline/${key}`, fixture, snapshot[key], expected[key], width, key === 'angles' ? { atol: 1e-8, circular: true, allowed: downstream } : { allowed: downstream });
    compare('pipeline/shadeCodes', fixture, snapshot.shadeCodes, expected.shadeCodes, width);
    compare('pipeline/text', fixture, asciiBytes(outputText, width, height, fixture.id), expectedText, width, { allowed: downstream, minimumAgreement: 0.995 });
    // Re-run exported stages on candidate DoG, not reference DoG, to prove
    // every exempt downstream coordinate still follows the real algorithm.
    checkSobel(fixture, snapshot.dog, snapshot, width, height, 'workspace/sobel-propagation');
    const propagatedShade = new Uint8Array(size), propagatedEdge = new Uint8Array(size);
    fillCodes(snapshot.gray, snapshot.magnitude, snapshot.angles, propagatedShade, propagatedEdge, fixture.settings.edgeIndexThreshold);
    compare('workspace/shade-propagation', fixture, snapshot.shadeCodes, propagatedShade, width);
    compare('workspace/edge-propagation', fixture, snapshot.edgeCodes, propagatedEdge, width);
    compare('workspace/text-propagation', fixture, asciiBytes(outputText, width, height, fixture.id),
      asciiBytes(codesToText(propagatedShade, propagatedEdge, width, height), width, height, fixture.id), width);
    // Exercise reuse and prove diagnostics do not alias production buffers or inputs.
    const retained = Object.fromEntries(Object.entries(snapshot).filter(([, value]) => ArrayBuffer.isView(value)).map(([key, value]) => [key, value.slice()]));
    expect(renderer.renderGray(gray)).toBe(outputText);
    for (const [key, saved] of Object.entries(retained)) compare(`workspace/snapshot-stability/${key}`, fixture, snapshot[key], saved, width);
    const second = renderer.snapshot();
    for (const key of Object.keys(expected)) {
      expect(snapshot[key]).not.toBe(second[key]);
      expect(snapshot[key].buffer).not.toBe(second[key].buffer);
    }
    compare('workspace/input-unchanged', fixture, gray, array(fixture.input ?? fixture.outputs.gray), width);
  });
});

for (const fixture of manifest.sobelCases) test(`frozen independent Sobel ${fixture.id}`, () => {
  const [height, width] = fixture.input.shape;
  const expected = Object.fromEntries(Object.entries(fixture.outputs).map(([key, spec]) => [key, array(spec)]));
  const actual = checkSobel(fixture, array(fixture.input), expected, width, height);
  // Frozen radians additionally constrain conversion, without introducing a second Sobel implementation.
  compare('sobel/radians', fixture, Float64Array.from(actual.angles, (degree) => degree * Math.PI / 180), expected.radians, width, { atol: 1e-8 * Math.PI / 180 });
});

for (const fixture of manifest.rgbCases) test(`frozen RGB grayscale ${fixture.id}`, () => {
  const rgba = array(fixture.rgba), rgb = array(fixture.rgb), expected = array(fixture.gray);
  const [height, width] = fixture.gray.shape;
  compare('grayscale/opaque-source', fixture, Uint8Array.from(expected, (_, i) => rgba[i * 4 + 3]), new Uint8Array(expected.length).fill(255), width);
  for (let channel = 0; channel < 3; channel++) {
    compare(`grayscale/source-channel-${channel}`, fixture, Uint8Array.from(expected, (_, i) => rgba[i * 4 + channel]), Uint8Array.from(expected, (_, i) => rgb[i * 3 + channel]), width);
  }
  const out = new Uint8Array(expected.length);
  expect(rgbaToGray(rgba, out)).toBe(out);
  const rounded = Uint8Array.from(expected, (_, i) => Math.round(0.299 * rgb[i * 3] + 0.587 * rgb[i * 3 + 1] + 0.114 * rgb[i * 3 + 2]));
  compare('grayscale/rounding', fixture, out, rounded, width);
  compare('grayscale/Pillow', fixture, out, expected, width, { atol: fixture.id === 'all-grayscale-values' ? 0 : 1 });
  const clamped = new Uint8ClampedArray(rgba);
  compare('grayscale/clamped-input', fixture, rgbaToGray(clamped, new Uint8Array(expected.length)), rounded, width);
  const renderer = createRenderer(width, height, {});
  renderer.renderRgba(rgba);
  compare('grayscale/renderRgba', fixture, renderer.snapshot().gray, rounded, width);
  expect(renderer.renderRgba(clamped)).toBe(renderer.renderGray(rounded));
});

const frozenAngles = array(manifest.angles.input), frozenAngleCodes = array(manifest.angles.codes);
const angleOracle = new Map(Array.from(frozenAngles, (value, i) => [value, frozenAngleCodes[i]]));
test('original angle sweep, keys, boundary probes, modulo, and first ties', () => {
  const actual = Uint8Array.from(frozenAngles, (degree) => {
    const glyph = angleToChar(degree);
    expect(glyph.length).toBe(1);
    return glyph.charCodeAt(0);
  });
  compare('angles/frozen-sweep', { id: 'angles' }, actual, frozenAngleCodes, frozenAngles.length);
  for (const probe of manifest.angles.explicitProbes) expect(angleToChar(probe.degrees)).toBe(probe.char);
  for (const [degree, char] of [[-90, '/'], [22.5, '/'], [67.5, '/'], [180, '|']]) expect(angleToChar(degree)).toBe(char);
  for (const boundary of [0, 22.5, 67.5, 112.5, 157.5, 202.5, 360]) {
    for (const offset of [-360, 0, 360]) for (const epsilon of [-1e-7, 0, 1e-7]) {
      const degree = boundary + offset + epsilon;
      // The frozen exporter includes these exact float expressions.
      if (!angleOracle.has(degree)) throw new Error(`Missing frozen angle boundary ${degree}`);
      expect(angleToChar(degree).charCodeAt(0)).toBe(angleOracle.get(degree));
    }
  }
});

test('exhaustive frozen ramp and custom first-index ties', () => {
  const actual = buildRampLookup(manifest.ramp.inkUsage), expected = array(manifest.ramp.indices);
  expect(actual).toBeInstanceOf(Uint8Array);
  expect(actual.length).toBe(256);
  compare('ramp/indices', { id: 'ramp256' }, actual, expected, 256);
  compare('ramp/codes', { id: 'ramp256' }, Uint8Array.from(actual, (index) => manifest.ramp.inkUsage[index][0].charCodeAt(0)), array(manifest.ramp.codes), 256);
  const normalized = array(manifest.ramp.normalized);
  for (let value = 0; value < 256; value++) {
    let best = Infinity, first = -1;
    for (let index = 0; index < normalized.length; index++) {
      const distance = Math.abs(normalized[index] - value);
      if (distance < best) { best = distance; first = index; }
    }
    expect(actual[value]).toBe(first);
  }
  expect(buildRampLookup([['a', 0], ['b', 10], ['c', 10]])[255]).toBe(1);
  expect(buildRampLookup([['a', 0], ['b', 254], ['c', 255]])[127]).toBe(0);
});

test('malformed ramp arrays fail explicitly and a valid retry succeeds', () => {
  for (const entries of [null, {}, new Uint8Array(4), [], [['x', NaN]], [['x', Infinity]], [['xx', 1], ['y', 2]], [['x']], ['not-an-entry']]) {
    expect(() => buildRampLookup(entries)).toThrow(TypeError);
  }
  expect(() => buildRampLookup([['a', 1], ['b', 1]])).toThrow(RangeError);
  expect(buildRampLookup(manifest.ramp.inkUsage)).toEqual(array(manifest.ramp.indices));
});

for (const threshold of [0, 90, 95]) test(`all luminances and magnitudes, frozen transition indices, threshold ${threshold}`, () => {
  const width = 256, height = 256, size = width * height;
  const indices = array(manifest.ramp.indices), rampCodes = array(manifest.ramp.codes);
  const probes = [-135, -90, -45, 0, 22.5, 45, 67.5, 90, 135, 180];
  for (const degree of probes) if (!angleOracle.has(degree)) throw new Error(`Missing frozen angle key ${degree}`);
  const gray = new Uint8Array(size), magnitude = new Uint8Array(size), angles = new Float64Array(size);
  const shadeExpected = new Uint8Array(size), edgeExpected = new Uint8Array(size), combined = new Uint8Array(size);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const i = y * width + x, degree = probes[(x + y) % probes.length];
    gray[i] = x; magnitude[i] = y; angles[i] = degree;
    shadeExpected[i] = rampCodes[x];
    edgeExpected[i] = indices[y] < threshold ? angleOracle.get(degree) : 32;
    combined[i] = edgeExpected[i] === 32 ? shadeExpected[i] : edgeExpected[i];
  }
  const fixture = { id: `threshold-${threshold}`, settings: { edgeIndexThreshold: threshold } };
  const shade = new Uint8Array(size), edge = new Uint8Array(size), rows = new Array(height);
  expect(fillCodes(gray, magnitude, angles, shade, edge, threshold)).toBeUndefined();
  compare('selection/shadeCodes', fixture, shade, shadeExpected, width);
  compare('selection/edgeCodes', fixture, edge, edgeExpected, width);
  compare('selection/composeAscii', fixture, asciiBytes(composeAscii(gray, magnitude, angles, width, height, threshold), width, height, fixture.id), combined, width);
  const composed = codesToText(shade, edge, width, height, rows);
  compare('selection/codesToText', fixture, asciiBytes(composed, width, height, fixture.id), combined, width);
  expect(rows.join('\n')).toBe(composed);
  expect(codesToText(shade, edge, width, height, rows)).toBe(composed);
});

for (const fixture of manifest.gridCases) test(`Python grid policy ${fixture.id}`, () => {
  if (fixture.error) expect(() => computeGrid(fixture.sourceWidth, fixture.sourceHeight, fixture.settings)).toThrow(RangeError);
  else expect(computeGrid(fixture.sourceWidth, fixture.sourceHeight, fixture.settings)).toEqual({ width: fixture.width, height: fixture.height });
});

describe('explicit invalid inputs and valid retries', () => {
  const u8 = () => new Uint8Array(4);
  const f64 = () => new Float64Array(4);
  const badDimensions = [0, -1, 0.5, NaN, Infinity, -Infinity, '2', null, true];
  for (const invalid of badDimensions) test(`invalid dimensions ${String(invalid)}`, () => {
    for (const [width, height] of [[invalid, 2], [2, invalid]]) {
      expect(() => computeGrid(width, height, {})).toThrow(RangeError);
      expect(() => createRenderer(width, height, {})).toThrow(RangeError);
      expect(() => blurNormalized(f64(), width, height, makeGaussianKernel(1), f64(), f64())).toThrow(RangeError);
      expect(() => sobelWithAngle(u8(), width, height, sobelOutput(4))).toThrow(RangeError);
      expect(() => composeAscii(u8(), u8(), f64(), width, height)).toThrow(RangeError);
      expect(() => codesToText(u8(), u8(), width, height, [])).toThrow(RangeError);
    }
    expect(createRenderer(2, 2, {}).renderGray(u8())).toBe('  \n  ');
  });

  for (const key of ['maxWidthChars', 'maxHeightLines', 'sigma1', 'sigma2', 'edgeIndexThreshold']) {
    const invalidValues = key.startsWith('max') ? badDimensions
      : key.startsWith('sigma') ? [0, -1, NaN, Infinity, -Infinity, '1', null, true]
        : [-1, 96, 0.5, NaN, Infinity, -Infinity, '90', null, true];
    for (const invalid of invalidValues) test(`invalid ${key} ${String(invalid)}`, () => {
      const settings = { [key]: invalid };
      expect(() => createRenderer(2, 2, settings)).toThrow(RangeError);
      expect(() => computeGrid(2, 2, settings)).toThrow(RangeError);
      if (key.startsWith('sigma')) expect(() => makeGaussianKernel(invalid)).toThrow(RangeError);
      if (key === 'edgeIndexThreshold') {
        expect(() => fillCodes(u8(), u8(), f64(), u8(), u8(), invalid)).toThrow(RangeError);
        expect(() => composeAscii(u8(), u8(), f64(), 2, 2, invalid)).toThrow(RangeError);
      }
      expect(createRenderer(2, 2, { sigma1: 2, sigma2: 1, edgeIndexThreshold: 95 }).renderGray(u8())).toBe('  \n  ');
    });
  }

  test('malformed settings and angle scalar types', () => {
    for (const settings of [null, [], 'settings', 1, false]) {
      expect(() => createRenderer(2, 2, settings)).toThrow(TypeError);
      expect(() => computeGrid(2, 2, settings)).toThrow(TypeError);
    }
    for (const angle of [NaN, Infinity, -Infinity, '0', null, true]) expect(() => angleToChar(angle)).toThrow(RangeError);
    expect(angleToChar(180)).toBe('|');
  });

  test('RGBA and grayscale output types and lengths', () => {
    const rgba = new Uint8Array(16), renderer = createRenderer(2, 2, {});
    const initial = renderer.renderGray(u8());
    for (const wrong of [[], new Float64Array(16), new Uint16Array(16), null, new ArrayBuffer(16)]) {
      expect(() => rgbaToGray(wrong, u8())).toThrow(TypeError);
      expect(() => renderer.renderRgba(wrong)).toThrow(TypeError);
    }
    for (const wrong of [[], f64(), new Uint8ClampedArray(4), null, new ArrayBuffer(4)]) {
      expect(() => rgbaToGray(rgba, wrong)).toThrow(TypeError);
      expect(() => renderer.renderGray(wrong)).toThrow(TypeError);
    }
    for (const length of [0, 3, 5, 15, 17]) {
      expect(() => renderer.renderRgba(new Uint8Array(length))).toThrow(RangeError);
      expect(() => rgbaToGray(new Uint8Array(length), u8())).toThrow(RangeError);
    }
    for (const length of [0, 3, 5]) {
      expect(() => renderer.renderGray(new Uint8Array(length))).toThrow(RangeError);
      expect(() => rgbaToGray(rgba, new Uint8Array(length))).toThrow(RangeError);
    }
    expect(renderer.snapshot().text).toBe(initial);
    expect(renderer.renderRgba(rgba)).toBe(initial);
    expect(renderer.renderGray(u8())).toBe(initial);
    expect(rgbaToGray(rgba, u8())).toEqual(u8());
  });

  test('Gaussian input, output, kernel, and scratch types and counts', () => {
    const argumentsForBlur = () => [f64(), 2, 2, makeGaussianKernel(1), f64(), f64()];
    for (const slot of [0, 3, 4, 5]) {
      for (const wrong of [[], u8(), new Float32Array(4), null, new ArrayBuffer(32)]) {
        const args = argumentsForBlur();
        args[slot] = wrong;
        expect(() => blurNormalized(...args)).toThrow(TypeError);
      }
    }
    for (const slot of [0, 4, 5]) for (const length of [0, 3, 5]) {
      const args = argumentsForBlur();
      args[slot] = new Float64Array(length);
      expect(() => blurNormalized(...args)).toThrow(RangeError);
    }
    for (const kernel of [new Float64Array(0), new Float64Array(2), Float64Array.of(NaN), Float64Array.of(Infinity)]) {
      const args = argumentsForBlur();
      args[3] = kernel;
      expect(() => blurNormalized(...args)).toThrow(RangeError);
    }
    const args = argumentsForBlur();
    expect(blurNormalized(...args)).toBe(args[4]);
    expect(args[4]).toEqual(f64());
    // Scratch aliasing would corrupt separable passes rather than just save space.
    const input = f64(), out = f64(), kernel = makeGaussianKernel(1);
    expect(() => blurNormalized(input, 2, 2, kernel, out, input)).toThrow(RangeError);
    expect(() => blurNormalized(input, 2, 2, kernel, out, out)).toThrow(RangeError);
    expect(makeGaussianKernel(0.01)).toEqual(Float64Array.of(1));
  });

  test('DoG inputs and output array types and counts', () => {
    const valid = () => [f64(), f64(), u8()];
    for (const slot of [0, 1, 2]) {
      const wrongTypes = slot === 2 ? [[], f64(), new Uint8ClampedArray(4), null] : [[], u8(), new Float32Array(4), null];
      for (const wrong of wrongTypes) {
        const args = valid(); args[slot] = wrong;
        expect(() => dogFromBlurs(...args)).toThrow(TypeError);
      }
    }
    for (const slot of [0, 1, 2]) for (const length of [0, 3, 5]) {
      const args = valid(); args[slot] = slot === 2 ? new Uint8Array(length) : new Float64Array(length);
      expect(() => dogFromBlurs(...args)).toThrow(RangeError);
    }
    expect(() => dogFromBlurs(new Float64Array(0), new Float64Array(0), new Uint8Array(0))).toThrow(RangeError);
    const out = u8();
    expect(dogFromBlurs(Float64Array.of(0, 0.5, 1, 2), Float64Array.of(1, 0, 0, 0), out)).toBe(out);
    expect(out).toEqual(Uint8Array.of(0, 127, 255, 255));
  });

  test('Sobel input and every output type and count', () => {
    for (const wrong of [[], f64(), new Uint8ClampedArray(4), null]) expect(() => sobelWithAngle(wrong, 2, 2, sobelOutput(4))).toThrow(TypeError);
    for (const length of [0, 3, 5]) expect(() => sobelWithAngle(new Uint8Array(length), 2, 2, sobelOutput(4))).toThrow(RangeError);
    for (const wrong of [null, undefined, 1, 'outputs']) expect(() => sobelWithAngle(u8(), 2, 2, wrong)).toThrow(TypeError);
    for (const key of ['gx', 'gy', 'angles', 'magnitude']) {
      for (const wrong of [undefined, null, [], key === 'magnitude' ? f64() : u8(), key === 'magnitude' ? new Uint8ClampedArray(4) : new Float32Array(4)]) {
        const out = sobelOutput(4); out[key] = wrong;
        expect(() => sobelWithAngle(u8(), 2, 2, out)).toThrow(TypeError);
      }
      for (const length of [0, 3, 5]) {
        const out = sobelOutput(4);
        out[key] = key === 'magnitude' ? new Uint8Array(length) : new Float64Array(length);
        expect(() => sobelWithAngle(u8(), 2, 2, out)).toThrow(RangeError);
      }
    }
    const out = sobelOutput(4);
    expect(sobelWithAngle(u8(), 2, 2, out)).toBe(out);
    expect(out.gx).toEqual(f64()); expect(out.gy).toEqual(f64());
    expect(out.magnitude).toEqual(u8()); expect(out.angles).toEqual(f64());
  });

  test('composition input, shade output, and edge output types and counts', () => {
    const valid = () => [u8(), u8(), f64(), u8(), u8()];
    for (const slot of [0, 1, 2, 3, 4]) {
      for (const wrong of [[], null, slot === 2 ? u8() : f64(), slot === 2 ? new Float32Array(4) : new Uint8ClampedArray(4)]) {
        const args = valid(); args[slot] = wrong;
        expect(() => fillCodes(...args)).toThrow(TypeError);
        if (slot < 3) expect(() => composeAscii(...args.slice(0, 3), 2, 2)).toThrow(TypeError);
      }
      for (const length of [0, 3, 5]) {
        const args = valid(); args[slot] = slot === 2 ? new Float64Array(length) : new Uint8Array(length);
        expect(() => fillCodes(...args)).toThrow(RangeError);
        if (slot < 3) expect(() => composeAscii(...args.slice(0, 3), 2, 2)).toThrow(RangeError);
      }
    }
    expect(() => fillCodes(new Uint8Array(0), new Uint8Array(0), new Float64Array(0), new Uint8Array(0), new Uint8Array(0))).toThrow(RangeError);
    const args = valid();
    expect(fillCodes(...args)).toBeUndefined();
    expect(codesToText(args[3], args[4], 2, 2)).toBe('  \n  ');
    expect(composeAscii(...args.slice(0, 3), 2, 2)).toBe('  \n  ');
  });

  test('text conversion rejects wrong code planes and lengths, then reuses rows', () => {
    for (const slot of [0, 1]) {
      for (const wrong of [[], null, f64(), new Uint8ClampedArray(4)]) {
        const args = [u8(), u8(), 2, 2]; args[slot] = wrong;
        expect(() => codesToText(...args)).toThrow(TypeError);
      }
      for (const length of [0, 3, 5]) {
        const args = [u8(), u8(), 2, 2]; args[slot] = new Uint8Array(length);
        expect(() => codesToText(...args)).toThrow(RangeError);
      }
    }
    const rows = new Array(2), spaces = new Uint8Array(4).fill(32), dense = new Uint8Array(4).fill(64);
    expect(codesToText(dense, spaces, 2, 2, rows)).toBe('@@\n@@');
    expect(codesToText(spaces, spaces, 2, 2, rows)).toBe('  \n  ');
    expect(rows).toEqual(['  ', '  ']);
  });
});

test('frozen diagnostic fixtures cover the complete sweep and overflowing Sobel magnitudes', () => {
  expect(frozenAngles.length).toBe(2919);
  expect(frozenAngleCodes.length).toBe(2919);
  expect(angleOracle.size).toBe(2919);
  for (let step = 0; step <= 2880; step++) expect(angleOracle.has(-720 + step * 0.5)).toBe(true);
  for (const key of [-135, -90, -45, 45, 90, 135, 0, 180]) expect(angleOracle.has(key)).toBe(true);
  expect(array(manifest.ramp.normalized).length).toBe(manifest.ramp.inkUsage.length);
  expect(array(manifest.ramp.indices).length).toBe(256);
  expect(array(manifest.ramp.codes).length).toBe(256);
  let overflowing = 0;
  for (const fixture of manifest.sobelCases) {
    const gx = array(fixture.outputs.gx), gy = array(fixture.outputs.gy), magnitude = array(fixture.outputs.magnitude);
    for (let i = 0; i < gx.length; i++) {
      const raw = Math.sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
      if (raw > 255) {
        overflowing++;
        expect(magnitude[i]).toBe(Math.trunc(raw) % 256);
      }
    }
  }
  expect(overflowing).toBeGreaterThan(0);
});
