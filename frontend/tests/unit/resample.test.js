import { afterAll, describe, expect, test } from 'bun:test';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { createLanczosResampler, resizeLanczosRgba } from '../../src/resample.js';
import { array, canonicalIds, manifest, manifestSha256, repositoryRoot, sha256 } from './reference.js';

const artifactRoot = resolve(repositoryRoot, 'qa/artifacts/unit');
mkdirSync(artifactRoot, { recursive: true });
const boundaryShapes = [
  ['tiny-1x1', 1, 1, [[1, 1], [7, 9], [1, 13], [11, 1]]],
  ['tiny-1x17', 1, 17, [[1, 17], [1, 5], [1, 31], [9, 7]]],
  ['tiny-17x1', 17, 1, [[17, 1], [5, 1], [31, 1], [7, 9]]],
  ['awkward', 13, 11, [[5, 3], [29, 23], [7, 11], [13, 5], [23, 11], [13, 19], [13, 11]]],
];
const expectedIds = [
  ...canonicalIds.flatMap((id) => ['comparison', 'default'].map((profile) => `${profile}:${id}`)),
  ...boundaryShapes.flatMap(([label, , , targets]) => targets.map(([w, h]) => `boundary:${label}:${w}x${h}`)),
];
const report = {
  schemaVersion: 1,
  reference: 'qa/artifacts/reference/manifest.json', referenceSha256: manifestSha256,
  pillowVersion: manifest.dependencies.Pillow,
  tolerance: { maximumRgbChannelError: 1, maximumAlphaError: 0 },
  expectedCaseCount: expectedIds.length, comparisons: [],
};
afterAll(() => {
  report.summary = {
    cases: manifest.resizeCases?.length ?? 0,
    comparisons: report.comparisons.length,
    maximumRgbChannelError: Math.max(0, ...report.comparisons.map((item) => item.maxError)),
    failedComparisons: report.comparisons.filter((item) => item.violations > 0).length,
  };
  writeFileSync(resolve(artifactRoot, 'resize-report.json'), `${JSON.stringify(report, null, 2)}\n`);
});

function compare(fixture, stage, actual, expected) {
  expect(actual).toBeInstanceOf(Uint8ClampedArray);
  expect(actual.length).toBe(fixture.width * fixture.height * 4);
  expect(actual.length).toBe(expected.length);
  const errors = Buffer.alloc(fixture.width * fixture.height);
  let maxError = 0, mismatches = 0, violations = 0;
  const mismatchCoordinates = [];
  for (let i = 0; i < actual.length; i++) {
    const channel = i % 4, error = Math.abs(actual[i] - expected[i]);
    const pixel = Math.floor(i / 4);
    if (channel < 3) maxError = Math.max(maxError, error);
    errors[pixel] = Math.max(errors[pixel], error);
    if (error) {
      mismatches++;
      if (mismatchCoordinates.length < 16) mismatchCoordinates.push({
        x: pixel % fixture.width, y: Math.floor(pixel / fixture.width),
        channel: 'RGBA'[channel], actual: actual[i], expected: expected[i], error,
      });
    }
    if (error > (channel === 3 ? 0 : 1) || (channel === 3 && actual[i] !== 255)) violations++;
  }
  const result = {
    id: fixture.id, stage, srcWidth: fixture.srcWidth, srcHeight: fixture.srcHeight,
    width: fixture.width, height: fixture.height, maxError, mismatches,
    mismatchFraction: mismatches / actual.length, violations, mismatchCoordinates,
  };
  report.comparisons.push(result);
  if (!violations) return;
  const name = `resize-${sha256(Buffer.from(`${fixture.id}:${stage}`)).slice(0, 16)}.pgm`;
  let peak = 1;
  for (const error of errors) peak = Math.max(peak, error);
  for (let i = 0; i < errors.length; i++) errors[i] = Math.round(errors[i] * 255 / peak);
  writeFileSync(resolve(artifactRoot, name), Buffer.concat([
    Buffer.from(`P5\n${fixture.width} ${fixture.height}\n255\n`), errors,
  ]));
  result.heatmap = `qa/artifacts/unit/${name}`;
  throw new Error(`Pillow Lanczos resize disagreement: ${JSON.stringify(result, null, 2)}`);
}

// Fail closed before creating case tests: missing reference data is not a skipped suite.
if (!Array.isArray(manifest.resizeCases)) throw new Error('Python reference resizeCases is missing; regenerate the gated reference export');
if (JSON.stringify(manifest.resizeCases.map((item) => item.id).sort()) !== JSON.stringify([...expectedIds].sort())) {
  throw new Error('Python reference resizeCases has missing, duplicate or unknown case IDs');
}
for (const fixture of manifest.resizeCases) {
  for (const key of ['srcWidth', 'srcHeight', 'width', 'height']) {
    if (!Number.isSafeInteger(fixture[key]) || fixture[key] <= 0) throw new Error(`${fixture.id}: invalid ${key}`);
  }
  for (const [spec, shape] of [
    [fixture.input, [fixture.srcHeight, fixture.srcWidth, 4]],
    [fixture.outputs?.rgba, [fixture.height, fixture.width, 4]],
  ]) {
    if (spec?.dtype !== 'u8' || JSON.stringify(spec.shape) !== JSON.stringify(shape)) throw new Error(`${fixture.id}: invalid independent RGBA specification`);
  }
  if (fixture.id.startsWith('boundary:')) {
    const source = boundaryShapes.find(([label]) => fixture.id.startsWith(`boundary:${label}:`));
    if (!source || source[1] !== fixture.srcWidth || source[2] !== fixture.srcHeight) throw new Error(`${fixture.id}: unexpected boundary source dimensions`);
  } else {
    const full = manifest.fullImages.find((item) => item.id === fixture.id);
    if (!full || fixture.srcWidth !== full.sourceWidth || fixture.srcHeight !== full.sourceHeight || fixture.width !== full.width || fixture.height !== full.height) throw new Error(`${fixture.id}: resize/full-image dimensions disagree`);
  }
}

describe('Locked Pillow RGB 8-bit Lanczos independent resize answers', () => {
  test('reference uses the locked Pillow version and complete matrix', () => {
    expect(manifest.dependencies.Pillow).toBe('13.0.0');
    expect(manifest.resizeCases.length).toBe(51);
  });
  for (const fixture of manifest.resizeCases) test(fixture.id, () => {
    const input = array(fixture.input), expected = array(fixture.outputs.rgba);
    const original = input.slice();
    expect(expected.every((value, i) => i % 4 !== 3 || value === 255)).toBe(true);
    expect(input.every((value, i) => i % 4 !== 3 || value === 255)).toBe(true);
    const dimensions = [fixture.srcWidth, fixture.srcHeight, fixture.width, fixture.height];
    compare(fixture, 'wrapper/uint8', resizeLanczosRgba(input, ...dimensions), expected);
    const cached = createLanczosResampler(...dimensions);
    const first = cached.resize(input);
    compare(fixture, 'cached/uint8', first, expected);
    const retained = first.slice();
    const second = cached.resize(new Uint8ClampedArray(input));
    expect(second).not.toBe(first);
    expect(second.buffer).not.toBe(first.buffer);
    expect(first).toEqual(retained);
    compare(fixture, 'cached/clamped', second, expected);
    structuredClone(second.buffer, { transfer: [second.buffer] });
    expect(second.byteLength).toBe(0);
    compare(fixture, 'cached/after-transfer', cached.resize(input), expected);
    const black = new Uint8Array(input.length);
    for (let i = 3; i < black.length; i += 4) black[i] = 255;
    const blackOutput = cached.resize(black);
    expect(blackOutput.every((value, i) => value === (i % 4 === 3 ? 255 : 0))).toBe(true);
    compare(fixture, 'cached/after-changed-input', cached.resize(input), expected);
    expect(input).toEqual(original);
    if (fixture.srcWidth === fixture.width && fixture.srcHeight === fixture.height) expect(first).toEqual(new Uint8ClampedArray(input));
  });
});

describe('Lanczos explicit validation and valid retry', () => {
  const rgba = new Uint8Array([17, 103, 239, 255]);
  for (const invalid of [0, -1, 1.5, NaN, Infinity, -Infinity, undefined, null, '1']) {
    for (let axis = 0; axis < 4; axis++) test(`invalid dimension ${String(invalid)} axis ${axis}`, () => {
      const dimensions = [1, 1, 1, 1];
      dimensions[axis] = invalid;
      expect(() => createLanczosResampler(...dimensions)).toThrow(RangeError);
      expect(() => resizeLanczosRgba(rgba, ...dimensions)).toThrow(RangeError);
      expect(resizeLanczosRgba(rgba, 1, 1, 1, 1)).toEqual(new Uint8ClampedArray(rgba));
    });
  }
  for (const invalid of [null, undefined, [], [17, 103, 239, 255], new ArrayBuffer(4), new Uint16Array(4), new Float64Array(4), new Int8Array(4), new DataView(new ArrayBuffer(4))]) {
    test(`rejects incorrect array type ${Object.prototype.toString.call(invalid)}`, () => {
      const cached = createLanczosResampler(1, 1, 1, 1);
      expect(() => cached.resize(invalid)).toThrow(TypeError);
      expect(() => resizeLanczosRgba(invalid, 1, 1, 1, 1)).toThrow(TypeError);
      expect(cached.resize(rgba)).toEqual(new Uint8ClampedArray(rgba));
    });
  }
  for (const length of [0, 1, 3, 5, 8]) test(`rejects incorrect byte count ${length}`, () => {
    const cached = createLanczosResampler(1, 1, 1, 1);
    for (const Type of [Uint8Array, Uint8ClampedArray]) {
      expect(() => cached.resize(new Type(length))).toThrow(RangeError);
      expect(() => resizeLanczosRgba(new Type(length), 1, 1, 1, 1)).toThrow(RangeError);
    }
    expect(cached.resize(rgba)).toEqual(new Uint8ClampedArray(rgba));
  });
});
