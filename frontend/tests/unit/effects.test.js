import { describe, expect, test } from 'bun:test';
import inkUsage from '../../../ink-usage.json';
import {
  applyTone, unsharpMask, buildMeasuredRamp, getCompactRamp,
  nearestMeasuredIndex, floydSteinberg, bayerOrdered,
} from '../../src/effects.js';
import { normalizeLook } from '../../src/looks.js';

const bytes = (...values) => Uint8Array.from(values);
const glyphs = (codes) => String.fromCharCode(...codes);
const binary = buildMeasuredRamp([[' ', 0], ['#', 255]]);
const uneven = buildMeasuredRamp([['#', 100], [' ', 0], ['.', 20]]);

function floyd(input, width, height, ramp = binary, errors = new Float64Array(2 * (width + 2))) {
  return floydSteinberg(input, width, height, ramp, new Uint8Array(input.length), errors);
}

function bayer(input, width, height, ramp = binary) {
  return bayerOrdered(input, width, height, ramp, new Uint8Array(input.length));
}

describe('measured coverage quantization', () => {
  test('uses measured intervals, clamps endpoints and resolves midpoint ties downward', () => {
    expect(Array.from(uneven.levels)).toEqual([0, 51, 255]);
    expect(glyphs(uneven.codes)).toBe(' .#');
    expect(nearestMeasuredIndex(uneven.levels, -20)).toBe(0);
    expect(nearestMeasuredIndex(uneven.levels, 25.5)).toBe(0);
    expect(nearestMeasuredIndex(uneven.levels, 25.5001)).toBe(1);
    expect(nearestMeasuredIndex(uneven.levels, 153)).toBe(1);
    expect(nearestMeasuredIndex(uneven.levels, 153.0001)).toBe(2);
    expect(nearestMeasuredIndex(uneven.levels, 300)).toBe(2);
  });

  test('equal measurements retain the first glyph, independent of later sorting', () => {
    const ramp = buildMeasuredRamp([['#', 255], ['x', 80], ['.', 80], [' ', 0]]);
    expect(glyphs(ramp.codes)).toBe(' x#');
    expect(nearestMeasuredIndex(ramp.levels, 80)).toBe(1);
  });

  test('compact levels are actual ink measurements, not equal ordinal spacing', () => {
    const ramp = getCompactRamp();
    const measurements = new Map(inkUsage);
    const maximum = measurements.get('@');
    for (let i = 0; i < ramp.codes.length; i++) {
      const glyph = String.fromCharCode(ramp.codes[i]);
      expect(ramp.levels[i]).toBeCloseTo(255 * measurements.get(glyph) / maximum, 12);
    }
    expect(ramp.levels[0]).toBe(0);
    expect(ramp.levels[ramp.levels.length - 1]).toBe(255);
    expect(ramp.levels[1]).not.toBeCloseTo(255 / 9, 5);
  });

  test('rejects absent coverage span and invalid glyph measurements', () => {
    for (const entries of [[], [[' ', 0]], [[' ', 1], ['#', 1]], [[' ', 0], ['xx', 1]], [[' ', 0], ['#', NaN]]]) {
      expect(() => buildMeasuredRamp(entries)).toThrow();
    }
  });
});

describe('Floyd–Steinberg diffusion', () => {
  test('diffuses measured error rather than ordinal ramp error', () => {
    // 140 -> 51, error 89. Next value = 140 + 89*7/16 = 178.9375 -> 255.
    expect(glyphs(floyd(bytes(140, 140), 2, 1, uneven))).toBe('.#');
  });

  test('uses all four standard weights without wrapping edge error', () => {
    // First row: 100 -> 0, then 143.75 -> 255. Second row becomes
    // 110.390625 -> 0, then 119.7802734375 -> 0.
    expect(glyphs(floyd(bytes(100, 100, 100, 100), 2, 2))).toBe(' #  ');
    // In a one-column image only the 5/16 downward weight survives.
    expect(glyphs(floyd(bytes(100, 100, 100), 1, 3))).toBe(' # ');
  });

  test('resets reusable errors and keeps exact ramp endpoints', () => {
    const errors = new Float64Array(8).fill(999);
    const input = bytes(100, 100, 100, 100);
    const first = floyd(input, 2, 2, binary, errors);
    expect(floyd(input, 2, 2, binary, errors)).toEqual(first);
    expect(glyphs(floyd(bytes(0, 255, 0, 255), 2, 2))).toBe(' # #');
    expect(input).toEqual(bytes(100, 100, 100, 100));
  });
});

describe('Bayer ordered shading', () => {
  test('the standard 4x4 half-tone is spatial, balanced and repeatable', () => {
    const input = new Uint8Array(16).fill(128);
    expect(glyphs(bayer(input, 4, 4))).toBe('# #  # ## #  # #');
    expect(bayer(input, 4, 4)).toEqual(bayer(input, 4, 4));
  });

  test('thresholds interpolate between measured neighbors, including exact boundaries', () => {
    // x=0 threshold is 1/32; between measured 51 and 255 it is 57.375.
    // x=1 threshold is 17/32, so 80 remains at the 51-coverage glyph.
    expect(glyphs(bayer(bytes(58, 80), 2, 1, uneven))).toBe('#.');
    const exact = buildMeasuredRamp([[' ', 0], ['.', 32], ['#', 255]]);
    expect(glyphs(bayer(bytes(1), 1, 1, exact))).toBe(' ');
    expect(glyphs(bayer(bytes(2), 1, 1, exact))).toBe('.');
    expect(glyphs(bayer(bytes(0, 255), 2, 1))).toBe(' #');
  });
});

describe('opt-in tone', () => {
  test('neutral tone is exact and requires no workspace', () => {
    const input = Uint8Array.from({ length: 256 }, (_, i) => i);
    const expected = input.slice();
    expect(applyTone(input, normalizeLook(), null)).toBe(input);
    expect(input).toEqual(expected);
  });

  test('midtone gamma has fixed endpoints and inversion runs after gamma', () => {
    const lut = new Uint8Array(256);
    const input = bytes(0, 64, 128, 255);
    applyTone(input, normalizeLook({ midtone: 2 }), lut);
    expect(input).toEqual(bytes(0, 16, 64, 255));
    const inverted = bytes(0, 64, 128, 255);
    applyTone(inverted, normalizeLook({ midtone: 2, invert: true }), lut);
    expect(inverted).toEqual(bytes(255, 239, 191, 0));
    const lighter = bytes(64);
    applyTone(lighter, normalizeLook({ midtone: 0.5 }), lut);
    expect(lighter).toEqual(bytes(128));
  });

  test('contrast stretches extrema and leaves a flat field unchanged', () => {
    const lut = new Uint8Array(256);
    const contrast = normalizeLook({ autoContrast: true });
    const input = bytes(50, 75, 100);
    applyTone(input, contrast, lut);
    expect(input).toEqual(bytes(0, 128, 255));
    for (const value of [0, 1, 127, 255]) {
      const flat = new Uint8Array(9).fill(value);
      applyTone(flat, contrast, lut);
      expect(flat).toEqual(new Uint8Array(9).fill(value));
    }
    const flat = bytes(80, 80);
    applyTone(flat, normalizeLook({ autoContrast: true, invert: true }), lut);
    expect(flat).toEqual(bytes(175, 175));
  });
});

describe('bounded unsharp masking', () => {
  test('subtracts a real 2D binomial blur and clips overshoot', () => {
    const input = bytes(100, 100, 100, 100, 200, 100, 100, 100, 100);
    // Center blur=125, cross-neighbor blur=112.5, corner blur=106.25.
    unsharpMask(input, 3, 3, 1, new Float64Array(9));
    expect(input).toEqual(bytes(94, 88, 94, 88, 255, 88, 94, 88, 94));
    const clipped = bytes(0, 255, 0);
    unsharpMask(clipped, 3, 1, 1.5, new Float64Array(3));
    expect(clipped).toEqual(bytes(0, 255, 0));
  });

  test('constant, singleton and neutral images are unchanged', () => {
    const flat = new Uint8Array(12).fill(96);
    unsharpMask(flat, 4, 3, 1.5, new Float64Array(12));
    expect(flat).toEqual(new Uint8Array(12).fill(96));
    const single = bytes(125);
    unsharpMask(single, 1, 1, 1, new Float64Array(1));
    expect(single).toEqual(bytes(125));
    const neutral = bytes(20, 100, 240);
    expect(unsharpMask(neutral, 3, 1, 0, null)).toBe(neutral);
    expect(neutral).toEqual(bytes(20, 100, 240));
    for (const amount of [-1, 2, NaN]) {
      expect(() => unsharpMask(neutral, 3, 1, amount, new Float64Array(3))).toThrow();
    }
  });
});
