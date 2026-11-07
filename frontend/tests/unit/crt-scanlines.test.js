import { describe, expect, test } from 'bun:test';
import {
  DEFAULTS,
  GLSL,
  SCANLINE_BEAM_DUTY,
  SCANLINE_BEAM_EXPONENT,
  SCANLINE_MEAN_DARKNESS_BUDGET,
  normalizeScanlines,
  scanlineModulation,
} from '../../src/crt/scanlines.js';

const approx = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;

describe('normalizeScanlines', () => {
  test('accepts the full range and freezes the result', () => {
    for (const v of [0, 0.3, 0.55, 1]) {
      const n = normalizeScanlines(v);
      expect(n.intensity).toBe(v);
      expect(Object.isFrozen(n)).toBe(true);
    }
  });

  test('rejects mutation of the frozen result', () => {
    const n = normalizeScanlines(0.4);
    expect(() => { n.intensity = 1; }).toThrow();
    expect(n.intensity).toBe(0.4);
  });

  test('rejects out-of-range values with RangeError', () => {
    expect(() => normalizeScanlines(-0.1)).toThrow(RangeError);
    expect(() => normalizeScanlines(1.001)).toThrow(RangeError);
    expect(() => normalizeScanlines(-0.1)).toThrow(/between 0 and 1/);
  });

  test('rejects non-numbers with TypeError', () => {
    expect(() => normalizeScanlines('0.3')).toThrow(TypeError);
    expect(() => normalizeScanlines(NaN)).toThrow(TypeError);
    expect(() => normalizeScanlines(Infinity)).toThrow(TypeError);
    expect(() => normalizeScanlines(null)).toThrow(TypeError);
    expect(() => normalizeScanlines(undefined)).toThrow(TypeError);
    expect(() => normalizeScanlines('0.3')).toThrow(/must be a number/);
  });
});

describe('scanlineModulation', () => {
  test('intensity 0 is an exact 1.0 bypass at every phase', () => {
    const n = normalizeScanlines(0);
    for (const phase of [0, 0.07, 0.25, 0.5, 0.93, 1, 2.5, -0.75]) {
      expect(scanlineModulation(phase, n)).toBe(1);
    }
  });

  test('peaks at exactly 1.0 in the row center', () => {
    for (const intensity of [0.3, 0.55, 1]) {
      expect(scanlineModulation(0.5, normalizeScanlines(intensity))).toBe(1);
    }
  });

  test('bottoms at exactly 1 - intensity in the inter-line gap', () => {
    for (const intensity of [0.3, 0.55, 1]) {
      const n = normalizeScanlines(intensity);
      expect(scanlineModulation(0, n)).toBe(1 - intensity);
      expect(scanlineModulation(1, n)).toBe(1 - intensity);
    }
  });

  test('stays within [1 - intensity, 1] for every phase', () => {
    for (const intensity of [0.15, 0.3, 0.55, 0.85, 1]) {
      const n = normalizeScanlines(intensity);
      for (let k = 0; k <= 512; k++) {
        const m = scanlineModulation(k / 512, n);
        expect(m).toBeGreaterThanOrEqual(1 - intensity - 1e-12);
        expect(m).toBeLessThanOrEqual(1 + 1e-12);
      }
    }
  });

  test('rises to the row center and falls back monotonically', () => {
    const n = normalizeScanlines(0.3);
    const steps = 400;
    for (let k = 1; k <= steps; k++) {
      const before = scanlineModulation((0.5 * (k - 1)) / steps, n);
      const at = scanlineModulation((0.5 * k) / steps, n);
      expect(at).toBeGreaterThanOrEqual(before);
      // Strict through the visible lobe; only the tail within ~1e-13 of the
      // peak flattens under float rounding.
      if (0.5 * k / steps <= 0.38) expect(at).toBeGreaterThan(before);
    }
    for (let k = 1; k <= steps; k++) {
      const before = scanlineModulation(0.5 + (0.5 * (k - 1)) / steps, n);
      const at = scanlineModulation(0.5 + (0.5 * k) / steps, n);
      expect(at).toBeLessThanOrEqual(before);
      if (0.5 + 0.5 * (k - 1) / steps >= 0.62) expect(at).toBeLessThan(before);
    }
  });

  test('repeats with period 1', () => {
    const n = normalizeScanlines(0.42);
    const exactPairs = [
      [0, 1],
      [0.25, 1.25],
      [0.5, 1.5],
      [0.75, 1.75],
      [-0.25, 0.75],
      [-0.5, 0.5],
    ];
    for (const [a, b] of exactPairs) {
      expect(scanlineModulation(a, n)).toBe(scanlineModulation(b, n));
    }
    for (let k = 0; k <= 100; k++) {
      const phase = k / 100;
      expect(approx(scanlineModulation(phase, n), scanlineModulation(phase + 1, n), 1e-12)).toBe(true);
    }
  });

  test('dark band width matches the beam duty constant', () => {
    const n = normalizeScanlines(1);
    let low = 0.05;
    let high = 0.1;
    for (let i = 0; i < 60; i++) {
      const mid = (low + high) / 2;
      if (1 - scanlineModulation(mid, n) > 0.5) low = mid;
      else high = mid;
    }
    const duty = 2 * ((low + high) / 2);
    expect(Math.abs(duty - SCANLINE_BEAM_DUTY)).toBeLessThan(0.001);
  });

  test('keeps the row-average multiplier inside the documented bound', () => {
    const samples = 4096;
    const dark = normalizeScanlines(1);
    let darkSum = 0;
    for (let k = 0; k < samples; k++) darkSum += 1 - scanlineModulation(k / samples, dark);
    const meanDarkness = darkSum / samples;
    expect(meanDarkness).toBeLessThanOrEqual(SCANLINE_MEAN_DARKNESS_BUDGET);

    const n = normalizeScanlines(DEFAULTS.intensity);
    let sum = 0;
    for (let k = 0; k < samples; k++) sum += scanlineModulation(k / samples, n);
    const average = sum / samples;
    expect(average).toBeGreaterThanOrEqual(0.95);
    expect(average).toBeLessThanOrEqual(1.05);

    // Documented tradeoff: even at full intensity the frame dims by at most
    // the compensation budget.
    expect(1 - meanDarkness).toBeGreaterThanOrEqual(1 - SCANLINE_MEAN_DARKNESS_BUDGET - 1e-9);
  });

  test('matches the documented lobe formula', () => {
    const n = normalizeScanlines(0.3);
    for (const phase of [0.05, 0.123, 0.5, 0.777]) {
      const expected = 1 - 0.3 * Math.pow(
        0.5 + 0.5 * Math.cos(2 * Math.PI * phase),
        SCANLINE_BEAM_EXPONENT,
      );
      expect(scanlineModulation(phase, n)).toBe(expected);
    }
  });
});

describe('DEFAULTS and GLSL', () => {
  test('DEFAULTS is frozen at the default intensity', () => {
    expect(Object.isFrozen(DEFAULTS)).toBe(true);
    expect(DEFAULTS.intensity).toBe(0.3);
    expect(() => { DEFAULTS.intensity = 1; }).toThrow();
    expect(DEFAULTS.intensity).toBe(0.3);
  });

  test('GLSL declares only uCrtScan uniforms and the row-locked function', () => {
    expect(GLSL).toContain('uniform float uCrtScanIntensity;');
    expect(GLSL).toContain('uniform float uCrtScanRows;');
    expect(GLSL).toContain('float crtScanline(vec2 warpedUV)');
    // No other uniform leaks (all must be prefixed uCrtScan).
    const uniforms = GLSL.match(/uniform\s+\w+\s+(\w+);/g) ?? [];
    expect(uniforms.every((u) => u.includes('uCrtScan'))).toBe(true);
    // WebGL1 / GLSL ES 1.00: the composer owns sampling, precision, and the
    // varying stage; this snippet must not redeclare any of them.
    expect(GLSL).not.toContain('texture(');
    expect(GLSL).not.toContain('texture2D');
    expect(GLSL).not.toContain('precision');
    expect(GLSL).not.toContain('varying');
  });

  test('GLSL stays in sync with the JS mirror constants', () => {
    expect(GLSL).toContain(SCANLINE_BEAM_EXPONENT.toFixed(1));
    expect(GLSL).toContain('fract(warpedUV.y * uCrtScanRows)');
  });
});
