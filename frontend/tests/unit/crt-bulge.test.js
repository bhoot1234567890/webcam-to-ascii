import { describe, expect, test } from 'bun:test';
import {
  DEFAULTS,
  GLSL,
  bezelFactor,
  normalizeBulge,
  warpPoint,
} from '../../src/crt/bulge.js';

const approx = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;

describe('normalizeBulge', () => {
  test('accepts the full range and freezes the result', () => {
    for (const v of [0, 0.12, 0.5, 1]) {
      const n = normalizeBulge(v);
      expect(n.amount).toBe(v);
      expect(Object.isFrozen(n)).toBe(true);
    }
  });

  test('rejects out-of-range values with RangeError', () => {
    expect(() => normalizeBulge(-0.1)).toThrow(RangeError);
    expect(() => normalizeBulge(1.7)).toThrow(RangeError);
    expect(() => normalizeBulge(-0.1)).toThrow(/between 0 and 1/);
  });

  test('rejects non-numbers with TypeError', () => {
    expect(() => normalizeBulge('x')).toThrow(TypeError);
    expect(() => normalizeBulge(NaN)).toThrow(TypeError);
    expect(() => normalizeBulge('x')).toThrow(/must be a number/);
  });
});

describe('warpPoint', () => {
  const aspect = 4 / 3;
  const sampleUVs = [
    [0.5, 0.5],
    [0, 0],
    [1, 1],
    [0, 1],
    [1, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 0.5],
    [1, 0.5],
    [0.25, 0.75],
  ];

  test('aspect correction keeps the warp circular for aspect 2', () => {
    const n = normalizeBulge(0.4);
    const aspect = 2;
    // Two points at the SAME aspect-corrected radius on different axes warp
    // by the same radial factor, so the iso-warp contour is a circle, not an
    // ellipse. Pick r = 0.5 in corrected space on x and y axes.
    const pX = [0.625, 0.5]; // corrected x = 0.5, y = 0
    const pY = [0.5, 0.75]; // corrected x = 0, y = 0.5
    const oX = warpPoint(pX, aspect, n, [0, 0]);
    const oY = warpPoint(pY, aspect, n, [0, 0]);
    const dX = Math.hypot((oX[0] - 0.5) * aspect, oX[1] - 0.5);
    const dY = Math.hypot((oY[0] - 0.5) * aspect, oY[1] - 0.5);
    expect(approx(dX, dY, 1e-9)).toBe(true);
    // Corner symmetry: mirrored corners warp to mirrored coordinates.
    const oA = warpPoint([1, 1], aspect, n, [0, 0]);
    const oB = warpPoint([0, 0], aspect, n, [0, 0]);
    expect(approx(oA[0], 1 - oB[0], 1e-9)).toBe(true);
    expect(approx(oA[1], 1 - oB[1], 1e-9)).toBe(true);
  });

  test('amount 0 is exact identity everywhere', () => {
    const n = normalizeBulge(0);
    for (const uv of sampleUVs) {
      const out = warpPoint(uv, aspect, n, [0, 0]);
      expect(approx(out[0], uv[0])).toBe(true);
      expect(approx(out[1], uv[1])).toBe(true);
    }
  });

  test('center maps to center for any amount', () => {
    for (const amount of [0.05, 0.3, 0.7, 1]) {
      const n = normalizeBulge(amount);
      const out = warpPoint([0.5, 0.5], aspect, n, [0, 0]);
      expect(approx(out[0], 0.5)).toBe(true);
      expect(approx(out[1], 0.5)).toBe(true);
    }
  });

  test('corners pull inward monotonically as amount grows', () => {
    const corner = [1, 1];
    let prevDist = 0;
    for (const amount of [0, 0.2, 0.4, 0.6, 0.8, 1]) {
      const n = normalizeBulge(amount);
      const out = warpPoint(corner, aspect, n, [0, 0]);
      const dist = Math.hypot(out[0] - 0.5, out[1] - 0.5);
      expect(dist >= prevDist).toBe(true);
      prevDist = dist;
    }
    // The visible picture corner moves inward: the sampled point sits outside
    // the frame at max amount, i.e. the corner is cut by the bezel.
    expect(prevDist > Math.hypot(corner[0] - 0.5, corner[1] - 0.5)).toBe(true);
  });
  test('writes into the provided out array', () => {
    const out = [0, 0];
    const returned = warpPoint([0.5, 0.5], 1, normalizeBulge(0.12), out);
    expect(returned).toBe(out);
  });
});

describe('bezelFactor', () => {
  const n = normalizeBulge(0.12);

  test('is ~1 near the center', () => {
    expect(bezelFactor([0.5, 0.5], n)).toBeGreaterThan(0.99);
  });

  test('is 0 fully outside and on the frame edge', () => {
    expect(bezelFactor([-0.05, 0.5], n)).toBe(0);
    expect(bezelFactor([1.2, 0.5], n)).toBe(0);
    expect(bezelFactor([0.5, -1], n)).toBe(0);
    expect(bezelFactor([0, 0.5], n)).toBe(0);
    expect(bezelFactor([1, 1], n)).toBe(0);
  });

  test('is strictly between 0 and 1 inside the border band', () => {
    const w = n.bezelWidth;
    for (const x of [w * 0.5, w, w * 1.5]) {
      const f = bezelFactor([x, 0.5], n);
      expect(f).toBeGreaterThan(0);
      expect(f).toBeLessThan(1);
    }
    // Strictly decreasing across the band.
    const a = bezelFactor([w * 0.5, 0.5], n);
    const b = bezelFactor([w, 0.5], n);
    const c = bezelFactor([w * 1.5, 0.5], n);
    expect(a < b).toBe(true);
    expect(b < c).toBe(true);
  });

  test('vignette darkens corners of the frame interior', () => {
    const center = bezelFactor([0.5, 0.5], n);
    const near = bezelFactor([0.9, 0.5], n);
    expect(near).toBeLessThan(center);
    expect(near).toBeGreaterThan(0);
  });
});

describe('DEFAULTS and GLSL', () => {
  test('DEFAULTS is frozen with a sane amount', () => {
    expect(Object.isFrozen(DEFAULTS)).toBe(true);
    expect(DEFAULTS.amount).toBeGreaterThanOrEqual(0);
    expect(DEFAULTS.amount).toBeLessThanOrEqual(1);
  });

  test('GLSL declares only uCrtBulge uniforms and both functions', () => {
    expect(GLSL).toContain('uniform float uCrtBulgeAmount;');
    expect(GLSL).toContain('uniform float uCrtBulgeAspect;');
    expect(GLSL).toContain('vec2 crtWarpUV(vec2 uv)');
    expect(GLSL).toContain('float crtBezel(vec2 warpedUV)');
    // No other uniform leaks (all must be prefixed uCrtBulge).
    const uniforms = GLSL.match(/uniform\s+\w+\s+(\w+);/g) ?? [];
    expect(uniforms.every((u) => u.includes('uCrtBulge'))).toBe(true);
    // WebGL1 / GLSL ES 1.00 syntax (no texture() sampling calls).
    expect(GLSL).not.toContain('texture(');
  });
});
