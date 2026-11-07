import { describe, expect, test } from 'bun:test';
import {
  DEFAULTS, GLSL, normalizeAberration, aberrationOffset,
} from '../../src/crt/aberration.js';

const offsets = new Float64Array(6);

function sampleOffsets(x, y, amount = DEFAULTS.amount) {
  aberrationOffset({ x, y }, normalizeAberration(amount), offsets);
  return offsets;
}

function redMagnitude(x, y, amount = DEFAULTS.amount) {
  const out = sampleOffsets(x, y, amount);
  return Math.hypot(out[0], out[1]);
}

/** The caller-side aspect correction the GLSL derives internally. */
function cornerUnit(u, v, width, height) {
  const aspect = width / height;
  const corner = Math.hypot(aspect, 1);
  return { x: (u * 2 - 1) * aspect / corner, y: (v * 2 - 1) / corner };
}

describe('normalizeAberration', () => {
  test('defaults to the tuned corner separation and freezes the result', () => {
    expect(DEFAULTS).toEqual({ amount: 0.003 });
    expect(Object.isFrozen(DEFAULTS)).toBe(true);
    expect(normalizeAberration()).toEqual({ amount: 0.003 });
    expect(normalizeAberration({})).toEqual({ amount: 0.003 });
    const settings = normalizeAberration(0.004);
    expect(settings).toEqual({ amount: 0.004 });
    expect(Object.isFrozen(settings)).toBe(true);
    expect(() => { settings.amount = 0.02; }).toThrow();
  });

  test('accepts the full amount range including both bounds', () => {
    expect(normalizeAberration(0).amount).toBe(0);
    expect(normalizeAberration(0.02).amount).toBe(0.02);
    expect(normalizeAberration({ amount: 0.001 }).amount).toBe(0.001);
  });

  test('rejects out-of-range amounts with RangeError', () => {
    expect(() => normalizeAberration(-0.001)).toThrow(RangeError);
    expect(() => normalizeAberration(0.5)).toThrow(RangeError);
    expect(() => normalizeAberration({ amount: -0.001 })).toThrow(RangeError);
    expect(() => normalizeAberration({ amount: 0.5 })).toThrow(RangeError);
  });

  test('rejects non-number amounts with TypeError', () => {
    expect(() => normalizeAberration('x')).toThrow(TypeError);
    expect(() => normalizeAberration({ amount: 'x' })).toThrow(TypeError);
    expect(() => normalizeAberration(Number.NaN)).toThrow(TypeError);
    expect(() => normalizeAberration(Number.POSITIVE_INFINITY)).toThrow(TypeError);
    expect(() => normalizeAberration(null)).toThrow(TypeError);
    expect(() => normalizeAberration([0.003])).toThrow(TypeError);
  });
});

describe('aberrationOffset', () => {
  test('amount 0 offsets every channel by exactly zero at every point', () => {
    const stops = [-1, -0.75, -0.3, 0, 0.3, 0.75, 1, 1.2];
    for (const x of stops) {
      for (const y of stops) {
        const out = sampleOffsets(x, y, 0);
        for (let i = 0; i < 6; i += 1) expect(out[i]).toBe(0);
      }
    }
  });

  test('the center never moves, for any amount', () => {
    for (const amount of [0.001, DEFAULTS.amount, 0.02]) {
      const out = sampleOffsets(0, 0, amount);
      for (let i = 0; i < 6; i += 1) expect(out[i]).toBe(0);
    }
  });

  test('magnitude follows amount * r^2 and grows strictly with radius', () => {
    let previous = -1;
    for (const radius of [0.1, 0.25, 0.5, 0.75, 0.9, 1]) {
      const magnitude = redMagnitude(0.8 * radius, 0.6 * radius);
      expect(magnitude).toBeCloseTo(DEFAULTS.amount * radius * radius, 12);
      expect(magnitude).toBeGreaterThan(previous);
      previous = magnitude;
    }
    // Corner semantics: at radius 1 the separation is exactly the amount.
    expect(redMagnitude(1, 0)).toBe(DEFAULTS.amount);
    expect(redMagnitude(0, 1)).toBe(DEFAULTS.amount);
  });

  test('R is radially outward, B exactly opposite, G exactly zero', () => {
    const points = [[0.3, 0.4], [-0.6, 0.2], [0.5, -0.5], [0.9, 0.05]];
    for (const [x, y] of points) {
      const out = sampleOffsets(x, y);
      const radius = Math.hypot(x, y);
      // Collinear with the position vector, pointing away from the center.
      expect(out[0] * y - out[1] * x).toBeCloseTo(0, 15);
      expect(out[0] * x + out[1] * y).toBeGreaterThan(0);
      // B is the exact negation of R; G never moves.
      expect(out[4]).toBe(-out[0]);
      expect(out[5]).toBe(-out[1]);
      expect(out[2]).toBe(0);
      expect(out[3]).toBe(0);
      expect(Math.hypot(out[0], out[1])).toBeCloseTo(DEFAULTS.amount * radius * radius, 12);
    }
  });

  test('equal corrected radii at different angles give equal magnitudes', () => {
    const axis = redMagnitude(0.6, 0);
    expect(redMagnitude(0, 0.6)).toBe(axis);
    expect(redMagnitude(-0.6, 0)).toBe(axis);
    expect(redMagnitude(0, -0.6)).toBe(axis);
    const diagonal = Math.SQRT1_2 * 0.6;
    expect(redMagnitude(diagonal, diagonal)).toBeCloseTo(axis, 12);
    expect(redMagnitude(-diagonal, diagonal)).toBeCloseTo(axis, 12);
  });

  test('aspect correction keeps equal pixel distances equal in magnitude', () => {
    // 30 source pixels from the center of a 200x100 source, along both axes
    // and diagonally: identical corrected radius, identical magnitude.
    const width = 200;
    const height = 100;
    const across = cornerUnit(0.5 + 30 / width, 0.5, width, height);
    const down = cornerUnit(0.5, 0.5 + 30 / height, width, height);
    const diagonalStep = 30 / Math.SQRT2;
    const diagonal = cornerUnit(
      0.5 + diagonalStep / width, 0.5 + diagonalStep / height, width, height,
    );
    const magnitude = redMagnitude(across.x, across.y);
    expect(redMagnitude(down.x, down.y)).toBeCloseTo(magnitude, 12);
    expect(redMagnitude(diagonal.x, diagonal.y)).toBeCloseTo(magnitude, 12);
    // Without the correction those same pixel distances would not match.
    expect(redMagnitude(0.3, 0)).not.toBe(redMagnitude(0, 0.6));
  });

  test('writes into the caller buffer and validates inputs', () => {
    expect(aberrationOffset({ x: 0.3, y: 0.4 }, normalizeAberration(), offsets)).toBe(offsets);
    expect(() => aberrationOffset({ x: 0.3, y: 0.4 }, normalizeAberration(), new Float64Array(5)))
      .toThrow(RangeError);
    expect(() => aberrationOffset({ x: 0.3, y: 0.4 }, normalizeAberration(), [0, 0, 0, 0, 0, 0]))
      .toThrow(TypeError);
    expect(() => aberrationOffset({ x: Number.NaN, y: 0.4 }, normalizeAberration(), offsets))
      .toThrow(TypeError);
    expect(() => aberrationOffset({ x: 0.3 }, normalizeAberration(), offsets)).toThrow(TypeError);
    expect(() => aberrationOffset({ x: 0.3, y: 0.4 }, { amount: 'x' }, offsets)).toThrow(TypeError);
    expect(() => aberrationOffset({ x: 0.3, y: 0.4 }, { amount: 0.5 }, offsets))
      .toThrow(RangeError);
  });
});

describe('crtSampleChromatic GLSL', () => {
  test('declares only its own uniform and the three-tap sampling function', () => {
    expect(GLSL).toContain('uniform float uCrtCaAmount;');
    expect(GLSL).toContain('vec3 crtSampleChromatic(vec2 warpedUV)');
    expect((GLSL.match(/texture2D/g) || []).length).toBe(3);
    // The composer owns the header and the bulge warp; never redeclare them.
    expect(GLSL).not.toContain('precision ');
    expect(GLSL).not.toContain('varying ');
    expect(GLSL).not.toContain('uCrtSource;');
    expect(GLSL).not.toContain('crtWarpUV');
  });
});
