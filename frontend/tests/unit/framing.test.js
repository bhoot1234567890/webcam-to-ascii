import { test, expect } from 'bun:test';
import { computeCropRect, FRAMINGS } from '../../src/framing.js';

test('Full keeps every original pixel, including single-pixel and odd sources', () => {
  for (const [width, height] of [[1, 1], [101, 83], [640, 480], [480, 640]]) {
    expect(computeCropRect(width, height)).toEqual({ x: 0, y: 0, width, height });
  }
});

test('center crop removes only the intended outer bands without stretching', () => {
  expect(computeCropRect(640, 480, 'square')).toEqual({ x: 80, y: 0, width: 480, height: 480 });
  expect(computeCropRect(480, 640, 'square')).toEqual({ x: 0, y: 80, width: 480, height: 480 });
  expect(computeCropRect(512, 512, 'story')).toEqual({ x: 112, y: 0, width: 288, height: 512 });
  expect(computeCropRect(512, 512, 'wide')).toEqual({ x: 0, y: 112, width: 512, height: 288 });
  expect(computeCropRect(101, 83, 'portrait')).toEqual({ x: 20, y: 1, width: 60, height: 80 });
});

for (const frame of FRAMINGS.filter(({ id }) => id !== 'full')) {
  test(`${frame.label} stays centered, in bounds and exact at odd source sizes`, () => {
    for (const [width, height] of [[101, 83], [83, 101], [101, 101], [640, 481], [481, 640]]) {
      const rect = computeCropRect(width, height, frame.id);
      expect(rect.width * frame.height).toBe(rect.height * frame.width);
      expect(rect.x).toBeGreaterThanOrEqual(0);
      expect(rect.y).toBeGreaterThanOrEqual(0);
      expect(rect.x + rect.width).toBeLessThanOrEqual(width);
      expect(rect.y + rect.height).toBeLessThanOrEqual(height);
      expect(Math.abs(rect.x + rect.width / 2 - width / 2)).toBeLessThanOrEqual(0.5);
      expect(Math.abs(rect.y + rect.height / 2 - height / 2)).toBeLessThanOrEqual(0.5);
      // It is maximal among exact whole-pixel rectangles, not an arbitrary smaller crop.
      expect(rect.width + frame.width > width || rect.height + frame.height > height).toBe(true);
    }
  });
}

test('invalid IDs and source dimensions do not silently fall back to Full', () => {
  for (const id of ['', '16:9', 'missing', null, 1, {}, []]) {
    expect(() => computeCropRect(100, 100, id)).toThrow('Choose a supported frame');
  }
  for (const value of [0, -1, 1.5, NaN, Infinity, '100', null, Number.MAX_SAFE_INTEGER + 1]) {
    expect(() => computeCropRect(value, 100)).toThrow(RangeError);
    expect(() => computeCropRect(100, value)).toThrow(RangeError);
  }
});

test('tiny unrepresentable crops fail; the minimum exact rectangle succeeds', () => {
  for (const frame of FRAMINGS.filter(({ id }) => id !== 'full')) {
    expect(computeCropRect(frame.width, frame.height, frame.id))
      .toEqual({ x: 0, y: 0, width: frame.width, height: frame.height });
    if (frame.width > 1) expect(() => computeCropRect(frame.width - 1, frame.height, frame.id)).toThrow('too small');
    if (frame.height > 1) expect(() => computeCropRect(frame.width, frame.height - 1, frame.id)).toThrow('too small');
  }
});
