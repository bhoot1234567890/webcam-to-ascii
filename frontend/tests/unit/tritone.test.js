import { describe, expect, test } from 'bun:test';
import {
  NEON_ACCENT,
  NEON_PRIMARY,
  TRITONE_DITHER_BAND,
  TRITONE_THRESHOLD,
  buildTritoneLayers,
  buildTritonePixels,
  createTritoneFilter,
  tritoneAccent,
} from '../../src/tritone.js';

const rgb = (hex) => [1, 3, 5].map((offset) => Number.parseInt(hex.slice(offset, offset + 2), 16));

describe('tritoneAccent', () => {
  test('keeps deep shadows purple and highlights pink', () => {
    expect(tritoneAccent(0, 0, 0)).toBe(false);
    expect(tritoneAccent(255, 0, 0)).toBe(true);
    expect(tritoneAccent(255, 3, 3)).toBe(true);
  });

  test('only the narrow midtone seam dithers', () => {
    const low = Math.round((TRITONE_THRESHOLD - TRITONE_DITHER_BAND) * 255);
    const high = Math.round((TRITONE_THRESHOLD + TRITONE_DITHER_BAND) * 255);
    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < 4; x++) {
        expect(tritoneAccent(low, x, y)).toBe(false);
        expect(tritoneAccent(high, x, y)).toBe(true);
      }
    }
    const seam = [0, 1, 2, 3].flatMap((y) => [0, 1, 2, 3].map((x) => tritoneAccent(128, x, y)));
    expect(seam).toContain(true);
    expect(seam).toContain(false);
  });
});

describe('buildTritonePixels', () => {
  test('packs one RGBA pixel per cell and uses the exact ink colors', () => {
    const pixels = buildTritonePixels(Uint8Array.from([0, 255, 255, 0]), 2, 2);
    expect(pixels).toHaveLength(16);
    expect([...pixels.subarray(0, 4)]).toEqual([...rgb(NEON_PRIMARY), 255]);
    expect([...pixels.subarray(4, 8)]).toEqual([...rgb(NEON_ACCENT), 255]);
    expect([...pixels.subarray(8, 12)]).toEqual([...rgb(NEON_ACCENT), 255]);
    expect([...pixels.subarray(12, 16)]).toEqual([...rgb(NEON_PRIMARY), 255]);
  });

  test('rejects mismatched grids and tone planes', () => {
    expect(() => buildTritonePixels(new Uint8Array(3), 2, 2)).toThrow('tone plane');
    expect(() => buildTritonePixels(Uint8Array.from([0, 0, 0, 0]), 0, 2)).toThrow(RangeError);
  });
});

describe('createTritoneFilter', () => {
  test('a still commit is exact and live sensor jitter does not repaint the seam', () => {
    const filter = createTritoneFilter();
    const tones = Uint8Array.from([0, 96, 128, 144, 200, 255]);
    const first = filter.update(tones, 3, 2);
    expect(first.changed).toBe(true);
    expect([...first.tones]).toEqual([...tones]);
    for (let frame = 0; frame < 24; frame++) {
      const jitter = Uint8Array.from(tones, (tone, index) => {
        const delta = ((frame + index) & 1) === 0 ? 6 : -6;
        return Math.max(0, Math.min(255, tone + delta));
      });
      const next = filter.update(jitter, 3, 2);
      expect(next.changed).toBe(false);
      expect([...next.tones]).toEqual([...tones]);
    }
  });

  test('a real tone shift crosses the deadband and a reset restores exact colors', () => {
    const filter = createTritoneFilter();
    filter.update(Uint8Array.from([130, 130, 130, 130]), 2, 2);
    let changed = false;
    for (let frame = 0; frame < 12; frame++) {
      changed ||= filter.update(Uint8Array.from([190, 190, 190, 190]), 2, 2).changed;
    }
    expect(changed).toBe(true);
    filter.reset();
    const reset = filter.update(Uint8Array.from([64, 64, 64, 64]), 2, 2);
    expect(reset.changed).toBe(true);
    expect([...reset.tones]).toEqual([64, 64, 64, 64]);
    const nextGrid = filter.update(Uint8Array.from([10, 20, 30, 40, 50, 60]), 3, 2);
    expect(nextGrid.changed).toBe(true);
    expect([...nextGrid.tones]).toEqual([10, 20, 30, 40, 50, 60]);
  });

  test('auto-contrast pumping does not animate the color seam', () => {
    const filter = createTritoneFilter();
    filter.update(Uint8Array.from([128, 128, 128, 128]), 2, 2);
    for (let frame = 0; frame < 30; frame++) {
      const tone = frame & 1 ? 178 : 78;
      const next = filter.update(Uint8Array.from([tone, tone, tone, tone]), 2, 2);
      expect(next.changed).toBe(false);
    }
  });

  test('rejects invalid timing and tone grids', () => {
    expect(() => createTritoneFilter({ alpha: 0 })).toThrow(RangeError);
    expect(() => createTritoneFilter({ alpha: 1.1 })).toThrow(RangeError);
    expect(() => createTritoneFilter({ margin: -1 })).toThrow(RangeError);
    const filter = createTritoneFilter();
    expect(() => filter.update(new Uint8Array(3), 2, 2)).toThrow('tone plane');
    expect(() => filter.update(Uint8Array.from([0, 0, 0, 0]), 0, 2)).toThrow(RangeError);
  });
});

describe('buildTritoneLayers', () => {
  test('splits glyphs without changing either row width', () => {
    const layers = buildTritoneLayers('ab\ncd', Uint8Array.from([0, 255, 255, 0]), 2, 2);
    expect(layers.primary).toEqual(['a ', ' d']);
    expect(layers.accent).toEqual([' b', 'c ']);
  });

  test('preserves spaces in both layers', () => {
    const layers = buildTritoneLayers('a b\nc d', Uint8Array.from([0, 0, 255, 0, 0, 255]), 3, 2);
    expect(layers.primary).toEqual(['a  ', 'c  ']);
    expect(layers.accent).toEqual(['  b', '  d']);
  });

  test('rejects malformed text and tones', () => {
    expect(() => buildTritoneLayers('ab\nc', Uint8Array.from([0, 0, 0, 0]), 2, 2)).toThrow('grid');
    expect(() => buildTritoneLayers('ab\ncd', new Uint8Array(3), 2, 2)).toThrow('tone plane');
    expect(() => buildTritoneLayers(null, Uint8Array.from([0, 0, 0, 0]), 2, 2)).toThrow(TypeError);
  });
});
