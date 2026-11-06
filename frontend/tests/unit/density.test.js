import { describe, expect, test } from 'bun:test';
import { DENSITIES, findDensity, findDensityBySettings, normalizeDensity } from '../../src/density.js';

describe('DENSITIES presets', () => {
  test('exports 4 frozen density presets with required properties', () => {
    expect(DENSITIES).toHaveLength(4);
    expect(DENSITIES.map((d) => d.id)).toEqual(['chunky', 'balanced', 'fine', 'detailed']);
    expect(Object.isFrozen(DENSITIES)).toBe(true);
    for (const d of DENSITIES) {
      expect(Object.isFrozen(d)).toBe(true);
      expect(typeof d.label).toBe('string');
      expect(typeof d.description).toBe('string');
      expect(typeof d.maxWidthChars).toBe('number');
      expect(typeof d.maxHeightLines).toBe('number');
    }
  });

  test('findDensity looks up by ID or returns null', () => {
    expect(findDensity('chunky')?.maxWidthChars).toBe(30);
    expect(findDensity('balanced')?.maxWidthChars).toBe(50);
    expect(findDensity('fine')?.maxWidthChars).toBe(75);
    expect(findDensity('detailed')?.maxWidthChars).toBe(150);
    expect(findDensity('unknown')).toBeNull();
    expect(findDensity(123)).toBeNull();
    expect(findDensity(null)).toBeNull();
  });

  test('findDensityBySettings matches maxWidthChars and maxHeightLines exactly', () => {
    expect(findDensityBySettings({ maxWidthChars: 50, maxHeightLines: 65 })?.id).toBe('balanced');
    expect(findDensityBySettings({ maxWidthChars: 30, maxHeightLines: 40 })?.id).toBe('chunky');
    expect(findDensityBySettings({ maxWidthChars: 75, maxHeightLines: 100 })?.id).toBe('fine');
    expect(findDensityBySettings({ maxWidthChars: 150, maxHeightLines: 200 })?.id).toBe('detailed');
    expect(findDensityBySettings({ maxWidthChars: 50, maxHeightLines: 50 })).toBeNull();
    expect(findDensityBySettings(null)).toBeNull();
    expect(findDensityBySettings({})).toBeNull();
  });

  test('normalizeDensity validates and normalizes density IDs', () => {
    expect(normalizeDensity()).toBe('balanced');
    expect(normalizeDensity('chunky')).toBe('chunky');
    expect(normalizeDensity('balanced')).toBe('balanced');
    expect(normalizeDensity({ id: 'fine' })).toBe('fine');
    expect(() => normalizeDensity('invalid')).toThrow(RangeError);
    expect(() => normalizeDensity(123)).toThrow(TypeError);
    expect(() => normalizeDensity(null)).toThrow(TypeError);
  });
});
