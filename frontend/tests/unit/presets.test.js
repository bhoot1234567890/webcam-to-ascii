import { describe, expect, test } from 'bun:test';
import {
  PRESET_LIMIT,
  PRESET_NAME_MAX,
  normalizePreset,
  normalizePresetList,
  normalizePresetName,
  removePreset,
  upsertPreset,
} from '../../src/presets.js';
import { normalizeSettings } from '../../src/renderer.js';
import { normalizeLook } from '../../src/looks.js';

function captured(fn) {
  try {
    fn();
  } catch (error) {
    return error;
  }
  throw new Error('Expected the call to throw');
}

function presetNamed(name) {
  return normalizePreset({ name, settings: { sigma1: 1.5 }, look: { id: 'print' }, framing: 'full', tint: 'default', glow: false });
}

describe('normalizePresetName', () => {
  test('trims surrounding space and keeps the inner name', () => {
    expect(normalizePresetName('  Studio  ')).toBe('Studio');
    expect(normalizePresetName('a'.repeat(PRESET_NAME_MAX))).toBe('a'.repeat(PRESET_NAME_MAX));
  });

  test('rejects non-strings, blank names, long names, and control characters', () => {
    const notString = captured(() => normalizePresetName(7));
    expect(notString).toBeInstanceOf(TypeError);
    expect(notString.message).toBe('Preset name must be a string');

    for (const blank of ['', '   ', '\n\t ']) {
      const error = captured(() => normalizePresetName(blank));
      expect(error).toBeInstanceOf(RangeError);
      expect(error.message).toBe('Enter a preset name.');
    }

    const long = captured(() => normalizePresetName('a'.repeat(PRESET_NAME_MAX + 1)));
    expect(long).toBeInstanceOf(RangeError);
    expect(long.message).toBe('Preset names are 1 to 24 characters.');

    for (const control of ['Null\u0000Byte', 'Bell\u0007', 'Delete\u007f']) {
      const error = captured(() => normalizePresetName(control));
      expect(error).toBeInstanceOf(RangeError);
      expect(error.message).toBe('Preset names cannot contain control characters.');
    }
  });
});

describe('normalizePreset', () => {
  test('normalizes a complete preset, trimming the name and validating each field', () => {
    const preset = normalizePreset({
      name: '  Studio  ',
      settings: { maxWidthChars: 75, maxHeightLines: 100, sigma1: 1.5, sigma2: 2.5, edgeIndexThreshold: 70 },
      look: { id: 'photo', midtone: 1.2, autoContrast: true },
      framing: 'square',
      tint: '#ffd663',
      glow: true,
    });
    expect(preset).toEqual({
      name: 'Studio',
      settings: normalizeSettings({ maxWidthChars: 75, maxHeightLines: 100, sigma1: 1.5, sigma2: 2.5, edgeIndexThreshold: 70 }),
      look: normalizeLook({ id: 'photo', midtone: 1.2, autoContrast: true }),
      framing: 'square',
      tint: '#ffd663',
      glow: true,
    });
  });

  test('falls back to the default settings and look when they are absent', () => {
    const preset = normalizePreset({ name: 'Bare', framing: 'full', tint: 'default', glow: false });
    expect(preset.settings).toEqual(normalizeSettings({}));
    expect(preset.look).toEqual(normalizeLook({}));
    expect(normalizePreset({ name: 'Neon', framing: 'full', tint: 'neon', glow: true }).tint).toBe('neon');
  });

  test('rejects non-object presets, unsupported frames, bad tints, and bad glow values', () => {
    for (const value of [null, undefined, 'Studio', 42, ['Studio']]) {
      const error = captured(() => normalizePreset(value));
      expect(error).toBeInstanceOf(TypeError);
      expect(error.message).toBe('Preset must be an object');
    }

    const frame = captured(() => normalizePreset({ name: 'X', framing: 'panorama', tint: 'default', glow: false }));
    expect(frame).toBeInstanceOf(RangeError);
    expect(frame.message).toBe('Choose a supported frame');

    for (const tint of ['Amber', '#fff', '#ffd663ff', 12]) {
      const error = captured(() => normalizePreset({ name: 'X', framing: 'full', tint, glow: false }));
      expect(error).toBeInstanceOf(RangeError);
      expect(error.message).toBe('Choose a valid text color and glow setting');
    }

    for (const glow of ['true', 1, null]) {
      const error = captured(() => normalizePreset({ name: 'X', framing: 'full', tint: 'default', glow }));
      expect(error).toBeInstanceOf(TypeError);
      expect(error.message).toBe('Glow must be a boolean');
    }
  });
});

describe('upsertPreset', () => {
  test('appends a new name without mutating the input array', () => {
    const first = presetNamed('Studio');
    const before = [first];
    const next = upsertPreset(before, presetNamed('Night'));
    expect(before).toEqual([first]);
    expect(next.map((preset) => preset.name)).toEqual(['Studio', 'Night']);
  });

  test('replaces a same-name entry in place and keeps its position', () => {
    const before = [presetNamed('Studio'), presetNamed('Night')];
    const replacement = normalizePreset({
      name: 'Studio', settings: { sigma1: 3 }, look: { id: 'outline' }, framing: 'wide', tint: '#9be5a2', glow: true,
    });
    const next = upsertPreset(before, replacement);
    expect(before[0].look.id).toBe('print');
    expect(next.map((preset) => preset.name)).toEqual(['Studio', 'Night']);
    expect(next[0]).toEqual(replacement);
  });

  test('throws only when a new name would exceed the limit', () => {
    const full = Array.from({ length: PRESET_LIMIT }, (_, index) => presetNamed(`Preset ${index + 1}`));
    const error = captured(() => upsertPreset(full, presetNamed('One more')));
    expect(error).toBeInstanceOf(RangeError);
    expect(error.message).toBe('Preset limit reached (24). Delete one first.');
    const replaced = upsertPreset(full, presetNamed('Preset 12'));
    expect(replaced).toHaveLength(PRESET_LIMIT);
  });
});

describe('removePreset', () => {
  test('removes only the matching name', () => {
    const studio = presetNamed('Studio');
    const studioLower = presetNamed('studio');
    const night = presetNamed('Night');
    expect(removePreset([studio, studioLower, night], 'Studio')).toEqual([studioLower, night]);
    expect(removePreset([studio, night], 'Missing')).toEqual([studio, night]);
  });
});

describe('normalizePresetList', () => {
  test('drops malformed entries and keeps the valid ones in order', () => {
    const list = normalizePresetList([
      { name: 'Studio', framing: 'square', tint: '#ffd663', glow: true },
      null,
      { name: '', framing: 'square', tint: 'default', glow: false },
      { name: 'Broken', framing: 'panorama', tint: 'default', glow: false },
      { name: 'Night', framing: 'full', tint: 'default', glow: false },
    ]);
    expect(list.map((preset) => preset.name)).toEqual(['Studio', 'Night']);
  });

  test('returns an empty list for non-arrays and caps the result', () => {
    expect(normalizePresetList(null)).toEqual([]);
    expect(normalizePresetList({ 0: presetNamed('Studio') })).toEqual([]);
    expect(normalizePresetList('Studio')).toEqual([]);
    const capped = normalizePresetList(Array.from({ length: PRESET_LIMIT + 6 }, (_, index) => (
      { name: `Preset ${index + 1}`, framing: 'full', tint: 'default', glow: false }
    )));
    expect(capped).toHaveLength(PRESET_LIMIT);
    expect(capped.at(-1).name).toBe(`Preset ${PRESET_LIMIT}`);
  });
});
