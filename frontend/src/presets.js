import { normalizeSettings } from './renderer.js';
import { normalizeLook } from './looks.js';
import { FRAMINGS } from './framing.js';
import { NEON_TINT } from './tritone.js';

/** Named visual presets: pure validation and list operations. No DOM, no storage. */
export const PRESET_NAME_MAX = 24;
export const PRESET_LIMIT = 24;

const CONTROL_CHARS = /[\u0000-\u001f\u007f]/;
const HEX_COLOR = /^#[0-9a-f]{6}$/i;

export function normalizePresetName(name) {
  if (typeof name !== 'string') throw new TypeError('Preset name must be a string');
  const trimmed = name.trim();
  if (!trimmed) throw new RangeError('Enter a preset name.');
  if (trimmed.length > PRESET_NAME_MAX) throw new RangeError(`Preset names are 1 to ${PRESET_NAME_MAX} characters.`);
  if (CONTROL_CHARS.test(trimmed)) throw new RangeError('Preset names cannot contain control characters.');
  return trimmed;
}

function requireFrame(frame) {
  if (FRAMINGS.some(({ id }) => id === frame)) return frame;
  throw new RangeError('Choose a supported frame');
}

function requireTint(tint) {
  if (tint === 'default' || tint === NEON_TINT || (typeof tint === 'string' && HEX_COLOR.test(tint))) return tint;
  throw new RangeError('Choose a valid text color and glow setting');
}

function requireGlow(glow) {
  if (typeof glow === 'boolean') return glow;
  throw new TypeError('Glow must be a boolean');
}

export function normalizePreset(value) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError('Preset must be an object');
  }
  return {
    name: normalizePresetName(value.name),
    settings: normalizeSettings(value.settings ?? {}),
    look: normalizeLook(value.look ?? {}),
    framing: requireFrame(value.framing),
    tint: requireTint(value.tint),
    glow: requireGlow(value.glow),
  };
}

export function normalizePresetList(value) {
  if (!Array.isArray(value)) return [];
  const presets = [];
  for (const entry of value) {
    if (presets.length >= PRESET_LIMIT) break;
    try {
      presets.push(normalizePreset(entry));
    } catch {
      // A stored preset that fails validation is dropped, never repaired.
    }
  }
  return presets;
}

export function upsertPreset(presets, preset) {
  const next = [...presets];
  const index = next.findIndex((entry) => entry.name === preset.name);
  if (index >= 0) next[index] = preset;
  else if (next.length >= PRESET_LIMIT) throw new RangeError(`Preset limit reached (${PRESET_LIMIT}). Delete one first.`);
  else next.push(preset);
  return next;
}

export function removePreset(presets, name) {
  return presets.filter((preset) => preset.name !== name);
}
