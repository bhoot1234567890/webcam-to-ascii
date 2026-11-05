const neutral = { autoContrast: false, midtone: 1, sharpness: 0, invert: false };

export const LOOKS = Object.freeze([
  { id: 'original', label: 'Original', description: 'Full measured shading with directional contours.', edges: true },
  { id: 'photo', label: 'Photo', description: 'Smooth photographic tones with diffused shading.', edges: false },
  { id: 'print', label: 'Print', description: 'Stable ordered halftone shading.', edges: false },
  { id: 'outline', label: 'Outline', description: 'Directional contours with empty interiors.', edges: true },
].map((recipe) => Object.freeze({ ...recipe, ...neutral })));

export function normalizeLook(look = {}) {
  if (look === null || typeof look !== 'object' || Array.isArray(look)) {
    throw new TypeError('Look must be an object');
  }
  const normalized = { id: 'original', ...neutral };
  for (const key of Object.keys(normalized)) {
    if (Object.hasOwn(look, key)) normalized[key] = look[key];
  }
  if (typeof normalized.id !== 'string' || !LOOKS.some((recipe) => recipe.id === normalized.id)) {
    throw new RangeError('Unknown look');
  }
  for (const key of ['autoContrast', 'invert']) {
    if (typeof normalized[key] !== 'boolean') throw new TypeError(`${key} must be a boolean`);
  }
  for (const [key, minimum, maximum] of [['midtone', 0.5, 2], ['sharpness', 0, 1.5]]) {
    if (typeof normalized[key] !== 'number' || !Number.isFinite(normalized[key])
        || normalized[key] < minimum || normalized[key] > maximum) {
      throw new RangeError(`${key} must be a finite number from ${minimum} to ${maximum}`);
    }
  }
  return normalized;
}
