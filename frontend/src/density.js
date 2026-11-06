export const DENSITIES = Object.freeze([
  { id: 'chunky', label: 'Chunky', description: 'Bold typewriter glyphs (~60 cols)', maxWidthChars: 30, maxHeightLines: 40 },
  { id: 'balanced', label: 'Balanced', description: 'Legible characters + crisp image (~100 cols)', maxWidthChars: 50, maxHeightLines: 65 },
  { id: 'fine', label: 'Fine', description: 'Sharper detail, smaller text (~150 cols)', maxWidthChars: 75, maxHeightLines: 100 },
  { id: 'detailed', label: 'Detailed', description: 'Ultra-fine halftone grain (~300 cols)', maxWidthChars: 150, maxHeightLines: 200 },
].map(Object.freeze));

export function findDensity(id) {
  if (typeof id !== 'string') return null;
  return DENSITIES.find((density) => density.id === id) || null;
}

export function findDensityBySettings(settings) {
  if (!settings || typeof settings !== 'object') return null;
  return DENSITIES.find((density) =>
    density.maxWidthChars === settings.maxWidthChars &&
    density.maxHeightLines === settings.maxHeightLines,
  ) || null;
}

export function normalizeDensity(density = 'balanced') {
  const id = typeof density === 'object' && density !== null ? density.id : density;
  if (typeof id !== 'string') {
    throw new TypeError('Density ID must be a string');
  }
  const found = findDensity(id);
  if (!found) {
    throw new RangeError('Choose a supported density');
  }
  return found.id;
}
