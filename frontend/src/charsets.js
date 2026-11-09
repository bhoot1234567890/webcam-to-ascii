import inkUsage from '../../ink-usage.json';

const COMPACT = ' .:-=+*#%@';
const MINIMAL = ' .-+*#';

function entriesFor(chars) {
  const allowed = new Set(chars);
  return Object.freeze(inkUsage.filter(([glyph]) => allowed.has(glyph)));
}

const compactEntries = entriesFor(COMPACT);
const minimalEntries = entriesFor(MINIMAL);

export const CHARSETS = Object.freeze([
  Object.freeze({
    id: 'dense',
    label: 'Dense',
    description: 'All 95 measured glyphs.',
    fullEntries: Object.freeze(inkUsage.map((entry) => Object.freeze(entry))),
    compactEntries,
  }),
  Object.freeze({
    id: 'simple',
    label: 'Simple',
    description: 'Ten tonal glyphs.',
    fullEntries: compactEntries,
    compactEntries,
  }),
  Object.freeze({
    id: 'minimal',
    label: 'Minimal',
    description: 'Six tonal glyphs.',
    fullEntries: minimalEntries,
    compactEntries: minimalEntries,
  }),
]);

export const DEFAULT_CHAR_SET = 'dense';

export function findCharSet(id) {
  return CHARSETS.find((set) => set.id === id) || null;
}

export function normalizeCharSet(id = DEFAULT_CHAR_SET) {
  if (typeof id !== 'string') throw new TypeError('Character set ID must be a string');
  if (!findCharSet(id)) throw new RangeError('Choose a supported character set');
  return id;
}
