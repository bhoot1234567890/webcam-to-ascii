import { expect, test } from 'bun:test';
import {
  EXPORT_FRAMES, DEFAULT_EXPORT_FRAME, normalizeExportFrame, createExportFrame,
} from '../../src/export-frames.js';

test('export frame ids are unique and the default is one of them', () => {
  const ids = EXPORT_FRAMES.map(({ id }) => id);
  expect(new Set(ids).size).toBe(ids.length);
  expect(ids).toContain(DEFAULT_EXPORT_FRAME);
  expect(Object.isFrozen(EXPORT_FRAMES)).toBe(true);
});

test('export frame selection falls back without inventing a style', () => {
  expect(normalizeExportFrame('mac')).toBe('mac');
  expect(normalizeExportFrame('none')).toBe('none');
  expect(normalizeExportFrame('winxp')).toBe('winxp');
  expect(normalizeExportFrame('bogus')).toBe(DEFAULT_EXPORT_FRAME);
  expect(normalizeExportFrame('bogus', 'none')).toBe('none');
  expect(normalizeExportFrame('bogus', 'bogus')).toBe('none');
});

test('none passes the already-rendered canvas through untouched', () => {
  const source = { width: 20, height: 10 };
  expect(createExportFrame(source, 'none')).toBe(source);
});
