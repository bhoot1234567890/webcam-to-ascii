import '../src/styles.css';
import { configureAsciiText } from '../src/presentation.js';

const left = document.querySelector('#left-grid');
const right = document.querySelector('#right-grid');
const source = document.querySelector('#source');
const fontReady = Promise.all([configureAsciiText(left, 12), configureAsciiText(right, 12)]);

window.setComparison = async ({ sourceUrl, leftText, rightText }) => {
  await fontReady;
  if (typeof sourceUrl !== 'string' || typeof leftText !== 'string' || typeof rightText !== 'string') {
    throw new TypeError('Comparison requires a source URL and two complete text grids');
  }
  const leftRows = leftText.split('\n');
  const rightRows = rightText.split('\n');
  const width = leftRows[0].length;
  if (!width || leftRows.length !== rightRows.length ||
      [...leftRows, ...rightRows].some((row) => row.length !== width || /[^\x20-\x7e]/.test(row))) {
    throw new RangeError('Comparison grids must have identical complete ASCII dimensions');
  }
  left.textContent = leftText;
  right.textContent = rightText;
  source.src = sourceUrl;
  await source.decode();
  if (!source.naturalWidth || !source.naturalHeight) throw new Error('Comparison source failed to decode');
  await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  // Integer outer boxes keep both native-resolution screenshots on the same pixel grid.
  // Character advance/line height remain the shared measured font geometry.
  for (const grid of [left, right]) {
    grid.style.width = 'max-content';
    grid.style.height = 'auto';
    grid.style.width = `${Math.ceil(grid.getBoundingClientRect().width)}px`;
    // Preserve descender ink extending below the final measured line box.
    const bottomGutter = parseFloat(getComputedStyle(grid).paddingBottom);
    grid.style.height = `${Math.ceil(parseFloat(getComputedStyle(grid).lineHeight) * leftRows.length) + bottomGutter}px`;
  }
};
