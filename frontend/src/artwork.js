export const ARTWORK_BACKGROUND = '#09090b';

function validateText(text, columns, rows) {
  if (typeof text !== 'string') throw new Error('The ASCII frame has no committed text.');
  const lines = text.split('\n');
  if (lines.length !== rows || lines.some((line) => line.length !== columns || /[^\x20-\x7e]/.test(line))) {
    throw new Error('The ASCII grid is incomplete or changed. Wait for the preview to settle and try again.');
  }
  return lines;
}

function validateFrame(frame) {
  if (!frame || ![frame.columns, frame.rows].every((value) => Number.isSafeInteger(value) && value > 0) ||
      ![frame.fontSize, frame.lineHeight].every((value) => Number.isFinite(value) && value > 0) ||
      typeof frame.fontFamily !== 'string' ||
      !frame.fontFamily.split(',').some((family) => family.trim().replace(/^["']|["']$/g, '') === 'Roboto Mono') ||
      typeof frame.color !== 'string' || !frame.color.trim() ||
      typeof frame.background !== 'string' || !frame.background.trim() || typeof frame.glow !== 'boolean' ||
      (frame.crt !== undefined && typeof frame.crt !== 'boolean')) {
    throw new Error('The ASCII frame is not ready. Wait for the preview and try again.');
  }
  validateText(frame.text, frame.columns, frame.rows);
}

// Computed styles and canvas fillStyle serialize alpha as rgba() or a slash component.
function isOpaque(color) {
  const value = color.trim().toLowerCase();
  if (!value || value === 'transparent') return false;
  const alpha = value.includes('/') ? value.slice(value.lastIndexOf('/') + 1, -1).trim() :
    /^rgba\(/.test(value) ? value.slice(value.lastIndexOf(',') + 1, -1).trim() : null;
  if (alpha !== null) return Number.parseFloat(alpha) === (alpha.endsWith('%') ? 100 : 1);
  if (/^#[\da-f]{4}$/.test(value)) return value.endsWith('f');
  if (/^#[\da-f]{8}$/.test(value)) return value.endsWith('ff');
  return true;
}

/** Snapshot the committed grid and its untransformed presentation, not the camera or fitted DOM size. */
export function readAsciiArtwork(element, { glow = false, crt = false } = {}) {
  if (!element?.dataset || typeof element.textContent !== 'string') {
    throw new TypeError('Expected the committed ASCII output element.');
  }
  const style = getComputedStyle(element);
  const viewport = element.closest('.ascii-viewport');
  const background = viewport ? getComputedStyle(viewport).backgroundColor : '';
  const frame = {
    text: element.textContent,
    columns: Number(element.dataset.columns),
    rows: Number(element.dataset.rows),
    fontFamily: style.fontFamily,
    fontSize: Number.parseFloat(style.fontSize),
    lineHeight: Number.parseFloat(style.lineHeight),
    color: style.color,
    background: isOpaque(background) ? background : ARTWORK_BACKGROUND,
    glow,
    crt,
  };
  validateFrame(frame);
  return Object.freeze(frame);
}

/** Ink gutters and one uniform scale keep every row intact, even at odd encoder bounds. */
export function computeArtworkLayout(frame, metrics, { scale: requestedScale = 1, maxEdge = 1280, evenDimensions = true } = {}) {
  const { columns, rows, lineHeight, glow } = frame;
  const { advance, ascent, descent, left, right } = metrics;
  if (![columns, rows].every((value) => Number.isSafeInteger(value) && value > 0) ||
      ![lineHeight, advance, requestedScale].every((value) => Number.isFinite(value) && value > 0) ||
      ![ascent, descent, left, right].every((value) => Number.isFinite(value) && value >= 0) ||
      !(maxEdge > 0 && (Number.isFinite(maxEdge) || maxEdge === Infinity)) || typeof evenDimensions !== 'boolean') {
    throw new RangeError('Unable to measure the ASCII artwork grid.');
  }
  const quantum = evenDimensions ? 2 : 1;
  const edge = Math.floor(maxEdge / quantum) * quantum;
  if (edge < quantum) throw new RangeError('The ASCII artwork bound is too small.');
  const baseline = (lineHeight - ascent - descent) / 2 + ascent;
  const halo = glow ? 12 : 0;
  const gutterLeft = Math.ceil(left + halo + 1);
  const gutterRight = Math.ceil(right + halo + 1);
  const gutterTop = Math.ceil(Math.max(0, ascent - baseline) + halo + 1);
  const gutterBottom = Math.ceil(Math.max(0, baseline + descent - lineHeight) + halo + 1);
  const naturalWidth = columns * advance + gutterLeft + gutterRight;
  const naturalHeight = rows * lineHeight + gutterTop + gutterBottom;
  if (![baseline, naturalWidth, naturalHeight].every(Number.isFinite)) {
    throw new RangeError('The ASCII artwork grid is too large.');
  }
  const scale = Math.min(requestedScale, edge / Math.max(naturalWidth, naturalHeight));
  const width = Math.max(quantum, Math.min(edge, Math.ceil(naturalWidth * scale / quantum) * quantum));
  const height = Math.max(quantum, Math.min(edge, Math.ceil(naturalHeight * scale / quantum) * quantum));
  if (!(scale > 0) || ![width, height].every((value) => Number.isSafeInteger(value) && value <= 0xffffffff)) {
    throw new RangeError('The ASCII artwork dimensions are unsupported.');
  }
  return Object.freeze({
    width, height, scale, baseline, advance,
    x: gutterLeft * scale + (width - naturalWidth * scale) / 2,
    y: gutterTop * scale + (height - naturalHeight * scale) / 2,
  });
}

function configureFont(context, font) {
  context.font = font;
  context.fontKerning = 'none';
  context.textAlign = 'left';
  context.textBaseline = 'alphabetic';
}

function measureFont(context) {
  const advance = context.measureText('M').width;
  let ascent = 0;
  let descent = 0;
  let left = 0;
  let right = 0;
  // Include punctuation overhang and descenders, not merely glyphs in the first frame.
  for (let code = 32; code <= 126; code++) {
    const bounds = context.measureText(String.fromCharCode(code));
    ascent = Math.max(ascent, bounds.fontBoundingBoxAscent ?? 0, bounds.actualBoundingBoxAscent ?? 0);
    descent = Math.max(descent, bounds.fontBoundingBoxDescent ?? 0, bounds.actualBoundingBoxDescent ?? 0);
    left = Math.max(left, bounds.actualBoundingBoxLeft ?? 0);
    right = Math.max(right, (bounds.actualBoundingBoxRight ?? 0) - advance);
  }
  if (!(ascent > 0)) throw new Error('Unable to measure the ASCII font. Reload the page and try again.');
  return { advance, ascent, descent, left, right };
}

function resolveColor(context, color) {
  context.fillStyle = '#010203';
  context.fillStyle = color;
  if (context.fillStyle === '#010203') {
    context.fillStyle = '#040506';
    context.fillStyle = color;
    if (context.fillStyle === '#040506') throw new Error('The ASCII artwork color is invalid.');
  }
  return context.fillStyle;
}

/** Measure once without drawing; subsequent commits use only this immutable canvas presentation. */
export function createAsciiArtwork(frame, options = {}) {
  validateFrame(frame);
  const presentation = Object.freeze({ ...frame });
  const font = `400 ${presentation.fontSize}px ${presentation.fontFamily}`;
  if (!document.fonts?.check(font, 'M@ ')) {
    throw new Error('The ASCII font is still loading. Wait for the preview and try again.');
  }
  const canvas = document.createElement('canvas');
  try {
    const context = canvas.getContext('2d', { alpha: false });
    if (!context) throw new Error('Unable to create the ASCII artwork canvas. Try reloading the page.');
    configureFont(context, font);
    const layout = computeArtworkLayout(presentation, measureFont(context), options);
    const color = resolveColor(context, presentation.color);
    const background = resolveColor(context, presentation.background);
    if (!isOpaque(background)) throw new Error('The ASCII artwork background must be opaque.');
    canvas.width = layout.width;
    canvas.height = layout.height;
    // Setting dimensions resets context state, including the measured font and baseline.
    configureFont(context, font);
    return {
      canvas,
      layout,
      draw(text) {
        // Validate before touching pixels so a malformed commit cannot erase the last good frame.
        const lines = validateText(text, presentation.columns, presentation.rows);
        context.setTransform(1, 0, 0, 1, 0, 0);
        context.shadowBlur = 0;
        context.fillStyle = background;
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.setTransform(layout.scale, 0, 0, layout.scale, layout.x, layout.y);
        context.fillStyle = color;
        if (presentation.glow) {
          // Canvas shadow blur uses device pixels; gutters reserve twice this bounded radius.
          context.shadowColor = color;
          context.shadowBlur = 6 * layout.scale;
          for (let row = 0; row < lines.length; row++) context.fillText(lines[row], 0, layout.baseline + row * presentation.lineHeight);
          context.shadowBlur = 0;
        }
        for (let row = 0; row < lines.length; row++) context.fillText(lines[row], 0, layout.baseline + row * presentation.lineHeight);
      },
    };
  } catch (error) {
    canvas.width = 0;
    canvas.height = 0;
    throw error;
  }
}
