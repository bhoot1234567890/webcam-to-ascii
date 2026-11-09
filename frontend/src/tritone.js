export const NEON_TINT = 'neon';
export const NEON_PRIMARY = '#a855f7';
export const NEON_ACCENT = '#ff4fd8';
export const NEON_INK = Object.freeze({ color: NEON_PRIMARY, accent: NEON_ACCENT });

// A hard tritone split with a narrow ordered-dither seam keeps the two inks
// readable while avoiding a gradient or a contour line between them.
export const TRITONE_THRESHOLD = 0.5;
export const TRITONE_DITHER_BAND = 0.08;

// Live camera tones jitter by a few levels even when nothing moves. A deadband
// and a short exponential average keep the two-color decision from chattering
// at the seam; still images reset and use their exact committed tone plane.
export const TRITONE_TEMPORAL_ALPHA = 0.15;
export const TRITONE_HYSTERESIS = 10;

const BAYER_4 = [
  0, 8, 2, 10,
  12, 4, 14, 6,
  3, 11, 1, 9,
  15, 7, 13, 5,
];
const PRIMARY_RGB = [0xa8, 0x55, 0xf7];
const ACCENT_RGB = [0xff, 0x4f, 0xd8];

function requireTones(tones, length) {
  if (!(tones instanceof Uint8Array) || tones.length !== length) {
    throw new Error('The ASCII tone plane is incomplete or changed.');
  }
}

function requireGrid(columns, rows) {
  if (![columns, rows].every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError('The ASCII tone grid is invalid.');
  }
}

/** Local 4x4 Bayers decide only the midtone seam; flat highlights and shadows stay solid ink. */
export function tritoneAccent(tone, x, y) {
  if (!Number.isFinite(tone)) throw new TypeError('A tone sample must be finite');
  const value = Math.max(0, Math.min(255, tone)) / 255;
  const lower = TRITONE_THRESHOLD - TRITONE_DITHER_BAND / 2;
  const upper = TRITONE_THRESHOLD + TRITONE_DITHER_BAND / 2;
  if (value < lower) return false;
  if (value > upper) return true;
  return value > (BAYER_4[((y & 3) * 4) + (x & 3)] + 0.5) / 16;
}

/**
 * Stateful live-camera filter for the color plane. It averages tone updates and
 * only moves a cell's held value when the change exceeds a small deadband. The
 * returned tone plane is still fed through the same tritone split and Bayer
 * seam, so still images and saved frames keep the exact committed colors.
 */
export function createTritoneFilter({
  alpha = TRITONE_TEMPORAL_ALPHA, margin = TRITONE_HYSTERESIS,
} = {}) {
  if (!Number.isFinite(alpha) || alpha <= 0 || alpha > 1) {
    throw new RangeError('Tritone temporal alpha must be finite and in (0, 1]');
  }
  if (!Number.isFinite(margin) || margin < 0) {
    throw new RangeError('Tritone hysteresis must be finite and nonnegative');
  }
  let columns = 0;
  let rows = 0;
  let ready = false;
  let smoothed = null;
  let held = null;
  let stable = null;
  let bits = null;

  function reset() {
    ready = false;
  }

  function update(nextTones, nextColumns, nextRows) {
    const length = nextColumns * nextRows;
    requireGrid(nextColumns, nextRows);
    requireTones(nextTones, length);
    if (!ready || nextColumns !== columns || nextRows !== rows) {
      columns = nextColumns;
      rows = nextRows;
      smoothed = new Float32Array(length);
      held = new Float32Array(length);
      stable = new Uint8Array(length);
      bits = new Uint8Array(length);
      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < columns; x++) {
          const i = (y * columns) + x;
          const tone = nextTones[i];
          smoothed[i] = tone;
          held[i] = tone;
          stable[i] = tone;
          bits[i] = tritoneAccent(tone, x, y) ? 1 : 0;
        }
      }
      ready = true;
      return { tones: stable, changed: true };
    }

    let changed = false;
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < columns; x++) {
        const i = (y * columns) + x;
        const next = smoothed[i] + ((nextTones[i] - smoothed[i]) * alpha);
        smoothed[i] = next;
        if (next > held[i] + margin) held[i] = next;
        else if (next < held[i] - margin) held[i] = next;
        const tone = Math.max(0, Math.min(255, Math.round(held[i])));
        stable[i] = tone;
        const bit = tritoneAccent(tone, x, y) ? 1 : 0;
        if (bit !== bits[i]) {
          bits[i] = bit;
          changed = true;
        }
      }
    }
    return { tones: stable, changed };
  }

  return { reset, update };
}

/** Per-cell RGBA map used by the clipped preview; 1 pixel == 1 character cell. */
export function buildTritonePixels(tones, columns, rows) {
  requireGrid(columns, rows);
  requireTones(tones, columns * rows);
  const pixels = new Uint8ClampedArray(columns * rows * 4);
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < columns; x++) {
      const i = y * columns + x;
      const rgb = tritoneAccent(tones[i], x, y) ? ACCENT_RGB : PRIMARY_RGB;
      const offset = i * 4;
      pixels[offset] = rgb[0];
      pixels[offset + 1] = rgb[1];
      pixels[offset + 2] = rgb[2];
      pixels[offset + 3] = 255;
    }
  }
  return pixels;
}

/** The two ink strings share one grid; spaces preserve every cell and both line lengths. */
export function buildTritoneLayers(text, tones, columns, rows) {
  requireGrid(columns, rows);
  requireTones(tones, columns * rows);
  if (typeof text !== 'string') throw new TypeError('ASCII text must be a string');
  const lines = text.split('\n');
  if (lines.length !== rows || lines.some((line) => line.length !== columns)) {
    throw new Error('The ASCII grid is incomplete or changed.');
  }
  const primaryRows = new Array(rows);
  const accentRows = new Array(rows);
  for (let y = 0; y < rows; y++) {
    let primary = '';
    let accent = '';
    for (let x = 0; x < columns; x++) {
      const glyph = lines[y][x];
      if (glyph === ' ') {
        primary += ' ';
        accent += ' ';
      } else if (tritoneAccent(tones[(y * columns) + x], x, y)) {
        primary += ' ';
        accent += glyph;
      } else {
        primary += glyph;
        accent += ' ';
      }
    }
    primaryRows[y] = primary;
    accentRows[y] = accent;
  }
  return { primary: primaryRows, accent: accentRows };
}

let mapCanvas = null;
let mapContext = null;

/** Reuse one tiny canvas; PNG encoding stays below a frame's drawing budget. */
export function tritonePreviewUrl(tones, columns, rows) {
  requireGrid(columns, rows);
  requireTones(tones, columns * rows);
  if (typeof document === 'undefined') throw new Error('A canvas document is required for the tritone preview');
  if (!mapCanvas) {
    mapCanvas = document.createElement('canvas');
    mapContext = mapCanvas.getContext('2d');
    if (!mapContext) throw new Error('Unable to create the tritone preview canvas');
  }
  if (mapCanvas.width !== columns || mapCanvas.height !== rows) {
    mapCanvas.width = columns;
    mapCanvas.height = rows;
    mapContext = mapCanvas.getContext('2d');
    if (!mapContext) throw new Error('Unable to create the tritone preview canvas');
  }
  const image = mapContext.createImageData(columns, rows);
  image.data.set(buildTritonePixels(tones, columns, rows));
  mapContext.putImageData(image, 0, 0);
  return mapCanvas.toDataURL('image/png');
}
