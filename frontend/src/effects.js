import inkUsage from '../../ink-usage.json';

const compactCharacters = new Set(' .:-=+*#%@');
let compactRamp;

// Levels are measured coverage on the same 0..255 axis as Original, not
// evenly spaced positions in a string. Stable sorting preserves first-entry ties.
export function buildMeasuredRamp(entries) {
  if (!Array.isArray(entries) || entries.length < 2 || entries.length > 256) {
    throw new TypeError('Expected at least two measured glyphs');
  }
  for (const entry of entries) {
    if (!Array.isArray(entry) || typeof entry[0] !== 'string' || entry[0].length !== 1
        || entry[0].charCodeAt(0) < 32 || entry[0].charCodeAt(0) > 126
        || !Number.isFinite(entry[1])) throw new TypeError('Invalid measured glyph');
  }
  const ordered = entries.slice().sort((a, b) => a[1] - b[1]);
  const minimum = ordered[0][1];
  const span = ordered[ordered.length - 1][1] - minimum;
  if (!(span > 0)) throw new RangeError('Measured ramp must span a nonzero range');
  // Equal measurements cannot receive distinct quantization intervals.
  const unique = ordered.filter((entry, index) => index === 0 || entry[1] !== ordered[index - 1][1]);
  return {
    codes: Uint8Array.from(unique, ([glyph]) => glyph.charCodeAt(0)),
    levels: Float64Array.from(unique, ([, value]) => 255 * (value - minimum) / span),
  };
}

export function getCompactRamp() {
  if (!compactRamp) compactRamp = buildMeasuredRamp(inkUsage.filter(([glyph]) => compactCharacters.has(glyph)));
  return compactRamp;
}

function lowerBracket(levels, value) {
  let low = 0;
  let high = levels.length - 1;
  while (low + 1 < high) {
    const middle = (low + high) >>> 1;
    if (levels[middle] <= value) low = middle;
    else high = middle;
  }
  return low;
}

export function nearestMeasuredIndex(levels, value) {
  if (value <= levels[0]) return 0;
  const last = levels.length - 1;
  if (value >= levels[last]) return last;
  const low = lowerBracket(levels, value);
  // An exact midpoint deterministically chooses the lower-coverage glyph.
  return value - levels[low] <= levels[low + 1] - value ? low : low + 1;
}

// In-place processing. The renderer only allocates the 256-entry LUT when tone
// is enabled; a neutral Original never enters this stage.
export function applyTone(gray, look, lut) {
  if (!look.autoContrast && look.midtone === 1 && !look.invert) return gray;
  if (!(lut instanceof Uint8Array) || lut.length !== 256) throw new TypeError('Tone LUT must have 256 bytes');
  let minimum = 0;
  let maximum = 255;
  if (look.autoContrast) {
    minimum = 255;
    maximum = 0;
    for (let i = 0; i < gray.length; i++) {
      minimum = Math.min(minimum, gray[i]);
      maximum = Math.max(maximum, gray[i]);
    }
    // A flat field retains its brightness rather than dividing by zero or
    // manufacturing contrast. Midtone/inversion still apply when selected.
    if (minimum === maximum) { minimum = 0; maximum = 255; }
  }
  for (let i = 0; i < 256; i++) {
    const unit = Math.max(0, Math.min(1, (i - minimum) / (maximum - minimum)));
    const toned = look.midtone === 1 ? unit : Math.pow(unit, look.midtone);
    lut[i] = Math.round(255 * (look.invert ? 1 - toned : toned));
  }
  for (let i = 0; i < gray.length; i++) gray[i] = lut[gray[i]];
  return gray;
}

// A separable binomial low-pass followed by a bounded unsharp mask. Computing
// the complete horizontal plane first makes in-place output order independent.
export function unsharpMask(gray, width, height, amount, scratch) {
  if (!Number.isFinite(amount) || amount < 0 || amount > 1.5) throw new RangeError('Sharpness must be from 0 to 1.5');
  if (amount === 0) return gray;
  if (!(scratch instanceof Float64Array) || scratch.length !== gray.length) {
    throw new TypeError('Sharpness scratch must match the image');
  }
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      scratch[row + x] = (gray[row + Math.max(0, x - 1)] + 2 * gray[row + x]
        + gray[row + Math.min(width - 1, x + 1)]) / 4;
    }
  }
  for (let y = 0; y < height; y++) {
    const above = Math.max(0, y - 1) * width;
    const below = Math.min(height - 1, y + 1) * width;
    const row = y * width;
    for (let x = 0; x < width; x++) {
      const i = row + x;
      const blurred = (scratch[above + x] + 2 * scratch[i] + scratch[below + x]) / 4;
      gray[i] = Math.round(Math.max(0, Math.min(255, gray[i] + amount * (gray[i] - blurred))));
    }
  }
  return gray;
}

// Two padded error rows, not a full image copy. Boundary error is discarded,
// never wrapped into the next scanline; the workspace is reset each frame.
export function floydSteinberg(gray, width, height, ramp, out, errors) {
  const stride = width + 2;
  if (!(errors instanceof Float64Array) || errors.length !== 2 * stride) {
    throw new TypeError('Floyd error rows have an incorrect length');
  }
  errors.fill(0);
  let current = 0;
  let next = stride;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const value = gray[i] + errors[current + x + 1];
      const index = nearestMeasuredIndex(ramp.levels, value);
      out[i] = ramp.codes[index];
      const error = value - ramp.levels[index];
      errors[current + x + 2] += error * 7 / 16;
      errors[next + x] += error * 3 / 16;
      errors[next + x + 1] += error * 5 / 16;
      errors[next + x + 2] += error / 16;
    }
    errors.fill(0, current, current + stride);
    const previous = current;
    current = next;
    next = previous;
  }
  return out;
}

const bayer4 = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];

export function bayerOrdered(gray, width, height, ramp, out) {
  const { levels, codes } = ramp;
  const last = levels.length - 1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const value = gray[i];
      if (value <= levels[0]) out[i] = codes[0];
      else if (value >= levels[last]) out[i] = codes[last];
      else {
        const low = lowerBracket(levels, value);
        const fraction = (value - levels[low]) / (levels[low + 1] - levels[low]);
        const threshold = (bayer4[(y % 4) * 4 + x % 4] + 0.5) / 16;
        out[i] = codes[fraction > threshold ? low + 1 : low];
      }
    }
  }
  return out;
}
