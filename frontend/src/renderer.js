import { fillCodes, codesToText, buildRamp } from './ascii.js';
import { normalizeLook } from './looks.js';
import { findCharSet, normalizeCharSet } from './charsets.js';
import {
  applyTone, unsharpMask, buildMeasuredRamp, floydSteinberg, bayerOrdered,
} from './effects.js';

export const DEFAULT_SETTINGS = Object.freeze({
  maxWidthChars: 150,
  maxHeightLines: 200,
  sigma1: 1,
  sigma2: 2,
  edgeIndexThreshold: 90,
});

function positiveInteger(value, name) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${name} must be a positive integer`);
  }
}

function gridLength(width, height) {
  positiveInteger(width, 'Width');
  positiveInteger(height, 'Height');
  const length = width * height;
  if (!Number.isSafeInteger(length)) throw new RangeError('Grid is too large');
  return length;
}

function plane(value, Type, length, name) {
  if (!(value instanceof Type)) throw new TypeError(`${name} must be a ${Type.name}`);
  if (value.length !== length) throw new RangeError(`${name} has an incorrect buffer length`);
}

function positiveSigma(sigma) {
  if (!Number.isFinite(sigma) || sigma <= 0) {
    throw new RangeError('Sigma must be finite and positive');
  }
}

export function normalizeSettings(settings = {}) {
  if (settings === null || typeof settings !== 'object' || Array.isArray(settings)) {
    throw new TypeError('Settings must be an object');
  }
  const normalized = {};
  for (const key of Object.keys(DEFAULT_SETTINGS)) {
    normalized[key] = Object.hasOwn(settings, key) ? settings[key] : DEFAULT_SETTINGS[key];
  }
  positiveInteger(normalized.maxWidthChars, 'Maximum width');
  positiveInteger(normalized.maxHeightLines, 'Maximum height');
  positiveSigma(normalized.sigma1);
  positiveSigma(normalized.sigma2);
  if (!Number.isInteger(normalized.edgeIndexThreshold)
      || normalized.edgeIndexThreshold < 0 || normalized.edgeIndexThreshold > 95) {
    throw new RangeError('Edge index threshold must be an integer from 0 to 95');
  }
  return normalized;
}

export function computeGrid(width, height, settings = {}) {
  const { maxWidthChars, maxHeightLines } = normalizeSettings(settings);
  const error = 'Image aspect ratio is too extreme for this resolution';
  if (!Number.isSafeInteger(width) || width <= 0
      || !Number.isSafeInteger(height) || height <= 0) throw new RangeError(error);
  let columns;
  let rows;
  if (width > height) {
    columns = maxWidthChars;
    rows = Math.trunc(columns * (height / width));
  } else {
    rows = maxHeightLines;
    columns = Math.trunc(rows * (width / height));
  }
  columns *= 2;
  if (!Number.isSafeInteger(columns) || columns <= 0 || rows <= 0
      || !Number.isSafeInteger(columns * rows)) throw new RangeError(error);
  return { width: columns, height: rows };
}

export function rgbaToGray(rgba, out) {
  if (!(rgba instanceof Uint8Array) && !(rgba instanceof Uint8ClampedArray)) {
    throw new TypeError('RGBA must be a Uint8Array or Uint8ClampedArray');
  }
  if (!(out instanceof Uint8Array)) throw new TypeError('Gray output must be a Uint8Array');
  if (out.length === 0 || rgba.length !== out.length * 4) {
    throw new RangeError('RGBA or gray output has an incorrect buffer length');
  }
  for (let i = 0, j = 0; i < out.length; i++, j += 4) {
    out[i] = Math.round(0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2]);
  }
  return out;
}

export function makeGaussianKernel(sigma) {
  positiveSigma(sigma);
  const radius = Math.floor(4 * sigma + 0.5);
  if (!Number.isSafeInteger(radius) || radius * 2 + 1 > 0xffffffff) {
    throw new RangeError('Gaussian kernel is too large');
  }
  const kernel = new Float64Array(radius * 2 + 1);
  if (radius === 0) {
    kernel[0] = 1;
    return kernel;
  }
  const scale = -0.5 / (sigma * sigma);
  let sum = 0;
  for (let i = 0; i < kernel.length; i++) {
    const x = i - radius;
    const value = Math.exp(scale * (x * x));
    kernel[i] = value;
    sum += value;
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;
  return kernel;
}

function overlaps(a, b) {
  return a.buffer === b.buffer && a.byteOffset < b.byteOffset + b.byteLength
    && b.byteOffset < a.byteOffset + a.byteLength;
}

export function blurNormalized(input, width, height, kernel, out, scratch) {
  const length = gridLength(width, height);
  plane(input, Float64Array, length, 'Normalized input');
  plane(out, Float64Array, length, 'Blur output');
  plane(scratch, Float64Array, length, 'Blur scratch');
  if (!(kernel instanceof Float64Array)) throw new TypeError('Kernel must be a Float64Array');
  if (kernel.length === 0 || kernel.length % 2 !== 1) {
    throw new RangeError('Kernel must have a nonempty odd length');
  }
  if (overlaps(scratch, input) || overlaps(scratch, out)
      || overlaps(kernel, scratch) || overlaps(kernel, out)) {
    throw new RangeError('Blur scratch and kernel must not overlap working planes');
  }
  const radius = (kernel.length - 1) / 2;
  let symmetric = true;
  for (let k = 0; k < kernel.length; k++) {
    if (!Number.isFinite(kernel[k])) throw new RangeError('Kernel values must be finite');
    if (Math.abs(kernel[k] - kernel[kernel.length - 1 - k]) > Number.EPSILON) symmetric = false;
  }
  // SciPy correlates axis 0 before axis 1. Its symmetric path starts at
  // the center, then adds paired samples from the outermost coefficient in.
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      let value;
      if (symmetric) {
        value = input[i] * kernel[radius];
        for (let d = radius; d > 0; d--) {
          const above = Math.max(0, y - d) * width + x;
          const below = Math.min(height - 1, y + d) * width + x;
          value += (input[above] + input[below]) * kernel[radius - d];
        }
      } else {
        value = input[Math.min(height - 1, y + radius) * width + x] * kernel[2 * radius];
        for (let d = -radius; d < radius; d++) {
          value += input[Math.max(0, Math.min(height - 1, y + d)) * width + x] * kernel[radius + d];
        }
      }
      scratch[i] = value;
    }
  }
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      const i = row + x;
      let value;
      if (symmetric) {
        value = scratch[i] * kernel[radius];
        for (let d = radius; d > 0; d--) {
          value += (scratch[row + Math.max(0, x - d)] + scratch[row + Math.min(width - 1, x + d)])
            * kernel[radius - d];
        }
      } else {
        value = scratch[row + Math.min(width - 1, x + radius)] * kernel[2 * radius];
        for (let d = -radius; d < radius; d++) {
          value += scratch[row + Math.max(0, Math.min(width - 1, x + d))] * kernel[radius + d];
        }
      }
      out[i] = value;
    }
  }
  return out;
}

export function dogFromBlurs(blur1, blur2, out) {
  if (!(blur1 instanceof Float64Array)) throw new TypeError('First blur must be a Float64Array');
  if (blur1.length === 0) throw new RangeError('Blur input must be nonempty');
  plane(blur2, Float64Array, blur1.length, 'Second blur');
  plane(out, Uint8Array, blur1.length, 'DoG output');
  for (let i = 0; i < out.length; i++) {
    out[i] = Math.floor(Math.max(0, Math.min(1, blur1[i] - blur2[i])) * 255);
  }
  return out;
}

export function sobelWithAngle(dog, width, height, out) {
  const length = gridLength(width, height);
  plane(dog, Uint8Array, length, 'DoG input');
  if (out === null || typeof out !== 'object') throw new TypeError('Sobel output must be an object');
  plane(out.gx, Float64Array, length, 'Sobel gx');
  plane(out.gy, Float64Array, length, 'Sobel gy');
  plane(out.angles, Float64Array, length, 'Sobel angles');
  plane(out.magnitude, Uint8Array, length, 'Sobel magnitude');
  const degrees = 180 / Math.PI;
  for (let y = 0; y < height; y++) {
    const above = (y === 0 ? Math.min(1, height - 1) : y - 1) * width;
    const below = (y === height - 1 ? Math.max(0, height - 2) : y + 1) * width;
    const row = y * width;
    for (let x = 0; x < width; x++) {
      const left = x === 0 ? Math.min(1, width - 1) : x - 1;
      const right = x === width - 1 ? Math.max(0, width - 2) : x + 1;
      const gx = (dog[above + right] - dog[above + left])
        + 2 * (dog[row + right] - dog[row + left])
        + (dog[below + right] - dog[below + left]);
      const gy = (dog[below + left] + 2 * dog[below + x] + dog[below + right])
        - (dog[above + left] + 2 * dog[above + x] + dog[above + right]);
      const i = row + x;
      out.gx[i] = gx;
      out.gy[i] = gy;
      // Uint8Array intentionally truncates and wraps, unlike Uint8ClampedArray.
      out.magnitude[i] = Math.sqrt(gx * gx + gy * gy);
      out.angles[i] = Math.atan2(gy, gx) * degrees;
    }
  }
  return out;
}

export function createRenderer(width, height, settings = {}, look = {}, charSet = 'dense') {
  const length = gridLength(width, height);
  const normalizedSettings = normalizeSettings(settings);
  const processing = normalizeLook(look);
  const characters = findCharSet(normalizeCharSet(charSet));
  const edges = processing.id === 'original' || processing.id === 'outline';
  const tone = processing.autoContrast || processing.midtone !== 1 || processing.invert;
  const kernel1 = edges ? makeGaussianKernel(normalizedSettings.sigma1) : null;
  const kernel2 = edges ? makeGaussianKernel(normalizedSettings.sigma2) : null;
  const gray = new Uint8Array(length);
  const normalized = edges ? new Float64Array(length) : null;
  const scratch = edges ? new Float64Array(length) : null;
  const blur1 = edges ? new Float64Array(length) : null;
  const blur2 = edges ? new Float64Array(length) : null;
  const dog = edges ? new Uint8Array(length) : null;
  const gx = edges ? new Float64Array(length) : null;
  const gy = edges ? new Float64Array(length) : null;
  const magnitude = edges ? new Uint8Array(length) : null;
  const angles = edges ? new Float64Array(length) : null;
  const shadeCodes = new Uint8Array(length);
  const edgeCodes = new Uint8Array(length);
  if (!edges) edgeCodes.fill(32);
  const sobel = edges ? { gx, gy, magnitude, angles } : null;
  const toneLut = tone ? new Uint8Array(256) : null;
  const autoRange = processing.autoContrast ? { minimum: 0, maximum: 255, ready: false } : null;
  const sharpnessScratch = processing.sharpness > 0 ? (scratch ?? new Float64Array(length)) : null;
  const ramp = edges ? null : buildMeasuredRamp(characters.compactEntries);
  const shadeRamp = edges ? buildRamp(characters.fullEntries) : null;
  const errors = processing.id === 'photo' ? new Float64Array(2 * (width + 2)) : null;
  const rows = new Array(height);
  let text = '';
  let committed = false;

  function render() {
    if (tone) applyTone(gray, processing, toneLut, autoRange);
    if (sharpnessScratch) unsharpMask(gray, width, height, processing.sharpness, sharpnessScratch);
    if (edges) {
      for (let i = 0; i < length; i++) normalized[i] = gray[i] / 255;
      blurNormalized(normalized, width, height, kernel1, blur1, scratch);
      blurNormalized(normalized, width, height, kernel2, blur2, scratch);
      dogFromBlurs(blur1, blur2, dog);
      sobelWithAngle(dog, width, height, sobel);
      fillCodes(gray, magnitude, angles, shadeCodes, edgeCodes, normalizedSettings.edgeIndexThreshold, shadeRamp);
      if (processing.id === 'outline') shadeCodes.fill(32);
    } else if (processing.id === 'photo') {
      floydSteinberg(gray, width, height, ramp, shadeCodes, errors);
    } else {
      bayerOrdered(gray, width, height, ramp, shadeCodes);
    }
    text = codesToText(shadeCodes, edgeCodes, width, height, rows);
    committed = true;
    return text;
  }

  return {
    renderGray(input) {
      plane(input, Uint8Array, length, 'Gray input');
      gray.set(input);
      return render();
    },
    renderRgba(rgba) {
      if (!(rgba instanceof Uint8Array) && !(rgba instanceof Uint8ClampedArray)) {
        throw new TypeError('RGBA must be a Uint8Array or Uint8ClampedArray');
      }
      if (rgba.length !== length * 4) throw new RangeError('RGBA has an incorrect buffer length');
      rgbaToGray(rgba, gray);
      return render();
    },
    snapshot() {
      return {
        width, height, text,
        gray: gray.slice(), blur1: blur1?.slice() ?? null, blur2: blur2?.slice() ?? null,
        dog: dog?.slice() ?? null, gx: gx?.slice() ?? null, gy: gy?.slice() ?? null,
        magnitude: magnitude?.slice() ?? null, angles: angles?.slice() ?? null,
        shadeCodes: shadeCodes.slice(), edgeCodes: edgeCodes.slice(),
      };
    },
    tone() {
      if (!committed) throw new Error('The renderer has no committed ASCII frame.');
      return gray.slice();
    },
  };
}
