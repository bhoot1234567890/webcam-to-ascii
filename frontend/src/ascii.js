import inkUsage from '../../ink-usage.json';

export function buildRampLookup(entries) {
  if (!Array.isArray(entries) || entries.length === 0 || entries.length > 256) {
    throw new TypeError('Expected a nonempty ink ramp');
  }
  for (const entry of entries) {
    if (!Array.isArray(entry) || typeof entry[0] !== 'string' || entry[0].length !== 1 ||
        !Number.isFinite(entry[1])) throw new TypeError('Invalid ink ramp entry');
  }
  const values = entries.map((entry) => entry[1]);
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  if (maximum === minimum) throw new RangeError('Ink ramp must span a nonzero range');
  const normalized = values.map((value) => 255 * (value - minimum) / (maximum - minimum));
  const lookup = new Uint8Array(256);
  for (let value = 0; value < 256; value++) {
    let best = Infinity;
    for (let index = 0; index < normalized.length; index++) {
      const distance = Math.abs(normalized[index] - value);
      if (distance < best) {
        lookup[value] = index;
        best = distance;
      }
    }
  }
  return lookup;
}

const defaultLookup = buildRampLookup(inkUsage);
const defaultCodes = new Uint8Array(inkUsage.map(([char]) => char.charCodeAt(0)));
const angleKeys = [-135, -90, -45, 45, 90, 135, 0, 180];
const angleChars = ['/', '-', '\\', '/', '-', '\\', ' ', '|'];
const glyphs = Array.from({length: 128}, (_, code) => String.fromCharCode(code));

function createLinearRamp(charString) {
  const chars = [...charString];
  const count = chars.length;
  const codes = new Uint8Array(chars.map((c) => c.charCodeAt(0)));
  const values = new Float64Array(count);
  const step = 255 / (count - 1);
  for (let i = 0; i < count; i++) values[i] = i * step;
  const lookup = new Uint8Array(256);
  for (let v = 0; v < 256; v++) {
    lookup[v] = Math.min(count - 1, Math.max(0, Math.round(v / step)));
  }
  return { chars, codes, values, lookup, count, step };
}

const fullValues = inkUsage.map((e) => e[1]);
const fullMin = Math.min(...fullValues);
const fullMax = Math.max(...fullValues);
const fullNormalized = new Float64Array(fullValues.map((v) => 255 * (v - fullMin) / (fullMax - fullMin)));

export const RAMPS = Object.freeze({
  clean: createLinearRamp(' .:-=+*#%@'),
  detailed: createLinearRamp(' .\',^:";~-_+*#%@'),
  classic: createLinearRamp(' .\'`^",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$'),
  full: {
    chars: inkUsage.map(([c]) => c),
    codes: defaultCodes,
    values: fullNormalized,
    lookup: defaultLookup,
    count: inkUsage.length,
    step: 255 / 95,
  },
});

const BAYER_4X4 = [
  [  0 / 16,  8 / 16,  2 / 16, 10 / 16 ],
  [ 12 / 16,  4 / 16, 14 / 16,  6 / 16 ],
  [  3 / 16, 11 / 16,  1 / 16,  9 / 16 ],
  [ 15 / 16,  7 / 16, 13 / 16,  5 / 16 ],
];

export function stretchAutoLevels(input, out) {
  const length = input.length;
  const hist = new Int32Array(256);
  for (let i = 0; i < length; i++) hist[input[i]]++;
  const lowThreshold = Math.floor(length * 0.02);
  const highThreshold = Math.floor(length * 0.98);
  let count = 0;
  let min = 0;
  for (let i = 0; i < 256; i++) {
    count += hist[i];
    if (count >= lowThreshold) { min = i; break; }
  }
  count = 0;
  let max = 255;
  for (let i = 255; i >= 0; i--) {
    count += hist[i];
    if (count >= length - highThreshold) { max = i; break; }
  }
  if (max <= min) {
    out.set(input);
    return out;
  }
  const scale = 255 / (max - min);
  for (let i = 0; i < length; i++) {
    const v = (input[i] - min) * scale;
    out[i] = v < 0 ? 0 : v > 255 ? 255 : Math.round(v);
  }
  return out;
}

const lutCache = new Map();
export function getGammaInvertLut(gamma = 1.0, invert = false) {
  const roundedGamma = Math.round(gamma * 100) / 100;
  const key = `${roundedGamma}:${invert ? 1 : 0}`;
  let lut = lutCache.get(key);
  if (lut) return lut;
  lut = new Uint8Array(256);
  const invGamma = 1 / roundedGamma;
  for (let i = 0; i < 256; i++) {
    let v = Math.round(255 * Math.pow(i / 255, invGamma));
    if (invert) v = 255 - v;
    lut[i] = Math.min(255, Math.max(0, v));
  }
  lutCache.set(key, lut);
  return lut;
}
export function applySharpen(input, width, height, amount, out) {
  if (amount <= 0.01) {
    out.set(input);
    return out;
  }
  const s = amount * 0.5;
  const centerWeight = 1 + 4 * s;
  for (let y = 0; y < height; y++) {
    const row = y * width;
    const upRow = (y === 0 ? 0 : y - 1) * width;
    const downRow = (y === height - 1 ? height - 1 : y + 1) * width;
    for (let x = 0; x < width; x++) {
      const left = x === 0 ? 0 : x - 1;
      const right = x === width - 1 ? width - 1 : x + 1;
      const c = input[row + x];
      const neighbors = input[upRow + x] + input[downRow + x] + input[row + left] + input[row + right];
      const val = c * centerWeight - neighbors * s;
      out[row + x] = val < 0 ? 0 : val > 255 ? 255 : (val + 0.5) | 0;
    }
  }
  return out;
}

export function applyBloom(input, width, height, amount, threshold, out, scratch, blurBuf) {
  if (amount <= 0.01) {
    out.set(input);
    return out;
  }
  const length = width * height;
  const invRange = 1 / Math.max(1, 255 - threshold);
  for (let i = 0; i < length; i++) {
    const v = input[i];
    scratch[i] = v > threshold ? (v - threshold) * invRange * 255 : 0;
  }
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      const x0 = Math.max(0, x - 2), x1 = Math.max(0, x - 1), x2 = x;
      const x3 = Math.min(width - 1, x + 1), x4 = Math.min(width - 1, x + 2);
      blurBuf[row + x] = (scratch[row + x0] + 4 * scratch[row + x1] + 6 * scratch[row + x2]
        + 4 * scratch[row + x3] + scratch[row + x4]) / 16;
    }
  }
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      const y0 = Math.max(0, y - 2), y1 = Math.max(0, y - 1), y2 = y;
      const y3 = Math.min(height - 1, y + 1), y4 = Math.min(height - 1, y + 2);
      const b = (blurBuf[y0 * width + x] + 4 * blurBuf[y1 * width + x] + 6 * blurBuf[y2 * width + x]
        + 4 * blurBuf[y3 * width + x] + blurBuf[y4 * width + x]) / 16;
      const combined = input[y * width + x] + b * amount;
      const norm = combined / 255;
      const mapped = (norm / (1 + 0.3 * norm)) * 1.3 * 255;
      out[y * width + x] = mapped < 0 ? 0 : mapped > 255 ? 255 : (mapped + 0.5) | 0;
    }
  }
  return out;
}

const colorBurnCache = new Map();
export function getColorBurnLut(burn) {
  const rounded = Math.round(burn * 100) / 100;
  let lut = colorBurnCache.get(rounded);
  if (lut) return lut;
  lut = new Uint8Array(256);
  if (rounded <= 0.01) {
    for (let i = 0; i < 256; i++) lut[i] = i;
  } else {
    for (let i = 0; i < 256; i++) {
      const norm = i / 255;
      const denom = Math.max(0.01, 1 - rounded * (1 - norm));
      const burned = 1 - (1 - norm) / denom;
      lut[i] = Math.min(255, Math.max(0, Math.round(burned * 255)));
    }
  }
  colorBurnCache.set(rounded, lut);
  return lut;
}

function runAtkinson(source, width, height, ramp, out, scratch) {
  scratch.set(source);
  const { codes, values, lookup } = ramp;
  for (let y = 0; y < height; y++) {
    const row = y * width;
    const nextRow = (y + 1) * width;
    const next2Row = (y + 2) * width;
    for (let x = 0; x < width; x++) {
      const i = row + x;
      const v = Math.min(255, Math.max(0, scratch[i]));
      const charIndex = lookup[Math.round(v)];
      out[i] = codes[charIndex];
      const err = (v - values[charIndex]) * 0.125;
      if (x + 1 < width) scratch[i + 1] += err;
      if (x + 2 < width) scratch[i + 2] += err;
      if (y + 1 < height) {
        if (x > 0) scratch[nextRow + x - 1] += err;
        scratch[nextRow + x] += err;
        if (x + 1 < width) scratch[nextRow + x + 1] += err;
      }
      if (y + 2 < height) {
        scratch[next2Row + x] += err;
      }
    }
  }
}

function runFloydSteinberg(source, width, height, ramp, out, scratch) {
  scratch.set(source);
  const { codes, values, lookup } = ramp;
  for (let y = 0; y < height; y++) {
    const row = y * width;
    const nextRow = (y + 1) * width;
    for (let x = 0; x < width; x++) {
      const i = row + x;
      const v = Math.min(255, Math.max(0, scratch[i]));
      const charIndex = lookup[Math.round(v)];
      out[i] = codes[charIndex];
      const err = v - values[charIndex];
      if (x + 1 < width) scratch[i + 1] += err * (7 / 16);
      if (y + 1 < height) {
        if (x > 0) scratch[nextRow + x - 1] += err * (3 / 16);
        scratch[nextRow + x] += err * (5 / 16);
        if (x + 1 < width) scratch[nextRow + x + 1] += err * (1 / 16);
      }
    }
  }
}

function runBayer(source, width, height, ramp, out) {
  const { codes, lookup, step } = ramp;
  for (let y = 0; y < height; y++) {
    const row = y * width;
    const bayerRow = BAYER_4X4[y & 3];
    for (let x = 0; x < width; x++) {
      const i = row + x;
      const offset = (bayerRow[x & 3] - 0.5) * step * 0.9;
      const v = Math.min(255, Math.max(0, source[i] + offset));
      out[i] = codes[lookup[Math.round(v)]];
    }
  }
}

export function angleToChar(degrees) {
  if (!Number.isFinite(degrees)) throw new RangeError('Angle must be finite');
  let degree = degrees % 360;
  if (degree < 0) degree += 360;
  let best = Infinity;
  let closest = 0;
  for (let index = 0; index < angleKeys.length; index++) {
    const difference = Math.abs(degree - angleKeys[index]);
    const distance = Math.min(difference, 360 - difference);
    if (distance < best) {
      best = distance;
      closest = index;
    }
  }
  return angleChars[closest];
}

function plane(value, Type, length, name) {
  if (!(value instanceof Type)) throw new TypeError(`${name} must be ${Type.name}`);
  if (value.length !== length) throw new RangeError(`${name} must have ${length} elements`);
}

function thresholdValue(threshold) {
  if (!Number.isInteger(threshold) || threshold < 0 || threshold > 95) {
    throw new RangeError('Edge index threshold must be an integer from 0 to 95');
  }
}

function gridSize(width, height) {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width <= 0 || height <= 0 ||
      !Number.isSafeInteger(width * height)) throw new RangeError('Grid dimensions must be positive integers');
  return width * height;
}

export function fillCodes(
  gray,
  magnitude,
  angles,
  shadeCodes,
  edgeCodes,
  edgeIndexThreshold = 90,
  options = {},
  scratchBuffers = {}
) {
  if (!(gray instanceof Uint8Array)) throw new TypeError('gray must be Uint8Array');
  if (gray.length === 0) throw new RangeError('gray must be nonempty');
  const length = gray.length;
  plane(magnitude, Uint8Array, length, 'magnitude');
  plane(angles, Float64Array, length, 'angles');
  plane(shadeCodes, Uint8Array, length, 'shadeCodes');
  plane(edgeCodes, Uint8Array, length, 'edgeCodes');
  thresholdValue(edgeIndexThreshold);

  const dither = options.dither || 'none';
  const rampKey = options.ramp || 'full';
  const invert = Boolean(options.invert);
  const autoLevels = Boolean(options.autoLevels);
  const gamma = options.gamma ?? 1.0;
  const edgeBlend = options.edgeBlend || 'overlay';
  const width = options.width || length;
  const height = options.height || 1;

  const sharpen = Number(options.sharpen) || 0;
  const bloom = Number(options.bloom) || 0;
  const colorBurn = Number(options.colorBurn) || 0;

  // 1. Strict legacy fast path
  if (sharpen <= 0.01 && colorBurn <= 0.01 && dither === 'none' && rampKey === 'full' && !invert && !autoLevels && gamma === 1.0 && edgeBlend === 'overlay') {
    for (let index = 0; index < length; index++) {
      shadeCodes[index] = defaultCodes[defaultLookup[gray[index]]];
      edgeCodes[index] = defaultLookup[magnitude[index]] < edgeIndexThreshold ?
        angleToChar(angles[index]).charCodeAt(0) : 32;
    }
    return;
  }

  // 2. Preprocessing
  let processed = gray;
  if (autoLevels) {
    const autoBuf = scratchBuffers.autoLevels || new Uint8Array(length);
    stretchAutoLevels(gray, autoBuf);
    processed = autoBuf;
  }

  if (sharpen > 0.01) {
    const sharpBuf = scratchBuffers.sharpen || new Uint8Array(length);
    applySharpen(processed, width, height, sharpen, sharpBuf);
    processed = sharpBuf;
  }

  if (colorBurn > 0.01) {
    const burnBuf = scratchBuffers.colorBurn || new Uint8Array(length);
    const burnLut = getColorBurnLut(colorBurn);
    for (let i = 0; i < length; i++) {
      burnBuf[i] = burnLut[processed[i]];
    }
    processed = burnBuf;
  }

  let toneMapped = processed;
  if (gamma !== 1.0 || invert) {
    const toneBuf = scratchBuffers.tone || new Uint8Array(length);
    const lut = getGammaInvertLut(gamma, invert);
    for (let i = 0; i < length; i++) {
      toneBuf[i] = lut[processed[i]];
    }
    toneMapped = toneBuf;
  }

  // 3. Shading generation
  const ramp = RAMPS[rampKey] || RAMPS.full;
  if (edgeBlend === 'edges-only') {
    shadeCodes.fill(32);
  } else if (dither === 'atkinson') {
    const scratch = scratchBuffers.dither || new Float64Array(length);
    runAtkinson(toneMapped, width, height, ramp, shadeCodes, scratch);
  } else if (dither === 'floyd-steinberg') {
    const scratch = scratchBuffers.dither || new Float64Array(length);
    runFloydSteinberg(toneMapped, width, height, ramp, shadeCodes, scratch);
  } else if (dither === 'bayer') {
    runBayer(toneMapped, width, height, ramp, shadeCodes);
  } else {
    const { codes: rCodes, lookup: rLookup } = ramp;
    for (let i = 0; i < length; i++) {
      shadeCodes[i] = rCodes[rLookup[toneMapped[i]]];
    }
  }

  // 4. Edge codes
  if (edgeBlend === 'luminance-only') {
    edgeCodes.fill(32);
  } else {
    for (let index = 0; index < length; index++) {
      edgeCodes[index] = defaultLookup[magnitude[index]] < edgeIndexThreshold ?
        angleToChar(angles[index]).charCodeAt(0) : 32;
    }
  }
}

export function codesToText(shadeCodes, edgeCodes, width, height, rows = new Array(height)) {
  const length = gridSize(width, height);
  plane(shadeCodes, Uint8Array, length, 'shadeCodes');
  plane(edgeCodes, Uint8Array, length, 'edgeCodes');
  for (let y = 0; y < height; y++) {
    let row = '';
    for (let x = 0; x < width; x++) {
      const index = y * width + x;
      row += glyphs[edgeCodes[index] === 32 ? shadeCodes[index] : edgeCodes[index]];
    }
    rows[y] = row;
  }
  return rows.join('\n');
}

export function composeAscii(gray, magnitude, angles, width, height, edgeIndexThreshold = 90, options = {}) {
  const length = gridSize(width, height);
  plane(gray, Uint8Array, length, 'gray');
  const shadeCodes = new Uint8Array(length);
  const edgeCodes = new Uint8Array(length);
  fillCodes(gray, magnitude, angles, shadeCodes, edgeCodes, edgeIndexThreshold, {
    ...options,
    width,
    height,
  });
  return codesToText(shadeCodes, edgeCodes, width, height);
}
