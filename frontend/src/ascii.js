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

const lookup = buildRampLookup(inkUsage);
const codes = new Uint8Array(inkUsage.map(([char]) => char.charCodeAt(0)));
const angleKeys = [-135, -90, -45, 45, 90, 135, 0, 180];
const angleChars = ['/', '-', '\\', '/', '-', '\\', ' ', '|'];
const glyphs = Array.from({length: 128}, (_, code) => String.fromCharCode(code));

export function buildRamp(entries) {
  const rampLookup = buildRampLookup(entries);
  const rampCodes = Uint8Array.from(entries, ([char]) => char.charCodeAt(0));
  return { lookup: rampLookup, codes: rampCodes };
}

const defaultRamp = { lookup, codes };

export function angleToChar(degrees) {
  if (!Number.isFinite(degrees)) throw new RangeError('Angle must be finite');
  // Unlike ((x % 360) + 360) % 360, this avoids rounding positive probes.
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

// Shared by the standalone component API and the production reusable workspace.
export function fillCodes(gray, magnitude, angles, shadeCodes, edgeCodes, edgeIndexThreshold = 90, ramp = defaultRamp) {
  if (!(gray instanceof Uint8Array)) throw new TypeError('gray must be Uint8Array');
  if (gray.length === 0) throw new RangeError('gray must be nonempty');
  const length = gray.length;
  plane(magnitude, Uint8Array, length, 'magnitude');
  plane(angles, Float64Array, length, 'angles');
  plane(shadeCodes, Uint8Array, length, 'shadeCodes');
  plane(edgeCodes, Uint8Array, length, 'edgeCodes');
  thresholdValue(edgeIndexThreshold);
  for (let index = 0; index < length; index++) {
    shadeCodes[index] = ramp.codes[ramp.lookup[gray[index]]];
    edgeCodes[index] = lookup[magnitude[index]] < edgeIndexThreshold ?
      angleToChar(angles[index]).charCodeAt(0) : 32;
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

export function composeAscii(gray, magnitude, angles, width, height, edgeIndexThreshold = 90, ramp = defaultRamp) {
  const length = gridSize(width, height);
  plane(gray, Uint8Array, length, 'gray');
  const shadeCodes = new Uint8Array(length);
  const edgeCodes = new Uint8Array(length);
  fillCodes(gray, magnitude, angles, shadeCodes, edgeCodes, edgeIndexThreshold, ramp);
  return codesToText(shadeCodes, edgeCodes, width, height);
}
