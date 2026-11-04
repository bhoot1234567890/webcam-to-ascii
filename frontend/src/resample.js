/*
 * RGB 8-bit Lanczos port of Pillow 13.0.0 src/libImaging/Resample.c:
 * https://github.com/python-pillow/Pillow/blob/13.0.0/src/libImaging/Resample.c
 * Licensed under MIT-CMU (https://github.com/python-pillow/Pillow/blob/13.0.0/LICENSE).
 *
 * The Python Imaging Library (PIL) is
 * Copyright © 1997-2011 by Secret Labs AB
 * Copyright © 1995-2011 by Fredrik Lundh and contributors
 * Pillow is the friendly PIL fork. It is
 * Copyright © 2010 by Jeffrey 'Alex' Clark and contributors
 *
 * Permission to use, copy, modify and distribute this software and its
 * documentation for any purpose and without fee is hereby granted,
 * provided that the above copyright notice appears in all copies, and that
 * both that copyright notice and this permission notice appear in supporting
 * documentation, and that the name of Secret Labs AB or the author not be
 * used in advertising or publicity pertaining to distribution of the software
 * without specific, written prior permission.
 *
 * SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
 * SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
 * IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
 * INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE.
 */

const PRECISION = 2 ** 22;
const ROUNDING = PRECISION / 2;

function rgbaLength(width, height) {
  if (!Number.isInteger(width) || !Number.isInteger(height)
      || width <= 0 || height <= 0 || width > 0x7fffffff || height > 0x7fffffff) {
    throw new RangeError('Image dimensions must be positive 32-bit integers');
  }
  const length = width * height * 4;
  if (!Number.isSafeInteger(length)) {
    throw new RangeError('Image byte count exceeds the safe integer range');
  }
  return length;
}

function lanczos(x) {
  if (x < -3 || x >= 3) return 0;
  if (x === 0) return 1;
  const first = x * Math.PI;
  const second = (x / 3) * Math.PI;
  return (Math.sin(first) / first) * (Math.sin(second) / second);
}

function coefficients(inSize, outSize) {
  // Pillow's full-image box is stored as float before precompute_coeffs.
  const scale = Math.fround(inSize) / outSize;
  const filterScale = Math.max(1, scale);
  const support = 3 * filterScale;
  const inverseScale = 1 / filterScale;
  const starts = new Int32Array(outSize);
  const offsets = new Float64Array(outSize + 1);
  let total = 0;
  for (let position = 0; position < outSize; position++) {
    const center = (position + 0.5) * scale;
    const start = Math.max(0, Math.trunc(center - support + 0.5));
    const end = Math.min(inSize, Math.trunc(center + support + 0.5));
    starts[position] = start;
    offsets[position] = total;
    total += end - start;
  }
  offsets[outSize] = total;
  const weights = new Int32Array(total);
  // Only coefficient construction uses floating scratch; no frame allocates it.
  const scratch = new Float64Array(Math.min(inSize, Math.ceil(support) * 2 + 1));
  for (let position = 0; position < outSize; position++) {
    const center = (position + 0.5) * scale;
    const start = starts[position];
    const offset = offsets[position];
    const count = offsets[position + 1] - offset;
    let sum = 0;
    for (let tap = 0; tap < count; tap++) {
      const weight = lanczos((tap + start - center + 0.5) * inverseScale);
      scratch[tap] = weight;
      sum += weight;
    }
    for (let tap = 0; tap < count; tap++) {
      const weight = sum !== 0 ? scratch[tap] / sum : scratch[tap];
      // A C int cast truncates toward zero, including negative coefficients.
      weights[offset + tap] = Math.trunc(weight * PRECISION + (weight < 0 ? -0.5 : 0.5));
    }
  }
  return { starts, offsets, weights };
}

function clip(sum) {
  // Arithmetic right shift in Pillow rounds negative accumulators down.
  // Number accumulation avoids JS bitwise truncation of intermediate products.
  const value = Math.floor(sum / PRECISION);
  return value < 0 ? 0 : value > 255 ? 255 : value;
}

function horizontal(input, inputWidth, height, output, outputWidth, table) {
  const { starts, offsets, weights } = table;
  for (let y = 0; y < height; y++) {
    const inputRow = y * inputWidth * 4;
    const outputRow = y * outputWidth * 4;
    for (let x = 0; x < outputWidth; x++) {
      let red = ROUNDING;
      let green = ROUNDING;
      let blue = ROUNDING;
      let source = inputRow + starts[x] * 4;
      const end = offsets[x + 1];
      for (let tap = offsets[x]; tap < end; tap++, source += 4) {
        const weight = weights[tap];
        red += input[source] * weight;
        green += input[source + 1] * weight;
        blue += input[source + 2] * weight;
      }
      const target = outputRow + x * 4;
      output[target] = clip(red);
      output[target + 1] = clip(green);
      output[target + 2] = clip(blue);
      output[target + 3] = 255;
    }
  }
}

function vertical(input, width, output, outputHeight, table) {
  const { starts, offsets, weights } = table;
  const stride = width * 4;
  for (let y = 0; y < outputHeight; y++) {
    const firstRow = starts[y] * stride;
    const begin = offsets[y];
    const end = offsets[y + 1];
    const outputRow = y * stride;
    for (let x = 0; x < width; x++) {
      let red = ROUNDING;
      let green = ROUNDING;
      let blue = ROUNDING;
      let source = firstRow + x * 4;
      for (let tap = begin; tap < end; tap++, source += stride) {
        const weight = weights[tap];
        red += input[source] * weight;
        green += input[source + 1] * weight;
        blue += input[source + 2] * weight;
      }
      const target = outputRow + x * 4;
      output[target] = clip(red);
      output[target + 1] = clip(green);
      output[target + 2] = clip(blue);
      output[target + 3] = 255;
    }
  }
}

/** Cache coefficients and the two-pass intermediate for opaque black-backed RGBA. */
export function createLanczosResampler(srcWidth, srcHeight, dstWidth, dstHeight) {
  const inputLength = rgbaLength(srcWidth, srcHeight);
  const outputLength = rgbaLength(dstWidth, dstHeight);
  const needHorizontal = srcWidth !== dstWidth;
  const needVertical = srcHeight !== dstHeight;
  // Pillow 13.0.0 uses absolute pixel reductions, not scale ratios:
  // vertical first iff height shrinks and its reduction exceeds twice width's.
  const horizontalFirst = !((srcHeight - dstHeight) > 0
    && (srcHeight - dstHeight) > (srcWidth - dstWidth) * 2);
  const horizontalTable = needHorizontal ? coefficients(srcWidth, dstWidth) : null;
  const verticalTable = needVertical ? coefficients(srcHeight, dstHeight) : null;
  const intermediate = needHorizontal && needVertical
    ? new Uint8Array(horizontalFirst
      ? rgbaLength(dstWidth, srcHeight)
      : rgbaLength(srcWidth, dstHeight))
    : null;

  return {
    resize(rgba) {
      if (!(rgba instanceof Uint8Array) && !(rgba instanceof Uint8ClampedArray)) {
        throw new TypeError('Expected Uint8Array or Uint8ClampedArray RGBA input');
      }
      if (rgba.length !== inputLength) {
        throw new RangeError('RGBA byte count does not match source dimensions');
      }
      // The caller transfers/detaches this buffer; it must never be reused.
      const output = new Uint8ClampedArray(outputLength);
      if (needHorizontal && needVertical) {
        if (horizontalFirst) {
          horizontal(rgba, srcWidth, srcHeight, intermediate, dstWidth, horizontalTable);
          vertical(intermediate, dstWidth, output, dstHeight, verticalTable);
        } else {
          vertical(rgba, srcWidth, intermediate, dstHeight, verticalTable);
          horizontal(intermediate, srcWidth, dstHeight, output, dstWidth, horizontalTable);
        }
      } else if (needHorizontal) {
        horizontal(rgba, srcWidth, srcHeight, output, dstWidth, horizontalTable);
      } else if (needVertical) {
        vertical(rgba, srcWidth, output, dstHeight, verticalTable);
      } else {
        output.set(rgba);
        for (let alpha = 3; alpha < outputLength; alpha += 4) output[alpha] = 255;
      }
      return output;
    },
  };
}

/** One-shot convenience entry point, using the same cached production implementation. */
export function resizeLanczosRgba(rgba, srcWidth, srcHeight, dstWidth, dstHeight) {
  return createLanczosResampler(srcWidth, srcHeight, dstWidth, dstHeight).resize(rgba);
}
