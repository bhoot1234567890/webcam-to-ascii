// Chromatic aberration for the CRT post-process. Lens-like channel dispersion
// is applied to the texture LOOKUP coordinates, in the same warped space as
// the barrel distortion: R samples outward along the radial direction, B
// inward, G at the plain warped lookup. The radial weight is r^2 measured in
// corner-normalized space, so the image center stays clean and fringing
// concentrates at the glyph edges near the frame corners, where CRT and lens
// dispersion is most visible on the ink-black artwork.
//
// Composer contract: this snippet is concatenated after the shared header
// (precision, vUV, uCrtSource, uCrtSourceSize, and the bulge warp function)
// and declares only its own uniform and its own sampling function. The pass
// calls crtSampleChromatic with the already-warped uv, so the offsets ride
// the curved geometry instead of fighting it.

export const DEFAULTS = Object.freeze({
  // Channel separation at the texture corner, as a fraction of the UV span.
  // 0.003 is ~1px of fringe on a 356px-wide phone pane (about a third of a
  // glyph on the 100-column grid) and under a subpixel for the inner half of
  // the frame (weight r^2 <= 0.25 there) — readable as lens dispersion
  // without smearing glyph legibility.
  amount: 0.003,
});

const AMOUNT_MIN = 0;
const AMOUNT_MAX = 0.02;
// Quadratic radial weight: dispersion grows toward the edges, never the center.
const RADIAL_EXPONENT = 2;
const OFFSET_COMPONENTS = 6;

/*
 * Live-view CSS recipe (integrator note). The live camera preview stays
 * CSS-only, so approximate this pass on the text output with a chromatic
 * text-shadow pair, scaled with DEFAULTS.amount:
 *
 *   shadowPx = max(1, round(amount * paneWidth))  // 0.003 -> 1px at 356px
 *
 *   #ascii-output {
 *     text-shadow:
 *       -1px 0 0 rgba(255, 64, 64, 0.4),
 *        1px 0 0 rgba(64, 255, 255, 0.4);
 *   }
 *
 * Red lands on one side of every glyph and cyan (the green + blue the red tap
 * leaves behind) on the other, echoing the pass's R-outward / B-inward split
 * at glyph edges. text-shadow cannot fade toward the center the way the r^2
 * weight does, so keep the alpha low (0.4 at the default amount) to stay in
 * family with the WebGL output.
 */

/**
 * Accepts a bare amount number (slider output) or an `{ amount }` settings
 * object (persisted state), mirroring normalizeDensity's dual form. Returns a
 * frozen `{ amount }`; the amount must be a finite number in [0, 0.02].
 */
export function normalizeAberration(value = {}) {
  let amount;
  if (typeof value === 'number') {
    amount = value;
  } else if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
    amount = Object.hasOwn(value, 'amount') ? value.amount : DEFAULTS.amount;
  } else {
    throw new TypeError('CRT aberration settings must be a number or an object');
  }
  if (typeof amount !== 'number' || !Number.isFinite(amount)) {
    throw new TypeError('CRT aberration amount must be a finite number');
  }
  if (amount < AMOUNT_MIN || amount > AMOUNT_MAX) {
    throw new RangeError('CRT aberration amount is out of range.');
  }
  return Object.freeze({ amount });
}

/**
 * Pure JS mirror of crtSampleChromatic's per-channel lookup offsets.
 *
 * `centeredUV` is the CALLER-aspect-corrected, corner-normalized coordinate:
 * radius 0 at the pane center, radius 1 at the texture corners. For a source
 * of `width x height` pixels and uv in [0, 1]:
 *
 *   const aspect = width / height;
 *   const corner = Math.hypot(aspect, 1);
 *   centeredUV = {
 *     x: (uv.x * 2 - 1) * aspect / corner,
 *     y: (uv.y * 2 - 1) / corner,
 *   };
 *
 * The GLSL derives the identical space inside crtSampleChromatic from vUV and
 * uCrtSourceSize, so both implementations agree, and equal pixel distances
 * from the center land at equal radii (isotropic dispersion).
 *
 * Writes `[rx, ry, gx, gy, bx, by]` into `offsets`, a caller-reused
 * Float64Array of exactly six components: R positive along the radial
 * direction, B negative, G exactly zero. Allocates nothing; returns
 * `offsets`. An amount of 0 writes exact zeros — single-tap equivalence.
 */
export function aberrationOffset(centeredUV, normalized, offsets) {
  if (centeredUV === null || typeof centeredUV !== 'object'
      || typeof centeredUV.x !== 'number' || typeof centeredUV.y !== 'number'
      || !Number.isFinite(centeredUV.x) || !Number.isFinite(centeredUV.y)) {
    throw new TypeError('Centered UV coordinates must be finite numbers');
  }
  if (normalized === null || typeof normalized !== 'object'
      || typeof normalized.amount !== 'number' || !Number.isFinite(normalized.amount)) {
    throw new TypeError('CRT aberration amount must be a finite number');
  }
  if (normalized.amount < AMOUNT_MIN || normalized.amount > AMOUNT_MAX) {
    throw new RangeError('CRT aberration amount is out of range.');
  }
  if (!(offsets instanceof Float64Array)) {
    throw new TypeError('Aberration offsets must be a Float64Array');
  }
  if (offsets.length !== OFFSET_COMPONENTS) {
    throw new RangeError('Aberration offsets need exactly six components');
  }
  const { x, y } = centeredUV;
  const radius = Math.hypot(x, y);
  if (normalized.amount === 0 || radius === 0) {
    // Plain-sample equivalence: no channel moves, not even by -0.
    offsets[0] = 0; offsets[1] = 0; offsets[2] = 0;
    offsets[3] = 0; offsets[4] = 0; offsets[5] = 0;
    return offsets;
  }
  const weight = radius ** RADIAL_EXPONENT;
  const push = normalized.amount * weight;
  const dirX = x / radius;
  const dirY = y / radius;
  offsets[0] = dirX * push;
  offsets[1] = dirY * push;
  offsets[2] = 0;
  offsets[3] = 0;
  // 0 - dir keeps axis-aligned points from emitting -0.
  offsets[4] = (0 - dirX) * push;
  offsets[5] = (0 - dirY) * push;
  return offsets;
}

export const GLSL = /* glsl */ `
// Chromatic aberration: three taps of uCrtSource, one per channel, offset in
// the warped space passed in by the composer. Direction and radius come from
// the un-warped centered vUV, aspect-corrected and normalized so the texture
// corners sit at radius 1; the quadratic weight keeps the center clean and
// puts uCrtCaAmount of UV separation at each corner. A uCrtCaAmount of 0.0
// makes every offset exactly 0.0 — a plain single-tap sample.
uniform float uCrtCaAmount;

vec3 crtSampleChromatic(vec2 warpedUV) {
  vec2 unitByHeight = vec2(uCrtSourceSize.x / uCrtSourceSize.y, 1.0);
  vec2 corrected = (vUV * 2.0 - 1.0) * unitByHeight;
  float corner = length(unitByHeight);
  float radius = length(corrected);
  vec2 direction = radius > 0.0 ? corrected / radius : vec2(0.0);
  float weight = radius / corner;
  vec2 offset = direction * (uCrtCaAmount * weight * weight);
  return vec3(
    texture2D(uCrtSource, warpedUV + offset).r,
    texture2D(uCrtSource, warpedUV).g,
    texture2D(uCrtSource, warpedUV - offset).b
  );
}
`;
