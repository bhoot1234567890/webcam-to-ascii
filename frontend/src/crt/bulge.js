// CRT bulge (barrel distortion + smooth bezel) — one post-process snippet for
// the composer in frontend/src/crt/compose.js.
//
// Derivation (technique verified from libretro crt-geom; do not re-search):
// - INVERSE MAPPING: the shader runs per OUTPUT pixel, so we compute which
//   SOURCE texel to sample. Sampling outward at the edges (source radius grows
//   faster than the output radius) minifies the artwork away from the tube
//   center, which reads as the picture bulging toward the viewer.
// - POLYNOMIAL: source offset = p * (1 + k1*r2 + k2*r4) where p = uv*2-1 is
//   aspect-corrected (p.x *= aspect) so dot(q,q) is circular, not elliptical.
//   k1 = amount * K1_MAX, k2 = k1 * K2_RATIO. amount 0 => k1 = k2 = 0 =>
//   EXACT identity, so the toggle can bypass the pass cheaply.
// - SCALE COMPENSATION: the polynomial is divided by its value at unit radius
//   (comp = 1 + k1 + k2), so the tube's inscribed circle is held fixed and only
//   the corners push past the frame; the visible image corners pull inward
//   monotonically as amount grows.
// - BLACK BEFORE SAMPLE: a warped coordinate outside [0,1] must resolve to
//   black BEFORE texture2D is called — clamping would smear edge texels
//   across the bezel. crtBezel returns the smooth (smoothstep border *
//   vignette) multiplicative factor instead of a hard clip so the tube edge
//   fades like glass; the caller multiplies the sampled color by it.

const K1_MAX = 1.2;
const K2_RATIO = 0.35;
export const BEZEL_WIDTH = 0.015;
export const VIGNETTE_STRENGTH = 0.22;

export const DEFAULTS = Object.freeze({ amount: 0.12 });

export function normalizeBulge(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError('CRT bulge amount must be a number.');
  }
  if (value < 0 || value > 1) {
    throw new RangeError('CRT bulge amount must be between 0 and 1.');
  }
  return Object.freeze({
    amount: value,
    k1: value * K1_MAX,
    k2: value * K1_MAX * K2_RATIO,
    bezelWidth: BEZEL_WIDTH,
    vignette: VIGNETTE_STRENGTH,
  });
}

// Pure JS mirror of crtWarpUV. NOT reentrant: writes into `out` when provided,
// otherwise into a shared module-level scratch array that the next call
// overwrites. Per-frame rendering uses the GLSL path; this exists to pin the
// shader semantics in unit tests.
const warpScratch = [0, 0];
export function warpPoint(uv, aspect, normalized, out = warpScratch) {
  const px = (uv[0] * 2 - 1) * aspect;
  const py = uv[1] * 2 - 1;
  const r2 = px * px + py * py;
  const factor = 1 + normalized.k1 * r2 + normalized.k2 * r2 * r2;
  const comp = 1 + normalized.k1 + normalized.k2;
  const scale = comp === 0 ? 1 : factor / comp;
  out[0] = (px * scale) / aspect * 0.5 + 0.5;
  out[1] = py * scale * 0.5 + 0.5;
  return out;
}

// Pure JS mirror of crtBezel. Returns the 0..1 border*vignette factor for a
// WARPED (post-warp) uv; write it as a plain function so tests can pin values.
export function bezelFactor(warpedUV, normalized) {
  const w = normalized.bezelWidth;
  const bx = smoothstep(0, w, warpedUV[0]) * smoothstep(0, w, 1 - warpedUV[0]);
  const by = smoothstep(0, w, warpedUV[1]) * smoothstep(0, w, 1 - warpedUV[1]);
  const cx = warpedUV[0] * 2 - 1;
  const cy = warpedUV[1] * 2 - 1;
  const vig = 1 - normalized.vignette * (cx * cx + cy * cy);
  return clamp01(bx * by * clamp01(vig));
}

function smoothstep(edge0, edge1, x) {
  const t = clamp01((x - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
}

function clamp01(x) {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

export const GLSL = `
uniform float uCrtBulgeAmount;
uniform float uCrtBulgeAspect;

// Inverse mapping: output uv -> source uv. Aspect-corrected barrel warp with
// unit-radius scale compensation; amount 0 is exact identity.
vec2 crtWarpUV(vec2 uv) {
  vec2 p = uv * 2.0 - 1.0;
  p.x *= uCrtBulgeAspect;
  float r2 = dot(p, p);
  float k1 = uCrtBulgeAmount * ${K1_MAX.toFixed(1)};
  float k2 = k1 * ${K2_RATIO.toFixed(2)};
  float comp = 1.0 + k1 + k2;
  float factor = (1.0 + k1 * r2 + k2 * r2 * r2) / comp;
  p *= factor;
  p.x /= uCrtBulgeAspect;
  return p * 0.5 + 0.5;
}

// Smooth bezel: 1.0 at the tube center, 0.0 at/after the frame edge, a smooth
// glass-like fade across the border band, times a mild radial vignette.
float crtBezel(vec2 warpedUV) {
  float w = ${BEZEL_WIDTH.toFixed(3)};
  vec2 e = vec2(smoothstep(0.0, w, warpedUV.x), smoothstep(0.0, w, warpedUV.y));
  vec2 f = vec2(smoothstep(0.0, w, 1.0 - warpedUV.x), smoothstep(0.0, w, 1.0 - warpedUV.y));
  vec2 c = warpedUV * 2.0 - 1.0;
  float vig = 1.0 - ${VIGNETTE_STRENGTH.toFixed(2)} * dot(c, c);
  return e.x * e.y * f.x * f.y * clamp(vig, 0.0, 1.0);
}
`;
