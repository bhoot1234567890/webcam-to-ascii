// CRT scanlines — one post-process snippet for the composer in
// frontend/src/crt/compose.js. The composer interpolates this GLSL between
// the sampling stage and main(), and multiplies the sampled color by
// crtScanline(warped).
//
// Derivation (technique verified from libretro crt-easymode / crt-royale; do
// not re-search):
// - ROW LOCK: the artwork is monospace ASCII with N glyph rows, so one
//   scanline period equals ONE GLYPH ROW: rowPhase = warpedUV.y * rows. That
//   is the analog of emulator shaders locking to the console's scanlines: the
//   dark bands always land in the gaps between text lines, and they curve
//   with the tube because the phase comes from the warped uv, never from the
//   flat screen uv.
// - BEAM RESPONSE: a scanline is the electron beam's brightness profile, so
//   it multiplies AFTER sampling; it is not a warp of the sampling uv.
// - SMOOTH LOBE: beams are not square waves. The profile is a raised cosine
//   raised to SCANLINE_BEAM_EXPONENT — smooth everywhere, exactly 1 at the
//   row center, exactly 0 at the lobe tip — which reads as glass, not blinds.
// - EXACT BYPASS: intensity 0 must mean the effect is off: the multiplier is
//   exactly 1.0 at every phase, on both the JS and GLSL paths.
// - MEAN BRIGHTNESS: a multiplier spends its average below 1 and its peak may
//   not exceed 1.0 (the row center stays exact), so the only brightness
//   compensation available is the lobe SHAPE. See SCANLINE_MEAN_DARKNESS_BUDGET.

// Duty (full width at half darkness) of the dark band as a fraction of one
// glyph-row period: ~15% line gap against an ~85% bright beam, the narrow
// profile that keeps the period-average multiplier near 1.
export const SCANLINE_BEAM_DUTY = 0.1523;

// Sharpness of the raised-cosine lobe: darkness = (0.5 + 0.5*cos(2*pi*phase))
// ** exponent. 12 lands the half-darkness duty at SCANLINE_BEAM_DUTY and the
// period-average darkness at C(24,12)/4^12 = 0.1612. Interpolated into the
// GLSL below so the JS mirror and the shader cannot drift apart.
export const SCANLINE_BEAM_EXPONENT = 12;

// Brightness compensation budget: the period-average darkness may not exceed
// 1/6. At the default intensity 0.3 the row-average multiplier is therefore
// 1 - 0.3 * 0.1612 = 0.9516 — inside the documented [0.95, 1.05] band, flat
// to the eye. The tradeoff this buys: the peak is pinned at exactly 1.0 and
// the floor at exactly 1 - intensity, so at full intensity the whole frame
// dims by the budgeted ~16% (0.8388 average), like a real tube driven hard.
export const SCANLINE_MEAN_DARKNESS_BUDGET = 1 / 6;

export const DEFAULTS = Object.freeze({ intensity: 0.3 });

export function normalizeScanlines(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError('CRT scanline intensity must be a number.');
  }
  if (value < 0 || value > 1) {
    throw new RangeError('CRT scanline intensity must be between 0 and 1.');
  }
  return Object.freeze({ intensity: value });
}

// Pure JS mirror of crtScanline: rowPhase is the fractional position within
// one glyph-row period (0 = the gap between text lines, 0.5 = the row
// center); any real input is valid and wraps by one period exactly like
// fract() in the shader. Per-frame rendering uses the GLSL path; this exists
// to pin the shader semantics in unit tests. Returns the beam multiplier.
export function scanlineModulation(rowPhase, normalized) {
  const intensity = normalized.intensity;
  if (intensity === 0) return 1;
  const phase = rowPhase - Math.floor(rowPhase);
  const base = 0.5 + 0.5 * Math.cos(2 * Math.PI * phase);
  const darkness = Math.pow(base, SCANLINE_BEAM_EXPONENT);
  return 1 - intensity * darkness;
}

export const GLSL = `
uniform float uCrtScanIntensity;
uniform float uCrtScanRows;

// Beam response for the sampled color. uCrtScanRows is the committed glyph
// row count supplied by the composer, so one dark band per glyph row lands in
// the gap between text lines; the phase comes from the warped uv, so the
// bands curve with the tube. Smooth lobe, never a square wave; intensity 0 is
// an exact 1.0 bypass.
float crtScanline(vec2 warpedUV) {
  if (uCrtScanIntensity == 0.0) return 1.0;
  float rowPhase = fract(warpedUV.y * uCrtScanRows);
  float base = 0.5 + 0.5 * cos(6.283185307179586 * rowPhase);
  float darkness = pow(base, ${SCANLINE_BEAM_EXPONENT.toFixed(1)});
  return 1.0 - uCrtScanIntensity * darkness;
}
`;

// CSS live-view recipe (for the integrator to transpose into styles.css when
// the WebGL pass is unavailable). The overlay reproduces the same beam lobe
// flat, with the dark band centered on the gradient seam so it falls in the
// gap between text lines.
//
// Period: P = the DISPLAYED line height of the artwork element, e.g.
//   const style = getComputedStyle(artwork);
//   const period = Math.round(parseFloat(style.lineHeight)
//     || 1.2 * parseFloat(style.fontSize));
// If the artwork has padding, shift the band onto the first line gap with
// background-position.
//
// Stops below are the default intensity 0.3 (alpha = intensity * lobe, i.e.
// the exact complement of scanlineModulation), shown for a 20px period; the
// comment on each stop is its fraction of the period. Scale every stop by
// (intensity / 0.3) for other intensities and remove the overlay at 0.
//
//   .scanline-overlay {
//     position: absolute;
//     inset: 0;
//     pointer-events: none;
//     background: repeating-linear-gradient(
//       to bottom,
//       rgba(0, 0, 0, 0.3)   0px,    /* 0.000 band center, the line seam   */
//       rgba(0, 0, 0, 0.28)  0.5px,  /* 0.025 */
//       rgba(0, 0, 0, 0.22)  1px,    /* 0.050 */
//       rgba(0, 0, 0, 0.15)  1.5px,  /* 0.075 */
//       rgba(0, 0, 0, 0.09)  2px,    /* 0.100 */
//       rgba(0, 0, 0, 0.045) 2.5px,  /* 0.125 */
//       rgba(0, 0, 0, 0.02)  3px,    /* 0.150 */
//       rgba(0, 0, 0, 0)     4px,    /* 0.200 band tail                   */
//       rgba(0, 0, 0, 0)     16px,   /* 0.800 bright beam holds clear     */
//       rgba(0, 0, 0, 0.02)  17px,   /* 0.850 mirror rise                 */
//       rgba(0, 0, 0, 0.045) 17.5px,
//       rgba(0, 0, 0, 0.09)  18px,
//       rgba(0, 0, 0, 0.15)  18.5px,
//       rgba(0, 0, 0, 0.22)  19px,
//       rgba(0, 0, 0, 0.28)  19.5px,
//       rgba(0, 0, 0, 0.3)   20px    /* seam: the next band center        */
//     );
//   }
//
// Unlike the shader, these bands stay straight (no tube curvature); the
// WebGL pass above remains the source of truth.
