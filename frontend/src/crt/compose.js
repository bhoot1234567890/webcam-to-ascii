// One-pass CRT post-process over the rasterized ASCII artwork canvas.
//
// Layer order follows emulator CRT shaders (libretro crt-geom / crt-easymode):
//   1. bulge: inverse barrel warp of the output UV (where to sample)
//   2. aberration: R/G/B sampled at radial offsets in the SAME warped space
//   3. scanlines: beam modulation multiplied AFTER sampling, row-locked so
//      one period equals one glyph row and the raster curves with the tube
//   4. bezel: smooth border + vignette from the bulge module
//
// The stage is presentation-only: it never touches the committed text, the
// worker, or the render pipeline. WebGL1 (GLSL ES 1.00) for broad support.

import { GLSL as BULGE_GLSL, DEFAULTS as BULGE_DEFAULTS, normalizeBulge } from './bulge.js';
import { GLSL as CA_GLSL, DEFAULTS as CA_DEFAULTS, normalizeAberration } from './aberration.js';
import { GLSL as SCAN_GLSL, DEFAULTS as SCAN_DEFAULTS, normalizeScanlines } from './scanlines.js';

const VERTEX_SRC = `
attribute vec2 aPos;
varying vec2 vUV;
void main() {
  // Canvas texture uploads put row 0 at v = 0 while clip space y points up: flip once here.
  vUV = vec2(aPos.x * 0.5 + 0.5, 0.5 - aPos.y * 0.5);
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const FRAGMENT_SRC = `
precision mediump float;
varying vec2 vUV;
uniform sampler2D uCrtSource;
uniform vec2 uCrtSourceSize;
${BULGE_GLSL}
${CA_GLSL}
${SCAN_GLSL}
void main() {
  vec2 warped = crtWarpUV(vUV);
  vec3 rgb = crtSampleChromatic(warped);
  float beam = crtScanline(warped);
  float bezel = crtBezel(warped);
  gl_FragColor = vec4(rgb * beam * bezel, 1.0);
}`;

const QUAD = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);

function compile(gl, type, source, label) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const detail = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`The CRT shader could not be compiled (${label}): ${detail}`);
  }
  return shader;
}

/** One shared lookup so per-frame renders only set values, never query state. */
function createCrtStage({ width, height } = {}) {
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl', {
    alpha: false,
    depth: false,
    stencil: false,
    antialias: false,
    preserveDrawingBuffer: true,
  });
  if (!gl) throw new Error('This browser has no WebGL, so the CRT effect is unavailable.');
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX_SRC, 'vertex'));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAGMENT_SRC, 'fragment'));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`The CRT shader could not be linked: ${gl.getProgramInfoLog(program)}`);
  }
  gl.useProgram(program);

  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, QUAD, gl.STATIC_DRAW);
  const aPos = gl.getAttribLocation(program, 'aPos');
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  const uniforms = {
    source: gl.getUniformLocation(program, 'uCrtSource'),
    sourceSize: gl.getUniformLocation(program, 'uCrtSourceSize'),
    bulgeAmount: gl.getUniformLocation(program, 'uCrtBulgeAmount'),
    bulgeAspect: gl.getUniformLocation(program, 'uCrtBulgeAspect'),
    caAmount: gl.getUniformLocation(program, 'uCrtCaAmount'),
    scanIntensity: gl.getUniformLocation(program, 'uCrtScanIntensity'),
    scanRows: gl.getUniformLocation(program, 'uCrtScanRows'),
  };
  const bulge = normalizeBulge(BULGE_DEFAULTS.amount);
  const aberration = normalizeAberration(CA_DEFAULTS.amount);
  const scanlines = normalizeScanlines(SCAN_DEFAULTS.intensity);
  gl.uniform1f(uniforms.bulgeAmount, bulge.amount);
  gl.uniform1f(uniforms.caAmount, aberration.amount);
  gl.uniform1f(uniforms.scanIntensity, scanlines.intensity);
  gl.uniform1i(uniforms.source, 0);

  let lastWidth = 0;
  let lastHeight = 0;

  return {
    canvas,
    /** Draw one filtered frame from a source 2D canvas. No allocations. */
    render(source, { rows } = {}) {
      if (!source || source.width < 1 || source.height < 1) {
        throw new TypeError('Expected a painted ASCII artwork canvas to filter.');
      }
      if (typeof rows !== 'number' || !Number.isFinite(rows) || rows < 1) {
        throw new RangeError('The CRT scanline renderer needs the committed grid row count.');
      }
      if (lastWidth !== source.width || lastHeight !== source.height) {
        canvas.width = source.width;
        canvas.height = source.height;
        gl.viewport(0, 0, source.width, source.height);
        lastWidth = source.width;
        lastHeight = source.height;
      }
      gl.uniform2f(uniforms.sourceSize, source.width, source.height);
      gl.uniform1f(uniforms.bulgeAspect, source.width / source.height);
      gl.uniform1f(uniforms.scanRows, rows);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    },
    dispose() {
      gl.deleteTexture(texture);
      gl.deleteBuffer(buffer);
      gl.deleteProgram(program);
      canvas.width = 0;
      canvas.height = 0;
    },
  };
}

/** Probed once at boot: the CRT toggle stays disabled without WebGL. */
export function crtSupported() {
  try {
    const probe = document.createElement('canvas');
    return Boolean(probe.getContext('webgl') || probe.getContext('experimental-webgl'));
  } catch {
    return false;
  }
}

export { createCrtStage };
