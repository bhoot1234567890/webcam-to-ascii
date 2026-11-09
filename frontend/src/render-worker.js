import { createRenderer, normalizeSettings } from './renderer.js';
import { normalizeLook } from './looks.js';
import { normalizeCharSet } from './charsets.js';

let renderer = null;
let cacheKey = '';
self.onmessage = ({ data }) => {
  const id = data?.id;
  try {
    if (data?.type !== 'render' || !Number.isSafeInteger(id) || id <= 0) {
      throw new TypeError('Invalid render request');
    }
    const { width, height, rgba } = data;
    if (!(rgba instanceof ArrayBuffer)) throw new TypeError('RGBA must be an ArrayBuffer');
    const settings = normalizeSettings(data.settings);
    const look = normalizeLook(data.look);
    const charSet = normalizeCharSet(data.charSet);
    const key = `${width}:${height}:${JSON.stringify(settings)}:${JSON.stringify(look)}:${charSet}`;
    if (key !== cacheKey) {
      renderer = createRenderer(width, height, settings, look, charSet);
      cacheKey = key;
    }
    const start = performance.now();
    const text = renderer.renderRgba(new Uint8Array(rgba));
    self.postMessage({ type: 'result', id, width, height, text, processingMs: performance.now() - start });
  } catch (error) {
    self.postMessage({ type: 'error', id, message: error.message });
  }
};
