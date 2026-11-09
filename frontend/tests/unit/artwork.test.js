import { expect, test } from 'bun:test';
import { ARTWORK_BACKGROUND, computeArtworkLayout, createAsciiArtwork, readAsciiArtwork } from '../../src/artwork.js';
import { asciiToPng } from '../../src/export.js';

const frame = {
  text: 'Mg\n@_', columns: 2, rows: 2,
  fontFamily: '"Roboto Mono"', fontSize: 12, lineHeight: 14.4,
  color: '#eeeeee', background: ARTWORK_BACKGROUND, glow: false,
};
const metrics = { advance: 7.2, ascent: 12, descent: 4, left: 1.5, right: 2 };

test('an odd edge bound reserves even dimensions before scaling rather than cropping the ink', () => {
  const input = { ...frame, columns: 400, rows: 300, glow: true };
  const layout = computeArtworkLayout(input, metrics, { maxEdge: 1279 });
  expect(layout.width % 2).toBe(0);
  expect(layout.height % 2).toBe(0);
  expect(Math.max(layout.width, layout.height)).toBe(1278);
  expect(layout.x - (metrics.left + 12) * layout.scale).toBeGreaterThanOrEqual(0);
  expect(layout.x + (input.columns * metrics.advance + metrics.right + 12) * layout.scale).toBeLessThanOrEqual(layout.width);
  expect(layout.y + (layout.baseline - metrics.ascent - 12) * layout.scale).toBeGreaterThanOrEqual(0);
  expect(layout.y + (layout.baseline + (input.rows - 1) * input.lineHeight + metrics.descent + 12) * layout.scale).toBeLessThanOrEqual(layout.height);
});

test('still-image scaling is explicit while default artwork remains downscale-only', () => {
  const plain = computeArtworkLayout(frame, metrics);
  const still = computeArtworkLayout(frame, metrics, { scale: 2, maxEdge: Infinity, evenDimensions: false });
  expect(plain.scale).toBe(1);
  expect(still.scale).toBe(2);
  expect(still.width).toBe(Math.ceil((frame.columns * metrics.advance + 6) * 2));
  expect(still.height).toBe(Math.ceil((frame.rows * frame.lineHeight + 4) * 2));
  const odd = computeArtworkLayout({ ...frame, columns: 400 }, metrics, { maxEdge: 1279, evenDimensions: false });
  expect(odd.width).toBe(1279);
});

test('invalid output policies and overflowing geometry fail before canvas dimensions can wrap', () => {
  for (const options of [
    { scale: 0 }, { scale: NaN }, { scale: Infinity },
    { maxEdge: 0 }, { maxEdge: 1 }, { maxEdge: NaN }, { maxEdge: -Infinity },
    { evenDimensions: 'yes' },
  ]) {
    expect(() => computeArtworkLayout(frame, metrics, options)).toThrow(RangeError);
  }
  expect(() => computeArtworkLayout(frame, { ...metrics, advance: Number.MAX_VALUE })).toThrow(RangeError);
  expect(() => computeArtworkLayout({ ...frame, columns: 1_000_000_000 }, metrics, { maxEdge: Infinity })).toThrow(RangeError);
});

async function withGlobals(values, run) {
  const saved = new Map();
  for (const [key, value] of Object.entries(values)) {
    saved.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  try { return await run(); } finally {
    for (const [key, descriptor] of saved) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  }
}

test('nonopaque or missing viewport backgrounds use the shared opaque fallback', async () => {
  const viewport = {};
  const element = {
    dataset: { columns: '2', rows: '2' }, textContent: frame.text,
    closest: () => viewport,
  };
  const style = { fontFamily: frame.fontFamily, fontSize: '12px', lineHeight: '14.4px', color: 'rgb(22, 44, 66)' };
  let backgroundColor = '';
  await withGlobals({ getComputedStyle: (node) => node === viewport ? { backgroundColor } : style }, () => {
    for (const background of ['transparent', 'rgba(9, 10, 11, 0)', 'rgb(9 10 11 / 50%)']) {
      backgroundColor = background;
      expect(readAsciiArtwork(element).background).toBe(ARTWORK_BACKGROUND);
    }
    element.closest = () => null;
    expect(readAsciiArtwork(element).background).toBe(ARTWORK_BACKGROUND);
    element.textContent = '';
    expect(() => readAsciiArtwork(element)).toThrow('grid');
  });
});

test('a neon snapshot overrides transparent CSS and carries the tone plane', async () => {
  const element = {
    dataset: { columns: '2', rows: '2' }, textContent: frame.text,
    closest: () => ({}),
  };
  const style = { fontFamily: frame.fontFamily, fontSize: '12px', lineHeight: '14.4px', color: 'rgba(0, 0, 0, 0)' };
  const tones = Uint8Array.from([0, 255, 255, 0]);
  await withGlobals({ getComputedStyle: () => ({ ...style, backgroundColor: 'rgb(9, 9, 11)' }) }, () => {
    const snapshot = readAsciiArtwork(element, {
      ink: { color: '#a855f7', accent: '#ff4fd8' }, tones,
    });
    expect(snapshot.color).toBe('#a855f7');
    expect(snapshot.accentColor).toBe('#ff4fd8');
    expect(snapshot.tones).toBe(tones);
  });
});

// These doubles check setup, validation and resource ownership, not raster or encoded image quality.
async function withCanvas(run, { fontReady = true, badMetrics = false } = {}) {
  const canvases = [];
  const document = {
    fonts: { check: () => fontReady },
    createElement() {
      const calls = [];
      const context = {
        measureText() {
          return {
            width: badMetrics ? NaN : metrics.advance,
            fontBoundingBoxAscent: metrics.ascent, fontBoundingBoxDescent: metrics.descent,
            actualBoundingBoxLeft: metrics.left, actualBoundingBoxRight: metrics.advance + metrics.right,
          };
        },
        setTransform(...args) { calls.push(['transform', ...args]); },
        fillRect(...args) { calls.push(['background', this.fillStyle, ...args]); },
        fillText(...args) { calls.push(['text', this.textBaseline, this.shadowBlur, this.fillStyle, ...args]); },
      };
      const canvas = {
        width: 300, height: 150, context, calls,
        getContext: () => context,
        toBlob(callback, type) { this.serialization = { callback, type }; },
      };
      canvases.push(canvas);
      return canvas;
    },
  };
  return withGlobals({ document }, () => run(canvases));
}

test('deferred drawing fixes the grid at setup and rejects malformed commits before touching pixels', async () => {
  await withCanvas(() => {
    const mutable = { ...frame, glow: true };
    const artwork = createAsciiArtwork(mutable);
    const { context, calls } = artwork.canvas;
    expect(calls).toHaveLength(0);
    context.measureText = () => { throw new Error('A later commit must not remeasure the font.'); };
    mutable.columns = 1;
    mutable.lineHeight = 100;
    mutable.color = '#000000';
    artwork.draw(frame.text);
    const painted = calls.length;
    for (const text of ['', 'A\nAA', 'AA\nAA\nAA', 'A\t\nAA', 'éA\nAA', null]) {
      expect(() => artwork.draw(text)).toThrow();
      expect(calls).toHaveLength(painted);
    }
    artwork.draw('[]\ng_');
    expect(Object.isFrozen(artwork.layout)).toBe(true);
  });
});

test('tritone artwork draws the two ink layers and rejects a mismatched tone plane', async () => {
  await withCanvas(() => {
    const tones = Uint8Array.from([0, 255, 255, 0]);
    const tritone = {
      ...frame, text: 'Mg\n@_', color: '#a855f7', accentColor: '#ff4fd8', tones,
    };
    const artwork = createAsciiArtwork(tritone);
    artwork.draw(tritone.text);
    const painted = artwork.canvas.calls.filter(([type]) => type === 'text');
    expect(painted.map((call) => call[4])).toEqual(['M ', ' _', ' g', '@ ']);
    expect(new Set(painted.map((call) => call[3]))).toEqual(new Set(['#a855f7', '#ff4fd8']));
    const count = artwork.canvas.calls.length;
    expect(() => artwork.draw(tritone.text, new Uint8Array(3))).toThrow('tone map');
    expect(artwork.canvas.calls).toHaveLength(count);
  });
});

test('malformed setup or an unloaded font never allocates a canvas', async () => {
  await withCanvas((canvases) => {
    for (const input of [{ ...frame, text: '' }, { ...frame, background: '' }, { ...frame, fontFamily: 'Not Roboto Mono' }]) {
      expect(() => createAsciiArtwork(input)).toThrow();
    }
    expect(canvases).toHaveLength(0);
  });
  await withCanvas((canvases) => {
    expect(() => createAsciiArtwork(frame)).toThrow('font is still loading');
    expect(canvases).toHaveLength(0);
  }, { fontReady: false });
});

test('failed measurement and transparent backgrounds release their standalone canvas', async () => {
  await withCanvas((canvases) => {
    expect(() => createAsciiArtwork(frame)).toThrow(RangeError);
    expect(canvases[0].width).toBe(0);
    expect(canvases[0].height).toBe(0);
  }, { badMetrics: true });
  await withCanvas((canvases) => {
    expect(() => createAsciiArtwork({ ...frame, background: 'rgba(0, 0, 0, 0.5)' })).toThrow('opaque');
    expect(canvases[0].width).toBe(0);
    expect(canvases[0].height).toBe(0);
  });
});

test('PNG serialization retains its canvas until completion and releases it on success or failure', async () => {
  await withCanvas(async (canvases) => {
    const pending = asciiToPng(frame);
    const canvas = canvases[0];
    expect(canvas.width).toBeGreaterThan(0);
    expect(canvas.height).toBeGreaterThan(0);
    const blob = new Blob(['serialization lifecycle double'], { type: 'image/png' });
    canvas.serialization.callback(blob);
    await pending;
    expect(canvas.width).toBe(0);
    expect(canvas.height).toBe(0);

    const failed = asciiToPng(frame, { scale: 1, maxEdge: 1280, evenDimensions: true });
    canvases[1].serialization.callback(null);
    await expect(failed).rejects.toThrow('Unable to render the PNG');
    expect(canvases[1].width).toBe(0);
    expect(canvases[1].height).toBe(0);
  });
});
