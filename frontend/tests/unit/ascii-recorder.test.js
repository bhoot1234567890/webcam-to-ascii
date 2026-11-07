import { expect, test } from 'bun:test';
import {
  RECORDING_LIMITS,
  createAsciiRecorder,
  getRecordingSupport,
  recordingExtension,
} from '../../src/ascii-recorder.js';
import { ARTWORK_BACKGROUND, computeArtworkLayout } from '../../src/artwork.js';

const frame = {
  text: 'Mg\n@_', columns: 2, rows: 2,
  fontFamily: '"Roboto Mono"', fontSize: 12, lineHeight: 14.4,
  color: '#eeeeee', background: ARTWORK_BACKGROUND, glow: false,
};
const metrics = { advance: 7.2, ascent: 12, descent: 4, left: 1.5, right: 2 };


test('actual MIME container chooses the extension, including codec parameters and casing', () => {
  for (const mime of ['video/mp4', 'video/mp4;codecs=avc1.640028', ' VIDEO/MP4 ; codecs="avc1"']) {
    expect(recordingExtension(mime)).toBe('mp4');
  }
  for (const mime of ['video/webm', 'video/webm;codecs=vp8', 'video/webm;codecs=vp9']) {
    expect(recordingExtension(mime)).toBe('webm');
  }
  for (const mime of ['', 'audio/webm', 'video/ogg', 'application/octet-stream']) {
    expect(() => recordingExtension(mime)).toThrow('unsupported recording format');
  }
});

test('layout preserves uniform grid scaling and complete ink bounds at every edge', () => {
  for (const columns of [1, 2, 80, 400]) {
    for (const rows of [1, 2, 60, 300]) {
      for (const glow of [false, true]) {
        const input = { ...frame, columns, rows, glow };
        const layout = computeArtworkLayout(input, metrics);
        const halo = glow ? 12 : 0;
        expect(layout.width % 2).toBe(0);
        expect(layout.height % 2).toBe(0);
        expect(layout.width).toBeLessThanOrEqual(1280);
        expect(layout.height).toBeLessThanOrEqual(1280);
        expect(layout.scale).toBeGreaterThan(0);
        expect(layout.scale).toBeLessThanOrEqual(1);
        // Baselines include half-leading, even when ink is taller than the line box.
        expect(layout.baseline).toBeCloseTo(11.2, 10);
        const left = layout.x - (metrics.left + halo) * layout.scale;
        const right = layout.x + (columns * metrics.advance + metrics.right + halo) * layout.scale;
        const top = layout.y + (layout.baseline - metrics.ascent - halo) * layout.scale;
        const bottom = layout.y + (layout.baseline + (rows - 1) * input.lineHeight + metrics.descent + halo) * layout.scale;
        expect(left).toBeGreaterThanOrEqual(0);
        expect(top).toBeGreaterThanOrEqual(0);
        expect(right).toBeLessThanOrEqual(layout.width);
        expect(bottom).toBeLessThanOrEqual(layout.height);
        // Horizontal and vertical cells share one scale; neither axis is stretched to even dimensions.
        expect(layout.advance * layout.scale / (input.lineHeight * layout.scale)).toBeCloseTo(0.5, 10);
      }
    }
  }
});

test('a small grid is not enlarged and glow adds bounded room without altering baselines', () => {
  const plain = computeArtworkLayout(frame, metrics);
  const glow = computeArtworkLayout({ ...frame, glow: true }, metrics);
  expect(plain.scale).toBe(1);
  expect(glow.scale).toBe(1);
  expect(glow.width - plain.width).toBe(24);
  expect(glow.height - plain.height).toBe(24);
  expect(glow.baseline).toBe(plain.baseline);
});

test('invalid geometry is rejected instead of producing a cropped or zero-size video', () => {
  for (const value of [0, -1, 1.5, NaN, Infinity]) {
    expect(() => computeArtworkLayout({ ...frame, columns: value }, metrics)).toThrow(RangeError);
    expect(() => computeArtworkLayout({ ...frame, rows: value }, metrics)).toThrow(RangeError);
  }
  for (const key of ['advance', 'ascent', 'descent', 'left', 'right']) {
    expect(() => computeArtworkLayout(frame, { ...metrics, [key]: NaN })).toThrow(RangeError);
  }
});

// These doubles exercise only feature selection and failure/resource lifecycles. They never
// manufacture a successful video or claim that an encoded container can actually be played.
function withRuntime(run, {
  supported = () => true, failStart = false, fontReady = true, manualCapture = true, failRequest = false,
} = {}) {
  const saved = new Map();
  const canvases = [];
  const tracks = [];
  const encoders = [];
  const calls = [];
  function install(name, value) {
    if (!saved.has(name)) saved.set(name, Object.getOwnPropertyDescriptor(globalThis, name));
    Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
  }
  class CaptureTrack extends EventTarget {
    readyState = 'live';
    stopped = 0;
    stop() { this.stopped++; this.readyState = 'ended'; }
    constructor() {
      super();
      if (manualCapture) this.requestFrame = () => {
        calls.push(['request', encoders.at(-1)?.state, canvases.at(-1).paintedText.slice()]);
        if (failRequest) throw new Error('Canvas capture failed');
      };
    }
  }
  class Canvas {
    width = 300;
    height = 150;
    constructor() {
      this.paintedText = [];
      canvases.push(this);
      this.context = {
        measureText: () => ({
          width: 7.2, fontBoundingBoxAscent: 12, fontBoundingBoxDescent: 4,
          actualBoundingBoxAscent: 11, actualBoundingBoxDescent: 3,
          actualBoundingBoxLeft: 1.5, actualBoundingBoxRight: 9.2,
        }),
        setTransform() {},
        fillRect: () => { this.paintedText = []; calls.push(['background']); },
        fillText: (...args) => { this.paintedText.push(args[0]); calls.push(['text', ...args]); },
      };
    }
    getContext() { return this.context; }
    captureStream(rate) {
      calls.push(['capture', rate]);
      const track = new CaptureTrack();
      tracks.push(track);
      return { getTracks: () => [track], getVideoTracks: () => [track] };
    }
  }
  class Encoder {
    state = 'inactive';
    stopCalls = 0;
    static isTypeSupported(type) { return supported(type); }
    constructor(stream, options) {
      this.mimeType = options.mimeType;
      this.stream = stream;
      encoders.push(this);
    }
    start() {
      if (failStart) throw new Error('Encoder startup failed');
      this.state = 'recording';
      calls.push(['start']);
    }
    stop() { this.stopCalls++; this.state = 'inactive'; }
  }
  install('MediaRecorder', Encoder);
  install('HTMLCanvasElement', Canvas);
  install('document', {
    fonts: { check: () => fontReady },
    createElement() { return new Canvas(); },
  });
  const recorder = createAsciiRecorder();
  try {
    run({ recorder, canvases, tracks, encoders, calls, install });
  } finally {
    recorder.dispose();
    for (const [name, descriptor] of saved) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else delete globalThis[name];
    }
  }
}

function installClock(install) {
  let now = 0;
  let nextTimer = 1;
  const timers = new Map();
  const intervals = new Map();
  install('performance', { now: () => now });
  install('setTimeout', (callback, delay) => {
    const id = nextTimer++;
    timers.set(id, { callback, delay });
    return id;
  });
  install('clearTimeout', (id) => timers.delete(id));
  install('setInterval', (callback, delay) => {
    const id = nextTimer++;
    intervals.set(id, { callback, delay });
    return id;
  });
  install('clearInterval', (id) => intervals.delete(id));
  return {
    timers, intervals,
    setNow(value) { now = value; },
    fire(id) {
      const timer = timers.get(id);
      timers.delete(id);
      timer.callback();
    },
  };
}

function installPaintFrames(install) {
  let nextFrame = 1;
  const frames = new Map();
  install('requestAnimationFrame', (callback) => {
    const id = nextFrame++;
    frames.set(id, callback);
    return id;
  });
  install('cancelAnimationFrame', (id) => frames.delete(id));
  return {
    frames,
    paint() {
      const [id, callback] = [...frames][0];
      frames.delete(id);
      callback();
    },
  };
}

test('Stop accepts the latest pending artwork within budget and fences its boundary before encoder stop', () => {
  withRuntime(({ recorder, calls, install, encoders, tracks }) => {
    const clock = installClock(install);
    const paints = installPaintFrames(install);
    recorder.start(frame);
    recorder.pushFrame({ ...frame, text: 'ZZ\nZZ' });
    clock.setNow(10);
    expect(recorder.stop()).toBe(true);
    expect(recorder.state).toBe('stopping');
    expect(encoders[0].stopCalls).toBe(0);
    const [boundaryId] = [...clock.timers].find(([, { delay }]) => delay < 70);
    clock.setNow(67);
    clock.fire(boundaryId);
    expect(calls.filter(([type]) => type === 'request').map(([, , lines]) => lines)).toEqual([
      ['Mg', '@_'], ['ZZ', 'ZZ'],
    ]);
    paints.paint();
    expect(encoders[0].stopCalls).toBe(0);
    paints.paint();
    expect(encoders[0].stopCalls).toBe(1);
    expect(paints.frames.size).toBe(0);
    expect([...clock.timers.values()].map(({ delay }) => delay)).toEqual([10_000]);
    recorder.reset();
    expect(tracks[0].stopped).toBe(1);
    expect(clock.timers.size).toBe(0);
    expect(clock.intervals.size).toBe(0);
  });
});


test('hidden or stalled painting cannot leave finalization waiting indefinitely on RAF', () => {
  for (const visibilityState of ['hidden', 'visible']) {
    withRuntime(({ recorder, install, encoders, tracks }) => {
      const clock = installClock(install);
      const paints = installPaintFrames(install);
      install('document', {
        fonts: { check: () => true }, visibilityState,
        createElement: () => new HTMLCanvasElement(),
      });
      recorder.start(frame);
      clock.setNow(650);
      recorder.stop();
      expect(paints.frames.size).toBe(visibilityState === 'hidden' ? 0 : 1);
      const stalePaint = [...paints.frames.values()][0];
      const [fenceId] = [...clock.timers].find(([, { delay }]) => delay === 250);
      clock.fire(fenceId);
      expect(encoders[0].stopCalls).toBe(1);
      expect(paints.frames.size).toBe(0);
      recorder.reset();
      recorder.start(frame);
      stalePaint?.();
      expect(encoders[1].state).toBe('recording');
      expect(tracks[1].stopped).toBe(0);
      recorder.reset();
      expect(clock.timers.size).toBe(0);
      expect(clock.intervals.size).toBe(0);
    });
  }
});

test('reset during either final boundary wait cancels all queued work and stale callbacks', () => {
  for (const afterBoundary of [false, true]) {
    withRuntime(({ recorder, install, calls, tracks, encoders }) => {
      const clock = installClock(install);
      const paints = installPaintFrames(install);
      recorder.start(frame);
      if (afterBoundary) clock.setNow(650);
      recorder.stop();
      const staleTimers = [...clock.timers.values()].map(({ callback }) => callback);
      const stalePaints = [...paints.frames.values()];
      recorder.reset();
      const cleaned = calls.slice();
      for (const callback of [...staleTimers, ...stalePaints]) callback();
      expect(recorder.state).toBe('idle');
      expect(calls).toEqual(cleaned);
      expect(tracks[0].stopped).toBe(1);
      expect(encoders[0].stopCalls).toBe(1);
      expect(clock.timers.size).toBe(0);
      expect(clock.intervals.size).toBe(0);
      expect(paints.frames.size).toBe(0);
    });
  }
});


test('a returned track without requestFrame uses a timed stream and releases its zero-rate probe', () => {
  withRuntime(({ recorder, tracks }) => {
    expect(recorder.start(frame)).toBe(true);
    expect(tracks[0].stopped).toBe(1);
    expect(tracks[1].readyState).toBe('live');
    recorder.reset();
    expect(tracks.map((track) => track.stopped)).toEqual([1, 1]);
  }, { manualCapture: false });
});

test('failed fallback startup and failed manual capture release every allocated track', () => {
  withRuntime(({ recorder, tracks, canvases }) => {
    expect(recorder.start(frame)).toBe(false);
    expect(tracks.map((track) => track.stopped)).toEqual([1, 1]);
    expect(canvases[0].width).toBe(0);
  }, { manualCapture: false, failStart: true });
  withRuntime(({ recorder, tracks, encoders, canvases }) => {
    expect(recorder.start(frame)).toBe(false);
    expect(tracks[0].stopped).toBe(1);
    expect(encoders[0].state).toBe('inactive');
    expect(canvases[0].width).toBe(0);
    expect(recorder.clip).toBeNull();
  }, { failRequest: true });
});

test('capture failures on a live update or final boundary discard output and clear scheduling resources', () => {
  for (const boundary of [false, true]) {
    withRuntime(({ recorder, install, tracks, encoders, canvases }) => {
      const clock = installClock(install);
      const paints = installPaintFrames(install);
      recorder.start(frame);
      tracks[0].requestFrame = () => { throw new Error('Capture request failed'); };
      clock.setNow(650);
      expect(boundary ? recorder.stop() : recorder.pushFrame({ ...frame, text: 'BB\nBB' })).toBe(false);
      expect(recorder.state).toBe('idle');
      expect(recorder.clip).toBeNull();
      expect(tracks[0].stopped).toBe(1);
      expect(encoders[0].state).toBe('inactive');
      expect(canvases[0].width).toBe(0);
      expect(clock.timers.size).toBe(0);
      expect(clock.intervals.size).toBe(0);
      expect(paints.frames.size).toBe(0);
    });
  }
});

test('unchanged commits cancel obsolete pending work without periodic painting or capture requests', () => {
  withRuntime(({ recorder, calls, install, canvases }) => {
    const clock = installClock(install);
    recorder.start(frame);
    const initial = calls.slice();
    recorder.pushFrame({ ...frame, text: 'BB\nBB' });
    const [oldId, oldTimer] = [...clock.timers].find(([, { delay }]) => delay < 70);
    recorder.pushFrame({ ...frame });
    expect(clock.timers.has(oldId)).toBe(false);
    clock.setNow(500);
    oldTimer.callback();
    recorder.pushFrame({ ...frame });
    expect(calls).toEqual(initial);
    expect(canvases[0].paintedText).toEqual(['Mg', '@_']);
    expect([...clock.timers.values()].map(({ delay }) => delay)).toEqual([30_000]);
    recorder.stop();
    expect(calls.filter(([type]) => type === 'request').map(([, , lines]) => lines)).toEqual([
      ['Mg', '@_'], ['Mg', '@_'],
    ]);
    recorder.reset();
    expect(clock.timers.size).toBe(0);
    expect(clock.intervals.size).toBe(0);
  });
});

test('stale canceled draw callbacks cannot replace the next commit timer', () => {
  withRuntime(({ recorder, calls, install, canvases }) => {
    const clock = installClock(install);
    recorder.start(frame);
    recorder.pushFrame({ ...frame, text: 'BB\nBB' });
    const [, oldTimer] = [...clock.timers].find(([, { delay }]) => delay < 70);
    recorder.pushFrame({ ...frame });
    recorder.pushFrame({ ...frame, text: 'CC\nCC' });
    const [newId] = [...clock.timers].find(([, { delay }]) => delay < 70);
    oldTimer.callback();
    expect([...clock.timers.keys()]).toContain(newId);
    expect(clock.timers.size).toBe(2);
    clock.setNow(67);
    clock.fire(newId);
    expect(canvases[0].paintedText).toEqual(['CC', 'CC']);
    expect(calls.filter(([type]) => type === 'request').map(([, , lines]) => lines)).toEqual([
      ['Mg', '@_'], ['CC', 'CC'],
    ]);
    recorder.reset();
    expect(clock.timers.size).toBe(0);
  });
});

test('malformed or drifting commits fail before unchanged-text suppression or draw scheduling', () => {
  for (const changed of [
    { ...frame, background: '#ffffff' },
    { ...frame, lineHeight: NaN },
    { ...frame, text: 'B\nBB' },
    { ...frame, text: 'BB\nB\t' },
  ]) {
    withRuntime(({ recorder, tracks, calls, install }) => {
      const clock = installClock(install);
      recorder.start(frame);
      const initial = calls.slice();
      expect(recorder.pushFrame(changed)).toBe(false);
      expect(recorder.state).toBe('idle');
      expect(recorder.clip).toBeNull();
      expect(tracks[0].stopped).toBe(1);
      expect(calls).toEqual(initial);
      expect(clock.timers.size).toBe(0);
      expect(clock.intervals.size).toBe(0);
    });
  }
});


test('unsupported codecs and absent platform APIs give actionable Photo fallback without a canvas', () => {
  withRuntime(({ recorder, canvases, install }) => {
    expect(getRecordingSupport().supported).toBe(false);
    expect(recorder.start(frame)).toBe(false);
    expect(canvases).toHaveLength(0);
    install('MediaRecorder', undefined);
    expect(getRecordingSupport().supported).toBe(false);
  }, { supported: () => false });
});


test('reset releases tracks/canvas and stale final or overflow events cannot revive a recording', () => {
  withRuntime(({ recorder, encoders, tracks, canvases }) => {
    recorder.start(frame);
    const oldData = encoders[0].ondataavailable;
    const oldStop = encoders[0].onstop;
    const oldError = encoders[0].onerror;
    recorder.reset();
    expect(recorder.state).toBe('idle');
    expect(tracks[0].stopped).toBe(1);
    expect(canvases[0].width).toBe(0);
    expect(canvases[0].height).toBe(0);
    expect(encoders[0].ondataavailable).toBeNull();
    recorder.start(frame);
    oldData({ data: { size: RECORDING_LIMITS.maxBytes + 1, type: 'video/webm' } });
    oldStop();
    oldError();
    expect(recorder.state).toBe('recording');
    expect(recorder.clip).toBeNull();
    expect(tracks[1].stopped).toBe(0);
  });
});

test('overflow discards the entire recording rather than truncating encoded chunks', () => {
  withRuntime(({ recorder, encoders, tracks, canvases }) => {
    recorder.start(frame);
    encoders[0].ondataavailable({ data: { size: RECORDING_LIMITS.maxBytes, type: 'video/mp4' } });
    expect(recorder.state).toBe('recording');
    encoders[0].ondataavailable({ data: { size: 1, type: 'video/mp4' } });
    expect(recorder.state).toBe('idle');
    expect(recorder.clip).toBeNull();
    expect(tracks[0].stopped).toBe(1);
    expect(canvases[0].width).toBe(0);
  });
});

test('unexpected track end and encoder failures discard output and clean resources', () => {
  for (const event of ['track', 'encoder', 'unexpected-stop', 'stopping-track', 'stopping-encoder', 'stopping-stop', 'empty']) {
    withRuntime(({ recorder, encoders, tracks, canvases, install }) => {
      const clock = installClock(install);
      recorder.start(frame);
      if (event.startsWith('stopping-') || event === 'empty') {
        clock.setNow(650);
        recorder.stop();
      }
      if (event.endsWith('track')) tracks[0].dispatchEvent(new Event('ended'));
      if (event.endsWith('encoder')) encoders[0].onerror();
      if (event.endsWith('stop')) encoders[0].onstop();
      if (event === 'empty') {
        const [fenceId] = [...clock.timers].find(([, { delay }]) => delay === 250);
        clock.fire(fenceId);
        encoders[0].onstop();
      }
      expect(recorder.state).toBe('idle');
      expect(recorder.clip).toBeNull();
      expect(tracks[0].stopped).toBe(1);
      expect(canvases[0].width).toBe(0);
      expect(clock.timers.size).toBe(0);
      expect(clock.intervals.size).toBe(0);
    });
  }
});

test('failed start releases already-created resources, while an unloaded font never allocates them', () => {
  withRuntime(({ recorder, tracks, canvases }) => {
    expect(recorder.start(frame)).toBe(false);
    expect(tracks[0].stopped).toBe(1);
    expect(canvases[0].width).toBe(0);
    expect(recorder.state).toBe('idle');
  }, { failStart: true });
  withRuntime(({ recorder, canvases }) => {
    expect(recorder.start(frame)).toBe(false);
    expect(canvases).toHaveLength(0);
  }, { fontReady: false });
});

test('presentation changes discard recording; dispose is idempotent and permanently disables start', () => {
  withRuntime(({ recorder, tracks }) => {
    recorder.start(frame);
    expect(recorder.pushFrame({ ...frame, columns: 3 })).toBe(false);
    expect(recorder.state).toBe('idle');
    expect(tracks[0].stopped).toBe(1);
    recorder.start(frame);
    const { dispose } = recorder;
    dispose();
    dispose();
    expect(recorder.start(frame)).toBe(false);
    expect(recorder.state).toBe('idle');
    expect(tracks[1].stopped).toBe(1);
  });
});

test('draw throttling keeps only the newest commit, deadline stops, and finishing timeout releases resources', () => {
  withRuntime(({ install, calls, tracks, encoders }) => {
    const clock = installClock(install);
    const { timers, intervals } = clock;
    const times = [];
    const errors = [];
    const engine = createAsciiRecorder({ onTime: (time) => times.push(time), onError: (error) => errors.push(error) });
    try {
      engine.start(frame);
      engine.pushFrame({ ...frame, text: 'AA\nAA' });
      engine.pushFrame({ ...frame, text: 'ZZ\nZZ' });
      expect(calls.filter(([type]) => type === 'text')).toHaveLength(2);
      const [drawId] = [...timers].find(([, { delay }]) => delay < 70);
      clock.setNow(67);
      clock.fire(drawId);
      expect(calls.filter(([type]) => type === 'text').map(([, text]) => text)).toEqual(['Mg', '@_', 'ZZ', 'ZZ']);
      clock.setNow(250);
      [...intervals.values()][0].callback();
      expect(times).toEqual([0, 250]);
      const [deadlineId] = [...timers].find(([, { delay }]) => delay === 30_000);
      clock.setNow(30_000);
      clock.fire(deadlineId);
      expect(engine.state).toBe('stopping');
      const [fenceId] = [...timers].find(([, { delay }]) => delay === 250);
      clock.fire(fenceId);
      expect(encoders[0].stopCalls).toBe(1);
      expect(intervals.size).toBe(0);
      const [finishId] = [...timers].find(([, { delay }]) => delay === 10_000);
      clock.fire(finishId);
      expect(engine.state).toBe('idle');
      expect(tracks[0].stopped).toBe(1);
      expect(errors[0]).toContain('did not finish');
      expect(timers.size).toBe(0);
    } finally { engine.dispose(); }
  });
});

test('reentrant state consumers cannot leave encoder resources running after reset', () => {
  withRuntime(({ tracks, canvases }) => {
    let engine;
    const states = [];
    const errors = [];
    engine = createAsciiRecorder({
      onState({ state }) {
        states.push(state);
        if (state === 'recording') engine.reset();
      },
      onError(message) { errors.push(message); },
    });
    try {
      expect(engine.start(frame)).toBe(false);
      expect(engine.state).toBe('idle');
      expect(states).toEqual(['recording', 'idle']);
      expect(tracks[0].stopped).toBe(1);
      expect(canvases[0].width).toBe(0);
      expect(engine.start({ ...frame, columns: 0 })).toBe(false);
      expect(errors).toHaveLength(1);
    } finally { engine.dispose(); }
  });
});

test('reset from the stopping notification cannot leave a boundary draw or paint fence running', () => {
  withRuntime(({ install, tracks, calls, encoders }) => {
    const clock = installClock(install);
    const paints = installPaintFrames(install);
    let engine;
    engine = createAsciiRecorder({
      onState({ state }) { if (state === 'stopping') engine.reset(); },
    });
    try {
      engine.start(frame);
      const initial = calls.slice();
      expect(engine.stop()).toBe(false);
      expect(engine.state).toBe('idle');
      expect(calls).toEqual(initial);
      expect(tracks[0].stopped).toBe(1);
      expect(encoders[0].stopCalls).toBe(1);
      expect(clock.timers.size).toBe(0);
      expect(clock.intervals.size).toBe(0);
      expect(paints.frames.size).toBe(0);
    } finally { engine.dispose(); }
  });
});
