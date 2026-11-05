export const RECORDING_LIMITS = Object.freeze({
  maxDurationMs: 30_000,
  maxBytes: 16_777_216,
  maxEdge: 1280,
  frameRate: 15,
  videoBitsPerSecond: 2_500_000,
});

const MIME_TYPES = [
  'video/mp4;codecs=avc1.42E028',
  'video/mp4',
  'video/webm;codecs=vp8',
  'video/webm;codecs=vp9',
  'video/webm',
];
const BACKGROUND = '#0b0b0d';
const FRAME_INTERVAL = 1000 / RECORDING_LIMITS.frameRate;
const PRESENTATION_KEYS = ['columns', 'rows', 'fontFamily', 'fontSize', 'lineHeight', 'color', 'glow'];

/** The encoder's actual container, not the requested codec, determines the filename. */
export function recordingExtension(mimeType) {
  const container = mimeType.split(';', 1)[0].trim().toLowerCase();
  if (container === 'video/mp4') return 'mp4';
  if (container === 'video/webm') return 'webm';
  throw new Error('The browser returned an unsupported recording format. Try another browser.');
}

export function getRecordingSupport() {
  if (typeof MediaRecorder === 'undefined' || typeof MediaRecorder.isTypeSupported !== 'function' ||
      typeof HTMLCanvasElement === 'undefined' || typeof HTMLCanvasElement.prototype.captureStream !== 'function') {
    return { supported: false, mimeType: '', reason: 'Video recording is unavailable in this browser. Use Photo or try a current browser.' };
  }
  for (const mimeType of MIME_TYPES) {
    try {
      if (MediaRecorder.isTypeSupported(mimeType)) return { supported: true, mimeType, reason: '' };
    } catch { /* A codec probe can fail independently of the remaining formats. */ }
  }
  return { supported: false, mimeType: '', reason: 'This browser has no supported MP4 or WebM recorder. Use Photo or try another browser.' };
}

/** Ink bounds are measured once at the preview's font size, before uniform downscaling. */
export function computeRecordingLayout(frame, metrics) {
  const { columns, rows, lineHeight, glow } = frame;
  const { advance, ascent, descent, left, right } = metrics;
  if (![columns, rows].every((value) => Number.isSafeInteger(value) && value > 0) ||
      ![lineHeight, advance].every((value) => Number.isFinite(value) && value > 0) ||
      ![ascent, descent, left, right].every((value) => Number.isFinite(value) && value >= 0)) {
    throw new RangeError('Unable to measure the ASCII recording grid.');
  }
  const baseline = (lineHeight - ascent - descent) / 2 + ascent;
  const halo = glow ? 12 : 0;
  const gutterLeft = Math.ceil(left + halo + 1);
  const gutterRight = Math.ceil(right + halo + 1);
  const gutterTop = Math.ceil(Math.max(0, ascent - baseline) + halo + 1);
  const gutterBottom = Math.ceil(Math.max(0, baseline + descent - lineHeight) + halo + 1);
  const naturalWidth = columns * advance + gutterLeft + gutterRight;
  const naturalHeight = rows * lineHeight + gutterTop + gutterBottom;
  if (!Number.isFinite(naturalWidth) || !Number.isFinite(naturalHeight)) {
    throw new RangeError('The ASCII recording grid is too large.');
  }
  const scale = Math.min(1, RECORDING_LIMITS.maxEdge / Math.max(naturalWidth, naturalHeight));
  const width = Math.max(2, Math.min(RECORDING_LIMITS.maxEdge, Math.ceil(naturalWidth * scale / 2) * 2));
  const height = Math.max(2, Math.min(RECORDING_LIMITS.maxEdge, Math.ceil(naturalHeight * scale / 2) * 2));
  return {
    width, height, scale, baseline, advance,
    x: gutterLeft * scale + (width - naturalWidth * scale) / 2,
    y: gutterTop * scale + (height - naturalHeight * scale) / 2,
  };
}

function validateFrame(frame) {
  if (!frame || typeof frame.text !== 'string' ||
      ![frame.columns, frame.rows].every((value) => Number.isSafeInteger(value) && value > 0) ||
      ![frame.fontSize, frame.lineHeight].every((value) => Number.isFinite(value) && value > 0) ||
      typeof frame.fontFamily !== 'string' || !frame.fontFamily.includes('Roboto Mono') ||
      typeof frame.color !== 'string' || !frame.color || typeof frame.glow !== 'boolean') {
    throw new Error('The ASCII frame is not ready. Wait for the live preview and try Record again.');
  }
}

function measureFont(context) {
  const advance = context.measureText('M').width;
  let ascent = 0;
  let descent = 0;
  let left = 0;
  let right = 0;
  // Cover every printable ASCII glyph, including punctuation overhang and descenders.
  for (let code = 32; code <= 126; code++) {
    const bounds = context.measureText(String.fromCharCode(code));
    ascent = Math.max(ascent, bounds.fontBoundingBoxAscent || 0, bounds.actualBoundingBoxAscent || 0);
    descent = Math.max(descent, bounds.fontBoundingBoxDescent || 0, bounds.actualBoundingBoxDescent || 0);
    left = Math.max(left, bounds.actualBoundingBoxLeft || 0);
    right = Math.max(right, (bounds.actualBoundingBoxRight || 0) - advance);
  }
  if (!(ascent > 0)) throw new Error('Unable to measure the recording font. Reload the page and try again.');
  return { advance, ascent, descent, left, right };
}

/** Records only committed ASCII text. No camera stream, microphone, or DOM capture is used. */
export function createAsciiRecorder({ onState, onTime, onError } = {}) {
  let state = 'idle';
  let clip = null;
  let session = null;
  let disposed = false;

  function notify(callback, value) {
    // Finish lifecycle cleanup, but surface application callback bugs rather than hiding them.
    try { callback?.(value); } catch (error) { queueMicrotask(() => { throw error; }); }
  }

  function transition(next) {
    if (state === next) return;
    state = next;
    notify(onState, { state, clip });
  }

  function clearTimers(active) {
    clearInterval(active.clock);
    clearTimeout(active.deadline);
    clearTimeout(active.drawTimer);
    clearTimeout(active.finishTimer);
    active.pendingText = null;
  }

  function release(active) {
    if (!active) return;
    clearTimers(active);
    if (active.recorder) {
      active.recorder.ondataavailable = null;
      active.recorder.onstop = null;
      active.recorder.onerror = null;
      try { if (active.recorder.state !== 'inactive') active.recorder.stop(); } catch { /* Already ended. */ }
    }
    for (const track of active.stream?.getTracks() || []) {
      track.removeEventListener('ended', active.trackEnded);
      try { track.stop(); } catch { /* Release the remaining tracks even if one fails. */ }
    }
    if (active.canvas) {
      active.canvas.width = 0;
      active.canvas.height = 0;
    }
    active.chunks.length = 0;
    active.context = null;
    active.canvas = null;
    active.stream = null;
    active.recorder = null;
  }

  function revokeClip() {
    if (!clip) return;
    URL.revokeObjectURL(clip.url);
    clip = null;
  }

  function fail(active, message) {
    if (active && session !== active) return;
    session = null; // Invalidate queued encoder events before stopping tracks or notifying callers.
    release(active);
    revokeClip();
    transition('idle');
    notify(onError, message);
  }

  function elapsed(active) {
    return Math.min(RECORDING_LIMITS.maxDurationMs, Math.max(0, performance.now() - active.startedAt));
  }

  function draw(active, text) {
    const { context, canvas, frame, layout } = active;
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.shadowBlur = 0;
    context.fillStyle = BACKGROUND;
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.setTransform(layout.scale, 0, 0, layout.scale, layout.x, layout.y);
    context.fillStyle = frame.color;
    const lines = text.split('\n');
    if (lines.length !== frame.rows || lines.some((line) => line.length !== frame.columns)) {
      throw new Error('The live ASCII grid changed. Start a new recording after the preview settles.');
    }
    if (frame.glow) {
      // Canvas shadows use device pixels, so scale the bounded preview-like halo explicitly.
      context.shadowColor = frame.color;
      context.shadowBlur = 6 * layout.scale;
      for (let row = 0; row < lines.length; row++) context.fillText(lines[row], 0, layout.baseline + row * frame.lineHeight);
      context.shadowBlur = 0;
    }
    for (let row = 0; row < lines.length; row++) context.fillText(lines[row], 0, layout.baseline + row * frame.lineHeight);
    active.frames++;
    active.lastDraw = performance.now();
  }

  function flush(active) {
    active.drawTimer = null;
    if (session !== active || state !== 'recording' || active.pendingText === null) return;
    const remaining = FRAME_INTERVAL - (performance.now() - active.lastDraw);
    if (remaining > 0) {
      active.drawTimer = setTimeout(() => flush(active), remaining);
      return;
    }
    const text = active.pendingText;
    active.pendingText = null;
    try { draw(active, text); } catch (error) { fail(active, error.message); }
  }

  function finish(active) {
    if (session !== active) return;
    try {
      if (active.bytes === 0) throw new Error('No video was recorded. Keep the live preview running and try again.');
      const mimeType = active.chunkMime || active.recorder.mimeType;
      const extension = recordingExtension(mimeType);
      const blob = new Blob(active.chunks, { type: mimeType });
      const file = new File([blob], `clack-${new Date().toISOString().replace(/[:.]/g, '-')}.${extension}`, { type: mimeType });
      const url = URL.createObjectURL(blob);
      clip = {
        blob, url, file, mimeType,
        width: active.layout.width,
        height: active.layout.height,
        durationMs: active.durationMs ?? elapsed(active),
        frames: active.frames,
      };
      session = null;
      release(active);
      transition('ready');
    } catch (error) {
      fail(active, error.message || 'Unable to finish this recording. Try recording again.');
    }
  }

  function stop() {
    const active = session;
    if (!active || state !== 'recording') return false;
    active.durationMs = elapsed(active);
    clearTimers(active);
    active.finishTimer = setTimeout(() => fail(active, 'The video encoder did not finish. Try a shorter recording or another browser.'), 10_000);
    transition('stopping');
    if (session !== active) return false;
    try {
      active.recorder.stop();
      return true;
    } catch {
      fail(active, 'The video encoder stopped unexpectedly. Try recording again.');
      return false;
    }
  }
  function reset() {
    const active = session;
    session = null;
    release(active);
    revokeClip();
    transition('idle');
  }


  function start(frame) {
    if (disposed || state === 'recording' || state === 'stopping') return false;
    const active = { chunks: [], bytes: 0, chunkMime: '', frames: 0, pendingText: null };
    session = active;
    revokeClip();
    try {
      const support = getRecordingSupport();
      if (!support.supported) throw new Error(support.reason);
      validateFrame(frame);
      active.frame = Object.fromEntries(PRESENTATION_KEYS.map((key) => [key, frame[key]]));
      const font = `400 ${frame.fontSize}px ${frame.fontFamily}`;
      if (!document.fonts?.check(font, 'M@ ')) throw new Error('The ASCII font is still loading. Wait for the preview and try Record again.');
      active.canvas = document.createElement('canvas');
      active.context = active.canvas.getContext('2d', { alpha: false });
      if (!active.context) throw new Error('Unable to create the recording canvas. Try reloading the page.');
      active.context.font = font;
      active.layout = computeRecordingLayout(frame, measureFont(active.context));
      active.canvas.width = active.layout.width;
      active.canvas.height = active.layout.height;
      // Resizing resets context settings; match the bundled preview font without ligatures or kerning.
      active.context.font = font;
      active.context.fontKerning = 'none';
      active.context.textAlign = 'left';
      active.context.textBaseline = 'alphabetic';
      draw(active, frame.text);
      active.stream = active.canvas.captureStream(RECORDING_LIMITS.frameRate);
      if (!active.stream.getVideoTracks().length || active.stream.getVideoTracks().some((track) => track.readyState !== 'live')) {
        throw new Error('The browser could not capture the ASCII canvas. Try another browser.');
      }
      active.trackEnded = () => {
        if (session === active && state === 'recording') fail(active, 'The ASCII capture ended unexpectedly. Keep this tab open and try recording again.');
      };
      for (const track of active.stream.getTracks()) track.addEventListener('ended', active.trackEnded);
      active.recorder = new MediaRecorder(active.stream, {
        mimeType: support.mimeType,
        videoBitsPerSecond: RECORDING_LIMITS.videoBitsPerSecond,
      });
      active.recorder.ondataavailable = (event) => {
        if (session !== active || !event.data?.size) return;
        try {
          if (active.bytes + event.data.size > RECORDING_LIMITS.maxBytes) {
            throw new Error('This recording exceeded the 16 MB limit and was discarded. Try a shorter recording.');
          }
          if (event.data.type) {
            const extension = recordingExtension(event.data.type);
            if (active.chunkMime && recordingExtension(active.chunkMime) !== extension) {
              throw new Error('The encoder changed video formats. Try another browser.');
            }
            active.chunkMime ||= event.data.type;
          }
          active.bytes += event.data.size;
          active.chunks.push(event.data);
        } catch (error) { fail(active, error.message); }
      };
      active.recorder.onerror = () => fail(active, 'The browser could not encode this recording. Try a shorter recording or another browser.');
      active.recorder.onstop = () => {
        if (session !== active) return;
        if (state !== 'stopping') {
          fail(active, 'The video encoder ended unexpectedly. Try recording again.');
          return;
        }
        finish(active);
      };
      active.startedAt = performance.now();
      active.recorder.start(1000);
      active.clock = setInterval(() => {
        if (session === active && state === 'recording') notify(onTime, elapsed(active));
      }, 250);
      active.deadline = setTimeout(() => { if (session === active) stop(); }, RECORDING_LIMITS.maxDurationMs);
      transition('recording');
      if (session === active) notify(onTime, 0);
      return session === active;
    } catch (error) {
      fail(active, error.message || 'Unable to start recording. Try another browser.');
      return false;
    }
  }

  return {
    start,
    pushFrame(frame) {
      const active = session;
      if (!active || state !== 'recording') return false;
      try {
        if (!frame || typeof frame.text !== 'string') {
          throw new Error('The live ASCII frame is unavailable. Try recording again.');
        }
        if (PRESENTATION_KEYS.some((key) => key in frame && frame[key] !== active.frame[key])) {
          throw new Error('Recording settings changed. Keep the same ASCII grid and look while recording.');
        }
        active.pendingText = frame.text;
        if (!active.drawTimer) flush(active);
        return session === active;
      } catch (error) {
        fail(active, error.message);
        return false;
      }
    },
    stop,
    reset,
    dispose() {
      disposed = true;
      reset();
    },
    get state() { return state; },
    get clip() { return clip; },
  };
}
