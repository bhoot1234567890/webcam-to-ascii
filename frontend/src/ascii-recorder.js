import { createAsciiArtwork } from './artwork.js';

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
const FRAME_INTERVAL = 1000 / RECORDING_LIMITS.frameRate;
const PAINT_FENCE_TIMEOUT = 250;
const PRESENTATION_KEYS = ['columns', 'rows', 'fontFamily', 'fontSize', 'lineHeight', 'color', 'background', 'glow'];

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


function validateFrame(frame) {
  if (!frame || typeof frame.text !== 'string' ||
      ![frame.columns, frame.rows].every((value) => Number.isSafeInteger(value) && value > 0) ||
      ![frame.fontSize, frame.lineHeight].every((value) => Number.isFinite(value) && value > 0) ||
      typeof frame.fontFamily !== 'string' || !frame.fontFamily.includes('Roboto Mono') ||
      typeof frame.color !== 'string' || !frame.color ||
      typeof frame.background !== 'string' || !frame.background || typeof frame.glow !== 'boolean') {
    throw new Error('The ASCII frame is not ready. Wait for the live preview and try Record again.');
  }
  validateGrid(frame.text, frame.columns, frame.rows);
}

function validateGrid(text, columns, rows) {
  const lines = text.split('\n');
  if (lines.length !== rows || lines.some((line) => line.length !== columns || /[^\x20-\x7e]/.test(line))) {
    throw new Error('The live ASCII grid changed. Start a new recording after the preview settles.');
  }
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
    clearTimeout(active.stopTimer);
    if (active.paintFrame != null) cancelAnimationFrame(active.paintFrame);
    active.paintFrame = null;
    active.drawTimer = null;
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
    active.artwork = null;
    active.canvas = null;
    active.stream = null;
    active.captureTrack = null;
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
    active.artwork.draw(text);
    // Request on the actual stream track, after painting and after the encoder has started.
    // A request is a capture opportunity, not a promise of one encoded frame per commit.
    if (active.captureTrack) active.captureTrack.requestFrame();
    active.frames++;
    active.lastText = text;
    active.lastDraw = performance.now();
  }

  function flush(active) {
    active.drawTimer = null;
    if (session !== active || state !== 'recording' || active.pendingText === null) return;
    const remaining = FRAME_INTERVAL - (performance.now() - active.lastDraw);
    if (remaining > 0) {
      const timer = setTimeout(() => {
        if (active.drawTimer === timer) flush(active);
      }, remaining);
      active.drawTimer = timer;
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

  function stopEncoder(active) {
    if (session !== active || state !== 'stopping' || active.encoderStopped) return;
    active.encoderStopped = true;
    clearTimeout(active.stopTimer);
    if (active.paintFrame != null) cancelAnimationFrame(active.paintFrame);
    active.paintFrame = null;
    try {
      active.recorder.stop();
    } catch {
      fail(active, 'The video encoder stopped unexpectedly. Try recording again.');
    }
  }

  function captureFinalBoundary(active, text) {
    active.drawTimer = null;
    if (session !== active || state !== 'stopping') return;
    try {
      // Sparse/static canvas streams need an endpoint to retain their elapsed hold. This one
      // explicit repeat is a Stop boundary, not periodic capture of unchanged preview commits.
      draw(active, text);
      // Native encoders can miss an immediately stopped canvas update. Give visible canvases
      // two paint opportunities, but never wait indefinitely for RAF in a backgrounded tab.
      active.stopTimer = setTimeout(() => stopEncoder(active), PAINT_FENCE_TIMEOUT);
      if (document.visibilityState !== 'hidden' && typeof requestAnimationFrame === 'function') {
        active.paintFrame = requestAnimationFrame(() => {
          active.paintFrame = null;
          if (session !== active || state !== 'stopping' || active.encoderStopped) return;
          active.paintFrame = requestAnimationFrame(() => {
            active.paintFrame = null;
            stopEncoder(active);
          });
        });
      }
    } catch (error) {
      fail(active, error.message || 'The final ASCII frame could not be captured. Try recording again.');
    }
  }

  function stop() {
    const active = session;
    if (!active || state !== 'recording') return false;
    active.durationMs = elapsed(active);
    const finalText = active.pendingText ?? active.lastText;
    clearTimers(active);
    active.finishTimer = setTimeout(() => fail(active, 'The video encoder did not finish. Try a shorter recording or another browser.'), 10_000);
    transition('stopping');
    if (session !== active) return false;
    const remaining = Math.max(0, FRAME_INTERVAL - (performance.now() - active.lastDraw));
    if (remaining > 0) {
      active.drawTimer = setTimeout(() => captureFinalBoundary(active, finalText), remaining);
    } else {
      captureFinalBoundary(active, finalText);
    }
    return session === active;
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
    const active = { chunks: [], bytes: 0, chunkMime: '', frames: 0, pendingText: null, drawTimer: null };
    session = active;
    revokeClip();
    try {
      const support = getRecordingSupport();
      if (!support.supported) throw new Error(support.reason);
      validateFrame(frame);
      active.frame = Object.fromEntries(PRESENTATION_KEYS.map((key) => [key, frame[key]]));
      active.artwork = createAsciiArtwork(frame, { maxEdge: RECORDING_LIMITS.maxEdge });
      active.canvas = active.artwork.canvas;
      active.layout = active.artwork.layout;
      active.stream = active.canvas.captureStream(0);
      if (!active.stream.getVideoTracks().length || active.stream.getVideoTracks().some((track) => track.readyState !== 'live')) {
        throw new Error('The browser could not capture the ASCII canvas. Try another browser.');
      }
      const captureTrack = active.stream.getVideoTracks()[0];
      if (typeof captureTrack.requestFrame === 'function') {
        active.captureTrack = captureTrack;
      } else {
        // Firefox lacks requestFrame: discard the zero-rate probe, then use a genuinely timed
        // stream. Drawing remains latest-wins and bounded; capture/encoding cadence is browser-owned.
        for (const track of active.stream.getTracks()) {
          try { track.stop(); } catch { /* Release every discarded probe track. */ }
        }
        active.stream = active.canvas.captureStream(RECORDING_LIMITS.frameRate);
        if (!active.stream.getVideoTracks().length || active.stream.getVideoTracks().some((track) => track.readyState !== 'live')) {
          throw new Error('The browser could not capture the ASCII canvas. Try another browser.');
        }
      }
      active.trackEnded = () => {
        if (session === active) fail(active, 'The ASCII capture ended unexpectedly. Keep this tab open and try recording again.');
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
        if (state !== 'stopping' || !active.encoderStopped) {
          fail(active, 'The video encoder ended unexpectedly. Try recording again.');
          return;
        }
        finish(active);
      };
      active.startedAt = performance.now();
      active.recorder.start(1000);
      draw(active, frame.text);
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
        if (frame.text === active.lastText) {
          // A queued B followed by the already drawn A must cancel B, not flash it later.
          active.pendingText = null;
          clearTimeout(active.drawTimer);
          active.drawTimer = null;
          return true;
        }
        validateGrid(frame.text, active.frame.columns, active.frame.rows);
        active.pendingText = frame.text;
        if (active.drawTimer === null) flush(active);
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
