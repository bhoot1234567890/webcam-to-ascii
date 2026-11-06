import './styles.css';
import { computeGrid, normalizeSettings } from './renderer.js';
import { createCameraController } from './camera.js';
import { configureAsciiText } from './presentation.js';
import { createLanczosResampler } from './resample.js';
import { createOutputView } from './output-view.js';
import { createCameraShell } from './camera-shell.js';
import { computeCropRect, FRAMINGS } from './framing.js';
import { normalizeLook } from './looks.js';
import { createLookControls } from './look-controls.js';
import { createAsciiRecorder, getRecordingSupport } from './ascii-recorder.js';
import { createRecordingView } from './recording-view.js';
import { findDensity, findDensityBySettings } from './density.js';
import { copyAscii, asciiToPng } from './export.js';
import { readAsciiArtwork } from './artwork.js';
import { loadSaved, save, loadPresets, savePresets } from './persist.js';
import { normalizePreset, normalizePresetList, upsertPreset, removePreset } from './presets.js';

const byId = (id) => document.getElementById(id);
const output = byId('ascii-output');
const video = byId('camera-preview');
const imagePreview = byId('image-preview');
const status = byId('status');
const errorLine = byId('error');
const errorText = byId('error-text');
const deviceSelect = byId('camera-select');
const facingSelect = byId('camera-facing');
const imageInput = byId('image-input');
const inputs = [...byId('settings-form').querySelectorAll('[data-field]')];
const canvas = document.createElement('canvas');
const context = canvas.getContext('2d', { willReadFrequently: true });
const initialDensity = findDensity('balanced');
let settings = normalizeSettings({
  maxWidthChars: initialDensity.maxWidthChars,
  maxHeightLines: initialDensity.maxHeightLines,
});
let settingsKey = JSON.stringify(settings);
let look = normalizeLook();
let lookKey = JSON.stringify(look);
let framing = 'full';
let tint = 'default';
let glow = false;
let lookControls = null;
let presets = [];
let source = null;
let sourceGeneration = 0;
let decodeToken = 0;
let startToken = 0;
let renderEpoch = 0;
let requestId = 0;
let cameraState = 'idle';
let worker = null;
let workerFailed = false;
let inFlight = null;
let pendingStill = false;
let disposed = false;
let sampler = null;
let samplerKey = '';
let outputView = null;
let cameraShell = null;
let captureBusy = false;
let captureMode = 'photo';
let recording = null;
let recordingView = null;
let recordingState = 'idle';
let pendingRecording = false;
let reviewPending = false;
let exportBusy = false;
const recordingSupport = getRecordingSupport();

function showError(message, photoAction = false) {
  errorText.textContent = message;
  errorLine.hidden = !message;
  byId('error-action').hidden = !photoAction;
}

let statusNoticeUntil = 0;

function setStatus(text, notice = false) {
  if (notice) statusNoticeUntil = performance.now() + 2500;
  if (status.textContent !== text) status.textContent = text;
}

function hasCurrentFrame() {
  return source && output.dataset.sourceGeneration === String(source.generation) &&
    output.dataset.settingsKey === settingsKey && output.dataset.lookKey === lookKey &&
    output.dataset.framing === framing && output.dataset.sourceWidth === String(source.width) &&
    output.dataset.sourceHeight === String(source.height);
}

function syncSettingsInputs() {
  for (const input of inputs) {
    if (Object.hasOwn(settings, input.dataset.field)) {
      input.value = settings[input.dataset.field];
    }
  }
}

function syncLookControls() {
  const currentDensity = findDensityBySettings(settings);
  lookControls?.sync({
    look,
    framing,
    tint,
    glow,
    settings,
    density: currentDensity ? currentDensity.id : null,
  });
}

function sourceGrid(width, height, proposedSettings = settings, proposedFraming = framing) {
  const crop = computeCropRect(width, height, proposedFraming);
  return { crop, ...computeGrid(crop.width, crop.height, proposedSettings) };
}

function videoBusy() {
  return pendingRecording || recordingState === 'recording' || recordingState === 'stopping';
}

function canShareClip() {
  try {
    return Boolean(recording?.clip && typeof navigator.share === 'function' &&
      typeof navigator.canShare === 'function' && navigator.canShare({ files: [recording.clip.file] }));
  } catch {
    return false;
  }
}


function startRecording() {
  if (disposed || captureMode !== 'video' || cameraState !== 'live' || !hasCurrentFrame()) return;
  stopPeek();
  lookControls.close();
  byId('camera-shutter').focus({ preventScroll: true });
  pendingRecording = false;
  showError('');
  recording.start(readAsciiArtwork(output, { glow }));
  updateControls();
}

function selectCaptureMode(mode) {
  if (videoBusy() || captureBusy || disposed || exportBusy) return;
  captureMode = mode;
  swallowNextActivation = false;
  showError('');
  if (mode === 'video' && !recordingSupport.supported) setStatus(recordingSupport.reason);
  else if (mode === 'video' && recording?.clip) setStatus('Video ready — review, save or share');
  else setStatus(mode === 'video' ? 'Video mode — press RECORD to begin.' : 'Photo mode — use the shutter or choose PHOTO.');
  updateControls();
}

function recordingChanged({ state, clip }) {
  if (disposed) return;
  const previous = recordingState;
  recordingState = state;
  if (state === 'stopping' || (state === 'idle' && previous === 'recording')) {
    pendingRecording = false;
    releaseCamera();
  }
  recordingView?.update({ state, clip, shareSupported: canShareClip() });
  if (state === 'recording') status.textContent = 'Recording silent ASCII video';
  if (state === 'stopping') status.textContent = 'Finishing video…';
  if (state === 'ready') {
    status.textContent = 'Video ready — review, save or share';
    reviewPending = document.visibilityState !== 'visible';
    if (!reviewPending) openVideoReview(byId('camera-shutter'));
  }
  updateControls();
}

function openVideoReview(opener) {
  if (disposed || videoBusy() || captureBusy || !recording?.clip) return;
  // Reviewing a saved clip must not leave a later Photo preview capturing behind the dialog.
  if (cameraState !== 'idle') releaseCamera();
  status.textContent = 'Video ready — review, save or share';
  recordingView.open(opener);
}

function retakeVideo() {
  if (exportBusy || videoBusy() || disposed) return;
  recordingView.close();
  reviewPending = false;
  recording.reset();
  captureMode = 'video';
  showError('');
  updateControls();
  // Retake returns to the live viewfinder; recording still requires a new tap.
  void startCamera();
}

function saveVideo() {
  const clip = recording?.clip;
  if (!clip || exportBusy || disposed) return;
  const link = document.createElement('a');
  link.href = clip.url;
  link.download = clip.file.name;
  document.body.append(link);
  link.click();
  link.remove();
  recordingView.setExportMessage('Download requested. Your browser may ask where to save the video.');
}

async function shareVideo() {
  if (exportBusy || disposed || !recording?.clip) return;
  if (!canShareClip()) {
    recordingView.setExportMessage('File sharing is unavailable here. Use Save video instead.');
    return;
  }
  const file = recording.clip.file;
  exportBusy = true;
  byId('video-share').disabled = true;
  byId('video-retake').disabled = true;
  try {
    // Invoke synchronously from the click's activation, before any asynchronous work.
    await navigator.share({ files: [file], title: 'clack — ASCII video' });
    if (!disposed) recordingView.setExportMessage('Video shared.');
  } catch (error) {
    if (!disposed) recordingView.setExportMessage(error.name === 'AbortError' ?
      'Sharing canceled. Your clip is still here.' : 'Unable to share this video. Use Save video instead.');
  } finally {
    exportBusy = false;
    if (!disposed) recordingView.update({ state: recordingState, clip: recording.clip, shareSupported: canShareClip() });
  }
}

function updateControls() {
  const locked = videoBusy() || captureBusy || disposed;
  lookControls?.setDisabled(locked);
  deviceSelect.disabled = cameraState === 'requesting' || locked;
  facingSelect.disabled = cameraState === 'requesting' || locked;
  imageInput.disabled = locked;
  for (const id of ['open-framing', 'open-looks', 'open-grid', 'open-ink', 'toggle-guides', 'fit-output', 'detail-output']) {
    byId(id).disabled = locked;
  }
  const captureReady = cameraState === 'live' && source?.kind === 'camera' && source.active && hasCurrentFrame();
  cameraShell?.update({
    cameraState, kind: source?.kind, captureReady, busy: captureBusy, disposed, mode: captureMode,
    recordingState, pendingRecording, recordingSupported: recordingSupport.supported,
    hasClip: Boolean(recording?.clip),
  });
  const exportable = hasCurrentFrame();
  byId('copy-output').disabled = !exportable || locked;
  byId('save-output').disabled = !exportable || locked;
  byId('compare-output').disabled = locked || !source || (source.kind === 'camera' && !source.active);
  const waiting = cameraState === 'requesting';
  byId('empty-start').hidden = waiting;
  byId('empty-photo').hidden = waiting;
  byId('empty-start').disabled = locked;
  byId('empty-photo').disabled = locked;
  byId('error-action').disabled = locked;
  syncEmptyState();
}

function syncEmptyState() {
  const empty = output.textContent === '';
  byId('empty-state').hidden = !empty;
  byId('empty-hint').hidden = !(empty && cameraState === 'requesting');
}

function cameraMessage(error) {
  if (error.message === 'Camera requires HTTPS or localhost') return error.message;
  return ({
    NotAllowedError: 'Camera access was denied. Allow camera in your browser site settings, then press CAMERA.',
    NotFoundError: 'No camera found.',
    NotReadableError: 'Camera is unavailable. Another app may be using it.',
  })[error.name] || 'Unable to start camera';
}

function releaseImage(previous) {
  if (previous?.kind === 'image') {
    previous.bitmap.close();
    URL.revokeObjectURL(previous.url);
  }
}

function workerFailure() {
  if (disposed) return;
  workerFailed = true;
  worker?.terminate();
  worker = null;
  inFlight = null;
  pendingStill = false;
  renderEpoch++;
  startToken++;
  if (source?.kind === 'camera') source.active = false;
  stopCamera();
  showError('Rendering failed');
  setStatus('Camera stopped');
}

function ensureWorker(explicitAction = false) {
  if (workerFailed && !explicitAction) return false;
  if (worker) return true;
  workerFailed = false;
  try {
    const instance = new Worker(new URL('./render-worker.js', import.meta.url), { type: 'module' });
    worker = instance;
    instance.onerror = (event) => {
      event.preventDefault();
      if (worker === instance) workerFailure();
    };
    instance.onmessageerror = () => { if (worker === instance) workerFailure(); };
    instance.onmessage = ({ data }) => {
      if (disposed || worker !== instance || !inFlight || data.id !== inFlight.id) return;
      const request = inFlight;
      inFlight = null;
      if (data.type !== 'result') {
        workerFailure();
        return;
      }
      const current = source && request.epoch === renderEpoch &&
        request.generation === source.generation && request.settingsKey === settingsKey &&
        request.lookKey === lookKey && request.framing === framing &&
        request.source === source && (source.kind === 'image' || source.active);
      if (current) {
        if (data.width !== request.width || data.height !== request.height || typeof data.text !== 'string') {
          workerFailure();
          return;
        }
        // One synchronous commit: identity describes precisely this text, not UI intent.
        const gridChanged = output.dataset.columns !== String(data.width) || output.dataset.rows !== String(data.height);
        const sourceChanged = output.dataset.sourceGeneration !== String(request.generation);
        const configurationChanged = output.dataset.settingsKey !== request.settingsKey ||
          output.dataset.lookKey !== request.lookKey || output.dataset.framing !== request.framing ||
          output.dataset.sourceWidth !== String(request.sourceWidth) ||
          output.dataset.sourceHeight !== String(request.sourceHeight);
        const wasEmpty = output.textContent === '';
        output.textContent = data.text;
        Object.assign(output.dataset, {
          renderId: String(request.id), sourceGeneration: String(request.generation),
          sourceKind: request.source.kind, sourceName: request.source.name,
          settingsKey: request.settingsKey, columns: String(data.width), rows: String(data.height),
          lookKey: request.lookKey, framing: request.framing,
          sourceWidth: String(request.sourceWidth), sourceHeight: String(request.sourceHeight),
          processingMs: String(data.processingMs),
        });
        if (wasEmpty && data.text) {
          output.classList.remove('just-revealed');
          void output.offsetWidth; // restart the one-shot if a prior run left the class
          output.classList.add('just-revealed');
        }
        output.setAttribute('aria-label', `ASCII art, ${data.width} by ${data.height} characters`);
        if (gridChanged) outputView.update();
        byId('grid-label').textContent = `${data.width} × ${data.height}`;
        if (recordingState !== 'recording' && recordingState !== 'stopping' &&
            (source.kind !== 'camera' || performance.now() > statusNoticeUntil)) {
          setStatus(source.kind === 'image' ? (source.captured ? 'Frame captured' : 'Image ready') : 'Camera live');
        }
        if (recordingState === 'recording' && source.kind === 'camera') {
          recording.pushFrame({ text: data.text, columns: data.width, rows: data.height });
        } else if (pendingRecording && source.kind === 'camera') startRecording();
        if (sourceChanged || configurationChanged) updateControls();
      }
      if (pendingStill) renderStill();
    };
    return true;
  } catch {
    workerFailure();
    return false;
  }
}

function submitRender() {
  if (disposed || inFlight || !worker || workerFailed || !source) return;
  if (source.kind === 'camera' && !source.active) return;
  try {
    const { crop, width, height } = sourceGrid(source.width, source.height);
    const key = `${crop.width}:${crop.height}:${width}:${height}`;
    if (key !== samplerKey) {
      sampler = createLanczosResampler(crop.width, crop.height, width, height);
      samplerKey = key;
    }
    if (canvas.width !== crop.width || canvas.height !== crop.height) {
      canvas.width = crop.width;
      canvas.height = crop.height;
    }
    // Acquire the selected native-pixel rectangle; Lanczos alone owns resizing.
    context.fillStyle = '#000';
    context.fillRect(0, 0, crop.width, crop.height);
    context.drawImage(source.kind === 'image' ? source.bitmap : video,
      crop.x, crop.y, crop.width, crop.height, 0, 0, crop.width, crop.height);
    const original = context.getImageData(0, 0, crop.width, crop.height).data;
    const rgba = sampler.resize(original);
    const id = ++requestId;
    inFlight = { id, width, height, source, generation: source.generation, settingsKey,
      lookKey, framing, sourceWidth: source.width, sourceHeight: source.height, epoch: renderEpoch };
    worker.postMessage({ type: 'render', id, width, height, settings, look, rgba: rgba.buffer }, [rgba.buffer]);
  } catch (error) {
    if (error instanceof RangeError) showError(error.message);
    else workerFailure();
  }
}

function renderStill() {
  pendingStill = false;
  if (source?.kind !== 'image' || workerFailed || disposed) return;
  if (inFlight) pendingStill = true;
  else submitRender();
}

const camera = createCameraController(video, {
  onState(state) {
    cameraState = state;
    updateControls();
    if (state === 'requesting') setStatus('Requesting camera…');
  },
  onFrame() {
    // Camera dimensions can change when a phone rotates.
    if (!inFlight && source?.kind === 'camera' && source.active && video.videoWidth && video.videoHeight) {
      if (source.width !== video.videoWidth || source.height !== video.videoHeight) {
        if (recordingState === 'recording') {
          stopCamera();
          return; // Finalize the existing fixed-size clip instead of stretching a rotated source.
        }
        source.width = video.videoWidth;
        source.height = video.videoHeight;
        renderEpoch++;
        updateControls();
      }
      submitRender();
    }
  },
  onError(error) {
    pendingRecording = false;
    if (recordingState === 'recording') recording.stop();
    if (source?.kind === 'camera') source.active = false;
    renderEpoch++;
    showError(cameraMessage(error), ['NotAllowedError', 'NotFoundError', 'NotReadableError'].includes(error.name));
    setStatus('Camera stopped');
    updateControls();
  },
});

async function refreshDevices(preferred, generation) {
  try {
    const devices = await camera.listDevices();
    if (disposed || source?.generation !== generation) return;
    deviceSelect.replaceChildren(new Option('Automatic camera', ''));
    for (const [index, device] of devices.entries()) {
      const option = document.createElement('option');
      option.value = device.deviceId;
      option.textContent = device.label || `Camera ${index + 1}`;
      deviceSelect.append(option);
    }
    if (devices.some((device) => device.deviceId === preferred)) deviceSelect.value = preferred;
  } catch {
    // Enumeration is supplementary; acquisition errors remain explicit and recoverable.
  }
}

async function startCamera(deviceId = deviceSelect.value) {
  if (disposed || captureBusy || recordingState === 'recording' || recordingState === 'stopping' ||
      cameraState === 'requesting') return;
  const token = ++startToken;
  decodeToken++;
  renderEpoch++;
  pendingStill = false;
  if (source?.kind === 'camera') source.active = false;
  showError('');
  if (!ensureWorker(true)) return;
  const started = await camera.start(deviceId || undefined, facingSelect.value);
  if (!started || token !== startToken || disposed) return;
  const actualDeviceId = video.srcObject?.getVideoTracks()[0]?.getSettings().deviceId || deviceId || '';
  releaseImage(source);
  source = { kind: 'camera', name: actualDeviceId, width: video.videoWidth,
    height: video.videoHeight, active: true, generation: ++sourceGeneration };
  renderEpoch++;
  imagePreview.hidden = true;
  imagePreview.removeAttribute('src');
  video.hidden = false;
  byId('source-label').textContent = 'Live camera';
  setStatus('Rendering camera…');
  void refreshDevices(actualDeviceId, source.generation);
  submitRender();
}

function releaseCamera() {
  startToken++;
  renderEpoch++;
  pendingStill = false;
  if (source?.kind === 'camera') source.active = false;
  camera.stop();
  if (recordingState !== 'stopping') setStatus('Camera stopped');
}

function stopCamera() {
  pendingRecording = false;
  if (recordingState === 'recording') recording.stop();
  else releaseCamera();
  updateControls();
}

async function freezeFrame() {
  if (captureBusy || videoBusy() || disposed || cameraState !== 'live' || source?.kind !== 'camera' || !hasCurrentFrame()) return;
  const token = ++decodeToken;
  captureBusy = true;
  updateControls();
  try {
    // A private snapshot canvas cannot be overwritten by live acquisition during PNG encoding.
    const snapshot = document.createElement('canvas');
    snapshot.width = video.videoWidth;
    snapshot.height = video.videoHeight;
    const snapshotContext = snapshot.getContext('2d', { alpha: false });
    if (!snapshotContext) throw new Error('Unable to capture frame');
    snapshotContext.drawImage(video, 0, 0);
    const encoded = new Promise((resolve) => snapshot.toBlob(resolve, 'image/png'));
    stopCamera();
    setStatus('Developing frame…');
    const blob = await encoded;
    if (disposed || token !== decodeToken) return;
    if (!blob) throw new Error('Unable to capture frame');
    await acceptImage(new File([blob], 'Captured frame.png', { type: 'image/png' }), true);
  } catch {
    if (!disposed && token === decodeToken) showError('Unable to capture frame');
  } finally {
    captureBusy = false;
    if (!disposed) updateControls();
  }
}

let swallowNextActivation = false;

function shutterAction() {
  if (recordingState === 'recording') {
    stopCamera();
  } else if (recordingState === 'stopping') {
    return;
  } else if (cameraState === 'requesting' || pendingRecording) {
    stopCamera();
    // Preserve Photo's protection against a double-tap replay after cancellation.
    swallowNextActivation = captureMode === 'photo';
  } else if (captureMode === 'video') {
    if (recording.clip) {
      openVideoReview(byId('camera-shutter'));
      return;
    }
    if (!recordingSupport.supported) {
      showError(recordingSupport.reason);
      return;
    }
    if (cameraState === 'live' && hasCurrentFrame()) startRecording();
    else {
      pendingRecording = true;
      updateControls();
      // A live source only needs its next matching commit, not another camera acquisition.
      if (cameraState !== 'live') void startCamera();
    }
  } else if (cameraState === 'live') {
    void freezeFrame();
  } else if (swallowNextActivation) {
    swallowNextActivation = false;
  } else {
    void startCamera();
  }
}

function flipCamera() {
  if (disposed || captureBusy || videoBusy() || cameraState === 'requesting') return;
  facingSelect.value = facingSelect.value === 'environment' ? 'user' : 'environment';
  deviceSelect.value = '';
  if (cameraState === 'live') void startCamera('');
}

let peeking = false;
let peekRaf = 0;

function peekDraw() {
  if (!peeking || !source) return;
  try {
    const media = source.kind === 'image' ? source.bitmap : video;
    const crop = computeCropRect(source.width, source.height, framing);
    const peek = byId('source-peek');
    if (peek.width !== crop.width || peek.height !== crop.height) {
      peek.width = crop.width;
      peek.height = crop.height;
    }
    peek.getContext('2d').drawImage(media, crop.x, crop.y, crop.width, crop.height, 0, 0, crop.width, crop.height);
  } catch {
    stopPeek();
    return;
  }
  peekRaf = requestAnimationFrame(peekDraw);
}

function startPeek() {
  if (!source || peeking) return;
  peeking = true;
  byId('source-peek').hidden = false;
  byId('compare-output').setAttribute('aria-pressed', 'true');
  peekDraw();
}

function stopPeek() {
  if (!peeking) return;
  peeking = false;
  cancelAnimationFrame(peekRaf);
  byId('source-peek').hidden = true;
  byId('compare-output').setAttribute('aria-pressed', 'false');
}

async function acceptImage(file, captured = false) {
  if (!file || disposed || videoBusy()) return;
  const token = ++decodeToken;
  let bitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    if (token === decodeToken && !disposed) showError('Unable to read this image');
    return;
  }
  if (token !== decodeToken || disposed) {
    bitmap.close();
    return;
  }
  try {
    sourceGrid(bitmap.width, bitmap.height);
  } catch (error) {
    bitmap.close();
    showError(error.message);
    return;
  }
  startToken++;
  camera.stop();
  renderEpoch++;
  releaseImage(source);
  source = { kind: 'image', name: file.name, width: bitmap.width, height: bitmap.height,
    bitmap, url: URL.createObjectURL(file), captured, generation: ++sourceGeneration };
  imagePreview.src = source.url;
  captureMode = 'photo';
  imagePreview.alt = `Original image: ${file.name}`;
  imagePreview.hidden = false;
  video.hidden = true;
  byId('source-label').textContent = file.name;
  showError('');
  setStatus('Rendering image…');
  if (ensureWorker(true)) renderStill();
  updateControls();
}

function changeSettings() {
  if (videoBusy()) {
    syncSettingsInputs();
    showError('Stop recording before changing the detail');
    return;
  }
  const proposed = { ...settings };
  for (const input of inputs) {
    const value = input.valueAsNumber;
    const valid = Number.isFinite(value) && value >= Number(input.min) && value <= Number(input.max) &&
      Number.isInteger((value - Number(input.min)) / Number(input.step));
    const fieldError = byId(`${input.id}-error`);
    input.setAttribute('aria-invalid', String(!valid));
    fieldError.textContent = valid ? '' : "Choose a value between the indicated min and max, using the field's step.";
    fieldError.hidden = valid;
    if (!valid) return;
    proposed[input.dataset.field] = value;
  }
  try {
    const next = normalizeSettings(proposed);
    if (source) sourceGrid(source.width, source.height, next);
    const nextKey = JSON.stringify(next);
    showError(workerFailed ? 'Rendering failed' : '');
    if (nextKey === settingsKey) return;
    settings = next;
    settingsKey = nextKey;
    renderEpoch++;
    updateControls();
    if (source?.kind === 'image') renderStill();
    else if (source?.kind === 'camera' && source.active) submitRender();
    syncLookControls();
    persist();
  } catch (error) {
    showError(error.message);
  }
}

function persist() {
  if (disposed) return;
  save({
    look,
    framing,
    tint,
    glow,
    density: { maxWidthChars: settings.maxWidthChars, maxHeightLines: settings.maxHeightLines },
    tuning: {
      sigma1: settings.sigma1,
      sigma2: settings.sigma2,
      edgeIndexThreshold: settings.edgeIndexThreshold,
    },
    facing: facingSelect.value,
    view: document.querySelector('.ascii-viewport').dataset.view,
    guides: byId('toggle-guides').getAttribute('aria-pressed') === 'true',
  });
}

function changeDensity(proposed) {
  if (videoBusy()) { syncLookControls(); return; }
  let nextWidth, nextHeight;
  if (typeof proposed === 'string') {
    const preset = findDensity(proposed);
    if (!preset) return;
    nextWidth = preset.maxWidthChars;
    nextHeight = preset.maxHeightLines;
  } else if (proposed && typeof proposed === 'object') {
    nextWidth = proposed.maxWidthChars;
    nextHeight = proposed.maxHeightLines;
  } else {
    return;
  }
  const proposedSettings = { ...settings, maxWidthChars: nextWidth, maxHeightLines: nextHeight };
  try {
    const next = normalizeSettings(proposedSettings);
    if (source) sourceGrid(source.width, source.height, next);
    const nextKey = JSON.stringify(next);
    showError(workerFailed ? 'Rendering failed' : '');
    if (nextKey !== settingsKey) {
      settings = next;
      settingsKey = nextKey;
      syncSettingsInputs();
      renderEpoch++;
      updateControls();
      if (source?.kind === 'image') renderStill();
      else if (source?.kind === 'camera' && source.active) submitRender();
    }
    persist();
  } catch (error) {
    showError(error.message);
  }
  syncLookControls();
}

function changeLook(proposed) {
  if (videoBusy()) { syncLookControls(); return; }
  try {
    const next = normalizeLook(proposed);
    const key = JSON.stringify(next);
    showError(workerFailed ? 'Rendering failed' : '');
    if (key !== lookKey) {
      look = next;
      lookKey = key;
      renderEpoch++;
      updateControls();
      if (source?.kind === 'image') renderStill();
      else if (source?.kind === 'camera' && source.active) submitRender();
    }
    persist();
  } catch (error) {
    showError(error.message);
  }
  syncLookControls();
}

function changeFraming(proposed) {
  if (videoBusy()) { syncLookControls(); return; }
  try {
    if (!FRAMINGS.some(({ id }) => id === proposed)) throw new RangeError('Choose a supported frame');
    if (source) sourceGrid(source.width, source.height, settings, proposed);
    showError(workerFailed ? 'Rendering failed' : '');
    if (proposed !== framing) {
      framing = proposed;
      renderEpoch++;
      updateControls();
      if (source?.kind === 'image') renderStill();
      else if (source?.kind === 'camera' && source.active) submitRender();
    }
    persist();
  } catch (error) {
    showError(error.message);
  }
  syncLookControls();
}

function applyPresentation({ tint: nextTint, glow: nextGlow }) {
  tint = nextTint;
  glow = nextGlow;
  if (tint === 'default') document.body.style.removeProperty('--ascii-tint');
  else document.body.style.setProperty('--ascii-tint', tint);
  document.body.dataset.glow = glow ? 'on' : 'off';
  outputView.update();
}

function changePresentation(proposed) {
  if (videoBusy()) { syncLookControls(); return; }
  if ((proposed.tint !== 'default' && !/^#[0-9a-f]{6}$/i.test(proposed.tint)) ||
      typeof proposed.glow !== 'boolean') {
    showError('Choose a valid text color and glow setting');
    syncLookControls();
    return;
  }
  applyPresentation({ tint: proposed.tint, glow: proposed.glow });
  syncLookControls();
  persist();
}

function savePreset(name) {
  if (videoBusy() || captureBusy || disposed) {
    return { ok: false, message: 'Finish capture before changing presets' };
  }
  try {
    const preset = normalizePreset({ name, settings, look, framing, tint, glow });
    presets = upsertPreset(presets, preset);
    savePresets(presets);
    lookControls?.syncPresets(presets);
    setStatus(`Saved ${preset.name}.`, true);
    return { ok: true };
  } catch (error) {
    return { ok: false, message: error.message };
  }
}

function deletePreset(name) {
  if (videoBusy() || captureBusy || disposed) return;
  presets = removePreset(presets, name);
  savePresets(presets);
  lookControls?.syncPresets(presets);
  setStatus('Preset deleted.', true);
}

function applyPreset(name) {
  if (videoBusy() || captureBusy || disposed) return;
  const preset = presets.find((entry) => entry.name === name);
  if (!preset || disposed) return;
  settings = preset.settings;
  settingsKey = JSON.stringify(settings);
  look = preset.look;
  lookKey = JSON.stringify(look);
  framing = preset.framing;
  renderEpoch++;
  applyPresentation({ tint: preset.tint, glow: preset.glow });
  syncSettingsInputs();
  syncLookControls();
  updateControls();
  if (source?.kind === 'image') renderStill();
  else if (source?.kind === 'camera' && source.active) submitRender();
  persist();
  setStatus(`Applied ${preset.name}.`, true);
}

function dispose() {
  if (disposed) return;
  disposed = true;
  startToken++;
  decodeToken++;
  renderEpoch++;
  stopPeek();
  camera.dispose();
  recording?.dispose();
  recordingView?.dispose();
  outputView?.dispose();
  lookControls?.dispose();
  worker?.terminate();
  worker = null;
  inFlight = null;
  pendingStill = false;
  releaseImage(source);
  source = null;
}

try {
  const saved = loadSaved();
  if (saved) {
    try {
      look = normalizeLook(saved.look);
      lookKey = JSON.stringify(look);
    } catch { /* keep the default recipe */ }
    if (FRAMINGS.some((frame) => frame.id === saved.framing)) framing = saved.framing;
    if (saved.tint === 'default' || /^#[0-9a-f]{6}$/i.test(saved.tint)) tint = saved.tint;
    if (typeof saved.glow === 'boolean') glow = saved.glow;
    try {
      settings = normalizeSettings({ ...settings, ...saved.density, ...saved.tuning });
      settingsKey = JSON.stringify(settings);
    } catch { /* keep the default density */ }
    if (['', 'user', 'environment'].includes(saved.facing)) facingSelect.value = saved.facing;
  }
  presets = normalizePresetList(loadPresets());
  await configureAsciiText(output);
  outputView = createOutputView(document.querySelector('.ascii-viewport'), output, byId('fit-output'), byId('detail-output'));
  if (saved?.view === 'detail') byId('detail-output').click();
  else if (saved?.view === 'fit') byId('fit-output').click();
  cameraShell = createCameraShell();
  recordingView = createRecordingView({ onRetake: retakeVideo, onSave: saveVideo, onShare: shareVideo });
  recording = createAsciiRecorder({
    onState: recordingChanged,
    onTime: (elapsedMs) => recordingView.tick(elapsedMs),
    onError: (message) => {
      if (disposed) return;
      pendingRecording = false;
      releaseCamera();
      showError(message);
      status.textContent = 'Video recording failed';
      updateControls();
    },
  });
  if (saved?.guides === true) {
    byId('toggle-guides').setAttribute('aria-pressed', 'true');
    document.body.dataset.guides = 'on';
  }
  lookControls = createLookControls({
    onLookChange: changeLook,
    onFramingChange: changeFraming,
    onPresentationChange: changePresentation,
    onDensityChange: changeDensity,
    onSavePreset: savePreset,
    onApplyPreset: applyPreset,
    onDeletePreset: deletePreset,
  });
  changePresentation({ tint, glow });
  syncSettingsInputs();
  syncLookControls();
  lookControls.syncPresets(presets);
  byId('camera-shutter').addEventListener('click', shutterAction);
  byId('camera-stop').addEventListener('click', stopCamera);
  byId('camera-import').addEventListener('click', () => imageInput.click());
  byId('photo-mode').addEventListener('click', () => selectCaptureMode('photo'));
  byId('video-mode').addEventListener('click', () => selectCaptureMode('video'));
  byId('review-video').addEventListener('click', () => openVideoReview(byId('review-video')));
  byId('studio-video').addEventListener('click', () => {
    selectCaptureMode('video');
    lookControls.close();
    const shutter = byId('camera-shutter');
    (shutter.disabled ? byId('photo-mode') : shutter).focus({ preventScroll: true });
  });
  byId('error-action').addEventListener('click', () => imageInput.click());
  byId('empty-start').addEventListener('click', () => {
    if (!disposed && !captureBusy) void startCamera();
  });
  byId('empty-photo').addEventListener('click', () => imageInput.click());
  byId('copy-output').addEventListener('click', async () => {
    if (videoBusy() || captureBusy || disposed || !hasCurrentFrame()) return;
    try {
      await copyAscii(output.textContent);
      setStatus('Copied. Paste it anywhere.', true);
    } catch {
      showError('Copy failed. Your browser blocked clipboard access.');
    }
  });
  byId('save-output').addEventListener('click', () => {
    if (videoBusy() || captureBusy || disposed || !hasCurrentFrame()) return;
    let frame;
    try {
      frame = readAsciiArtwork(output, { glow });
    } catch {
      showError('Unable to save the frame');
      return;
    }
    asciiToPng(frame).then((blob) => {
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = `clack-${new Date().toISOString().replace(/[-:T]/g, '').slice(0, 14)}.png`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 5000);
      setStatus('Saved.', true);
    }).catch(() => showError('Unable to save the frame'));
  });
  const compare = byId('compare-output');
  compare.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    startPeek();
  });
  compare.addEventListener('keydown', (event) => {
    if (event.key !== ' ' && event.key !== 'Enter') return;
    event.preventDefault();
    startPeek();
  });
  compare.addEventListener('keyup', (event) => {
    if (event.key !== ' ' && event.key !== 'Enter') return;
    stopPeek();
  });
  window.addEventListener('pointerup', stopPeek);
  window.addEventListener('pointercancel', stopPeek);
  byId('flip-camera').addEventListener('click', flipCamera);
  deviceSelect.addEventListener('change', () => {
    if (cameraState === 'live') void startCamera(deviceSelect.value);
  });
  facingSelect.addEventListener('change', () => {
    deviceSelect.value = '';
    persist();
    if (cameraState === 'live') void startCamera('');
  });
  byId('fit-output').addEventListener('click', persist);
  byId('detail-output').addEventListener('click', persist);
  byId('toggle-guides').addEventListener('click', persist);
  imageInput.addEventListener('change', () => { void acceptImage(imageInput.files[0]); });
  byId('settings-form').addEventListener('submit', (event) => event.preventDefault());
  inputs.forEach((input) => {
    input.addEventListener('input', changeSettings);
    input.addEventListener('change', changeSettings);
  });
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden' && cameraState !== 'idle') {
      stopCamera();
      setStatus('Camera paused. Press CAMERA to resume.');
    } else if (document.visibilityState === 'visible' && reviewPending) {
      reviewPending = false;
      openVideoReview(byId('camera-shutter'));
    }
  });
  window.addEventListener('pagehide', (event) => {
    if (event.persisted) {
      if (cameraState !== 'idle') stopCamera();
    } else dispose();
  });
  setStatus('Ready. Press CAMERA or PHOTO.');
  updateControls();
} catch {
  showError('Unable to load the ASCII font');
  byId('camera-shutter').disabled = true;
  byId('empty-start').disabled = true;
  byId('empty-photo').disabled = true;
  imageInput.disabled = true;
  setStatus('Font unavailable');
}
