import './styles.css';
import './camera-shell.css';
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

const byId = (id) => document.getElementById(id);
const output = byId('ascii-output');
const video = byId('camera-preview');
const imagePreview = byId('image-preview');
const status = byId('status');
const errorText = byId('error');
const startButton = byId('start-camera');
const stopButton = byId('stop-camera');
const deviceSelect = byId('camera-select');
const facingSelect = byId('camera-facing');
const imageInput = byId('image-input');
const inputs = [...byId('settings-form').querySelectorAll('[data-field]')];
const canvas = document.createElement('canvas');
const context = canvas.getContext('2d', { willReadFrequently: true });
let settings = normalizeSettings();
let settingsKey = JSON.stringify(settings);
let look = normalizeLook();
let lookKey = JSON.stringify(look);
let framing = 'full';
let tint = 'default';
let glow = false;
let lookControls = null;
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
let deliveredFrames = 0;
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

function showError(message) {
  errorText.textContent = message;
  errorText.hidden = !message;
}

function hasCurrentFrame() {
  return source && output.dataset.sourceGeneration === String(source.generation) &&
    output.dataset.settingsKey === settingsKey && output.dataset.lookKey === lookKey &&
    output.dataset.framing === framing && output.dataset.sourceWidth === String(source.width) &&
    output.dataset.sourceHeight === String(source.height);
}

function syncLookControls() {
  lookControls?.sync({ look, framing, tint, glow });
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

function recordingFrame() {
  const style = getComputedStyle(output);
  return {
    text: output.textContent, columns: Number(output.dataset.columns), rows: Number(output.dataset.rows),
    fontFamily: style.fontFamily, fontSize: parseFloat(style.fontSize), lineHeight: parseFloat(style.lineHeight),
    color: style.color, glow,
  };
}

function startRecording() {
  if (disposed || captureMode !== 'video' || cameraState !== 'live' || !hasCurrentFrame()) return;
  pendingRecording = false;
  showError('');
  recording.start(recordingFrame());
  updateControls();
}

function selectCaptureMode(mode) {
  if (videoBusy() || captureBusy || disposed || exportBusy) return;
  captureMode = mode;
  if (mode === 'video' && !recordingSupport.supported) showError(recordingSupport.reason);
  else showError('');
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
    if (!reviewPending) openVideoReview();
  }
  updateControls();
}

function openVideoReview() {
  if (disposed || videoBusy() || captureBusy || !recording?.clip) return;
  // Reviewing a saved clip must not leave a later Photo preview capturing behind the dialog.
  if (cameraState !== 'idle') releaseCamera();
  status.textContent = 'Video ready — review, save or share';
  recordingView.open();
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
  const locked = videoBusy();
  startButton.disabled = cameraState !== 'idle' || disposed || captureBusy || locked;
  stopButton.disabled = cameraState === 'idle';
  deviceSelect.disabled = cameraState === 'requesting' || captureBusy || locked;
  facingSelect.disabled = cameraState === 'requesting' || captureBusy || locked;
  imageInput.disabled = disposed || captureBusy || locked;
  const shadingOnly = look.id === 'photo' || look.id === 'print';
  for (const input of inputs) {
    input.disabled = locked || (shadingOnly && ['sigma1', 'sigma2', 'edgeIndexThreshold'].includes(input.dataset.field));
  }
  byId('open-framing').disabled = locked;
  byId('open-looks').disabled = locked;
  const captureReady = cameraState === 'live' && source?.kind === 'camera' && source.active && hasCurrentFrame();
  cameraShell?.update({
    cameraState, kind: source?.kind, captureReady, busy: captureBusy, disposed, mode: captureMode,
    recordingState, pendingRecording, recordingSupported: recordingSupport.supported,
    hasClip: Boolean(recording?.clip),
  });
}

function cameraMessage(error) {
  if (error.message === 'Camera requires HTTPS or localhost') return error.message;
  return ({ NotAllowedError: 'Camera access was denied', NotFoundError: 'No camera found',
    NotReadableError: 'Camera is unavailable' })[error.name] || 'Unable to start camera';
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
  status.textContent = 'Rendering failed';
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
        output.textContent = data.text;
        Object.assign(output.dataset, {
          renderId: String(request.id), sourceGeneration: String(request.generation),
          sourceKind: request.source.kind, sourceName: request.source.name,
          settingsKey: request.settingsKey, columns: String(data.width), rows: String(data.height),
          lookKey: request.lookKey, framing: request.framing,
          sourceWidth: String(request.sourceWidth), sourceHeight: String(request.sourceHeight),
        });
        if (gridChanged) outputView.update();
        byId('grid-label').textContent = `${data.width} columns × ${data.height} rows`;
        const performanceLabel = byId('performance-label');
        if (source.kind === 'camera') deliveredFrames++;
        performanceLabel.dataset.processingMs = String(data.processingMs);
        performanceLabel.dataset.deliveredFrames = String(deliveredFrames);
        performanceLabel.textContent = `${data.processingMs.toFixed(1)} ms / frame`;
        if (recordingState !== 'recording' && recordingState !== 'stopping') {
          status.textContent = source.kind === 'image' ? (source.captured ? 'Frame captured' : 'Image ready') : 'Camera live';
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
    if (state === 'requesting') status.textContent = 'Requesting camera…';
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
    showError(cameraMessage(error));
    status.textContent = 'Camera stopped';
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
  status.textContent = 'Rendering camera…';
  void refreshDevices(actualDeviceId, source.generation);
  submitRender();
}

function releaseCamera() {
  startToken++;
  renderEpoch++;
  pendingStill = false;
  if (source?.kind === 'camera') source.active = false;
  camera.stop();
  if (recordingState !== 'stopping') status.textContent = 'Camera stopped';
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
    status.textContent = 'Developing frame…';
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

function shutterAction() {
  if (recordingState === 'recording') {
    stopCamera();
  } else if (recordingState === 'stopping') {
    return;
  } else if (cameraState === 'requesting' || pendingRecording) {
    stopCamera();
  } else if (captureMode === 'video') {
    if (recording.clip) {
      openVideoReview();
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
  } else if (cameraState === 'live') void freezeFrame();
  else void startCamera();
}

function flipCamera() {
  if (disposed || captureBusy || videoBusy() || cameraState === 'requesting') return;
  facingSelect.value = facingSelect.value === 'environment' ? 'user' : 'environment';
  deviceSelect.value = '';
  if (cameraState === 'live') void startCamera('');
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
  status.textContent = 'Rendering image…';
  if (ensureWorker(true)) renderStill();
  updateControls();
}

function changeSettings() {
  if (videoBusy()) {
    for (const input of inputs) input.value = settings[input.dataset.field];
    showError('Stop recording before changing the detail');
    return;
  }
  const proposed = {};
  for (const input of inputs) {
    const value = input.valueAsNumber;
    const valid = Number.isFinite(value) && value >= Number(input.min) && value <= Number(input.max) &&
      Number.isInteger((value - Number(input.min)) / Number(input.step));
    input.setCustomValidity(valid ? '' : 'Choose a value within the indicated range and step');
    input.setAttribute('aria-invalid', String(!valid));
    if (!valid) {
      showError('Invalid settings: choose a value within the indicated range and step');
      return;
    }
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
  } catch (error) {
    showError(error.message);
  }
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
    }
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
    }
  } catch (error) {
    showError(error.message);
  }
  syncLookControls();
}

function changePresentation(proposed) {
  if (videoBusy()) { syncLookControls(); return; }
  if ((proposed.tint !== 'default' && !/^#[0-9a-f]{6}$/i.test(proposed.tint)) ||
      typeof proposed.glow !== 'boolean') {
    showError('Choose a valid text color and glow setting');
    syncLookControls();
    return;
  }
  tint = proposed.tint;
  glow = proposed.glow;
  if (tint === 'default') document.body.style.removeProperty('--ascii-tint');
  else document.body.style.setProperty('--ascii-tint', tint);
  document.body.dataset.glow = glow ? 'on' : 'off';
  outputView.update();
  syncLookControls();
}

function dispose() {
  if (disposed) return;
  disposed = true;
  startToken++;
  decodeToken++;
  renderEpoch++;
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
  await configureAsciiText(output);
  outputView = createOutputView(document.querySelector('.ascii-viewport'), output, byId('fit-output'), byId('detail-output'));
  byId('fit-output').click();
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
  lookControls = createLookControls({
    onLookChange: changeLook, onFramingChange: changeFraming, onPresentationChange: changePresentation,
  });
  syncLookControls();
  byId('camera-shutter').addEventListener('click', shutterAction);
  byId('camera-stop').addEventListener('click', stopCamera);
  byId('camera-import').addEventListener('click', () => imageInput.click());
  byId('flip-camera').addEventListener('click', flipCamera);
  byId('photo-mode').addEventListener('click', () => selectCaptureMode('photo'));
  byId('video-mode').addEventListener('click', () => selectCaptureMode('video'));
  byId('review-video').addEventListener('click', openVideoReview);
  byId('studio-video').addEventListener('click', () => {
    selectCaptureMode('video');
    cameraShell.setLayout('camera');
  });
  byId('settings-panel').open = !matchMedia('(max-width: 1000px)').matches;
  startButton.addEventListener('click', () => { void startCamera(); });
  stopButton.addEventListener('click', stopCamera);
  deviceSelect.addEventListener('change', () => {
    if (cameraState === 'live') void startCamera(deviceSelect.value);
  });
  facingSelect.addEventListener('change', () => {
    deviceSelect.value = '';
    if (cameraState === 'live') void startCamera('');
  });
  imageInput.addEventListener('change', () => { void acceptImage(imageInput.files[0]); });
  byId('settings-form').addEventListener('submit', (event) => event.preventDefault());
  inputs.forEach((input) => input.addEventListener('change', changeSettings));
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden' && cameraState !== 'idle') {
      stopCamera();
      status.textContent = 'Camera paused — tap Start camera to resume';
    } else if (document.visibilityState === 'visible' && reviewPending) {
      reviewPending = false;
      openVideoReview();
    }
  });
  window.addEventListener('pagehide', (event) => {
    if (event.persisted) {
      if (cameraState !== 'idle') stopCamera();
    } else dispose();
  });
  status.textContent = 'Ready — choose a local source';
  updateControls();
} catch {
  showError('Unable to load the ASCII font');
  startButton.disabled = true;
  imageInput.disabled = true;
  status.textContent = 'Font unavailable';
}
