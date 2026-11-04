import './styles.css';
import './camera-shell.css';
import { computeGrid, normalizeSettings, computeCropRect } from './renderer.js';
import { createCameraController } from './camera.js';
import { configureAsciiText } from './presentation.js';
import { createLanczosResampler } from './resample.js';
import { createOutputView } from './output-view.js';
import { createCameraShell } from './camera-shell.js';

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
function readSettingsFromForm() {
  const proposed = {};
  for (const input of inputs) {
    if (input.type === 'checkbox') proposed[input.dataset.field] = input.checked;
    else if (input.tagName === 'SELECT' || input.type === 'color') proposed[input.dataset.field] = input.value;
    else proposed[input.dataset.field] = input.valueAsNumber;
  }
  return normalizeSettings(proposed);
}
let settings = readSettingsFromForm();
let settingsKey = JSON.stringify(settings);
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

function showError(message) {
  errorText.textContent = message;
  errorText.hidden = !message;
}

function updateControls() {
  startButton.disabled = cameraState !== 'idle' || disposed || captureBusy;
  stopButton.disabled = cameraState === 'idle';
  deviceSelect.disabled = cameraState === 'requesting' || captureBusy;
  facingSelect.disabled = cameraState === 'requesting' || captureBusy;
  imageInput.disabled = disposed || captureBusy;
  const captureReady = cameraState === 'live' && source?.kind === 'camera' && source.active &&
    output.dataset.sourceGeneration === String(source.generation);
  cameraShell?.update({ cameraState, kind: source?.kind, captureReady, busy: captureBusy, disposed });
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
  camera.stop();
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
        request.source === source && (source.kind === 'image' || source.active);
      if (current) {
        if (data.width !== request.width || data.height !== request.height || typeof data.text !== 'string') {
          workerFailure();
          return;
        }
        // One synchronous commit: identity describes precisely this text, not UI intent.
        const gridChanged = output.dataset.columns !== String(data.width) || output.dataset.rows !== String(data.height);
        const sourceChanged = output.dataset.sourceGeneration !== String(request.generation);
        output.textContent = data.text;
        Object.assign(output.dataset, {
          renderId: String(request.id), sourceGeneration: String(request.generation),
          sourceKind: request.source.kind, sourceName: request.source.name,
          settingsKey: request.settingsKey, columns: String(data.width), rows: String(data.height),
        });
        if (gridChanged) outputView.update();
        byId('grid-label').textContent = `${data.width} columns × ${data.height} rows`;
        const performanceLabel = byId('performance-label');
        if (source.kind === 'camera') deliveredFrames++;
        performanceLabel.dataset.processingMs = String(data.processingMs);
        performanceLabel.dataset.deliveredFrames = String(deliveredFrames);
        performanceLabel.textContent = `${data.processingMs.toFixed(1)} ms / frame`;
        status.textContent = source.kind === 'image' ? (source.captured ? 'Frame captured' : 'Image ready') : 'Camera live';
        if (sourceChanged) updateControls();
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
    const crop = computeCropRect(source.width, source.height, settings.aspectRatio);
    const { width, height } = computeGrid(source.width, source.height, settings);
    const key = `${crop.width}:${crop.height}:${width}:${height}`;
    if (key !== samplerKey) {
      sampler = createLanczosResampler(crop.width, crop.height, width, height);
      samplerKey = key;
    }
    if (canvas.width !== crop.width || canvas.height !== crop.height) {
      canvas.width = crop.width;
      canvas.height = crop.height;
    }
    context.fillStyle = '#000';
    context.fillRect(0, 0, crop.width, crop.height);
    context.drawImage(
      source.kind === 'image' ? source.bitmap : video,
      crop.x, crop.y, crop.width, crop.height,
      0, 0, crop.width, crop.height
    );
    const original = context.getImageData(0, 0, crop.width, crop.height).data;
    const rgba = sampler.resize(original);
    const id = ++requestId;
    inFlight = { id, width, height, source, generation: source.generation, settingsKey, epoch: renderEpoch };
    worker.postMessage({ type: 'render', id, width, height, settings, rgba: rgba.buffer }, [rgba.buffer]);
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
      source.width = video.videoWidth;
      source.height = video.videoHeight;
      submitRender();
    }
  },
  onError(error) {
    if (source?.kind === 'camera') source.active = false;
    renderEpoch++;
    showError(cameraMessage(error));
    status.textContent = 'Camera stopped';
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
  if (disposed || captureBusy || cameraState === 'requesting') return;
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

function stopCamera() {
  startToken++;
  renderEpoch++;
  pendingStill = false;
  if (source?.kind === 'camera') source.active = false;
  camera.stop();
  status.textContent = 'Camera stopped';
}

async function freezeFrame() {
  if (captureBusy || disposed || cameraState !== 'live' || source?.kind !== 'camera') return;
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
  if (cameraState === 'requesting') stopCamera();
  else if (cameraState === 'live') void freezeFrame();
  else void startCamera();
}

function flipCamera() {
  if (disposed || captureBusy || cameraState === 'requesting') return;
  facingSelect.value = facingSelect.value === 'environment' ? 'user' : 'environment';
  deviceSelect.value = '';
  if (cameraState === 'live') void startCamera('');
}

async function acceptImage(file, captured = false) {
  if (!file || disposed) return;
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
    computeGrid(bitmap.width, bitmap.height, settings);
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
  imagePreview.alt = `Original image: ${file.name}`;
  imagePreview.hidden = false;
  video.hidden = true;
  byId('source-label').textContent = file.name;
  showError('');
  status.textContent = 'Rendering image…';
  if (ensureWorker(true)) renderStill();
  updateControls();
}

function hexToRgb(hex) {
  const clean = String(hex).replace('#', '');
  let r = 255;
  let g = 255;
  let b = 255;
  if (clean.length === 3) {
    r = parseInt(clean[0] + clean[0], 16);
    g = parseInt(clean[1] + clean[1], 16);
    b = parseInt(clean[2] + clean[2], 16);
  } else if (clean.length === 6) {
    r = parseInt(clean.slice(0, 2), 16);
    g = parseInt(clean.slice(2, 4), 16);
    b = parseInt(clean.slice(4, 6), 16);
  }
  return `${Number.isFinite(r) ? r : 255}, ${Number.isFinite(g) ? g : 255}, ${Number.isFinite(b) ? b : 255}`;
}

const GLOW_COLORS = {
  white: { text: '#ffffff', rgb: '255, 255, 255' },
  green: { text: '#a7e8ce', rgb: '167, 232, 206' },
  amber: { text: '#ffc83b', rgb: '255, 200, 59' },
  cyan: { text: '#4deeea', rgb: '77, 238, 234' },
  magenta: { text: '#ff007f', rgb: '255, 0, 127' },
};

function updateAsciiBloom() {
  const bloomInput = byId('bloom-slider');
  const colorInput = byId('glow-color-select');
  const customInput = byId('custom-color-picker');
  const radius = Number(bloomInput?.value || 0);
  const colorKey = colorInput?.value || 'white';
  let text;
  let rgb;
  if (colorKey === 'custom') {
    text = customInput?.value || '#ff7700';
    rgb = hexToRgb(text);
  } else {
    const color = GLOW_COLORS[colorKey] || GLOW_COLORS.white;
    text = color.text;
    rgb = color.rgb;
  }
  if (!output) return;
  output.style.setProperty('--ascii-color', text);
  if (radius <= 0.05) {
    output.style.setProperty('--ascii-bloom', 'none');
  } else {
    const r = radius * 12;
    output.style.setProperty(
      '--ascii-bloom',
      `0 0 ${r * 0.25}px rgba(${rgb}, 0.95), 0 0 ${r * 0.75}px rgba(${rgb}, 0.65), 0 0 ${r * 1.5}px rgba(${rgb}, 0.35)`
    );
  }
}

const PRESETS = {
  crisp: {
    dither: 'atkinson',
    ramp: 'clean',
    invert: true,
    autoLevels: true,
    gamma: 1.0,
    sharpen: 0.8,
    colorBurn: 0.0,
    bloom: 0.0,
    glowColor: 'white',
    edgeBlend: 'overlay',
    edgeIndexThreshold: 60,
    desc: 'Atkinson dithering + unsharp mask sharpening. Razor-sharp characters with clean background.',
  },
  burnt: {
    dither: 'atkinson',
    ramp: 'clean',
    invert: true,
    autoLevels: true,
    gamma: 1.2,
    sharpen: 1.2,
    colorBurn: 0.5,
    bloom: 0.4,
    glowColor: 'amber',
    edgeBlend: 'overlay',
    edgeIndexThreshold: 60,
    desc: 'Deep color-burnt shadows, razor-sharp edges, with an amber phosphor CRT bloom.',
  },
  neon: {
    dither: 'floyd-steinberg',
    ramp: 'detailed',
    invert: true,
    autoLevels: true,
    gamma: 1.0,
    sharpen: 0.6,
    colorBurn: 0.15,
    bloom: 1.0,
    glowColor: 'green',
    edgeBlend: 'luminance-only',
    edgeIndexThreshold: 90,
    desc: 'Cyberpunk Matrix look. Characters emit an intense neon-green phosphor bloom.',
  },
  smooth: {
    dither: 'floyd-steinberg',
    ramp: 'detailed',
    invert: true,
    autoLevels: true,
    gamma: 1.0,
    sharpen: 0.0,
    colorBurn: 0.0,
    bloom: 0.0,
    glowColor: 'white',
    edgeBlend: 'luminance-only',
    edgeIndexThreshold: 90,
    desc: 'Floyd-Steinberg error diffusion + detailed ramp. Continuous, smooth photographic gradation.',
  },
  bayer: {
    dither: 'bayer',
    ramp: 'clean',
    invert: true,
    autoLevels: true,
    gamma: 1.0,
    sharpen: 0.6,
    colorBurn: 0.0,
    bloom: 0.0,
    glowColor: 'white',
    edgeBlend: 'overlay',
    edgeIndexThreshold: 75,
    desc: 'Bayer 4×4 crosshatch screen with sharpened contours. Vintage newspaper halftone.',
  },
  sketch: {
    dither: 'none',
    ramp: 'clean',
    invert: true,
    autoLevels: true,
    gamma: 1.0,
    sharpen: 1.2,
    colorBurn: 0.0,
    bloom: 0.0,
    glowColor: 'white',
    edgeBlend: 'edges-only',
    edgeIndexThreshold: 80,
    desc: 'Pure directional edge contours (/ - \\ |). Minimalist architectural line drawing.',
  },
  comic: {
    dither: 'atkinson',
    ramp: 'clean',
    invert: true,
    autoLevels: true,
    gamma: 1.5,
    sharpen: 1.0,
    colorBurn: 0.35,
    bloom: 0.2,
    glowColor: 'white',
    edgeBlend: 'overlay',
    edgeIndexThreshold: 50,
    desc: 'Punchy gamma curve with crisp Atkinson edges and burnt midtones. Graphic novel punch.',
  },
  classic: {
    dither: 'none',
    ramp: 'full',
    invert: false,
    autoLevels: false,
    gamma: 1.0,
    sharpen: 0.0,
    colorBurn: 0.0,
    bloom: 0.0,
    glowColor: 'white',
    edgeBlend: 'overlay',
    edgeIndexThreshold: 90,
    desc: 'Unprocessed linear luminance + 382 font glyphs. The raw legacy terminal stream.',
  },
};

function updateSliderBadges(input) {
  const id = input.id;
  const val = Number(input.value);
  if (id === 'gamma-slider') byId('gamma-val').textContent = val.toFixed(1);
  else if (id === 'sharpen-slider') byId('sharpen-val').textContent = val.toFixed(1);
  else if (id === 'bloom-slider') byId('bloom-val').textContent = val.toFixed(1);
  else if (id === 'colorburn-slider') byId('colorburn-val').textContent = val.toFixed(2);
}

function applyPreset(key) {
  const preset = PRESETS[key];
  if (!preset) return;
  for (const [field, val] of Object.entries(preset)) {
    if (field === 'desc') continue;
    const input = inputs.find((inp) => inp.dataset.field === field);
    if (!input) continue;
    if (input.type === 'checkbox') input.checked = val;
    else if (input.tagName === 'SELECT') input.value = val;
    else {
      input.value = String(val);
      updateSliderBadges(input);
    }
  }
  updateAsciiBloom();
  const descEl = byId('preset-desc');
  if (descEl && preset.desc) descEl.textContent = preset.desc;
  document.querySelectorAll('.preset-pill').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.preset === key);
  });
  syncPillGroups();
  changeSettings();
}

function changeSettings() {
  const proposed = {};
  for (const input of inputs) {
    let value;
    if (input.type === 'checkbox') {
      value = input.checked;
    } else if (input.tagName === 'SELECT' || input.type === 'select-one' || input.type === 'color') {
      value = input.value;
      input.setCustomValidity('');
      input.setAttribute('aria-invalid', 'false');
    } else {
      value = input.valueAsNumber;
      const min = Number(input.min);
      const max = Number(input.max);
      const step = Number(input.step) || 1;
      const ratio = (value - min) / step;
      const stepValid = Math.abs(Math.round(ratio) - ratio) < 1e-3;
      const valid = Number.isFinite(value) && value >= min && value <= max && stepValid;
      input.setCustomValidity(valid ? '' : 'Choose a value within the indicated range and step');
      input.setAttribute('aria-invalid', String(!valid));
      if (!valid) {
        showError('Invalid settings: choose a value within the indicated range and step');
        return;
      }
    }
    proposed[input.dataset.field] = value;
  }
  updateAsciiBloom();
  try {
    const next = normalizeSettings(proposed);
    if (source) computeGrid(source.width, source.height, next);
    const nextKey = JSON.stringify(next);
    showError(workerFailed ? 'Rendering failed' : '');
    if (nextKey === settingsKey) return;
    settings = next;
    settingsKey = nextKey;
    renderEpoch++;
    if (source?.kind === 'image') renderStill();
    else if (source?.kind === 'camera' && source.active) submitRender();
  } catch (error) {
    showError(error.message);
  }
  syncPillGroups();
}

function dispose() {
  if (disposed) return;
  disposed = true;
  startToken++;
  decodeToken++;
  renderEpoch++;
  camera.dispose();
  outputView?.dispose();
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
  updateAsciiBloom();
  cameraShell = createCameraShell();
  byId('camera-shutter').addEventListener('click', shutterAction);
  byId('camera-stop').addEventListener('click', stopCamera);
  byId('camera-import').addEventListener('click', () => imageInput.click());
  byId('flip-camera').addEventListener('click', flipCamera);
  byId('live-mode').addEventListener('click', () => {
    if (cameraState === 'idle') void startCamera();
  });
  byId('still-mode').addEventListener('click', () => { void freezeFrame(); });
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
  document.querySelectorAll('.preset-pill').forEach((btn) => {
    btn.addEventListener('click', () => applyPreset(btn.dataset.preset));
  });
  inputs.forEach((input) => {
    input.addEventListener('change', changeSettings);
    input.addEventListener('input', () => {
      updateSliderBadges(input);
      document.querySelectorAll('.preset-pill').forEach((btn) => btn.classList.remove('active'));
      changeSettings();
    });
  });
  byId('glow-color-select')?.addEventListener('change', () => {
    const bloomInput = byId('bloom-slider');
    if (bloomInput && Number(bloomInput.value) <= 0.05 && byId('glow-color-select').value !== 'white') {
      bloomInput.value = '0.7';
      const bloomVal = byId('bloom-val');
      if (bloomVal) bloomVal.textContent = '0.7';
    }
    updateAsciiBloom();
    changeSettings();
  });
  function syncPillGroups() {
    document.querySelectorAll('.pill-group[data-for]').forEach((group) => {
      const targetId = group.dataset.for;
      const select = byId(targetId);
      if (!select) return;
      const currentVal = select.value;
      group.querySelectorAll('.pill-btn').forEach((btn) => {
        btn.classList.toggle('active', btn.dataset.val === currentVal);
      });
    });
  }

  document.querySelectorAll('.pill-group[data-for]').forEach((group) => {
    const targetId = group.dataset.for;
    const select = byId(targetId);
    if (!select) return;
    group.querySelectorAll('.pill-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const val = btn.dataset.val;
        select.value = val;
        group.querySelectorAll('.pill-btn').forEach((b) => b.classList.toggle('active', b === btn));
        document.querySelectorAll('.preset-pill').forEach((p) => p.classList.remove('active'));
        if (targetId === 'glow-color-select') {
          const bloomInput = byId('bloom-slider');
          if (bloomInput && Number(bloomInput.value) <= 0.05 && val !== 'white') {
            bloomInput.value = '0.7';
            const bloomVal = byId('bloom-val');
            if (bloomVal) bloomVal.textContent = '0.7';
          }
        }
        updateAsciiBloom();
        changeSettings();
      });
    });
  });
  const customPicker = byId('custom-color-picker');
  const customPill = byId('custom-color-pill');
  const customSwatch = byId('custom-color-swatch');
  function onCustomColor() {
    const hex = customPicker.value;
    if (customPill) customPill.style.setProperty('--c', hex);
    if (customSwatch) customSwatch.style.background = hex;
    const glowSelect = byId('glow-color-select');
    if (glowSelect) glowSelect.value = 'custom';
    const bloomInput = byId('bloom-slider');
    if (bloomInput && Number(bloomInput.value) <= 0.05) {
      bloomInput.value = '0.7';
      const bloomVal = byId('bloom-val');
      if (bloomVal) bloomVal.textContent = '0.7';
    }
    syncPillGroups();
    document.querySelectorAll('.preset-pill').forEach((p) => p.classList.remove('active'));
    updateAsciiBloom();
    changeSettings();
  }
  customPicker?.addEventListener('input', onCustomColor);
  customPicker?.addEventListener('change', onCustomColor);
  syncPillGroups();
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden' && cameraState !== 'idle') {
      stopCamera();
      status.textContent = 'Camera paused — tap Start camera to resume';
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
