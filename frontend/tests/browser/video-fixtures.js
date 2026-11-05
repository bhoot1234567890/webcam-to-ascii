import path from 'node:path';
import { execFile } from 'node:child_process';
import { mkdir } from 'node:fs/promises';
import { promisify } from 'node:util';
import { createHash } from 'node:crypto';
import { ROOT, IMAGES, checkedBytes } from './fixtures.js';

/** A deterministic local camera adapter; encoding, worker rendering and playback stay real. */
export async function installVideoCamera(page) {
  const fixture = IMAGES.fixtures.find(({ id }) => id === 'astronaut');
  const png = checkedBytes(path.join(ROOT, 'qa/fixtures/images'), fixture).toString('base64');
  await page.addInitScript((base64) => {
    const nativeCapture = HTMLCanvasElement.prototype.captureStream;
    const NativeRecorder = window.MediaRecorder;
    const createURL = URL.createObjectURL.bind(URL);
    const revokeURL = URL.revokeObjectURL.bind(URL);
    const probe = window.__videoQA = {
      cameras: [], sources: [], captures: [], recorders: [], requests: [], urls: [], revoked: [], failNextEncoder: false,
    };
    const cameraCanvases = new WeakSet();
    if (nativeCapture) HTMLCanvasElement.prototype.captureStream = function (...args) {
      const stream = nativeCapture.apply(this, args);
      if (!cameraCanvases.has(this)) probe.captures.push({ canvas: this, stream });
      return stream;
    };
    if (NativeRecorder) window.MediaRecorder = class extends NativeRecorder {
      static isTypeSupported(type) { return NativeRecorder.isTypeSupported(type); }
      constructor(stream, options) {
        if (probe.failNextEncoder) {
          probe.failNextEncoder = false;
          throw new DOMException('Encoder unavailable', 'NotSupportedError');
        }
        super(stream, options);
        probe.recorders.push(this);
      }
    };
    URL.createObjectURL = (blob) => {
      const url = createURL(blob);
      probe.urls.push({ url, type: blob.type, bytes: blob.size });
      return url;
    };
    URL.revokeObjectURL = (url) => { probe.revoked.push(url); revokeURL(url); };
    Object.defineProperty(navigator.mediaDevices, 'getUserMedia', { configurable: true, value: async (constraints) => {
      probe.requests.push(constraints);
      const bytes = Uint8Array.from(atob(base64), (char) => char.charCodeAt(0));
      const bitmap = await createImageBitmap(new Blob([bytes], { type: 'image/png' }));
      const canvas = document.createElement('canvas');
      probe.sources.push(canvas);
      canvas.width = 320;
      canvas.height = 240;
      cameraCanvases.add(canvas);
      const ctx = canvas.getContext('2d');
      let frame = 0;
      function paint() {
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, 320, 240);
        ctx.drawImage(bitmap, 40, 0, 240, 240);
        // A moving bar makes temporal changes testable in the decoded recording.
        ctx.fillStyle = '#fff';
        ctx.fillRect((frame++ * 8) % 300, 192, 20, 48);
      }
      paint();
      const stream = canvas.captureStream(15);
      const interval = setInterval(paint, 1000 / 15);
      let released = false;
      function cleanup() {
        if (released) return;
        released = true;
        clearInterval(interval);
        bitmap.close();
      }
      for (const track of stream.getTracks()) {
        const stop = track.stop.bind(track);
        track.stop = () => { stop(); cleanup(); };
        track.addEventListener('ended', cleanup, { once: true });
      }
      addEventListener('pagehide', cleanup, { once: true });
      probe.cameras.push(stream);
      return stream;
    } });
  }, png);
}

/** Save through the real download control, then inspect the native encoded container. */
export async function downloadVideo(page, testInfo, variant = '') {
  const [download] = await Promise.all([
    page.waitForEvent('download'), page.locator('#video-save').click(),
  ]);
  const filename = download.suggestedFilename();
  const extension = filename.split('.').at(-1);
  const directory = path.join(ROOT, 'qa/artifacts/video', testInfo.project.name);
  await mkdir(directory, { recursive: true });
  const id = createHash('sha256').update(testInfo.testId + variant).digest('hex').slice(0, 10);
  const file = path.join(directory, `${id}.${extension}`);
  await download.saveAs(file);
  const { stdout } = await promisify(execFile)('ffprobe', [
    '-v', 'error', '-show_entries', 'format=duration,format_name:stream=codec_name,codec_type,width,height',
    '-of', 'json', file,
  ]);
  await testInfo.attach(`recorded-ascii-video${variant ? `-${variant}` : ''}`, { path: file, contentType: extension === 'mp4' ? 'video/mp4' : 'video/webm' });
  return { filename, path: file, ...JSON.parse(stdout) };
}
