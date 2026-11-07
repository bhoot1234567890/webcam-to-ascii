import path from 'node:path';
import { mkdir, writeFile } from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { test, expect, ROOT, openApp } from './fixtures.js';
import { installVideoCamera } from './video-fixtures.js';

const run = promisify(execFile);

test('a superseded queued picture is never encoded and a static hold retains its endpoint', async ({ page }, testInfo) => {
  await installVideoCamera(page);
  await openApp(page);
  try {
    const started = await page.evaluate(async () => {
      const { createAsciiRecorder } = await import('/src/ascii-recorder.js');
      await document.fonts.load('400 18px "Roboto Mono"');
      const text = Array(8).fill('@'.repeat(20)).join('\n');
      const frame = {
        text, columns: 20, rows: 8, fontFamily: '"Roboto Mono"', fontSize: 18,
        lineHeight: 21.6, color: '#ffffff', background: '#09090b', glow: false,
      };
      const engine = window.__queueRecorder = createAsciiRecorder();
      const ok = engine.start(frame);
      // Both commits arrive inside one draw budget. The blank B must be canceled
      // when the latest committed picture returns to the already painted A.
      engine.pushFrame({ text: Array(8).fill(' '.repeat(20)).join('\n'), columns: 20, rows: 8 });
      engine.pushFrame({ text, columns: 20, rows: 8 });
      return ok;
    });
    expect(started).toBe(true);
    // This interval is the media timeline under test, not a UI settling delay.
    await page.evaluate(() => new Promise(resolve => setTimeout(resolve, 850)));
    const beforeStop = await page.evaluate(() => {
      const probe = window.__videoQA;
      const capture = probe.captures.find(({ stream }) => stream === probe.recorders[0].stream);
      return { manual: typeof capture.stream.getVideoTracks()[0].requestFrame === 'function', requests: capture.frameRequests.length };
    });
    if (beforeStop.manual) expect(beforeStop.requests).toBe(1);
    await page.evaluate(() => window.__queueRecorder.stop());
    await page.waitForFunction(() => window.__queueRecorder.state === 'ready');
    const clip = await page.evaluate(async () => {
      const clip = window.__queueRecorder.clip;
      return { mimeType: clip.mimeType, bytes: [...new Uint8Array(await clip.blob.arrayBuffer())] };
    });
    const directory = path.join(ROOT, 'qa/artifacts/issue3', testInfo.project.name);
    await mkdir(directory, { recursive: true });
    const file = path.join(directory, `queued-revert.${clip.mimeType.startsWith('video/mp4') ? 'mp4' : 'webm'}`);
    await writeFile(file, new Uint8Array(clip.bytes));
    const { stdout } = await run('ffprobe', ['-v', 'error', '-show_entries', 'packet=pts_time', '-of', 'json', file]);
    const packets = JSON.parse(stdout).packets;
    const times = packets.map(packet => Number(packet.pts_time));
    expect(Math.max(...times) - Math.min(...times)).toBeGreaterThan(0.7);
    const decoded = await run('ffmpeg', [
      '-hide_banner', '-loglevel', 'error', '-i', file, '-vf', 'scale=8:8:flags=area,format=gray',
      '-fps_mode', 'passthrough', '-f', 'rawvideo', 'pipe:1',
    ], { encoding: 'buffer' });
    for (let offset = 0; offset < decoded.stdout.length; offset += 64) {
      const sample = decoded.stdout.subarray(offset, offset + 64);
      const mean = sample.reduce((sum, value) => sum + value, 0) / sample.length;
      expect(mean, 'Every decoded picture remains the bright @ grid, never the canceled blank frame').toBeGreaterThan(32);
    }
    expect(await page.evaluate(() => window.__videoQA.captures.every(({ stream }) => stream.getTracks().every(track => track.readyState === 'ended')))).toBe(true);
    await testInfo.attach('native-queued-revert', { path: file, contentType: clip.mimeType.split(';')[0] });
  } finally {
    await page.evaluate(() => window.__queueRecorder?.dispose());
  }
});
