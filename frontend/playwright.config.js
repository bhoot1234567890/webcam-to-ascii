import { defineConfig, devices } from '@playwright/test';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const frontend = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(frontend, '..');
const production = process.env.QA_PRODUCTION === '1';
const localLib = path.join(root, '.tools/host-libs/usr/lib/x86_64-linux-gnu');
const preload = ['libhidapi-hidraw.so.0', 'libmanette-0.2.so.0'].map((file) => path.join(localLib, file));
// The bundled WebKit shell replaces LD_LIBRARY_PATH. Keep optional host fixes local.
const webkitEnvironment = preload.every(existsSync) ? {
  ...process.env,
  LD_PRELOAD: [...preload, process.env.LD_PRELOAD].filter(Boolean).join(':'),
} : process.env;
const chromium = { browserName: 'chromium', launchOptions: { chromiumSandbox: true } };
const iphone = { ...devices['iPhone 14 Plus'], browserName: 'webkit', launchOptions: { env: webkitEnvironment } };

export default defineConfig({
  testDir: './tests/browser',
  timeout: 60000,
  expect: { timeout: 10000 },
  retries: 0,
  workers: 2,
  fullyParallel: true,
  forbidOnly: true,
  outputDir: path.join(root, 'qa/artifacts/playwright', production ? 'production-results' : 'results'),
  reporter: [
    ['list'],
    ['html', { outputFolder: path.join(root, 'qa/artifacts/playwright', production ? 'production-html' : 'html'), open: 'never' }],
    ['json', { outputFile: path.join(root, 'qa/artifacts/playwright', production ? 'production.json' : 'results.json') }],
  ],
  use: {
    baseURL: 'http://127.0.0.1:4173',
    headless: true,
    viewport: { width: 1440, height: 1000 },
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: production ? [
    { name: 'production-chromium', testMatch: 'production.spec.js', use: chromium },
    { name: 'production-iphone-webkit', testMatch: 'mobile.spec.js', use: iphone },
  ] : [
    { name: 'chromium', testMatch: 'image.spec.js', use: chromium },
    { name: 'firefox', testMatch: 'image.spec.js', use: { browserName: 'firefox' } },
    { name: 'webkit', testMatch: 'image.spec.js', use: { browserName: 'webkit', launchOptions: { env: webkitEnvironment } } },
    { name: 'iphone-webkit', testMatch: 'mobile.spec.js', use: iphone },
    { name: 'camera-chromium', testMatch: ['camera.spec.js', 'controller.spec.js'], use: {
      ...chromium,
      launchOptions: { chromiumSandbox: true, args: [
        '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream',
        `--use-file-for-fake-video-capture=${path.join(root, 'qa/artifacts/fake-camera.y4m')}`,
      ] },
    } },
    { name: 'ab-chromium', testMatch: 'ab.spec.js', use: chromium },
  ],
  webServer: {
    command: production ? 'bun run preview' : 'bun run dev',
    cwd: frontend,
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: false,
    timeout: 30000,
    env: { ...process.env, NO_COLOR: '1' },
  },
});
