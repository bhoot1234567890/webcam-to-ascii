import { test as base, expect } from '@playwright/test';
import { readFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';

export { expect };
export const ROOT = fileURLToPath(new URL('../../../', import.meta.url));
const imageRoot = path.join(ROOT, 'qa/fixtures/images');
const referenceRoot = path.join(ROOT, 'qa/artifacts/reference');
export const IMAGES = JSON.parse(readFileSync(path.join(imageRoot, 'manifest.json'), 'utf8'));
export const REFERENCE = JSON.parse(readFileSync(path.join(referenceRoot, 'manifest.json'), 'utf8'));
const ids = new Set(IMAGES.fixtures.map((entry) => entry.id));
if (ids.size !== 16 || REFERENCE.fullImages.length !== 32 || REFERENCE.canonicalIds.some((id) => !ids.has(id))) {
  throw new Error('Incomplete canonical/reference fixture matrix');
}
const origin = 'http://127.0.0.1:4173';
const production = process.env.QA_PRODUCTION === '1';
const policyPattern = /Permissions policy violation: camera is not allowed in this document/i;

export function checkedBytes(root, spec) {
  const file = path.resolve(root, spec.path);
  if (!file.startsWith(`${path.resolve(root)}${path.sep}`)) throw new Error(`Escaping fixture path: ${file}`);
  const data = readFileSync(file);
  if (data.byteLength !== spec.bytes || createHash('sha256').update(data).digest('hex') !== spec.sha256) {
    throw new Error(`Fixture length/SHA-256 mismatch: ${file}`);
  }
  return data;
}
for (const fixture of IMAGES.fixtures) checkedBytes(imageRoot, fixture);
export function readReferenceText(entry) { return checkedBytes(referenceRoot, entry.text).toString('utf8'); }

export const test = base.extend({
  allowCameraPolicyError: [false, { option: true }],
  policyDiagnostics: async ({}, use) => { await use([]); },
  privacy: [async ({ context, allowCameraPolicyError, policyDiagnostics }, use) => {
    const unexpected = [];
    const allowed = (urlString) => {
      const url = new URL(urlString);
      if (url.protocol === 'blob:') return url.origin === origin;
      if (url.origin !== origin) return false;
      if (url.pathname.startsWith('/__qa/images/')) {
        const match = /^\/__qa\/images\/([a-z0-9-]+)\.png$/.exec(url.pathname);
        return Boolean(match && ids.has(match[1]));
      }
      return true;
    };
    const allowedSocket = (urlString) => {
      const url = new URL(urlString);
      return !production && url.protocol === 'ws:' && url.host === '127.0.0.1:4173' && url.pathname === '/';
    };
    context.on('request', (request) => {
      if (!allowed(request.url()) || !['GET', 'HEAD'].includes(request.method())) {
        unexpected.push(`Forbidden request: ${request.method()} ${request.url()}`);
      }
    });
    await context.route('**/*', async (route) => {
      const request = route.request();
      if (!allowed(request.url()) || !['GET', 'HEAD'].includes(request.method())) await route.abort();
      else await route.fallback();
    });
    // Context routing observes dedicated-worker WebSocket traffic as well as pages.
    await context.routeWebSocket('**/*', (socket) => {
      if (allowedSocket(socket.url())) socket.connectToServer();
      else {
        unexpected.push(`Forbidden WebSocket: ${socket.url()}`);
        socket.close();
      }
    });
    function observePage(page) {
      page.on('console', (message) => {
        if (policyPattern.test(message.text())) {
          if (allowCameraPolicyError) policyDiagnostics.push(message.text());
          else unexpected.push(`Unexpected camera policy diagnostic: ${message.text()}`);
        } else if (message.type() === 'error') unexpected.push(`Console: ${message.text()}`);
      });
      page.on('pageerror', (error) => unexpected.push(`Page error: ${error.stack || error.message}`));
      page.on('websocket', (socket) => {
        if (!allowedSocket(socket.url())) unexpected.push(`Forbidden page WebSocket: ${socket.url()}`);
      });
    }
    context.on('page', observePage);
    context.pages().forEach(observePage);
    await use();
    expect(unexpected, 'Unexpected network, console, or page errors').toEqual([]);
    if (allowCameraPolicyError) expect(policyDiagnostics.length, 'Expected real camera policy denial diagnostic').toBeGreaterThan(0);
  }, { auto: true }],
});

export async function openApp(page, { layout = 'studio' } = {}) {
  await page.goto('/');
  await expect(page.locator('#status')).toContainText('Ready');
  expect(await page.evaluate(() => document.fonts.check('400 12px "Roboto Mono"', 'M@ '))).toBe(true);
  if (layout === 'studio') await page.locator('#open-studio').click();
}

export async function readCommit(page) {
  return page.locator('#ascii-output').evaluate((element) => ({
    text: element.textContent,
    renderId: Number(element.dataset.renderId || 0),
    sourceGeneration: Number(element.dataset.sourceGeneration || 0),
    sourceKind: element.dataset.sourceKind || '', sourceName: element.dataset.sourceName || '',
    settingsKey: element.dataset.settingsKey || '',
    columns: Number(element.dataset.columns || 0), rows: Number(element.dataset.rows || 0),
  }));
}

async function currentSettings(page) {
  return page.locator('#settings-form').evaluate((form) => {
    const values = {};
    for (const input of form.querySelectorAll('[data-field]')) values[input.dataset.field] = input.valueAsNumber;
    return values;
  });
}

export async function uploadFixture(page, id) {
  const fixture = IMAGES.fixtures.find((entry) => entry.id === id);
  if (!fixture) throw new Error(`Unknown canonical fixture ID: ${id}`);
  const before = await readCommit(page);
  const settingsKey = JSON.stringify(await currentSettings(page));
  const name = path.basename(fixture.path);
  await page.locator('#image-input').setInputFiles(path.join(imageRoot, fixture.path));
  await page.waitForFunction(({ before, name, settingsKey }) => {
    const data = document.querySelector('#ascii-output').dataset;
    return Number(data.renderId) > before.renderId && Number(data.sourceGeneration) > before.sourceGeneration &&
      data.sourceKind === 'image' && data.sourceName === name && data.settingsKey === settingsKey;
  }, { before, name, settingsKey });
  const commit = await readCommit(page);
  expect(commit.renderId).toBeGreaterThan(before.renderId);
  expect(commit.sourceGeneration).toBeGreaterThan(before.sourceGeneration);
  expect(commit.sourceName).toBe(name);
  expect(commit.settingsKey).toBe(settingsKey);
  return commit;
}

export async function changeSettings(page, patch) {
  if (!await page.locator('#settings-panel').evaluate((panel) => panel.open)) {
    await page.locator('#settings-panel > summary').click();
  }
  const before = await readCommit(page);
  const next = { ...await currentSettings(page), ...patch };
  const key = JSON.stringify(next);
  const selectors = { maxWidthChars: '#max-width', maxHeightLines: '#max-height', sigma1: '#sigma1', sigma2: '#sigma2', edgeIndexThreshold: '#edge-threshold' };
  for (const [name, value] of Object.entries(patch)) {
    if (!selectors[name]) throw new Error(`Unknown UI setting: ${name}`);
    await page.locator(selectors[name]).fill(String(value));
    await page.locator(selectors[name]).dispatchEvent('change');
  }
  if (!before.renderId) return null;
  await page.waitForFunction(({ before, key }) => {
    const data = document.querySelector('#ascii-output').dataset;
    return Number(data.renderId) > before.renderId && Number(data.sourceGeneration) === before.sourceGeneration &&
      data.sourceKind === before.sourceKind && data.sourceName === before.sourceName && data.settingsKey === key;
  }, { before, key });
  const commit = await readCommit(page);
  expect(commit.sourceGeneration).toBe(before.sourceGeneration);
  expect(commit.settingsKey).toBe(key);
  return commit;
}

export async function captureApp(page, testInfo, name) {
  const dir = path.join(ROOT, 'qa/artifacts/screenshots', testInfo.project.name);
  mkdirSync(dir, { recursive: true });
  const suffix = createHash('sha256').update(testInfo.testId).digest('hex').slice(0, 8);
  const file = path.join(dir, `${name.replace(/[^a-z0-9_-]/gi, '-')}-${suffix}.png`);
  await page.locator('.ascii-viewport').scrollIntoViewIfNeeded();
  await page.evaluate(async () => {
    const viewport = document.querySelector('.ascii-viewport');
    viewport.scrollTop = 0;
    viewport.scrollLeft = 0;
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  });
  await page.screenshot({ path: file, fullPage: true });
  await testInfo.attach(name, { path: file, contentType: 'image/png' });
  return file;
}
