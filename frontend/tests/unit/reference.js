import { createHash } from 'node:crypto';
import { readFileSync, realpathSync } from 'node:fs';
import { dirname, isAbsolute, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const repositoryRoot = fileURLToPath(new URL('../../../', import.meta.url));
const manifestPath = resolve(repositoryRoot, 'qa/artifacts/reference/manifest.json');
const root = realpathSync(dirname(manifestPath));
const cache = new Map();
const fail = (message) => { throw new Error(`Python reference ${message}`); };
export const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

export function bytes(spec) {
  if (!spec || typeof spec.path !== 'string' || isAbsolute(spec.path)) fail('has malformed relative path');
  const path = resolve(root, spec.path);
  const inside = relative(root, path);
  if (inside === '..' || inside.startsWith('../') || isAbsolute(inside)) fail(`path escapes manifest directory: ${spec.path}`);
  const real = realpathSync(path);
  const realInside = relative(root, real);
  if (realInside === '..' || realInside.startsWith('../') || isAbsolute(realInside)) fail(`symlink escapes manifest directory: ${path}`);
  let data = cache.get(real);
  if (!data) { data = readFileSync(real); cache.set(real, data); }
  if (!Number.isSafeInteger(spec.bytes) || spec.bytes < 0 || data.length !== spec.bytes) fail(`${path}: byte length mismatch`);
  if (typeof spec.sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(spec.sha256) || sha256(data) !== spec.sha256) fail(`${path}: SHA-256 mismatch`);
  return data;
}

export function array(spec) {
  const data = bytes(spec);
  if (!['u8', 'f64le'].includes(spec.dtype) || !Array.isArray(spec.shape) || !spec.shape.length ||
      spec.shape.some((n) => !Number.isSafeInteger(n) || n <= 0)) fail(`${spec.path}: malformed dtype/shape`);
  const count = spec.shape.reduce((a, b) => a * b, 1);
  if (!Number.isSafeInteger(count) || count * (spec.dtype === 'u8' ? 1 : 8) !== data.length) fail(`${spec.path}: array byte count disagrees with shape`);
  if (spec.dtype === 'u8') return Uint8Array.from(data);
  const out = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    out[i] = data.readDoubleLE(i * 8);
    if (!Number.isFinite(out[i])) fail(`${spec.path}: nonfinite float at ${i}`);
  }
  return out;
}

export function text(spec) {
  return new TextDecoder('utf-8', { fatal: true }).decode(bytes(spec));
}

function visit(value) {
  if (!value || typeof value !== 'object') return;
  if (!Array.isArray(value) && 'path' in value) {
    if ('dtype' in value || 'shape' in value) array(value);
    else bytes(value);
  }
  for (const nested of Object.values(value)) visit(nested);
}

const manifestBytes = readFileSync(manifestPath);
export const manifestSha256 = sha256(manifestBytes);
export const manifest = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(manifestBytes));
if (!manifest || manifest.schemaVersion !== 1 || manifest.selectedFixtureId !== undefined) fail('requires the complete schemaVersion=1 export, not a selected fixture');
visit(manifest);

export const canonicalIds = [
  'black', 'midgray', 'white', 'ramp', 'step-horizontal', 'step-vertical',
  'step-diagonal-down', 'step-diagonal-up', 'impulse-center', 'impulse-corner',
  'checker4', 'noise', 'rgb-patches', 'astronaut', 'astronaut-portrait', 'astronaut-landscape',
];
const pairs = [[1, 2], [0.5, 1.5], [2, 3], [1, 1], [2, 1], [0.5, 5]];
function exactIds(section, expected) {
  const entries = manifest[section];
  if (!Array.isArray(entries) || entries.some((entry) => !entry || typeof entry.id !== 'string')) fail(`malformed ${section}`);
  const actual = entries.map((entry) => entry.id).sort();
  if (JSON.stringify(actual) !== JSON.stringify([...expected].sort())) fail(`${section}: missing, duplicate, or unknown fixture IDs`);
}
if (JSON.stringify(manifest.canonicalIds) !== JSON.stringify(canonicalIds)) fail('canonicalIds differs from the complete frozen fixture set');
exactIds('components', [...canonicalIds, 'tiny-1x1', 'tiny-1x17', 'tiny-17x1'].flatMap((id) => pairs.map(([a, b]) => `${id}@${a},${b}`)));
exactIds('fullImages', canonicalIds.flatMap((id) => ['comparison', 'default'].map((profile) => `${profile}:${id}`)));
exactIds('rgbCases', [...canonicalIds, 'all-grayscale-values']);
exactIds('sobelCases', ['step-horizontal', 'step-vertical', 'step-diagonal-down', 'step-diagonal-up', 'checker4', 'tiny-1x1', 'tiny-1x17', 'tiny-17x1'].flatMap((id) => ['positive', 'negative'].map((polarity) => `${id}-${polarity}`)));
exactIds('gridCases', [...manifest.fullImages.map((entry) => entry.id), ...['square', 'tiny-square', 'tiny-portrait', 'tiny-landscape', 'zero-width', 'negative-height', 'extreme-portrait', 'extreme-landscape'].map((id) => `boundary:${id}`)]);
const expectedPlanes = ['gray', 'blur1', 'blur2', 'dog', 'gx', 'gy', 'magnitude', 'angles', 'shadeCodes', 'edgeCodes'];
for (const entry of [...manifest.components, ...manifest.fullImages]) {
  const { width, height } = entry;
  if (![width, height].every((n) => Number.isSafeInteger(n) && n > 0)) fail(`${entry.id}: invalid grid`);
  for (const key of expectedPlanes) {
    const spec = entry.outputs?.[key];
    const dtype = ['blur1', 'blur2', 'gx', 'gy', 'angles'].includes(key) ? 'f64le' : 'u8';
    if (!spec || spec.dtype !== dtype || JSON.stringify(spec.shape) !== JSON.stringify([height, width])) fail(`${entry.id}/${key}: invalid stage specification`);
  }
  const rows = text(entry.text).split('\n');
  if (rows.length !== height || rows.some((row) => row.length !== width || /[^\x20-\x7e]/.test(row))) fail(`${entry.id}: incomplete ASCII text rows`);
  if (entry.input && (entry.input.dtype !== 'u8' || JSON.stringify(entry.input.shape) !== JSON.stringify([height, width]))) fail(`${entry.id}: invalid gray input`);
  const settings = entry.settings;
  if (!settings || !['maxWidthChars', 'maxHeightLines', 'sigma1', 'sigma2', 'edgeIndexThreshold'].every((key) => Number.isFinite(settings[key]))) fail(`${entry.id}: malformed settings`);
  if (entry.fixtureId && ![...canonicalIds, 'tiny-1x1', 'tiny-1x17', 'tiny-17x1'].includes(entry.fixtureId)) fail(`${entry.id}: unknown fixtureId`);
  if (entry.input && entry.id !== `${entry.fixtureId}@${settings.sigma1},${settings.sigma2}`) fail(`${entry.id}: settings/ID mismatch`);
}
for (const entry of manifest.fullImages) {
  if (entry.id !== `${entry.profile}:${entry.fixtureId}` || !canonicalIds.includes(entry.fixtureId)) fail(`${entry.id}: malformed full-image identity`);
  if (JSON.stringify(entry.settings) !== JSON.stringify(manifest.profiles[entry.profile])) fail(`${entry.id}: profile/settings mismatch`);
}
for (const entry of manifest.components) {
  if (!entry.input) fail(`${entry.id}: missing independent gray input`);
}
for (const entry of manifest.sobelCases) {
  const shape = entry.input?.shape;
  if (entry.input?.dtype !== 'u8' || shape?.length !== 2) fail(`${entry.id}: invalid independent Sobel input`);
  for (const key of ['gx', 'gy', 'magnitude', 'angles', 'radians']) {
    const spec = entry.outputs?.[key];
    if (!spec || spec.dtype !== (key === 'magnitude' ? 'u8' : 'f64le') || JSON.stringify(spec.shape) !== JSON.stringify(shape)) fail(`${entry.id}/${key}: invalid independent Sobel plane`);
  }
}
for (const entry of manifest.rgbCases) {
  const shape = entry.gray?.shape;
  if (entry.gray?.dtype !== 'u8' || shape?.length !== 2) fail(`${entry.id}: invalid independent grayscale plane`);
  for (const [key, channels] of [['rgb', 3], ['rgba', 4]]) {
    if (entry[key]?.dtype !== 'u8' || JSON.stringify(entry[key].shape) !== JSON.stringify([...shape, channels])) fail(`${entry.id}: invalid independent ${key} plane`);
  }
}
