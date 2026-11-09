/** Settings persistence: versioned localStorage records, best-effort writes. */
const KEY = 'clack:settings:v2';
const PRESET_KEY = 'clack:presets:v1';

export function loadSaved() {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function save(state) {
  try {
    localStorage.setItem(KEY, JSON.stringify(state));
  } catch {
    // Private mode: persistence is best-effort, never fatal.
  }
}

export function loadPresets() {
  try {
    const parsed = JSON.parse(localStorage.getItem(PRESET_KEY) ?? '[]');
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function savePresets(presets) {
  try {
    localStorage.setItem(PRESET_KEY, JSON.stringify(presets));
  } catch {
    // Private mode: persistence is best-effort, never fatal.
  }
}
