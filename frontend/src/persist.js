/** Settings persistence: one versioned localStorage record, best-effort writes. */
const KEY = 'clack:settings:v1';

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
