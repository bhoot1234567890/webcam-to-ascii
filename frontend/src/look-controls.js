import { LOOKS, normalizeLook } from './looks.js';
import { FRAMINGS } from './framing.js';

/** Shared Camera/Studio controls. Only the owner commits processing and presentation state. */
export function createLookControls({ onLookChange, onFramingChange, onPresentationChange }) {
  const byId = (id) => document.getElementById(id);
  const sheet = byId('look-sheet');
  const listeners = [];
  let opener = null;
  let state = { look: normalizeLook(), framing: 'full', tint: 'default', glow: false };
  const listen = (node, event, handler) => {
    node.addEventListener(event, handler);
    listeners.push(() => node.removeEventListener(event, handler));
  };

  function addChoices(fieldset, name, choices) {
    for (const choice of choices) {
      const label = document.createElement('label');
      label.className = 'sheet-choice';
      const radio = document.createElement('input');
      radio.type = 'radio';
      radio.name = name;
      radio.value = choice.id;
      const text = document.createElement('span');
      const title = document.createElement('strong');
      title.textContent = choice.label;
      text.append(title);
      if (choice.description) {
        const description = document.createElement('small');
        description.textContent = choice.description;
        text.append(description);
      }
      label.append(radio, text);
      fieldset.append(label);
    }
  }
  addChoices(byId('look-choices'), 'look', LOOKS);
  addChoices(byId('framing-choices'), 'framing', FRAMINGS);

  function open(event, name) {
    opener = event.currentTarget;
    sheet.showModal();
    sheet.querySelector(`input[name="${name}"]:checked`).focus({ preventScroll: true });
    sheet.querySelector(`#${name === 'look' ? 'look' : 'framing'}-choices`).scrollIntoView({ block: 'nearest' });
  }
  listen(byId('open-framing'), 'click', (event) => open(event, 'framing'));
  listen(byId('open-looks'), 'click', (event) => open(event, 'look'));
  listen(byId('close-look-sheet'), 'click', () => sheet.close());
  listen(sheet, 'close', () => opener?.focus({ preventScroll: true }));
  listen(sheet, 'click', (event) => {
    if (event.target !== sheet) return;
    const box = sheet.getBoundingClientRect();
    if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) sheet.close();
  });
  listen(byId('look-choices'), 'change', (event) => {
    if (event.target.name === 'look') onLookChange(normalizeLook({ id: event.target.value }));
  });
  // Reapplying the current recipe also resets its manual adjustments.
  listen(byId('look-choices'), 'click', (event) => {
    if (event.target.name === 'look' && event.target.value === state.look.id &&
      (state.look.autoContrast || state.look.midtone !== 1 || state.look.sharpness !== 0 || state.look.invert)) {
      onLookChange(normalizeLook({ id: event.target.value }));
    }
  });
  listen(byId('framing-choices'), 'change', (event) => {
    if (event.target.name === 'framing') onFramingChange(event.target.value);
  });
  const adjustments = {
    'look-auto-contrast': ['autoContrast', (input) => input.checked],
    'look-midtone': ['midtone', (input) => input.valueAsNumber],
    'look-sharpness': ['sharpness', (input) => input.valueAsNumber],
    'look-invert': ['invert', (input) => input.checked],
  };
  for (const [id, [key, value]] of Object.entries(adjustments)) {
    listen(byId(id), byId(id).type === 'range' ? 'input' : 'change', (event) => {
      onLookChange(normalizeLook({ ...state.look, [key]: value(event.target) }));
    });
  }
  listen(byId('tint-choices'), 'change', (event) => {
    if (event.target.name === 'text-tint') onPresentationChange({ tint: event.target.value, glow: state.glow });
  });
  listen(byId('text-color'), 'input', (event) => onPresentationChange({ tint: event.target.value, glow: state.glow }));
  listen(byId('soft-glow'), 'change', (event) => onPresentationChange({ tint: state.tint, glow: event.target.checked }));

  function sync(next) {
    state = { ...next, look: normalizeLook(next.look) };
    const recipe = LOOKS.find((look) => look.id === state.look.id);
    const adjusted = state.look.autoContrast || state.look.midtone !== 1 || state.look.sharpness !== 0 || state.look.invert;
    for (const radio of sheet.querySelectorAll('input[name="look"]')) radio.checked = radio.value === state.look.id;
    for (const radio of sheet.querySelectorAll('input[name="framing"]')) radio.checked = radio.value === state.framing;
    byId('look-label').textContent = adjusted ? 'Adjusted' : recipe.label;
    byId('open-looks').setAttribute('aria-label', `Looks, ${recipe.label}${adjusted ? ', adjusted' : ''}`);
    byId('look-adjusted').hidden = !adjusted;
    const framing = FRAMINGS.find((frame) => frame.id === state.framing);
    byId('framing-label').textContent = framing.id === 'full' ? 'Full' : `${framing.width}:${framing.height}`;
    byId('look-auto-contrast').checked = state.look.autoContrast;
    byId('look-midtone').value = state.look.midtone;
    byId('midtone-value').textContent = `${state.look.midtone.toFixed(2)} · ${state.look.midtone < 1 ? 'Lighter' : state.look.midtone > 1 ? 'Darker' : 'Neutral'}`;
    byId('look-sharpness').value = state.look.sharpness;
    byId('sharpness-value').textContent = state.look.sharpness.toFixed(2);
    byId('look-invert').checked = state.look.invert;
    for (const radio of sheet.querySelectorAll('input[name="text-tint"]')) radio.checked = radio.value === state.tint;
    if (state.tint !== 'default') byId('text-color').value = state.tint;
    byId('soft-glow').checked = state.glow;
    for (const id of ['sigma1', 'sigma2', 'edge-threshold']) {
      byId(id).disabled = !recipe.edges;
      if (recipe.edges) byId(id).removeAttribute('aria-describedby');
      else byId(id).setAttribute('aria-describedby', 'edge-controls-help');
    }
    byId('edge-controls-help').hidden = recipe.edges;
  }
  sync(state);
  return {
    sync,
    dispose() {
      for (const remove of listeners) remove();
      if (sheet.open) sheet.close();
      for (const fieldset of [byId('look-choices'), byId('framing-choices')]) {
        for (const label of fieldset.querySelectorAll('.sheet-choice')) label.remove();
      }
    },
  };
}
