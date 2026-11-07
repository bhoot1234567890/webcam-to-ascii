import { LOOKS, normalizeLook } from './looks.js';
import { FRAMINGS } from './framing.js';
import { DENSITIES, findDensity, findDensityBySettings } from './density.js';

const TINT_LABELS = {
  default: 'Paper', '#ffffff': 'White', '#ffd663': 'Amber', '#9be5a2': 'Green',
  '#3aff81': 'Phosphor', '#6ce4ff': 'Ice', '#ffe14d': 'Yellow', '#ff5c5c': 'Red', '#ff6bd6': 'Magenta',
};
const FIELD_HELP = { sigma1: 'sigma1-help', sigma2: 'sigma2-help', 'edge-threshold': 'edge-threshold-help' };

/** Settings drawer. Only the owner (main.js) commits processing and presentation state. */
export function createLookControls({
  onLookChange, onFramingChange, onPresentationChange, onDensityChange,
  onSavePreset, onApplyPreset, onDeletePreset,
}) {
  const byId = (id) => document.getElementById(id);
  const sheet = byId('look-sheet');
  const listeners = [];
  let opener = null;
  let disabled = false;
  let state = { look: normalizeLook(), framing: 'full', tint: 'default', glow: false, crt: false, density: 'balanced' };

  const listen = (node, event, handler) => {
    node.addEventListener(event, handler);
    listeners.push(() => node.removeEventListener(event, handler));
  };

  function addChoices(fieldset, name, choices) {
    for (const choice of choices) {
      const label = document.createElement('label');
      label.className = 'choice';
      const radio = document.createElement('input');
      radio.type = 'radio';
      radio.name = name;
      radio.value = choice.id;
      const text = document.createElement('span');
      text.textContent = choice.label;
      label.append(radio, text);
      if (choice.description) {
        const note = document.createElement('small');
        note.className = 'choice-note';
        note.textContent = choice.description;
        label.append(note);
      }
      fieldset.append(label);
    }
  }
  addChoices(byId('look-choices'), 'look', LOOKS);
  addChoices(byId('framing-choices'), 'framing', FRAMINGS);
  if (byId('density-choices')) addChoices(byId('density-choices'), 'density', DENSITIES);

  const underline = sheet.querySelector('.tab-underline');
  const tabsBar = sheet.querySelector('.tabs');
  function positionUnderline(tab, animate) {
    if (!underline || !tab || sheet.hidden) return;
    if (!animate) underline.style.transition = 'none';
    underline.style.width = `${tab.offsetWidth}px`;
    underline.style.transform = `translateX(${tab.offsetLeft}px)`;
    if (!animate) {
      void underline.offsetWidth;
      underline.style.transition = '';
    }
  }
  function switchTab(panelId, animate = true) {
    let activeTab = null;
    for (const tab of sheet.querySelectorAll('.tab')) {
      const active = tab.getAttribute('aria-controls') === panelId;
      tab.classList.toggle('active', active);
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
      if (active) activeTab = tab;
    }
    for (const panel of sheet.querySelectorAll('.panel')) {
      panel.hidden = panel.id !== panelId;
    }
    positionUnderline(activeTab, animate);
  }
  const tabsObserver = new ResizeObserver(() => {
    positionUnderline(sheet.querySelector('.tab[aria-selected="true"]'), false);
  });
  tabsObserver.observe(tabsBar);
  for (const tab of sheet.querySelectorAll('.tab')) {
    listen(tab, 'click', () => switchTab(tab.getAttribute('aria-controls')));
  }
  // Automatic activation: an arrow key moves focus and selects together.
  listen(sheet.querySelector('.tabs'), 'keydown', (event) => {
    const keys = { ArrowRight: 1, ArrowLeft: -1, Home: 'first', End: 'last' };
    if (!(event.key in keys)) return;
    event.preventDefault();
    const tabs = [...sheet.querySelectorAll('.tab')];
    const index = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1
      : (tabs.indexOf(document.activeElement) + keys[event.key] + tabs.length) % tabs.length;
    tabs[index].focus();
    switchTab(tabs[index].getAttribute('aria-controls'));
  });

  function close() {
    sheet.hidden = true;
    opener?.focus({ preventScroll: true });
  }
  function open(event, panelId, focusSelector) {
    opener = event.currentTarget;
    const wasHidden = sheet.hidden;
    sheet.hidden = false;
    switchTab(panelId, !wasHidden);
    const checked = sheet.querySelector(`${focusSelector}:checked`) || sheet.querySelector(focusSelector);
    checked?.focus({ preventScroll: true });
  }
  listen(byId('open-framing'), 'click', (event) => open(event, 'panel-framing', 'input[name="framing"]'));
  listen(byId('open-looks'), 'click', (event) => open(event, 'panel-looks', 'input[name="look"]'));
  if (byId('open-grid')) listen(byId('open-grid'), 'click', (event) => open(event, 'panel-grid', 'input[name="density"]'));
  listen(byId('open-ink'), 'click', (event) => open(event, 'panel-style', 'input[name="text-tint"]'));
  listen(byId('close-look-sheet'), 'click', close);
  listen(document, 'keydown', (event) => {
    if (!sheet.hidden && event.key === 'Escape') close();
  });
  // The sheet is modal: a Tab at either end wraps inside it.
  listen(sheet, 'keydown', (event) => {
    if (event.key !== 'Tab' || sheet.hidden) return;
    const items = [...sheet.querySelectorAll('button, input, select')]
      .filter((element) => !element.disabled && element.offsetParent !== null);
    if (!items.length) return;
    const first = items[0];
    const last = items[items.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  });
  // The rail stays interactive while tuning; a tap on the dimmed stage or top bar closes.
  listen(document, 'click', (event) => {
    if (sheet.hidden || sheet.contains(event.target) || event.target.closest('.rail')) return;
    if (event.target.closest('.stage, .topbar')) close();
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
  if (byId('density-choices')) {
    listen(byId('density-choices'), 'change', (event) => {
      if (event.target.name === 'density') onDensityChange?.(event.target.value);
    });
  }
  const widthSlider = byId('grid-max-width');
  const heightSlider = byId('grid-max-height');
  if (widthSlider) {
    listen(widthSlider, 'input', (event) => {
      const val = event.target.valueAsNumber;
      const out = byId('grid-max-width-value');
      if (out) out.textContent = `${val} chars`;
      onDensityChange?.({ maxWidthChars: val, maxHeightLines: heightSlider ? heightSlider.valueAsNumber : 65 });
    });
  }
  if (heightSlider) {
    listen(heightSlider, 'input', (event) => {
      const val = event.target.valueAsNumber;
      const out = byId('grid-max-height-value');
      if (out) out.textContent = `${val} lines`;
      onDensityChange?.({ maxWidthChars: widthSlider ? widthSlider.valueAsNumber : 50, maxHeightLines: val });
    });
  }
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
    if (event.target.name === 'text-tint') onPresentationChange({ tint: event.target.value, glow: state.glow, crt: state.crt });
  });
  listen(byId('text-color'), 'input', (event) => onPresentationChange({ tint: event.target.value, glow: state.glow, crt: state.crt }));
  listen(byId('soft-glow'), 'change', (event) => onPresentationChange({ tint: state.tint, glow: event.target.checked, crt: state.crt }));
  listen(byId('crt-effect'), 'change', (event) => onPresentationChange({ tint: state.tint, glow: state.glow, crt: event.target.checked }));

  // Presets: rebuilt only when the saved list changes, so slider input never steals list focus.
  function syncPresets(presets) {
    const list = byId('preset-list');
    list.replaceChildren();
    byId('preset-empty').hidden = presets.length > 0;
    for (const preset of presets) {
      const item = document.createElement('li');
      const name = document.createElement('span');
      name.className = 'preset-name';
      name.textContent = preset.name;
      const apply = document.createElement('button');
      apply.type = 'button';
      apply.textContent = 'APPLY';
      apply.disabled = disabled;
      apply.dataset.presetApply = preset.name;
      apply.setAttribute('aria-label', `Apply preset ${preset.name}`);
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.textContent = 'DELETE';
      remove.disabled = disabled;
      remove.dataset.presetDelete = preset.name;
      remove.setAttribute('aria-label', `Delete preset ${preset.name}`);
      item.append(name, apply, remove);
      list.append(item);
    }
  }

  function savePresetFromInput() {
    const input = byId('preset-name');
    const error = byId('preset-error');
    const result = onSavePreset(input.value);
    if (result.ok) {
      error.hidden = true;
      input.removeAttribute('aria-invalid');
      input.value = '';
    } else {
      error.textContent = result.message;
      error.hidden = false;
      input.setAttribute('aria-invalid', 'true');
      input.focus();
    }
  }

  listen(byId('preset-list'), 'click', (event) => {
    const apply = event.target.closest('button[data-preset-apply]');
    if (apply) {
      onApplyPreset(apply.dataset.presetApply);
      return;
    }
    const remove = event.target.closest('button[data-preset-delete]');
    if (remove) {
      onDeletePreset(remove.dataset.presetDelete);
      // Rebuilding the list detaches the pressed button; keep focus inside the modal.
      byId('preset-name').focus();
    }
  });
  listen(byId('preset-save'), 'click', savePresetFromInput);
  listen(byId('preset-name'), 'keydown', (event) => {
    if (event.key !== 'Enter') return;
    event.preventDefault();
    savePresetFromInput();
  });

  function setDisabled(next) {
    if (disabled === next) return;
    disabled = next;
    const edges = LOOKS.find((recipe) => recipe.id === state.look.id).edges;
    for (const control of sheet.querySelectorAll('button, input, select')) {
      if (control.id === 'close-look-sheet') continue;
      control.disabled = disabled || (control.id === 'crt-effect' && state.crtUnsupported === true) ||
        (Object.hasOwn(FIELD_HELP, control.id) && !edges);
    }
  }

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
    byId('crt-effect').checked = state.crt === true;
    byId('crt-effect').disabled = disabled || state.crtUnsupported === true;
    const inkLabel = TINT_LABELS[state.tint] ?? String(state.tint).toUpperCase();
    byId('ink-label').textContent = inkLabel;
    byId('open-ink').setAttribute('aria-label', `Ink, ${inkLabel}`);
    byId('text-color').closest('.ink-row').classList.toggle('custom-selected', !Object.hasOwn(TINT_LABELS, state.tint));
    for (const id of Object.keys(FIELD_HELP)) {
      byId(id).disabled = disabled || !recipe.edges;
      byId(id).setAttribute('aria-describedby', recipe.edges ? FIELD_HELP[id] : 'edge-controls-help');
    }
    byId('edge-controls-help').hidden = recipe.edges;

    const matchedDensity = (typeof state.density === 'string' ? findDensity(state.density) : null) ||
      findDensityBySettings(state.settings);
    const densityId = matchedDensity ? matchedDensity.id : null;
    const densityLabel = matchedDensity ? matchedDensity.label : 'Custom';
    const gridLabel = byId('grid-density-label');
    if (gridLabel) gridLabel.textContent = densityLabel;
    const openGridBtn = byId('open-grid');
    if (openGridBtn) openGridBtn.setAttribute('aria-label', `Detail, ${densityLabel}`);
    for (const radio of sheet.querySelectorAll('input[name="density"]')) {
      radio.checked = densityId ? radio.value === densityId : false;
    }
    if (next.settings) {
      if (widthSlider) {
        widthSlider.value = next.settings.maxWidthChars;
        const out = byId('grid-max-width-value');
        if (out) out.textContent = `${next.settings.maxWidthChars} chars`;
      }
      if (heightSlider) {
        heightSlider.value = next.settings.maxHeightLines;
        const out = byId('grid-max-height-value');
        if (out) out.textContent = `${next.settings.maxHeightLines} lines`;
      }
    }
  }
  sync(state);
  return {
    sync,
    syncPresets,
    close,
    setDisabled,
    dispose() {
      tabsObserver.disconnect();
      sheet.hidden = true;
      byId('preset-list').replaceChildren();
      for (const fieldset of [byId('look-choices'), byId('framing-choices'), byId('density-choices')].filter(Boolean)) {
        for (const label of fieldset.querySelectorAll('.choice')) label.remove();
      }
    },
  };
}
