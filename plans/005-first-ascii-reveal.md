# 005 — Reveal the first ASCII render

- **Status**: DONE (2026-11-06)
- **Commit**: 774004e
- **Severity**: LOW-MEDIUM
- **Category**: Missed opportunity (rare, high-emotion moment)
- **Estimated scope**: 2 files (`frontend/src/main.js`, `frontend/src/styles.css`), ~12 lines; requires plan 001 merged first
- **Depends on**: 001 (uses `--dur-morph`, `--ease-out`)

## Problem

The app's money moment, your world rendered as characters for the first time,
appears in a single frame. The empty state vanishes and the full grid is simply
there, at final opacity, with no acknowledgment that anything began.

`frontend/src/main.js:122-125` (empty-state sync, reads the same signal):

```js
function syncEmptyState() {
  const output = byId('ascii-output');
  const empty = output.textContent === '';
  byId('empty-state').hidden = !empty;
  byId('empty-hint').hidden = !(empty && cameraState === 'requesting');
}
```

`frontend/src/main.js:194` (commit path; the render loop lands here for every
camera frame and every settings change):

```js
        output.textContent = data.text;
```

Frequency framing: this fires once per empty-to-populated transition (first
load, first camera start after a cleared state), which is the rare/delight tier
where a touch of motion is allowed.

Two deliberate constraints:

- Opacity only, no scale. `transform` on `#ascii-output` is owned by the view
  scaler (`output-view.js:26`), and scaling 12px text mid-entrance blurs it.
  A pure fade on a full-grid reveal is the right tool here, not a popover
  entrance.
- Reveal only when the grid was empty. Switching source later (live camera to
  photo) keeps today's instant swap: continuity between renders is correct;
  only emergence from nothing gets the moment.

## Target

`frontend/src/styles.css`, after the `#ascii-output` rule at line 73 (before
plan 004's `.view-morph` rule if that plan already landed; order between the
two does not matter):

```css
/* First render: one fade when the grid goes from empty to populated. A
   one-shot animation, not a transition: it is never reversed mid-flight by
   user input, so the keyframe restart hazard does not apply. */
@keyframes ascii-reveal { from { opacity: 0; } }
#ascii-output.just-revealed { animation: ascii-reveal var(--dur-morph) var(--ease-out); }
```

`frontend/src/main.js`, in the commit handler around line 194. Capture the
empty state before overwriting the text, then arm the reveal after the
dataset assignment:

```js
        const wasEmpty = output.textContent === '';
        output.textContent = data.text;
        Object.assign(output.dataset, {
          renderId: String(request.id), sourceGeneration: String(request.generation),
```

(the `Object.assign` block continues unchanged to its closing brace), then
immediately after that closing brace:

```js
        if (wasEmpty && data.text) {
          output.classList.remove('just-revealed');
          void output.offsetWidth; // restart the one-shot if a prior run left the class
          output.classList.add('just-revealed');
        }
```

Reduced motion: this animation is opacity-only, which reduced motion keeps by
design (it aids comprehension, drops no content). Do NOT add it to the
`prefers-reduced-motion` block at `frontend/src/styles.css:174-177`.

## Repo conventions to follow

- One-line rules and `/* comment above the thing */` style, as everywhere in
  `frontend/src/styles.css` (exemplar: the halo comment at
  `frontend/src/output-view.js:18` explains intent in-place; do the same).
- DOM access in `main.js` is via the module-level `output` reference already
  used at line 194; do not introduce `querySelector` calls.

## Steps

1. In `frontend/src/styles.css`, add the keyframes and the `.just-revealed`
   rule after the `#ascii-output` block.
2. In `frontend/src/main.js`, add the `const wasEmpty = ...` line directly
   above `output.textContent = data.text;` (line 194).
3. Add the `if (wasEmpty && data.text) { ... }` block directly after the
   `Object.assign(output.dataset, { ... });` statement that follows it. Match
   the surrounding indentation (it is inside the same handler, one level deeper
   than function body).
4. No other edits; in particular do not touch `syncEmptyState()`.

## Boundaries

- Do NOT add scale, translate, or per-line stagger to the reveal.
- Do NOT reveal on source switches where the grid was already populated (the
  `wasEmpty` guard is the whole point; do not widen it to
  `sourceGeneration` changes).
- Do NOT disable the animation under reduced motion (opacity-only is allowed
  to stay; state that in the PR/commit message if asked).
- Do NOT touch the empty-state hide (`main.js:123`): the overlay disappearing
  instantly while the art fades in underneath is correct layering (the overlay
  is a sibling, absolutely positioned, `styles.css:64`).
- If `main.js` around lines 188-200 no longer matches the excerpt, STOP and
  report.

## Verification

- **Mechanical**: `cd frontend && bun run build`; `bun run test:unit`; `bun run
  test:browser`. `tests/browser/ab.spec.js:168-170` screenshots use
  Playwright's `animations: 'disabled'`, which fast-forwards finite animations,
  so those must stay green unchanged. Camera specs assert `output.textContent`
  and `dataset`, which the class does not affect.
- **Feel check** (dev server):
  - Fresh load (empty state visible), press START CAMERA, allow the fake
    device: the grid fades in from transparent over 200ms with a fast start;
    it does not slide, scale, or stagger.
  - Once live, frames keep streaming with no per-frame flicker (the class is
    added once; subsequent commits with a non-empty grid never re-add it).
  - Freeze, then USE A PHOTO: the new image replaces the grid instantly (no
    reveal), by design.
  - DevTools Animations panel: exactly one `ascii-reveal` entry per emergence,
    never two stacked.
  - Rendering panel, `prefers-reduced-motion: reduce`: the first render still
    fades in over 200ms (allowed; opacity only), the cursor blink is off.
- **Done when**: exactly one fade per empty-to-populated transition, zero
  animation activity on steady-state streaming, and all mechanical commands
  pass.
