# 004 — Morph the FIT / 1:1 view toggle

- **Status**: REVERTED after implementation, see plans/README.md Outcome
- **Commit**: 774004e
- **Severity**: MEDIUM
- **Category**: Easing & duration (on-screen morph) / missed opportunity
- **Estimated scope**: 2 files (`frontend/src/output-view.js`, `frontend/src/styles.css`), ~12 lines; requires plan 001 merged first
- **Depends on**: 001 (uses `--dur-morph`, `--ease-in-out`)

## Problem

FIT vs 1:1 is a frequent control (`.view-controls`, `styles.css:148-151`). The
toggle teleports the artwork between fit-scale (often ~0.4-0.8x) and 1:1 in a
single frame: a jump cut on the app's main canvas.

`frontend/src/output-view.js:26` sets the scale inline on every layout pass:

```js
    output.style.transform = `scale(${scale})`;
```

`frontend/src/output-view.js:35-43`:

```js
  function select(next) {
    mode = next;
    fitButton.setAttribute('aria-pressed', String(mode === 'fit'));
    detailButton.setAttribute('aria-pressed', String(mode === 'detail'));
    viewport.dataset.view = mode;
    viewport.scrollTop = 0;
    viewport.scrollLeft = 0;
    layout();
  }
```

`layout()` also runs from a `ResizeObserver` (`output-view.js:49-50`) and from
`update()` (`output-view.js:54-62`) on every grid change (slider drags, glow
gutter, camera commits). A blanket `transition: transform` on `#ascii-output`
would rubber-band slider drags and window resizes. The morph must be gated to
the deliberate mode switch only.

Catalog: moving/morphing on screen gets `ease-in-out`
(`cubic-bezier(0.77, 0, 0.175, 1)`), 200ms, under the 300ms UI budget.

## Target

`frontend/src/styles.css`, after the `#ascii-output` rule at line 73:

```css
/* View toggle morph: class is added only around a deliberate FIT/1:1 switch so
   slider drags and resizes (same layout() path) stay instant. */
#ascii-output.view-morph { transition: transform var(--dur-morph) var(--ease-in-out); }
```

`frontend/src/output-view.js`: add a module-level timer next to the existing
`let` declarations (lines 5-9), gate the class inside `select()`, clean up in
`dispose()`.

```js
  let morphTimer = 0;
```

In `select()`, insert between the scroll reset and `layout()` (the class must
be present before the new transform is written, and the timer reset keeps the
class alive across rapid toggles):

```js
    viewport.scrollLeft = 0;
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
      clearTimeout(morphTimer);
      output.classList.remove('view-morph');
    } else {
      output.classList.add('view-morph');
      clearTimeout(morphTimer);
      morphTimer = setTimeout(() => output.classList.remove('view-morph'), 240);
    }
    layout();
```

In `dispose()` (`output-view.js:63-67`), prepend:

```js
      clearTimeout(morphTimer);
      output.classList.remove('view-morph');
```

240ms = 200ms transition + slack; the class outlives the transition slightly,
which is harmless (no transform change happens in that window outside
`select()`).

Interruptibility: CSS transitions retarget, so flipping FIT/1:1 mid-morph
reverses from the current scale; the cleared-and-reset timer keeps the
transition property armed across rapid flips.

## Repo conventions to follow

- `matchMedia` is already used in this file for viewport behavior
  (`output-view.js:5`: `let mode = matchMedia('(max-width: 1000px)').matches ? 'fit' : 'detail';`);
  the reduced-motion check follows the same inline style.
- `#ascii-output` keeps `transform-origin: top left` (`styles.css:73`): the
  grid is anchored top-left by `output-view.js:24-25`; do not change it. The
  catalog's scale-from-trigger rule is for popovers; this is a canvas zoom from
  its reading origin, which is correct as-is.

## Steps

1. In `frontend/src/styles.css`, add the `.view-morph` rule directly after the
   `#ascii-output` rule (line 73).
2. In `frontend/src/output-view.js`, add `let morphTimer = 0;` after
   `let artworkHeight = 0;` (line 9).
3. Replace the body of `select()` with the Target version (only the inserted
   if/else changes; the aria-pressed, dataset, and scroll lines stay identical).
4. Prepend the two cleanup lines in `dispose()`.
5. No other edits.

## Boundaries

- Do NOT add `width`, `height`, `left`, or `top` to any transition: stage box
  sizing (`output-view.js:22-25`) and the guides element (`:27-32`) snap
  instantly and must keep doing so.
- Do NOT touch `update()`, `layout()` internals, or the `ResizeObserver`.
- Do NOT animate on reduced motion (the branch removes the class: instant,
  exactly today's behavior).
- Do NOT change the aria-pressed/dataset/scroll-reset behavior in `select()`.
- If `output-view.js` does not match the excerpts (drift since 774004e), STOP
  and report.

## Verification

- **Mechanical**: `cd frontend && bun run build`; `bun run test:unit`; `bun run
  test:browser` (camera/image specs exercise the commit path that calls
  `update()`; they assert text and dataset, not transforms, so they must stay
  green unchanged).
- **Feel check** (dev server; upload a photo via USE A PHOTO so the grid is
  static):
  - Switch to FIT, then press 1:1: the grid zooms from its top-left anchor to
    full size over 200ms, symmetric ease (accelerate in, decelerate out); the
    stage box and scrollbars appear immediately, the content catches up.
  - Press FIT: zooms back down the same way.
  - Spam FIT / 1:1 mid-morph: direction reverses from the current scale, no
    snap to an endpoint then re-animate.
  - Open the look sheet, drag the WIDTH slider: the grid rescales instantly
    with zero lag or rubber-banding (this is the class gate doing its job).
  - Resize the window while in FIT: instant re-fit, no visible transition.
  - Rendering panel, `prefers-reduced-motion: reduce`: FIT/1:1 toggles are
    instant jumps, identical to today.
- **Done when**: the mode toggle is the only path that morphs, the morph
  retargets cleanly on spam, and all three mechanical commands pass.
