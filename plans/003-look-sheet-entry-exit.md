# 003 — Look sheet entry and exit motion

- **Status**: DONE (grid-track push, see Amendment)
- **Commit**: 774004e
- **Severity**: MEDIUM
- **Category**: Physicality & origin / missed opportunity (modal drawer)
- **Estimated scope**: 1 file (`frontend/src/styles.css`), ~20 lines; pure CSS, zero JS; requires plan 001 merged first
- **Depends on**: 001 (uses `--dur-drawer`, `--dur-fast`, `--ease-drawer`, `--ease-out`)

## Problem

The look sheet is the app's only modal surface (`role="dialog"`,
`aria-modal="true"`, a bottom drawer in normal flow between the stage and the
rail). It teleports: shown and hidden by flipping the `hidden` attribute.

`frontend/src/look-controls.js:72-78` (do not edit):

```js
  function close() {
    sheet.hidden = true;
    opener?.focus({ preventScroll: true });
  }
  function open(event, panelId, focusSelector) {
    opener = event.currentTarget;
    sheet.hidden = false;
```

`frontend/src/styles.css:27` and `:79`:

```css
[hidden] { display: none !important; }
...
.drawer { display: flex; flex-direction: column; min-height: 0; flex-shrink: 1; border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); background: var(--panel); }
```

Nothing explains where the sheet came from: one frame it does not exist, the
next it occupies up to 44svh (`styles.css:90`). Drawers are occasional-use UI
and get the standard drawer treatment.

Rejected alternative (do not implement): a full `translateY(100%)` slide-in
from the bottom edge. The sheet is in flow, and the stage resize happens on the
first frame the sheet un-hides, so a full-height slide would show an empty gap
above while the sheet crossed the rail. A short settle-up keeps it attached to
its final place.

## Target

Entry: settle up 12px with a fade over 220ms on the drawer curve. Exit: snap
back down over 140ms ease-out (asymmetric: the user asked to leave, the
response is fast). Implemented entirely with `@starting-style` and
`allow-discrete`, no JS changes.

Add after the `.drawer` rule (`frontend/src/styles.css:79`):

```css
/* Look sheet: entry settles up on the drawer curve; exit is a fast fade-down.
   [hidden] keeps display:none; allow-discrete holds the box only while the
   exit transition runs. */
#look-sheet {
  transition: opacity var(--dur-drawer) var(--ease-drawer), translate var(--dur-drawer) var(--ease-drawer), display var(--dur-drawer) allow-discrete;
}
#look-sheet[hidden] {
  opacity: 0;
  translate: 0 12px;
  transition: opacity 140ms var(--ease-out), translate 140ms var(--ease-out), display 140ms allow-discrete;
}
@starting-style {
  #look-sheet { opacity: 0; translate: 0 12px; }
}
```

Why the second transition on `[hidden]`: transitions take their timing from the
after-change state, so the entry (un-hide) reads the 220ms drawer curve and the
exit (hide) reads the 140ms override.

Reduced motion: fade only, no movement, shorter. Inside the existing
`prefers-reduced-motion` block (`frontend/src/styles.css:174-177`), append:

```css
  #look-sheet { transition: opacity var(--dur-fast) ease, display var(--dur-fast) allow-discrete; }
  #look-sheet[hidden] { transition: opacity var(--dur-fast) ease, display var(--dur-fast) allow-discrete; }
```

And after the media block (top level, so it always wins the starting style):

```css
@media (prefers-reduced-motion: reduce) {
  @starting-style { #look-sheet { opacity: 0; translate: none; } }
}
```

The nested `@starting-style` inside the media query overrides the top-level one
for reduced-motion users: no 12px movement, a 120ms opacity fade remains
(the sheet still explains its own appearance without motion).

## Repo conventions to follow

- ID selectors for one-off elements match existing usage (`#close-look-sheet`,
  `#error-action` in `styles.css:56,88`); this is the only look sheet.
- The reduced-motion block is the single place motion gets disabled
  (`styles.css:174-177`); extend it, do not create a second one elsewhere.
- `translate` (the independent transform property) is used because `#look-sheet`
  has no `transform` today and never will; keep it that way.

## Steps

1. Read `frontend/src/styles.css:79` and confirm the `.drawer` rule matches the
   excerpt above.
2. Insert the `#look-sheet` / `#look-sheet[hidden]` / `@starting-style` block
   from Target directly after the `.drawer` rule.
3. Append the two `#look-sheet` lines inside the `prefers-reduced-motion` block.
4. Add the reduced-motion `@starting-style` media block after it.
5. No edits to `look-controls.js`, `index.html`, or any other file.

## Boundaries

- Do NOT touch `look-controls.js`: focus handling, the Tab wrap, Escape, and
  outside-click close stay byte-identical. This plan is CSS-only.
- Do NOT convert the sheet to `position: fixed` or an overlay; it stays in flow.
- Do NOT animate the stage shrinking (the resize above the sheet stays instant;
  animating it means animating layout properties, which the catalog forbids).
- Do NOT add a backdrop/scrim, scale, or opacity to `.stage` or `.rail`.
- If `look-controls.js:72-78` or `styles.css:79` no longer match, STOP and
  report.

## Verification

- **Mechanical**: `cd frontend && bun run build`; `bun run test:browser` green.
  `tests/browser/image.spec.js:512-516` asserts Escape hides the sheet and
  focus returns to the opener; Playwright's `toBeHidden()` polls through the
  140ms exit, so it must still pass without modification. If that test times
  out, the exit duration is fighting the poll: report, do not relax the test.
- **Feel check** (dev server):
  - Click LOOKS in the rail: sheet fades in while settling up 12px, decelerating
    the whole way (drawer curve: fast start, long tail). 220ms total; no bounce.
  - Press Escape, click the stage, or CLOSE: sheet drops 12px and fades over
    140ms, clearly faster than the entry.
  - Spam LOOKS / Escape quickly: the sheet reverses from wherever it is
    (transitions retarget); it never restarts from opacity 0 while fully open,
    and never flashes fully opaque while closing.
  - Focus moves into the sheet on open and back to the opener on close
    (existing behavior, unchanged).
  - Animations panel at 10%: entry curve visibly front-loaded; exit visibly
    shorter.
  - Rendering panel, `prefers-reduced-motion: reduce`: sheet appears with a
    120ms fade, no upward movement; exit instant or 120ms fade.
- **Done when**: open and close each show one continuous opacity+translate move
  (or fade-only under reduced motion), the browser suite is green, and
  `git diff --stat` shows only `frontend/src/styles.css` changed.

## Amendment (2026-11-06, post-implementation)

The translate-based entry this plan specified was rejected: while a transform
transitions on the sheet, WebKit quantizes descendant rects on the Core
Animation 16.16 grid (`43.99998474121094` vs 44), failing the strict target
checks. A clip-path variant was shipped next, then also replaced: the clip
wipe ran inside space the stage had already paid for, so the push (the
expensive part) never animated, and the reveal ran bottom-up with the tab bar
arriving last.

Shipped instead, zero JS:

```css
#settings-form { display: grid; grid-template-rows: 0fr; min-height: 0; }
#settings-form:has(#look-sheet:not([hidden])) {
  grid-template-rows: 1fr;
  transition: grid-template-rows var(--dur-drawer) var(--ease-drawer);
}
#look-sheet { min-height: 0; overflow: hidden; }
#look-sheet > .drawer-head { flex-shrink: 0; }
```

The grid track is the layout push: the stage's ResizeObserver re-fits the
ASCII every frame, so the sheet's top edge drives the stage's bottom edge and
content is revealed in reading order (header first, then body). The transition
lives on the open state only, so the close path has none and layout is
restored in the same frame (the `mobile.spec.js:355` contract). Probes in
Chromium, Firefox, and WebKit all interpolate the track, and child 44px rects
stay exact.

Tab panels gained a sliding underline instead of a crossfade: one leaf element
positioned from the active tab's `offsetWidth`/`offsetLeft`, transitioned on
`--dur-fast` and `--ease-in-out`. Panel content swaps instantly, so there is
never an empty body frame (`plans/README.md` Outcome).
