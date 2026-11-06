# 006 — Soften the error banner and guides overlay entries

- **Status**: DONE (2026-11-06)
- **Commit**: 774004e
- **Severity**: LOW
- **Category**: Missed opportunity (small teleports)
- **Estimated scope**: 1 file (`frontend/src/styles.css`), ~15 lines; requires plan 001 merged first
- **Depends on**: 001 (uses `--dur-press`, `--dur-fast`, `--ease-out`)

## Problem

Two occasional surfaces appear by hard teleport:

1. The error banner. `frontend/src/main.js:59-61` flips `hidden`; the banner
   (`styles.css:55`) pops in at full opacity and pushes the entire stage down
   in one frame.

```js
  errorText.textContent = message;
  errorLine.hidden = !message;
  byId('error-action').hidden = !photoAction;
```

```css
.errorline { display: flex; align-items: center; gap: 12px; min-height: 34px; margin: 0; padding: 4px 16px; border-bottom: 1px solid var(--error-border); background: var(--error-bg); color: var(--error-ink); font-size: 12px; overflow-wrap: anywhere; flex-shrink: 0; }
```

2. The composition grid. `frontend/src/camera-shell.js:9` sets
  `document.body.dataset.guides`; the overlay snaps between invisible and
   visible (`styles.css:74-75`):

```css
.composition-grid { position: absolute; visibility: hidden; pointer-events: none; background-image: linear-gradient(#ffffff30, #ffffff30), linear-gradient(#ffffff30, #ffffff30), linear-gradient(#ffffff30, #ffffff30), linear-gradient(#ffffff30, #ffffff30); background-size: 1px 100%, 1px 100%, 100% 1px, 100% 1px; background-position: 33.333% 0, 66.667% 0, 0 33.333%, 0 66.667%; background-repeat: no-repeat; }
body[data-guides="on"] .composition-grid { visibility: visible; }
```

`visibility` is not interpolable, so no fade is possible without switching to
`opacity`.

The layout push of the error banner cannot be animated without animating layout
properties, which the catalog forbids; the entry treatment softens the
appearance, the push stays instant, and the exit stays instant (asymmetric:
entry explains, exit snaps).

## Target

Error banner: entry-only fade with a 4px settle-down from above, 160ms
ease-out. No `display` in the transition list, so hiding stays a hard cut.

Extend `.errorline` (`frontend/src/styles.css:55`) with:

```css
.errorline { ...existing declarations...; transition: opacity var(--dur-press) var(--ease-out), translate var(--dur-press) var(--ease-out); }
@starting-style { .errorline { opacity: 0; translate: 0 -4px; } }
```

Guides overlay: switch `visibility` for `opacity` (safe: the element is
`aria-hidden` markup, `pointer-events: none`, and `output-view.js:27-32` keeps
sizing it inline whether on or off). Edit the two rules at lines 74-75 to:

```css
.composition-grid { position: absolute; opacity: 0; transition: opacity var(--dur-fast) ease; pointer-events: none; background-image: linear-gradient(#ffffff30, #ffffff30), linear-gradient(#ffffff30, #ffffff30), linear-gradient(#ffffff30, #ffffff30), linear-gradient(#ffffff30, #ffffff30); background-size: 1px 100%, 1px 100%, 100% 1px, 100% 1px; background-position: 33.333% 0, 66.667% 0, 0 33.333%, 0 66.667%; background-repeat: no-repeat; }
body[data-guides="on"] .composition-grid { opacity: 1; }
```

(Only `visibility: hidden` was replaced by `opacity: 0; transition: ...` in the
first rule and `visibility: visible` by `opacity: 1` in the second; every other
declaration is byte-identical.)

Reduced motion, appended inside the block at `frontend/src/styles.css:174-177`:

```css
  .errorline { transition: none; }
```

And immediately after that media block, at top level:

```css
@media (prefers-reduced-motion: reduce) {
  @starting-style { .errorline { opacity: 0; translate: none; } }
}
```

Net effect under reduced motion: the banner appears instantly (the red banner
itself is the signal; no movement, no fade). The guides opacity fade stays
active under reduced motion: opacity-only, no movement, and it prevents the
jarring snap the policy prefers to avoid.

## Repo conventions to follow

- Keep the one-line rule format of `styles.css`; the `@starting-style` blocks
  sit directly under the rule they style (same pattern plan 003 introduces for
  `#look-sheet`).
- Reduced-motion overrides live in the single existing block at lines 174-177
  plus the paired `@starting-style` media block; do not scatter more media
  queries.

## Steps

1. Confirm `frontend/src/styles.css:55` and `:74-75` match the excerpts.
2. Append the two new declarations to the `.errorline` rule and add the
   `@starting-style` block right after it.
3. Edit the two `.composition-grid` rules per Target (visibility to opacity,
   add the transition to the base rule only).
4. Append `.errorline { transition: none; }` inside the reduced-motion block
   and add the reduced-motion `@starting-style` block after it.
5. No JS edits: `main.js` and `camera-shell.js` stay byte-identical.

## Boundaries

- Do NOT animate the error banner's exit, height, margin, or the stage shift.
- Do NOT add `display ... allow-discrete` to `.errorline` (that would hold the
  banner on screen during hide, exactly what we avoid).
- Do NOT change `camera-shell.js`; the `data-guides` body attribute contract is
  unchanged.
- Do NOT touch `#ascii-output` or the glow rule (`styles.css:76`).
- If the excerpts do not match, STOP and report.

## Verification

- **Mechanical**: `cd frontend && bun run build`; `bun run test:browser` green
  (error paths are exercised by camera denial specs; they assert text and
  `role="alert"` content, not timing).
- **Feel check** (dev server):
  - Trigger the banner (deny camera permission in the browser prompt, or an
    unsupported device): banner fades in settling 4px downward over 160ms while
    the stage shifts instantly; the USE A PHOTO button rides in with it.
  - Dismiss (grant permission / USE A PHOTO): banner vanishes in one frame; no
    ghost, no fade-out.
  - Toggle GUIDES in the topbar repeatedly: rule-of-thirds lines fade in/out
    over 120ms, retargeting mid-flight on quick toggles; the artwork under
    them never shifts or flickers.
  - With guides ON, drag the WIDTH slider: grid lines track the artwork
    instantly (they resize via inline styles, not the transition; only opacity
    transitions).
  - Rendering panel, `prefers-reduced-motion: reduce`: banner appears
    instantly; guides still fade (opacity only).
- **Done when**: `grep -c 'visibility' frontend/src/styles.css` returns 0,
  both entries animate (or respect reduced motion as specced), and the
  mechanical commands pass.
