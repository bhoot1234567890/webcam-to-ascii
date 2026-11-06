# 001 — Add motion tokens and fix the linear easing

- **Status**: DONE (2026-11-06)
- **Commit**: 774004e
- **Severity**: LOW (foundational for plans 002-006)
- **Category**: Easing & duration / Cohesion & tokens
- **Estimated scope**: 1 file (`frontend/src/styles.css`), ~15 lines changed

## Problem

The app has exactly one transition rule and it uses `linear` easing, which the
motion catalog reserves for constant motion (marquee, progress). Hover and color
changes should use `ease`. Two state changes teleport because they are outside
the rule: the tab underline (a `box-shadow`) and the statusline accent color.
There are no motion tokens, so every duration and curve in the app is a magic
number, and plans 002-006 need shared curves.

Current code, `frontend/src/styles.css:36`:

```css
button, .choice, .tab, input[type="number"], select { transition: color 120ms linear, border-color 120ms linear, background-color 120ms linear; }
```

Tab underline teleport, `frontend/src/styles.css:87` (not covered by rule 36,
`box-shadow` is not in its property list):

```css
.tab.active, .tab[aria-selected="true"] { color: var(--ink); box-shadow: inset 0 -2px 0 var(--ink); }
```

Statusline accent teleport, `frontend/src/styles.css:38` (`.statusline` is a
`p`, not in the rule 36 selector):

```css
.statusline[data-camera="live"] { color: var(--accent); }
```

`:root` (`frontend/src/styles.css:6-24`) defines color tokens only, no motion
tokens.

## Target

Add motion tokens inside `:root`, after the color tokens (before the closing
brace on line 24):

```css
  /* motion: strong curves for deliberate motion (press, morph, drawer); plain
     `ease` stays inline for color/state hovers. Catalog values, do not tune. */
  --ease-out: cubic-bezier(0.23, 1, 0.32, 1);
  --ease-in-out: cubic-bezier(0.77, 0, 0.175, 1);
  --ease-drawer: cubic-bezier(0.32, 0.72, 0, 1);
  --dur-fast: 120ms;
  --dur-press: 160ms;
  --dur-morph: 200ms;
  --dur-drawer: 220ms;
```

Replace rule 36 with:

```css
button, .choice, .tab, .statusline, input[type="number"], select { transition: color var(--dur-fast) ease, border-color var(--dur-fast) ease, background-color var(--dur-fast) ease, box-shadow var(--dur-fast) ease; }
```

Notes the executor must not second-guess:

- Hover/color changes use the built-in `ease` on purpose. The custom
  `--ease-out` / `--ease-in-out` / `--ease-drawer` curves are for deliberate
  motion introduced by plans 002-006 (press, scale morph, drawer), where
  built-in easings are too weak. Do not swap the hovers to a custom curve.
- `.statusline` joins the selector group so the camera-live accent color eases.
  It gains border/background transitions too; it never changes those, so this
  is inert.
- `box-shadow` joins the property list so the tab underline crossfades when the
  active tab moves. Inert for elements with no box-shadow.

## Repo conventions to follow

- Tokens are CSS custom properties in the single `:root` block in
  `frontend/src/styles.css`. Exemplar: `--accent: #ffd663;`
  (`frontend/src/styles.css:19`). Add the motion tokens in the same block.
- All product motion CSS lives in `frontend/src/styles.css`. There is no second
  stylesheet and no CSS-in-JS; do not create either.
- Section comments use the `/* ————— name ————— */` banner style
  (`frontend/src/styles.css:41`).

## Steps

1. In `frontend/src/styles.css`, insert the seven motion custom properties into
   the `:root` block (lines 6-24), after `--error-bg` on line 23, with the
   comment from the Target section.
2. Replace the rule on line 36 verbatim with the Target version (selector gains
   `.statusline`, property list gains `box-shadow`, every `120ms linear`
   becomes `var(--dur-fast) ease`).
3. Change nothing else. The reduced-motion block at lines 174-177 keeps its
   exact selector list for now; plan 002 extends it.

## Boundaries

- Do NOT touch `styles.css:45-46` (cursor blink keyframes): documented brand
  mark, settled decision.
- Do NOT change durations, add new properties (no `transform` yet; that is plan
  002), or restyle anything.
- Do NOT add JavaScript, dependencies, or files.
- If lines 36, 87, or the `:root` block no longer match the excerpts above
  (drift since commit 774004e), STOP and report instead of improvising.

## Verification

- **Mechanical**: `cd frontend && bun run build` succeeds; `bun run test:browser`
  passes unchanged.
- **Feel check** (`cd frontend && bun run dev`, open http://127.0.0.1:4173):
  - Hover a `.choice` card and a topbar button: border-color change is slightly
    softer at start than before (linear had constant speed); still 120ms.
  - Open the look sheet (LOOKS in the rail), click between the Look and Detail
    tabs: the underline fades between tabs instead of jumping.
  - Start the camera (fake device is fine): the statusline turns accent-colored
    with a short ease instead of snapping.
  - DevTools Rendering panel, emulate `prefers-reduced-motion: reduce`: all
    transitions are off (instant state changes), cursor stops blinking.
- **Done when**: `grep -c 'var(--dur-fast)' frontend/src/styles.css` returns at
  least 1, `grep -c 'linear' frontend/src/styles.css` returns 0, and the build
  plus browser suite are green.
