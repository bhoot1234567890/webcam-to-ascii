# 002 — Press feedback on buttons and choice cards

- **Status**: DONE (2026-11-06)
- **Commit**: 774004e
- **Severity**: MEDIUM
- **Category**: Physicality (press feedback)
- **Estimated scope**: 1 file (`frontend/src/styles.css`), ~6 lines; requires plan 001 merged first
- **Depends on**: 001 (uses `--dur-press` and `--ease-out`)

## Problem

No control in the app has press feedback: there is no `:active` rule anywhere in
`frontend/src/styles.css`. The shutter (`.shutter`, `styles.css:144`) is the
app's primary, highest-frequency control and it confirms a press only through a
border-color hover state. Every `.choice` card (look, density, framing, ink
selectors) and every rail/topbar button feels dead under the finger.

Current state: `frontend/src/styles.css:36` (after plan 001) transitions color,
border-color, background-color, box-shadow only:

```css
button, .choice, .tab, .statusline, input[type="number"], select { transition: color var(--dur-fast) ease, border-color var(--dur-fast) ease, background-color var(--dur-fast) ease, box-shadow var(--dur-fast) ease; }
```

There is no `:active` rule in the file (verify with
`grep -n ':active' frontend/src/styles.css` before starting; it must return
nothing).

## Target

Catalog values: press feedback is `transform: scale(0.97)` on `:active`,
transitioned with `transform 160ms ease-out` (subtle range 0.95-0.98; 160ms is
`--dur-press`, the strong ease-out from plan 001).

Extend the rule 36 transition list with `transform var(--dur-press) var(--ease-out)`
and add:

```css
button:active:not(:disabled), .choice:active { transform: scale(0.97); }
```

In the reduced-motion block (`frontend/src/styles.css:174-177`), press movement
is dropped entirely (scale without a transition is an instant jump: worse than
none), so append:

```css
  button:active:not(:disabled), .choice:active { transform: none; }
```

Resulting reduced-motion block:

```css
@media (prefers-reduced-motion: reduce) {
  .wordmark-cursor { animation: none; }
  button, .choice, .tab, .statusline, input[type="number"], select { transition: none; }
  button:active:not(:disabled), .choice:active { transform: none; }
}
```

## Repo conventions to follow

- Single-rule selectors on one line, grouped where they share a declaration
  (exemplar: `styles.css:50` `button:hover:not(:disabled)` guard pattern; keep
  the `:not(:disabled)` guard so disabled buttons do not scale).
- Reduced-motion overrides live only in the existing block at the end of the
  file (`styles.css:174-177`), not in a second media query.

## Steps

1. In the rule at `frontend/src/styles.css:36` (plan 001 applied), append
   `, transform var(--dur-press) var(--ease-out)` to the transition list.
2. Directly after that rule, add the two-selector `:active` rule from Target.
3. Extend the `prefers-reduced-motion` block with the `transform: none` line so
   it matches the Target block exactly.
4. Nothing else. No JS, no markup, no new files.

## Boundaries

- Do NOT add `:active` scale to `input[type="range"]`, `select`,
  `input[type="number"]`, or `input[type="color"]`: fields, not pressable
  cards. They are not `button` or `.choice`, so the selector already excludes
  them; do not "fix" them into it.
- Do NOT change hover rules, focus rings, or the shutter's state colors
  (`styles.css:144-147`).
- Watch item, not a change: `.view-controls button` (styles.css:149-150) is a
  segmented control (`border-left-width: 0`); pressing one button briefly opens
  a 1-2px gap at its right edge during the 160ms scale. That reads as press,
  not breakage. If it looks wrong in the feel check, narrow the selector to
  `button:active:not(:disabled):not(.view-controls button)` and report the
  deviation in the plan result.
- If the excerpts do not match (drift since 774004e or plan 001 not applied),
  STOP and report.

## Verification

- **Mechanical**: `cd frontend && bun run build`; `bun run test:browser` green
  (no spec asserts on `:active`).
- **Feel check** (dev server, mouse):
  - Press and hold CAMERA/STOP, a look `.choice`, a tab, CLOSE: element dips to
    97% with a fast start, springs back on release over 160ms.
  - Press the live shutter (accent-filled state, `styles.css:146`): the filled
    button scales the same way; background fill does not repaint separately.
  - Tap quickly twice on a choice: second press starts from wherever the first
    release was (CSS transitions retarget; no restart-from-1 jump).
  - DevTools Animations panel at 10% speed: scale-down and scale-up are both
    one smooth 160ms move; no layout shift anywhere (transform only; neighbors
    must not move).
  - Keyboard: Tab to a button, hold Space: `:active` shows in Chromium; Enter
    shows none (correct; keyboard users have the focus ring instead).
  - Rendering panel, `prefers-reduced-motion: reduce`: pressing does not move
    or flash; hover color changes still apply instantly.
- **Done when**: `grep -n ':active' frontend/src/styles.css` shows exactly the
  two new declarations and the reduced-motion override, and both mechanical
  commands pass.
