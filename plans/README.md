# Animation improvement plans

Written by the `improve-animations` audit at commit `774004e` (2026-11-06).
Scope: `frontend/` (the clack app). The root-level `terminal/`, `examples/`,
`templates/` directories belong to the embedded design-system kit, not the
product, and are out of scope.

The audit found no HIGH issues. Existing motion is color-only, interruptible,
and reduced-motion aware; the work below adds the missing feedback and
explanation layers. Every plan animates `transform`/`translate`/`opacity`
only.

| # | Plan | Severity | Depends on | Status |
|---|---|---|---|---|
| 001 | Motion tokens and easing fix (`001-motion-tokens-and-easing.md`) | LOW | - | DONE |
| 002 | Press feedback (`002-press-feedback.md`) | MEDIUM | 001 | DONE |
| 003 | Look sheet entry/exit (`003-look-sheet-entry-exit.md`) | MEDIUM | 001 | DONE (clip variant) |
| 004 | FIT/1:1 scale morph (`004-view-scale-morph.md`) | MEDIUM | 001 | REVERTED |
| 005 | First ASCII reveal (`005-first-ascii-reveal.md`) | LOW-MEDIUM | 001 | DONE |
| 006 | Error + guides fades (`006-error-and-guides-fades.md`) | LOW | 001 | DONE |

## Execution order

1. **001 first, always.** It creates the `--ease-*` and `--dur-*` tokens every
   other plan consumes.
2. **002 next.** Smallest diff, felt on every press; also the template for how
   the reduced-motion block grows.
3. **003, 004, 005, 006 in any order.** They touch different surfaces
   (sheet CSS only / view scaler + one CSS rule / commit path + one CSS rule /
   banner + overlay CSS) and do not conflict. 003 and 006 both add rules near
   each other in `styles.css` and both extend the reduced-motion block; if run
   in parallel, merge the media-block additions rather than duplicating the
   block.

After each plan: run its Verification section (mechanical commands plus the
feel check). Plans are stamped against commit `774004e`; every plan's
Boundaries section says STOP on drift rather than improvising.

## Outcome (2026-11-06)

001, 002, 003, 005, 006 are implemented and green in the browser suite. 004
was implemented, then reverted: it collides with the app's own test contract,
which measures geometry synchronously after a toggle.

- 004: `image.spec.js:117` clicks Detail and `image.spec.js:139` immediately
  reads `pre.getBoundingClientRect().width` on mobile, where the mode flips
  FIT to 1:1. Any transform transition makes that rect read mid-interpolation
  (failed by ~1300px on all three engines, mobile only). A visible zoom morph
  is incompatible with instant post-toggle geometry.
- 003 final design: the form is the layout channel. `#settings-form` is a grid
  whose `grid-template-rows` grows `0fr -> 1fr` on open, so the sheet's top
  edge pushes the stage and the ASCII re-fits every frame (ResizeObserver);
  the transition is declared only on the open state, so close restores layout
  in the same frame. Probed in Chromium, Firefox, and WebKit: the track
  interpolates and child 44px rects stay exactly 44 throughout. Transform
  remains forbidden on this chain: WebKit quantizes child rects mid-flight
  (`43.99998474121094` against the strict 44px checks).
- Tab switch: one sliding underline replaces the per-tab `box-shadow`
  crossfade. It is a leaf element whose width and translateX are written from
  the active tab's `offsetWidth`/`offsetLeft` (re-measured on resize), with a
  `--dur-fast` `--ease-in-out` transition; panel content swaps instantly, so
  there is never an empty body frame. Requested after the audit; not a
  numbered plan.
- Review refinements folded in after the parallel agents reported: press is
  asymmetric (160ms down, `--dur-fast` release), the 140ms panel literals are
  gone, and the guides fade uses `--dur-press` with `--ease-out` so the first
  frame is already visible.
- Baseline note: `mobile.spec.js:245` also failed in this working tree without
  any of these changes, earlier, at `mobile.spec.js:270` (guides delta of 8,
  the glow gutter). The final change set passes the whole suite.
