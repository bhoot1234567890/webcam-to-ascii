# Implementing the Cyberdeck UI in clack

**Scope:** translating `clack-cyberdeck-viewfinder.html` (this folder) into the real app (`frontend/`). This is the engineering plan: what changes, in what order, against which gates.
**Status: IMPLEMENTED 2026-11-09** (all phases, default decisions D-A..D-D). Verification: `test:unit` 504/504 pass; `test:browser` 134 passed with the exact 7 pre-existing baseline failures (identical set, zero test edits); `test:production` 14 passed with the exact 3 pre-existing baseline failures. Visual evidence captured at 280/428/1440 plus drawer, guides-over-artwork, and loaded-artwork states. Final palette contrast measured (text pairs 5.43-16.50, indicators 4.03-4.68). `design.md` D5 amendment recorded in its 3.4.1 log. Two post-audit refinements shipped: four-corner stage brackets (background-gradient technique) and SYS.OPTIC badge threshold 480px -> 360px.
**One-sentence translation rule:** keep every structural contract in `design.md` (ids, element types, aria, 44px targets, row order, drawer mechanism, instant FIT, pinned strings/colors) and re-express only the chrome's visual layer - tokens, surfaces, overlays, iconography, microcopy - in the cyberdeck language. The mockup is a look, not an architecture.

**Acceptance bar:** the full existing suite passes untouched - `test:unit` (needs `qa/artifacts/reference/manifest.json`, see `qa/setup.sh`), all `test:browser` projects, `test:production`. Zero test edits. A failed spec means the translation broke a contract, not that the test is wrong.

---

## 1. Owner decisions needed before code (defaults recommended)

| # | Decision | Default | Why |
|---|---|---|---|
| D-A | Chrome accent switches amber `#ffd663` -> cyan `#00f0ff`, amber demoted to secondary | **Adopt cyan as primary chrome accent; keep amber as secondary/tertiary** | The chrome accent value is test-free. Amber stays reachable as the AMBER tint (its hex is pinned as artwork data, unaffected). Note: this supersedes `design.md` D5 - record that in `design.md` when done. |
| D-B | Typeface | **Keep Roboto Mono everywhere.** Do not add Space Mono | `design.md` D3 pins the family; the mockup's own second design system lists `'Space Mono', 'Roboto Mono', monospace` - Roboto Mono is already its fallback. The cyberdeck read survives on color and structure. Adding `@fontsource/space-mono` for chrome-only is possible later without touching the artwork, if you still want it after seeing v1. |
| D-C | Scanlines + cyan glow on the artwork by default | **No. Artwork effects stay behind the SOFT GLOW / CRT toggles exactly as now.** Ambient glow goes on chrome labels only | `production.spec.js:177-183` asserts default state restores `text-shadow: none` on `#ascii-output`. Always-on art glow breaks a pinned contract and changes what PNG/video exports look like. |
| D-D | Icon set | **Inline SVG, lucide-grade, `currentColor`, injected once as a `<symbol>` sprite; icons hidden below 320px where labels must survive at 44px width** | Material Symbols via Google Fonts is a network icon font - banned by the same-origin policy and the house icon rules (`design.md` 8.4). |

Everything else in this document follows from the defaults above.

---

## 2. Palette plan (measured)

The mockup's own Tailwind config already keeps clack's neutrals (`matte #09090b`, `titanium #e8e8e2`, `brand #ffd663`, panels `#121215`/`#18181b`). So the base stays; cyan is added; the deep obsidian blues are chrome-only surface tweaks. Measured ratios (WCAG formula, on this app's surfaces):

| Pair | Ratio | AA 4.5 | UI 3.0 | Rule |
|---|---|---|---|---|
| cyan `#00f0ff` on `#09090b` / panel / panel-2 | 14.12 / 13.48 / 12.69 | pass | pass | Primary accent text, focus ring, live states |
| cyan @0.8 / @0.7 / @0.6 opacity on bg | 9.11 / 7.11 / 5.43 | pass | pass | Secondary cyan text - floor is **@0.6** |
| cyan @0.5 on bg | 4.00 | fail | pass | Indicator edges only, never text |
| cyan @0.3 on bg | 2.11 | fail | fail | Decorative only (glow washes), carries no meaning |
| amber `#ffd663` on bg | 14.26 | pass | pass | Secondary accent (selected values, cursor alt) |
| alert `#ff0055` on bg | 5.10 | pass | pass | Recording/alert family replacement candidate - or keep the existing salmon `#ffb4a3` (11.66) |
| bg `#09090b` on cyan fill | 14.12 | pass | pass | Filled cyan controls use dark text (the `--accent-ink` pattern) |
| amber on cyan fill | 1.01 | fail | fail | Never combine |
| mock muted `#71717a` on bg | 4.12 | fail | pass | Do not adopt; keep app muted `#8f8f96` (6.19) |

Token patch (`styles.css` `:root`, additive, nothing removed):

```css
  /* cyberdeck additions */
  --accent: #00f0ff;            /* was #ffd663 - value swap, see D-A */
  --accent-ink: #062126;        /* dark ink for filled cyan controls (≈bg family) */
  --amber: #ffd663;             /* demoted secondary: selected values, cursor option */
  --accent-dim: rgba(0, 240, 255, 0.6);  /* secondary cyan text floor (5.43:1) */
  --accent-edge: rgba(0, 240, 255, 0.5); /* indicator edges only (4.00:1 UI) */
  --accent-glow: rgba(0, 240, 255, 0.35);
  --hud-strip: #040608;         /* header HUD band, chrome-only */
  --hud-panel: rgba(5, 12, 22, 0.85);    /* peek overlay / floating panels, chrome-only */
```

Unchanged and untouchable: `--bg #09090b` (artwork, exports, `theme-color` - asserted as RGB `[9,9,11,255]`), default artwork ink `#e2e9e7`, the nine tint hexes, CRT chromatic red. Pinned-string review: the error family tokens are free values; if you adopt `#ff0055` for recording, keep the photo/video shutter color distinction (asserted) and `FLIP` color == `IMPORT` color (asserted).

---

## 3. Element-by-element disposition

| Mockup element | Disposition | Implementation note |
|---|---|---|
| `SYS.OPTIC` badge next to wordmark | Translate | Static badge in `.topbar`; wordmark + GUIDES geometry invariants stay (wordmark right edge <= GUIDES left; GUIDES x fixed across camera states) |
| `LIVE // 60FPS` header readout | **Replace with real state** | 60fps is fiction. Restyle the existing `#status` line as the HUD strip; its strings are pinned (`Camera live` etc.), styling and `data-camera` tinting are free. Do not invent telemetry |
| Blinking cyan cursor block | Translate | `.wordmark-cursor` swaps amber -> accent (cyan); the blink loop and its reduced-motion kill stay |
| Viewfinder corner brackets + center reticle + `AF: TGT_LOCKED` | Translate as the GUIDES overlay | `.composition-grid` visual is free; keep geometry (sized to artwork by `output-view.js`), the 160ms fade, and the guides-toggle-never-renders rule. Replace thirds lines with bracket frame + reticle; drop the fictional AF/RES/F-stop text or bind to real values only (grid `W x H` from `#grid-label`) |
| Scanlines overlay (always on) | Reject as default; keep as CRT toggle | D-C; the `data-crt` mechanism already renders scanlines + chromatic shadow, asserted by tests |
| Cyan glow on `#ascii-output` | Reject as default | D-C; `text-shadow: none` at rest is pinned. Glow rides chrome labels instead (`text-shadow: 0 0 6px var(--accent-glow)`) |
| `RAW SENSOR // DIRECT_FEED` peek overlay | Translate styling only | `#source-peek` canvas + hold behavior are wired in `main.js`; restyle the overlay label/border with `--hud-panel` + `backdrop-filter: blur(8px)` (chrome-only, acceptable perf off the render path) |
| Quick chips row (ORIG / 64x48 / FULL / CYAN) + tune button | Translate onto the existing readout tiles | The four tiles ARE these chips (`#open-looks` etc.); values come from the catalogs, `aria` labels from `look-controls.js`. Icons hidden <320px. Tile count stays exactly 4, uniform heights, values never overflow (all asserted) |
| PHOTO / VIDEO / FLIP cluster | Translate | Keep real ids/aria-pressed; FLIP keeps `aria-label="Switch front and back camera"` and color-equals-IMPORT |
| `ACTUATE` shutter block | Translate visually; keep functional labels | `camera-shell.js` writes `querySelector('span').textContent` - the **first span** is the label slot; keep the CAMERA/FREEZE/RESUME/... text and pinned aria-labels. Decorative micro-label (ACTUATE) must be a `::after` or `<small>`, never an extra leading `<span>`. 58px min target is fine (>=44 asserted) |
| Self-timer `DELAY 0s/3s/7s/10s` | **Reject (product feature)** | Does not exist in the app; wiring a timer is recorder/capture work, out of UI scope. Drop the button |
| `PEEK` hold-to-compare | Translate label/icon | Behavior (pointer/Space hold, `aria-pressed`) is wired; only presentation changes |
| `IMPORT // DIR`, `COPY // BUFF`, `SAVE // DUMP` | Translate cautiously | Titles/aria are pinned (`Choose photo from this device`); visible labels are restylable but `COPY`/`SAVE` button text is asserted by selectors - keep the words, add the `//` suffixes only if specs that locate buttons by text are checked first (they locate by id - safe). Keep it minimal: icon + word |
| `AIR-GAPPED // ZERO CLOUD TELEMETRY` | Translate | The pinned privacy line `nothing leaves this device` must remain verbatim; the cyberdeck line can ride as a `title`/decorative secondary or replace the free decorative slot - keep the pinned string visible |
| Mirror-flip (`scaleX(-1)` on the art) | Reject | Real FLIP ends the stream and restarts with the other lens; mirroring the pre is a hack that corrupts preview-vs-export parity |
| Modal/drawer surfaces | Extend language | Mockup has none: apply cyan hairlines (`--accent-edge`), `--hud-panel` fills, bracket corners to `#look-sheet`, tabs (underline mechanism unchanged), choice chips, review dialog |
| Material Symbols icons | Replace | Inline SVG sprite (D-D): grid_on->`grid-4x4`/lucide `grid-2x2`, photo_camera, videocam, flip, timer (dropped), visibility, folder_open, content_copy, save, tune, water_drop, verified_user |
| Space Mono | Reject (default) | D-B |
| Tailwind CDN | Reject | Hand-CSS in `styles.css`; translate utility classes to the existing token/block structure |

---

## 4. Surface work order (each phase ends at a gate)

| Phase | Work | Files | Gate |
|---|---|---|---|
| P1 | Tokens + focus/disabled/selection restyle pass (cyan accent, glow text-shadows on chrome, borders) | `styles.css` | `test:browser --grep image` green; screenshot 428px |
| P2 | Topbar (SYS badge, HUD strip) + status line + error banner + empty state (`NO SIGNAL` cyberdeck boot art; pinned strings kept) | `styles.css`, `index.html` | mobile specs green; wordmark/GUIDES geometry intact |
| P3 | Rail -> control deck: restyle all five rows + meta line; icons sprite; 44px + seam checks | `styles.css`, `index.html`, `look-controls.js` (icon hooks in tile/chip builders) | `camera.spec.js` layout block (356x784/375x667 seams, row order) + `mobile.spec.js` targets green |
| P4 | Drawer + review dialog surfaces (panels, chips, sliders, numbers, presets list, dialog chrome) | `styles.css`, `index.html` | image.spec tabs/dialog/focus-trap + recording review specs green |
| P5 | Stage overlays: guides brackets/reticle, peek overlay, pannable edge fades tint | `styles.css`, `index.html` | guides/peek/CRT specs green; `renderId` never bumps (presentation-only) |
| P6 | Motion: pulse/ping loops as tokens, reduced-motion parity for every addition | `styles.css` | manual `prefers-reduced-motion` pass; no console errors |
| P7 | Full acceptance: `test:unit`, all `test:browser` projects, `test:production`; screenshots at 280/320/428/832x378/1440; re-measure contrast of final palette | - | All green, zero test edits |

Scale: P3 and P4 are the bulk (the rail and drawer carry most asserted geometry); P1/P2/P5 are small; P6/P7 are passes, not builds.

---

## 5. Known traps (each has bitten this app before)

1. **No transforms on the drawer chain or FIT toggle** - WebKit quantizes descendant rects mid-flight and the strict 44px checks read `43.99998...`. Transitions on `grid-template-rows` (open-only) and instant close stay exactly as-is.
2. **First-span label slot in the shutter** - `camera-shell.js` overwrites the first `span` inside `#camera-shutter`. Decorative text must not occupy it.
3. **`text-shadow: none` at rest on `#ascii-output`** - pinned; chrome-only glow.
4. **Seam alignment at 356x784** - `.rail-output`, `.view-controls`, `.readout-tiles` share a right edge within 0.1px; keep the shared 4-track grid, change only its skin.
5. **280px tile budget** - icons drop out below 320px; label text has 44px inner width (the current 9px/`letter-spacing: 0` floor exists for this).
6. **Infinite loops** - `animate-pulse`/`ping` equivalents must stop under `prefers-reduced-motion`; keep them opacity-only, never on layout or box-shadow.
7. **`--ascii-tint` is the only JS-written custom property** - keep it that way; new state hooks use the existing `data-*` attributes (several are already written and unconsumed: `data-view`, `data-capture-mode`, `data-recording-state` - free styling hooks for the deck's state language).
8. **Backdrop-filter** - allowed on the peek overlay/dialog chrome, never inside the per-frame render path (the performance gate watches cadence and latency).

---

## 6. What this does not cover (flagged, not forgotten)

- Self-timer (product feature), any new telemetry (fictional in the mockup), a second typeface (D-B revisit), ambient scanlines on the artwork (D-C revisit), replacing visible shutter labels (`ACTUATE` as the functional label would touch `camera-shell.js` and the label assertions - decorative only in v1).
- `design.md` D5/D3 amendments should be recorded in `design.md` itself the day the decisions are made, so the two documents do not drift.
