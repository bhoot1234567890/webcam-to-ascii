# Source: Stitch project "Neon Cyberpunk Grid"

- Project: `15977922294196306755` ("Neon Cyberpunk Grid")
- Screen: `39cb35b260e34e2cbe7824f7ff3f2def` ("clack — Angular Cyberdeck Viewfinder")
- Retrieved: 2026-11-09 via Stitch MCP (`get_screen` + `download_file`) and the user's zip export.

## Files

| File | Origin | Notes |
|---|---|---|
| `clack-cyberdeck-viewfinder.html` | zip `code.html` (identical payload to MCP download) | Single-file mockup, 20.8KB. Self-contained logic; see caveats below. |
| `clack-cyberdeck-viewfinder.png` | MCP screenshot API | 780 x 1768 (original render). |
| `clack-cyberdeck-viewfinder-preview.png` | zip `screen.png` | 706 x 1600 (export pipeline scale of the same screen). |
| `DESIGN.md` | zip | Two Stitch design systems: full "Cyberdeck Tactical Terminal" token set + Space Mono type scale, and the distilled "eDEX-UI Sci-Fi Terminal Instrument" direction (cyan `#00f0ff` primary, `#ff0055` alert, `#ffd663` tertiary amber). |

## Concept summary

Cyberdeck viewfinder: obsidian casing (`#05070a`), glowing cyan phosphor (`#00f0ff`), viewfinder corner brackets + telemetry HUD (RES / AF / F-readouts), scanline overlay, `SYS.OPTIC` system header, an `ACTUATE` shutter block, mode chips (PHOTO / VIDEO / FLIP), quick chips for look / density / frame / ink, hold-to-PEEK source compare, and `AIR-GAPPED // ZERO CLOUD TELEMETRY` as the privacy line. Retains several real clack element ids (`ascii-output`, `camera-shutter`, `toggle-guides`, `compare-output`) and the amber `#ffd663` as a secondary accent.

## Implementation caveats (mockup vs product rules)

This is a design concept, not shippable code. If any of it is adopted:

1. Loads Tailwind via CDN, Google Fonts (Space Mono), and Material Symbols icons at runtime. The real app is same-origin only with locally bundled fonts (test-enforced network policy); every dependency here would need local bundling or re-expression in vanilla CSS.
2. Space Mono is not Roboto Mono. The artwork canvas is contractually Roboto Mono 400 @ 12px, 1:2 cells; chrome typography may adopt another treatment only within the redesign rules (`design.md` D3 currently says family stays Roboto Mono; changing that is an owner decision).
3. The mockup's ASCII `pre` is static placeholder art at 7.5-9.5px with its own leading; the real canvas metrics, FIT/100% behavior, and instant-toggle geometry contracts apply (`design.md` Section 9.2).
4. Cyan-on-dark glow treatments must be re-checked for WCAG AA (4.5:1 text, 3:1 indicators) and reduced-motion parity (the mockup uses `animate-pulse` / `animate-ping` loops).
5. Decorative telemetry (60FPS, F/1.8, ISO, AF lock) is fiction; the real status line narrates actual state (`design.md` Section 5.9).
