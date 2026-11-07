# QA evidence archive

Frozen verification records for clack: historical gate counts, physical-camera baselines, vision-review adjudications, performance samples, and artifact inventories. These are point-in-time records, not living documentation - current commands live in the README's QA section, and `AGENTS.md` holds the contributor contract. Verbatim as recorded; nothing here has been rewritten to look better in hindsight.

**Renderer and lifecycle baseline, before the clack branding cutover:** `bash qa/run-tests.sh` passed on Linux x64 with the locked dependencies:

| Gate | Observed result |
|---|---|
| Frozen Python regressions | 13 passed; all 16 original full-image outputs and 114 component cases unchanged |
| Bun numerical and resize tests | 387 passed; common-gray text exact; maximum Gaussian error `6.67e-16`; zero quantization exceptions |
| Pillow RGB resize oracle | 51 cases, maximum channel difference `0` |
| Browser acceptance | 70 passed; Chromium, Firefox, WebKit, iPhone emulation, shutter/capture/retake, cancellation, camera switching, privacy, and A/B evidence |
| Fresh static-production smoke | 4 passed; desktop/mobile Studio plus iPhone camera-library import, Fit/Detail, unobstructed controls, rotation and delayed-CSS startup |
| Visual inspection | 10 native-camera iPhone screenshots directly reviewed and image-judged; final camera/layout judgments all pass, with two artwork flags directly adjudicated |

**Branding verification:** a fresh `bun run --cwd frontend test:production` passed all four desktop/mobile production checks after the identity cutover. Direct browser interaction and screenshot inspection covered Camera and Studio, iPhone 14 Plus emulation, and the 280px minimum-width camera. Review caught a toolbar-label overlap at 280px; the corrected label now has a permanent geometry regression check. Layout switching retained identical source identity and ASCII text, all three icon URLs decoded successfully with HTTP 200, and no browser console errors were observed. These are emulated mobile checks, not physical iPhone testing.

**Integrated video verification:** `bash qa/run-tests.sh` passed **13 Python, 462 Bun, 132 browser and 17 fresh-production checks**, plus strict validation of 20 A/B cases and 20 fixed crop triplets after merging current `main` and implementing the adopted issue #3 recommendations.

Native-browser checks record real public-fixture streams through the worker and recorder, decode changing tinted frames, download the actual files, and inspect codec/container/dimensions/audio tracks with `ffprobe`. Coverage includes AVC MP4, native Firefox/VP8 WebM fallback, static hold duration, manually requested frames, latest-frame cancellation, shared PNG/video artwork, the full duration limit, pending-start cancellation, fresh-configuration gating, restored density/presets and locked controls, source rotation, encoder failure/recovery, visibility finalization, URL release, share cancellation, and camera release/focus restoration on review. Successful video output is never mocked.

An earlier approved physical C930e check on the original feature branch recorded a five-second silent AVC clip using Photo look and story framing. The 640×480 source committed 150 live ASCII frames during the measured interval; its file played as 718×1280 video, about 5.01 seconds and 855,218 bytes. Camera and canvas-capture tracks ended after Stop. Discard & retake released the clip; stopping the new preview ended the second camera stream. Permission was revoked, and no private clip or camera pixels were sent to image review. This is desktop Chromium evidence, not a post-integration hardware rerun or an iPhone encoding/sharing benchmark.

Video evidence is kept under `qa/artifacts/video/`, including native-encoded public clips, production screenshots, initial failure evidence, capability findings, and physical lifecycle metadata. QA caught and corrected an unavailable-browser caption reducing the narrow viewfinder, capture continuing behind a reopened saved-video review, and a stale-configuration recording start stalling an already-live camera. These behaviors have regression coverage.

Current stage/rail/drawer smoke evidence is under `qa/artifacts/pr-video/`. Direct browser interaction recorded and played a silent AVC MP4 at the retained Balanced default, verified configuration locks and ended tracks, and hit-tested the active drawer tab at 280×568 with a 140px artwork pane and no horizontal document overflow. Screenshots use a public fixture and native DPR 3. Integration fixed a collapsed narrow drawer, capability notices shifting the viewfinder, and focus targeting a disabled recording control. The strict A/B profile validator now matches upstream's already-updated 50×65 default exporter; schema/hash/source validation and frozen answers are unchanged.

Issue #3 evidence is under `qa/artifacts/issue3/`: native encoder timing probes, a directly inspected iPhone-sized review screenshot, and same-source preview cadence measurements. The new regressions exercise real Chromium/Firefox encoders and decoded files, including static and hidden-visibility finalization, cancellation of obsolete pending artwork, and PNG/video artwork geometry, background, tint and glow. The hidden-visibility regression dispatches the visibility lifecycle; it is not proof of actual OS-background rendering. QA also caught missing focus restoration after asynchronous encoding; review now receives its initiating control explicitly. Native mobile smoke confirmed shutter focus after automatic review, Review-button focus after reopening, ended tracks and zero horizontal overflow.

The cadence sample used a synthetic 320×240 camera requesting 30 fps, Photo look, a 428×745 DPR 3 viewport and the development server on this Linux desktop host. Each phase lasted about 2.2 seconds; recording kept the same camera generation and made no second camera request. These are observed preview commits, not phone benchmarks or guarantees about every encoded frame:

| Browser / density | Preview fps before | Preview fps recording | Commit interval p95 before / recording (ms) |
|---|---:|---:|---:|
| Chromium 158 / Balanced (100×37) | 30.4 | 30.4 | 33.7 / 34.4 |
| Chromium 158 / Detailed (300×112) | 30.4 | 30.0 | 34.1 / 36.8 |
| Firefox 159 / Balanced (100×37) | 16.3 | 16.3 | 68 / 69 |
| Firefox 159 / Detailed (300×112) | 16.4 | 15.9 | 69 / 69 |

**Camera-feature verification:** `bash qa/run-tests.sh` passed **13 Python, 417 Bun, 82 browser and 6 fresh-production checks**, including unchanged frozen Original answers, 20 legacy A/B cases, center landmarks, neutral-look restoration, tiny-crop rejection, held worker replies, capture/reframing, shared-sheet focus, tint consistency and glow bounds. Native-DPR iPhone 14 Plus screenshots were inspected in the production app; WebKit production coverage includes 280/320/428px, landscape, complete Fit and Detail's last glyph. This remains desktop emulation, not physical iPhone/Safari validation.

Same-host five-second production measurements used the same deterministic **320×240, 15 fps** camera source, **300×112** grid and **428×745, DPR 3** Chromium viewport. These are observed delivery/worker measurements, not phone benchmarks:

| Case | Delivered fps | Worker p50 / p95 (ms) |
|---|---:|---:|
| Pre-feature Original baseline | 15.00 | 3.3 / 3.5 |
| Original, glow off | 15.00 | 3.4 / 3.6 |
| Original, glow on | 15.00 | 3.3 / 4.3 |
| Photo | 15.00 | 1.0 / 1.2 |
| Print | 15.00 | 0.9 / 1.1 |
| Outline | 15.00 | 3.4 / 3.6 |

All six feature measurement cases (including auto contrast) had zero sampled animation intervals over 50ms and Stop completed in 31–33ms, ending video tracks and preventing later commits. A public low-contrast scene with a toggling bright corner exposed approximately 50 luminance units of global auto-contrast pumping in an otherwise unchanged region; the control remains opt-in and explicitly warns about this risk.

The approved physical **Logitech C930e** check acquired 640×480 video, delivered changing frames, preserved a full-size captured PNG and retuned it from Photo/story to Print/wide. Retake and Flip acquired new source generations; Stop ended all three retained video streams and prevented further commits. Original/Full delivered 151 frames in five seconds (worker p50/p95 3.0/3.3ms); Photo/story delivered 150 (1.2/1.4ms, a different 224×200 grid). One physical camera cannot prove distinct front/back lens selection. Camera-only permission was revoked; no private frames were submitted to image review.

Feature evidence lives in `qa/artifacts/camera-features/`: production screenshots and state hashes, WebKit geometry, same-host performance samples, physical lifecycle metadata, lighting/pumping evidence and unmodified independent vision judgments. All 15 independent usability selections passed; the separate purpose question flagged Photo and several crop/tint/baseline cases as weak. Direct native-glyph review retained Photo for its contour-free photographic shading and Print for its regular pattern, not as universal improvements. Center cropping intentionally removes outer regions; Full/Original restoration is text-exact. Raw flags are retained alongside direct adjudication rather than rewritten as unanimous approval. Selected public-fixture review images and the verification record accompany the feature PR; other generated artifacts remain ignored.

The initial independent image review caught a real native-Canvas impulse-contour difference; the verified Lanczos sampler fixed it. Final independent judgments and direct adjudication are preserved in `qa/artifacts/ab/vision-review.json` and `review.md`. One contradictory judge flag was resolved by identical full-resolution corner-image pixels and identical text; the raw judgments were not rewritten. All synthetic cases and comparison-profile photographs are text-exact. Default astronaut/portrait cases differ at five/four cells, with grayscale-rounding propagation recorded in `residual-provenance.json`.

Current outputs and retained historical evidence:

- `qa/artifacts/unit/numerical-report.json` and `resize-report.json`
- `qa/artifacts/playwright/html/index.html` and `production-html/index.html`
- `qa/artifacts/ab/index.html`, `review.md`, `vision-raw.json`, and `vision-review.json`
- `qa/artifacts/performance.json` (settings, browser and machine included)
- `qa/artifacts/clack/production-iphone-webkit/` (current branded camera screenshots and geometry, including 280px)
- `qa/artifacts/clack/brand-runtime.json`, `iphone-camera.png`, `iphone-studio.png`, `desktop-camera.png`, and `desktop-studio.png`
- `qa/artifacts/clack/brand-overlap-before.json`, `minimum-width-before.png`, and `minimum-width.png` (observed header regression and correction)
- `qa/artifacts/camera-ui/production-iphone-webkit/` (retained pre-branding camera screenshots and geometry)
- `qa/artifacts/camera-ui/vision-raw.json`, `vision-review.json`, `presentation-proof.json`, and `artwork-crops/`
- `qa/artifacts/camera-ui/physical-camera.json`, `physical-live.png`, and `physical-captured.png`
- `qa/artifacts/camera-ui/desktop-camera.png` and `final-surface-smoke.json`
- `qa/artifacts/camera-ui-before-detail-fix/` preserves the initial vision evidence that exposed the overlapping Detail controls
- Earlier Studio/mobile/font diagnostics remain under `qa/artifacts/mobile/`

The five-second fake-camera gate records delivered frame rate, warm worker p50/p95, machine/browser details, Stop response, and outstanding-request count in `performance.json`. These measurements are host-specific, not an iPhone FPS promise.

**Physical webcam baseline, before the branding cutover:** the production camera shell acquired real 640×480 Logitech C930e video without fake-capture flags or microphone tracks. Shutter capture ended the live stream and accepted a real PNG snapshot as a retunable still. Retake reacquired the camera; Flip ended the prior stream; Stop ended all retained video tracks, detached the video and prevented further commits. The hardware report records source generations 1 → 2 → 3 → 4 for live → captured still → retake → flip. This is desktop Chromium camera evidence at emulated phone dimensions, **not physical iPhone/Safari validation**.

Vision review caught glyphs scrolling under the Detail controls; the controls now sit outside the scrollport, with a production regression check for nonoverlap. A fractional CSS-height edge case in Fit is also corrected rather than hidden by relaxed assertions. Final independent judgments recognize all ten views as camera-like and usable. Two comparisons against the raw wide photograph still flag artwork; native-pixel inspection, complete-row/uniform-scale checks and unchanged Camera/Studio text are recorded in the direct adjudication. The renderer and reference answers were not changed to satisfy a visual judge.

The camera shell fits the tested 320×568 viewport; Studio remains a vertically scrolling interface. Earlier WebKit production testing also found and fixed a stylesheet/font-registration race; the delayed-CSS gate remains active, and the prior cold-start/missing-font evidence is retained under `qa/artifacts/mobile/`.
