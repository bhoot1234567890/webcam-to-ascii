# clack — product overview & open design brief

**What this is:** a short, deliberately light brief. We are handing you a working product and a lot of freedom. We have kept our input to the minimum a team needs to not waste effort: what clack is, who it serves, five ground truths, and an open field. The rest — including the big questions — is yours.

**For:** exploratory concept work (directions, not implementation).
**Companion document:** `design.md` in this repo is the opposite of this brief — an exhaustive, highly prescriptive handoff written for a separate refinement track. For this engagement, **ignore its decision register (D1-D16)**. Only the Ground Truths below bind you. If you want deep reference material, its Sections 4 and 5 (current-state audit, complete flows) are accurate.

---

## The product in one paragraph

clack is a camera that speaks characters. Point it at anything — yourself, a street, a photo from your library — and it redraws the world live as ASCII text, entirely on your device. Freeze a frame, tune how it reads (four rendering looks, character density, framing, ink color, glow, a CRT mode), then take the art with you: as plain text, as a PNG, or as a short silent video. There is no server, no account, no upload. The privacy is architectural, and the app says so quietly: `nothing leaves this device`.

## Lineage

clack began as a Python terminal viewer — still in this repo, still the numerical oracle the browser app is tested against (same measured ink ramp, same character-for-character math). The browser app is its descendant. Terminal heritage is lineage here, not costume.

## Who uses it

- **The casual mobile creator.** Phone in hand, wants ASCII of themselves in under ten seconds, plays until it looks good, shares a PNG or clip. Low patience for jargon, charmed by the terminal aesthetic — as long as it never blocks them.
- **The terminal-fluent tinkerer.** Appreciates a monospace instrument. Wants density, edge thresholds, tone controls, presets, and text output to paste somewhere.
- **The privacy-conscious.** Chose this app *because* nothing uploads. Needs the privacy stance to stay visible and believable.

## What exists today (facts, not endorsements)

One screen, dark, monospaced: a top bar, a status line, the ASCII stage, a bottom control rail, a settings drawer, one review dialog. Four looks (Original, Photo, Print, Outline), four density presets plus custom, six framings, nine ink tints plus custom, glow and CRT effects, presets, silent video recording up to 30 seconds. Best way to understand it is to run it:

```sh
bun run --cwd frontend dev   # then open http://127.0.0.1:4173
```

Five minutes with the living app beats this document.

## The soul — what must survive any reinvention

1. **Characters are the medium.** The art is real text, live, and it is the hero. Everything else is a frame around it.
2. **Instant instrument.** It responds like hardware: point, see, adjust, see. Nothing between the person and the mirror.
3. **Privacy you can feel.** On-device isn't a policy footer here; it is the product's honor. The design should make trust tangible without nagging.
4. **Craft over novelty.** This is an instrument, not a toy — precise, quiet, deliberate. Retro is allowed; kitsch is not.
5. **Terminal lineage, worn lightly.** Flirt with the aesthetic, never gatekeep with it. A first-time user must never need a manual.

## Ground truths — the only hard rules

1. **On-device only.** The app is a static page: no backend, no network fetches at runtime, no analytics, no accounts. Ever. Anything you propose must live inside one HTML page and its assets.
2. **The rendering vocabulary is fixed data.** Four looks, the density range, the framings, the ink tints, and the measured character ramp shared with the Python oracle. You design the experience around this vocabulary; you cannot change the words it speaks. (Proposing *additions* to the vocabulary is welcome — as a separately noted product suggestion, not as a design assumption.)
3. **The artwork is typographic.** The canvas is monospaced text at a fixed cell size (currently 12px in 1:2 cells, a parity-tested contract). Treat "text as the medium" as the constraint, not any particular size — if a concept needs a different cell size, flag it rather than designing around it silently.
4. **Accessibility is part of the product.** WCAG 2.2 AA, real keyboard paths, 44px touch targets, reduced-motion parity. The current build enforces much of this with automated tests; whatever you propose inherits that bar.
5. **Plain web, wide range.** Evergreen Chromium/Firefox/WebKit including iOS Safari, from 280px phones to desktop. Implementable in vanilla HTML/CSS/JS — no framework, no server rendering, English-only for now.

## The open field

Everything else is genuinely undecided, because we want your point of view, not our echo: information architecture, layout, the navigation model (must it stay one screen?), visual language, color, typography of the chrome, motion, voice and copy, brand expression, the first-run story, how tuning and settings are experienced, how capture and export feel, what the empty state says, what this product *is* when it grows up.

We would rather see a strong direction we did not ask for than a polished version of what already exists.

## Sparks (optional, non-binding)

- The chrome itself is typeset: what if every label, button, and state drew from the same character set as the art?
- Instrument metaphors: darkroom, oscilloscope, lab bench, broadcast deck — which one earns these controls?
- `NO SIGNAL` is the current empty state. What does a terminal-native product say before it has anything to show?
- One accent color exists today (amber, `#ffd663`). Law, or habit?
- What does "nothing leaves this device" look like as a design material rather than a disclaimer?

## What to bring back

- Two or three **distinct directions** — not iterations of one idea. Each with: a name, a one-line thesis, the primary screen(s) at phone scale, and a line on what it deliberately rejects.
- Rationale grounded in the soul points and personas above.
- Explicit notes on anything that collides with the ground truths, rather than silent workarounds.

Prototypes, motion sketches, and copy-voice samples are welcome; none are required at this stage.

## Small print

Product name: `clack` (lowercase). Live at `clack.chaitanyamalhotra.com`. This repo is AGPL-3.0-or-later; work delivered into it carries the same license. Engineering contributor rules: `AGENTS.md`.
