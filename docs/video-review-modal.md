# The video review modal

clack has exactly one modal: `<dialog id="video-review">`, the review surface for a finished
ASCII video. This is the detailed reference for its markup, state machine, actions, focus
behavior, styling, and edge cases. The recorder itself — capture, encoding, limits — is
documented in [ascii-video-recording.md](./ascii-video-recording.md); this doc covers only the
dialog.

## Role

After a recording finishes encoding, the clip exists only in memory as an object URL. The modal
is the single gate between that temporary clip and the device: play it, save it, share it, or
discard and retake. It never uploads anything and never appears without a ready clip.

## DOM contract (`frontend/index.html`)

| Element | Purpose |
|---|---|
| `#video-review` (`<dialog>`) | Native modal. `aria-labelledby="video-review-title"`, `aria-describedby="video-review-privacy"`. Opened with `showModal()`, so the background is inert and Escape is native. |
| `.video-review-header` | Title row: `<h2 id="video-review-title">Your ASCII video</h2>` + `#close-video-review` (DONE, `autofocus`). |
| `#recorded-video` (`<video>`) | Native controls, `playsinline`, `preload="metadata"`, `aria-label="Recorded ASCII video"`. The only playback surface. |
| `#video-clip-details` | One line: `FORMAT · seconds · width × height` (for example `WEBM · 8.3s · 848 × 636`). Format is the encoder's actual container, uppercased, parameters stripped. |
| `#video-review-privacy` | "Silent video · Stored only in this browser until you save or share. Nothing is shared automatically." Describes the whole dialog. |
| `#video-save` / `#video-share` / `#video-retake` | Action row. `#video-retake` carries `aria-describedby="video-retake-help"`. |
| `#video-retake-help` | "Retake discards this clip, including any unsaved video." Always visible. |
| `#video-share-help` | File-sharing-unavailable notice. Shown only when sharing is unsupported; `#video-share`'s `aria-describedby` swaps to it in that case. |
| `#video-export-status` | `role="status" aria-live="polite"` feedback line; empty (and `display: none` via `:empty`) when idle. |

## Opening rules

Three entry points, all funneling through `openVideoReview()` (`frontend/src/main.js`):

1. **Auto-open on finish.** When the recorder reaches `ready` and the tab is visible, the modal
   opens immediately with `#camera-shutter` as the retained opener
   (`recordingChanged`, main.js ~163-167). Encoding finishes after Stop surrenders focus, so
   the opener is captured explicitly rather than assumed.
2. **Deferred open.** If the tab was hidden when the clip finalized (`reviewPending`), the modal
   opens on the next `visibilitychange` to visible (main.js ~995-999). A clip that finishes in
   the background is never lost.
3. **Manual reopen.** `#review-video` (rail REVIEW button, visible while a clip exists) and the
   shutter in video mode with a clip (label REVIEW) both call `openVideoReview(trigger)`.

Guards: nothing opens while `videoBusy()` (pending/recording/stopping), during a photo capture,
when disposed, or without a clip. `openVideoReview` also **releases a live camera first** —
reviewing a saved clip must not leave a Photo viewfinder capturing behind an inert dialog
(main.js ~173-174). `recordingView.open()` itself no-ops unless `state === 'ready'`, a clip
exists, and the dialog is not already open.

## State and enablement

`recordingView.update({ state, clip, shareSupported })` is the single writer
(`frontend/src/recording-view.js`):

- `ready = state === 'ready' && clip` gates all three action buttons.
- SHARE additionally requires `shareSupported` — `navigator.share` + `navigator.canShare` with
  the clip's actual `File`, re-checked on every update (`canShareClip()`, main.js ~120-127).
- Clip source changes swap `video.src` (or clear it), call `load()`, pause, and clear the
  export message; a clip going to `null` closes the dialog.
- The details line rebuilds from the clip: `mimeType.split(';')[0]` minus `video/`, uppercased;
  `durationMs/1000` to one decimal; the recorder's layout width and height.

Share availability changes only the disabled state and the help wiring — never a hidden button.

## The clip object

Produced by the recorder on finalize (`frontend/src/ascii-recorder.js` ~167-176):

```text
{ blob, url, file, mimeType, width, height, durationMs }
```

- `file` is a `File` named `clack-<ISO timestamp with : and . replaced by ->.<container>`;
  the extension comes from the encoder's actual container, not the requested codec
  (`recordingExtension`), so a WebM fallback never ships as `.mp4`.
- `url` is an object URL owned by the recorder. It is revoked on discard/retake, on a new
  recording, and on dispose. The modal never revokes it itself.
- The recorder caps recording at 30 seconds and 16 MiB; overflow finalizes or discards per
  `RECORDING_LIMITS` — the modal only ever sees a finished clip.

## Actions

| Action | Behavior |
|---|---|
| **DONE** / Escape / backdrop | `close()`: pause video, `dialog.close()`. The native `cancel` (Escape) and the click handler both land here. Backdrop clicks close only when the click target is the dialog itself and the coordinates fall outside its bounding box — the standard "clicked the dark area" test, safe against clicks on the flex-spaced body. |
| **SAVE VIDEO** | Synthesizes an `<a download>` to the object URL, clicks it, removes it, and reports "Download requested. Your browser may ask where to save the video." No re-encode, no copy — the exact recorded blob. |
| **SHARE VIDEO** | `navigator.share({ files: [file], title: 'clack — ASCII video' })`, invoked synchronously from the click activation (the comment in source: before any asynchronous work) so mobile browsers treat it as a user gesture. `exportBusy` disables SHARE/RETAKE for the flight. `AbortError` → "Sharing canceled. Your clip is still here."; other errors → "Unable to share this video. Use Save video instead."; success → "Video shared." Every path restores buttons via `update()`. |
| **DISCARD & RETAKE** | Closes the modal, clears `reviewPending`, resets the recorder (revoking the URL), forces video mode, and restarts the live viewfinder. Recording itself still needs a new tap — retake returns you to a live camera, not to a running recording. |

## Focus management

- Open: `showModal()`, then `#close-video-review` focuses with `preventScroll` (plus its
  `autofocus` attribute as the markup-level fallback). Native dialog semantics trap focus
  inside while open.
- Close (every path, via the dialog `close` event → `restoreFocus`): pause the video, then
  focus the retained opener if it is still connected and enabled — otherwise `#camera-shutter`.
  This is why `open(trigger)` stores the trigger: the modal usually opens from a shutter whose
  label has since changed, and encoding completes asynchronously after Stop.
- `dispose()` closes, strips all listeners, clears `src`, and drops the opener reference.

## Playback notes

Native `<video controls>`; the app does not wrap or re-skin them. An `error` event with a clip
present swaps in "This browser could not play the recorded video. Save it to try a compatible
video player." — save still works even when inline playback does not. One platform wart: macOS
WebKit emits `Button failed to load … placard …` console noise from these native controls; the
browser-test fixtures filter it explicitly (verified pre-existing, not app output).

## Styling (`frontend/src/styles.css`)

- Width `min(640px, calc(100% - 24px))`, centered, max-height `calc(100svh - 24px)`;
  `[open]` switches to `display: flex; flex-direction: column`.
- Backdrop: `#video-review::backdrop { background: #000b; }` (one of the few literal colors,
  matching the app's ink-black stage).
- Body scrolls (`overflow-y: auto`, `overscroll-behavior: contain`) with safe-area padding;
  the video caps at `max-height: 52svh` (floor 120px) so actions stay reachable on short
  phones.
- Action buttons: 44px minimum, `flex: 1 1 120px`; `#video-save` carries the accent border and
  text as the primary action. `#video-export-status:empty { display: none }`.
- All other colors via tokens; reduced-motion users get no transitions.

## Edge cases covered by design

- **Background finalize** — deferred open (`reviewPending`), asserted by
  `camera.spec.js` ("background release finishes the existing video and opens review only
  after returning").
- **Camera left running** — `openVideoReview` releases it before showing the dialog.
- **Share unsupported** (desktop Firefox, Linux WebKit) — button disabled, help line wired via
  `aria-describedby`, status explains the Save fallback.
- **Encode failure during stop** — recorder surfaces the error; `ready` never arrives; the
  modal never opens with a dead URL.
- **Re-record over a reviewed clip** — new finalize swaps the source (pause + `load()`), and
  the old URL is revoked by the recorder.

## Tests that pin this behavior

- `frontend/tests/browser/recording.spec.js` — real encode → review → save/retake flows,
  duration cap, visibilitychange finalize.
- `frontend/tests/browser/camera.spec.js` — background-release deferred open, WebM fallback
  naming, native-share cancel path, review reopen after stop.
- `frontend/tests/browser/mobile.spec.js` — dialog and body scroll containment plus 44px
  targets at 280/320/428/832-wide viewports (the `#close-video-review`, `#video-save`,
  `#video-share`, `#video-retake` loop).
- `frontend/tests/browser/production.spec.js` — the same surface against the built bundle.

Privacy posture: the modal renders an object URL that never leaves the tab unless SAVE or SHARE
is explicitly activated; nothing is shared automatically (the `#video-review-privacy` line is
the product's promise, and the tests enforce the no-network contract globally).
