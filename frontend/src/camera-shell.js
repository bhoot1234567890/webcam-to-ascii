const byId = (id) => document.getElementById(id);

/** Presentation only: writes control state; rendering and sources live in main.js. */
export function createCameraShell() {
  byId('toggle-guides').addEventListener('click', (event) => {
    const enabled = event.currentTarget.getAttribute('aria-pressed') !== 'true';
    event.currentTarget.setAttribute('aria-pressed', String(enabled));
    document.body.dataset.guides = enabled ? 'on' : 'off';
  });

  return {
    update({
      cameraState, kind, captureReady, busy, disposed,
      mode = 'photo', recordingState = 'idle', pendingRecording = false,
      recordingSupported = true, hasClip = false,
    }) {
      const live = cameraState === 'live';
      const requesting = cameraState === 'requesting';
      const held = !live && Boolean(kind);
      const video = mode === 'video';
      const recording = recordingState === 'recording';
      const stopping = recordingState === 'stopping';
      const locked = pendingRecording || recording || stopping;
      const shutter = byId('camera-shutter');
      document.body.dataset.captureMode = mode;
      document.body.dataset.recordingState = pendingRecording ? 'pending' : recordingState;
      shutter.dataset.mode = mode;
      shutter.disabled = disposed || (video ?
        stopping || (!pendingRecording && !recording && !requesting && (busy || (!hasClip && !recordingSupported))) :
        busy || (live && !captureReady));
      shutter.dataset.state = video && locked ? pendingRecording ? 'pending' : recordingState :
        busy ? 'capturing' : requesting ? 'requesting' : live ? 'live' : held ? 'held' : 'idle';
      const label = video ?
        pendingRecording ? 'Cancel video start' : recording ? 'Stop recording' :
          stopping ? 'Finishing video' : requesting ? 'Cancel camera request' :
            hasClip ? 'Review ASCII recording' : 'Start ASCII recording' :
        busy ? 'Developing frame' : requesting ? 'Cancel camera request' :
          live ? 'Freeze ASCII frame' : 'Start camera';
      shutter.setAttribute('aria-label', label);
      shutter.title = label;
      shutter.querySelector('span').textContent = video ?
        pendingRecording || requesting ? 'CANCEL' : recording ? 'STOP' :
          stopping ? 'FINISHING' : hasClip ? 'REVIEW' : 'RECORD' :
        busy ? 'DEVELOPING' : requesting ? 'CANCEL' : live ? 'FREEZE' : 'CAMERA';
      shutter.setAttribute('aria-busy', String(stopping || (!video && busy)));
      byId('camera-stop').hidden = cameraState === 'idle' && !locked || disposed;
      byId('camera-stop').disabled = disposed || (!locked && (cameraState === 'idle' || busy));
      const flip = byId('flip-camera');
      flip.hidden = cameraState === 'idle' && !locked || disposed;
      flip.disabled = locked || requesting || busy || disposed;
      byId('camera-import').disabled = locked || busy || disposed;
      for (const id of ['photo-mode', 'video-mode', 'studio-video']) {
        byId(id).disabled = locked || requesting || busy || disposed;
      }
      byId('photo-mode').setAttribute('aria-pressed', String(!video));
      byId('video-mode').setAttribute('aria-pressed', String(video));
      byId('review-video').hidden = !hasClip;
      byId('review-video').disabled = locked || busy || disposed;
      byId('recording-indicator').hidden = !locked;
      byId('recording-label').textContent = pendingRecording ? 'Starting' : stopping ? 'Finishing' : 'Recording';
      if (!locked) {
        byId('recording-timer').textContent = '00:00';
        byId('recording-timer').dateTime = 'PT0S';
      }
      byId('shutter-help').textContent = video ?
        pendingRecording ? 'Tap CANCEL to cancel video start' : recording ? '' :
          stopping ? '' : requesting ? 'Tap CANCEL to cancel camera access' :
            !recordingSupported ? 'Video unavailable · Use Photo' : '' :
        busy ? '' : requesting ? 'Tap CANCEL to cancel camera access' :
          live ? '' : held ? 'COPY or SAVE the frame' : 'Tap CAMERA to open your camera';
      byId('status').dataset.camera = locked ? pendingRecording ? 'pending' : recordingState :
        busy ? 'capturing' : requesting ? 'requesting' : live ? 'live' : 'idle';
    },
  };
}
