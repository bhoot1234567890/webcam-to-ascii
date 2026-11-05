const byId = (id) => document.getElementById(id);

/** Presentation only: both layouts use the same source, worker, and ASCII node. */
export function createCameraShell() {
  function setLayout(layout) {
    document.body.dataset.layout = layout;
    window.scrollTo(0, 0);
    const target = layout === 'camera' ? byId('camera-shutter') : byId('page-title');
    target.focus({ preventScroll: true });
  }

  byId('open-studio').addEventListener('click', () => setLayout('studio'));
  byId('open-camera').addEventListener('click', () => setLayout('camera'));
  byId('toggle-guides').addEventListener('click', (event) => {
    const enabled = event.currentTarget.getAttribute('aria-pressed') !== 'true';
    event.currentTarget.setAttribute('aria-pressed', String(enabled));
    document.body.dataset.guides = enabled ? 'on' : 'off';
  });

  return {
    setLayout,
    update({
      cameraState, kind, captureReady, busy, disposed,
      mode = 'photo', recordingState = 'idle', pendingRecording = false,
      recordingSupported = true, hasClip = false,
    }) {
      const live = cameraState === 'live';
      const requesting = cameraState === 'requesting';
      const video = mode === 'video';
      const recording = recordingState === 'recording';
      const stopping = recordingState === 'stopping';
      const locked = pendingRecording || recording || stopping;
      const shutter = byId('camera-shutter');
      document.body.dataset.captureMode = mode;
      document.body.dataset.recordingState = pendingRecording ? 'pending' : recordingState;
      shutter.dataset.mode = mode;
      shutter.disabled = disposed || (video ?
        stopping || (!pendingRecording && !recording && (busy || (!hasClip && !recordingSupported))) :
        busy || (live && !captureReady));
      shutter.dataset.state = video && locked ? pendingRecording ? 'pending' : recordingState :
        busy ? 'capturing' : requesting ? 'requesting' : live ? 'live' : 'idle';
      shutter.setAttribute('aria-label', video ?
        pendingRecording ? 'Cancel video start' : recording ? 'Stop recording' :
          stopping ? 'Finishing video' : hasClip ? 'Review ASCII recording' : 'Start ASCII recording' :
        requesting ? 'Cancel camera request' : live ? 'Freeze ASCII frame' : 'Start camera');
      shutter.setAttribute('aria-busy', String(stopping));
      byId('camera-stop').disabled = cameraState === 'idle' || disposed;
      byId('flip-camera').disabled = locked || requesting || busy || disposed;
      byId('camera-import').disabled = locked || busy || disposed;
      for (const id of ['photo-mode', 'video-mode', 'studio-video']) {
        byId(id).disabled = locked || requesting || busy || disposed;
      }
      byId('photo-mode').setAttribute('aria-pressed', String(!video));
      byId('video-mode').setAttribute('aria-pressed', String(video));
      byId('open-studio').disabled = locked || disposed;
      byId('review-video').hidden = !hasClip;
      byId('review-video').disabled = locked || busy || disposed;
      const indicator = byId('recording-indicator');
      indicator.hidden = !locked;
      byId('recording-label').textContent = pendingRecording ? 'Starting' : stopping ? 'Finishing' : 'Recording';
      if (!locked) byId('recording-timer').textContent = '00:00';
      byId('shutter-help').textContent = video ?
        pendingRecording ? 'Tap the shutter to cancel video start' : recording ? 'Silent · Tap to stop · Up to 30s' :
          stopping ? 'Finishing your video…' : hasClip ? 'Tap to review · Silent · Up to 30s' :
            !recordingSupported ? 'Video unavailable · Use Photo' :
              'Silent · Up to 30s · Tap to record' :
        busy ? 'Developing your frame…' : requesting ? 'Tap the shutter to cancel' :
          live ? 'Tap the shutter to freeze a frame' :
            kind ? 'Tap the shutter to return to live' : 'Tap the shutter to open your camera';
    },
  };
}
