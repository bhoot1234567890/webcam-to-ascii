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
    update({ cameraState, kind, captureReady, busy, disposed }) {
      const live = cameraState === 'live';
      const requesting = cameraState === 'requesting';
      const shutter = byId('camera-shutter');
      shutter.disabled = disposed || busy || (live && !captureReady);
      shutter.dataset.state = busy ? 'capturing' : requesting ? 'requesting' : live ? 'live' : 'idle';
      shutter.setAttribute('aria-label', requesting ? 'Cancel camera request' : live ? 'Freeze ASCII frame' : 'Start camera');
      byId('camera-stop').disabled = cameraState === 'idle' || disposed;
      byId('flip-camera').disabled = requesting || busy || disposed;
      byId('camera-import').disabled = busy || disposed;
      byId('live-mode').disabled = requesting || busy || disposed;
      byId('still-mode').disabled = (!captureReady && kind !== 'image') || busy || disposed;
      byId('live-mode').setAttribute('aria-pressed', String(live || !kind));
      byId('still-mode').setAttribute('aria-pressed', String(!live && Boolean(kind)));
      byId('shutter-help').textContent = busy ? 'Developing your frame…' :
        requesting ? 'Tap the shutter to cancel' : live ? 'Tap the shutter to freeze a frame' :
          kind ? 'Tap the shutter to return to live' : 'Tap the shutter to open your camera';
    },
  };
}
