const byId = (id) => document.getElementById(id);


/** Presentation only: writes control state; rendering and sources live in main.js. */
export function createCameraShell() {
  byId('toggle-guides').addEventListener('click', (event) => {
    const enabled = event.currentTarget.getAttribute('aria-pressed') !== 'true';
    event.currentTarget.setAttribute('aria-pressed', String(enabled));
    document.body.dataset.guides = enabled ? 'on' : 'off';
  });

  return {
    update({ cameraState, kind, captureReady, busy, disposed }) {
      const live = cameraState === 'live';
      const requesting = cameraState === 'requesting';
      const held = !live && Boolean(kind);
      const shutter = byId('camera-shutter');
      shutter.disabled = disposed || busy || (live && !captureReady);
      shutter.dataset.state = busy ? 'capturing' : requesting ? 'requesting' : live ? 'live' : 'idle';
      const label = busy ? 'Developing frame' : requesting ? 'Cancel camera request' : live ? 'Freeze ASCII frame' : held ? 'Return to live camera' : 'Start camera';
      shutter.setAttribute('aria-label', label);
      shutter.title = label;
      shutter.querySelector('span').textContent = busy ? 'DEVELOPING' : requesting ? 'CANCEL' : live ? 'FREEZE' : held ? 'RESUME' : 'CAMERA';
      byId('camera-stop').hidden = cameraState === 'idle' || disposed;
      byId('camera-stop').disabled = cameraState === 'idle' || busy || disposed;
      byId('flip-camera').disabled = cameraState === 'idle' || requesting || busy || disposed;
      byId('camera-import').disabled = busy || disposed;
      byId('status').dataset.camera = busy ? 'capturing' : requesting ? 'requesting' : live ? 'live' : 'idle';
    },
  };
}
