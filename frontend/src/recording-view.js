const byId = (id) => document.getElementById(id);

function elapsedLabel(elapsedMs) {
  const seconds = Math.floor(Math.max(0, elapsedMs) / 1000);
  return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
}

/** Local playback presentation. The recorder owns clips and object-URL lifetime. */
export function createRecordingView({ onRetake, onSave, onShare }) {
  const dialog = byId('video-review');
  const video = byId('recorded-video');
  const listeners = [];
  let clip = null;
  let state = 'idle';
  let opener = null;
  let disposed = false;
  const listen = (node, event, handler) => {
    node.addEventListener(event, handler);
    listeners.push(() => node.removeEventListener(event, handler));
  };

  function setExportMessage(message) {
    byId('video-export-status').textContent = message;
  }

  function restoreFocus() {
    video.pause();
    const target = opener?.isConnected && !opener.disabled ? opener : byId('camera-shutter');
    if (!disposed && !target.disabled) target.focus({ preventScroll: true });
    opener = null;
  }

  function close() {
    video.pause();
    if (dialog.open) dialog.close();
  }

  // Keep export callbacks directly in the click gesture for native file sharing.
  listen(byId('video-retake'), 'click', () => onRetake());
  listen(byId('video-save'), 'click', () => onSave());
  listen(byId('video-share'), 'click', () => onShare());
  listen(byId('close-video-review'), 'click', close);
  listen(dialog, 'close', restoreFocus);
  listen(dialog, 'click', (event) => {
    if (event.target !== dialog) return;
    const box = dialog.getBoundingClientRect();
    if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) close();
  });
  listen(video, 'error', () => {
    if (clip) setExportMessage('This browser could not play the recorded video. Save it to try a compatible video player.');
  });

  return {
    update({ state: nextState, clip: nextClip, shareSupported = false }) {
      if (disposed) return;
      state = nextState;
      const sourceChanged = (nextClip?.url || '') !== (clip?.url || '');
      clip = nextClip || null;
      if (sourceChanged) {
        video.pause();
        if (clip) video.src = clip.url;
        else video.removeAttribute('src');
        video.load();
        setExportMessage('');
      }
      const ready = state === 'ready' && Boolean(clip);
      byId('video-save').disabled = !ready;
      byId('video-share').disabled = !ready || !shareSupported;
      byId('video-retake').disabled = !ready;
      byId('video-share-help').hidden = shareSupported;
      byId('video-share').setAttribute('aria-describedby', shareSupported ? 'video-review-privacy' : 'video-share-help');
      if (clip) {
        const format = clip.mimeType.split(';')[0].replace('video/', '').toUpperCase();
        byId('video-clip-details').textContent = `${format} · ${(clip.durationMs / 1000).toFixed(1)}s · ${clip.width} × ${clip.height}`;
      } else {
        byId('video-clip-details').textContent = '';
        close();
      }
    },
    tick(elapsedMs) {
      if (disposed) return;
      const timer = byId('recording-timer');
      const label = elapsedLabel(elapsedMs);
      if (timer.textContent !== label) timer.textContent = label;
      timer.dateTime = `PT${Math.floor(Math.max(0, elapsedMs) / 1000)}S`;
    },
    open(trigger) {
      if (disposed || !clip || state !== 'ready' || dialog.open) return;
      // Encoding completes after Stop loses focus; retain the actual return control.
      opener = trigger;
      dialog.showModal();
      byId('close-video-review').focus({ preventScroll: true });
    },
    close,
    setExportMessage,
    dispose() {
      if (disposed) return;
      disposed = true;
      close();
      for (const remove of listeners) remove();
      video.removeAttribute('src');
      video.load();
      clip = null;
      opener = null;
    },
  };
}
