const HAVE_CURRENT_DATA = 2;

function requireMediaDevices() {
  const mediaDevices = globalThis.navigator?.mediaDevices;
  if (!globalThis.isSecureContext || typeof mediaDevices?.getUserMedia !== "function") {
    throw new Error("Camera requires HTTPS or localhost");
  }
  return mediaDevices;
}

function cameraUnavailable() {
  return new DOMException("Camera is unavailable", "NotReadableError");
}

function stopTracks(tracks) {
  for (const track of tracks) track.stop();
}

/** Owns the video's stream, pending permission request, and frame callbacks. */
export function createCameraController(video, { onFrame = () => {}, onState = () => {}, onError = () => {} } = {}) {
  let disposed = false;
  let generation = 0;
  let current = null;

  video.muted = true;
  video.playsInline = true;

  function isCurrent(session) {
    return !disposed && current === session && session.generation === generation;
  }

  function isUsable() {
    return video.videoWidth > 0 && video.videoHeight > 0
      && video.readyState >= HAVE_CURRENT_DATA && !video.paused && !video.ended;
  }

  function cancelScheduledFrame(session) {
    if (session.frameId === null) return;
    const receiver = session.frameKind === "video" ? video : globalThis;
    session.cancelFrame.call(receiver, session.frameId);
    session.frameId = null;
    session.frameKind = null;
    session.cancelFrame = null;
  }

  function clearCurrent() {
    generation += 1;
    const previous = current;
    current = null;
    if (previous) {
      previous.cancelReadiness?.();
      cancelScheduledFrame(previous);
      previous.removeListeners?.();
      stopTracks(previous.tracks);
    }
    video.pause();
    video.srcObject = null;
  }

  function fail(session, error) {
    if (!isCurrent(session)) return;
    clearCurrent();
    onState("idle");
    if (!disposed) onError(error);
  }

  function scheduleFrame(session) {
    if (!isCurrent(session) || !session.live) return;
    if (typeof video.requestVideoFrameCallback === "function"
      && typeof video.cancelVideoFrameCallback === "function") {
      session.frameKind = "video";
      session.cancelFrame = video.cancelVideoFrameCallback;
      session.frameId = video.requestVideoFrameCallback(session.onVideoFrame);
    } else {
      session.frameKind = "animation";
      session.cancelFrame = globalThis.cancelAnimationFrame;
      session.frameId = globalThis.requestAnimationFrame(session.onAnimationFrame);
    }
  }

  function frameArrived(session, metadata) {
    session.frameId = null;
    session.frameKind = null;
    session.cancelFrame = null;
    if (!isCurrent(session) || !session.live) return;
    if (isUsable()) onFrame(video, metadata);
    // onFrame can synchronously stop, switch, or dispose the controller.
    if (isCurrent(session) && session.live) scheduleFrame(session);
  }

  function listenToStream(session) {
    const unavailable = () => fail(session, cameraUnavailable());
    const videoError = () => {
      if (video.error) fail(session, cameraUnavailable());
    };
    for (const track of session.tracks) track.addEventListener("ended", unavailable);
    session.stream.addEventListener("inactive", unavailable);
    video.addEventListener("error", videoError);
    session.removeListeners = () => {
      for (const track of session.tracks) track.removeEventListener("ended", unavailable);
      session.stream.removeEventListener("inactive", unavailable);
      video.removeEventListener("error", videoError);
      session.removeListeners = null;
    };
  }

  function waitForPlayback(session) {
    return new Promise((resolve, reject) => {
      let finished = false;
      let played = false;
      const events = ["loadedmetadata", "loadeddata", "canplay", "playing", "resize"];

      function cleanup() {
        for (const event of events) video.removeEventListener(event, check);
        session.cancelReadiness = null;
      }

      function finish(ready) {
        if (finished) return;
        finished = true;
        cleanup();
        resolve(ready);
      }

      function check() {
        if (!isCurrent(session)) finish(false);
        else if (played && isUsable()) finish(true);
      }

      session.cancelReadiness = () => finish(false);
      for (const event of events) video.addEventListener(event, check);
      let playback;
      try {
        playback = video.play();
      } catch (error) {
        finished = true;
        cleanup();
        reject(error);
        return;
      }
      // Both handlers remain attached after cancellation: a late play rejection
      // must not become an unhandled rejection or reactivate a stopped source.
      Promise.resolve(playback).then(() => {
        played = true;
        if (!finished) check();
      }, (error) => {
        if (finished) return;
        finished = true;
        cleanup();
        reject(error);
      });
    });
  }

  async function start(deviceId, facingMode = '') {
    if (disposed) return false;
    const requestedDeviceId = deviceId || "";
    const requestedFacingMode = deviceId ? "" : facingMode;
    if (current && (!current.live ||
      (current.deviceId === requestedDeviceId && current.facingMode === requestedFacingMode))) return false;

    // Releasing the old stream precedes even the replacement permission request.
    clearCurrent();
    const session = {
      generation,
      deviceId: requestedDeviceId,
      facingMode: requestedFacingMode,
      stream: null,
      tracks: [],
      live: false,
      frameId: null,
      frameKind: null,
      cancelFrame: null,
      cancelReadiness: null,
      removeListeners: null,
      lastMediaTime: null,
      presentedFrames: 0,
      onVideoFrame: null,
      onAnimationFrame: null,
    };
    session.onVideoFrame = (_now, metadata) => frameArrived(session, metadata);
    session.onAnimationFrame = (now) => {
      const mediaTime = video.currentTime;
      const fresh = mediaTime !== session.lastMediaTime;
      if (isCurrent(session) && session.live && isUsable() && fresh) {
        session.lastMediaTime = mediaTime;
        session.presentedFrames += 1;
        frameArrived(session, {
          mediaTime,
          presentationTime: now,
          expectedDisplayTime: now,
          presentedFrames: session.presentedFrames,
          width: video.videoWidth,
          height: video.videoHeight,
        });
      } else {
        session.frameId = null;
        session.frameKind = null;
        session.cancelFrame = null;
        if (isCurrent(session) && session.live) scheduleFrame(session);
      }
    };
    current = session;
    onState("requesting");
    if (!isCurrent(session)) return false;

    try {
      const mediaDevices = requireMediaDevices();
      const stream = await mediaDevices.getUserMedia({
        audio: false,
        video: deviceId ? { deviceId: { exact: deviceId } } :
          requestedFacingMode ? { facingMode: { ideal: requestedFacingMode } } : true,
      });
      if (!isCurrent(session)) {
        // Permission can resolve after Stop/dispose, without ever being attached.
        stopTracks(stream.getTracks());
        return false;
      }
      session.stream = stream;
      session.tracks = stream.getTracks();
      listenToStream(session);
      if (!session.tracks.some((track) => track.kind === "video" && track.readyState === "live")
        || session.tracks.some((track) => track.readyState === "ended")) {
        throw cameraUnavailable();
      }
      video.srcObject = stream;
      if (!isCurrent(session) || !await waitForPlayback(session) || !isCurrent(session)) return false;
      session.live = true;
      onState("live");
      if (!isCurrent(session)) return false;
      scheduleFrame(session);
      return true;
    } catch (error) {
      fail(session, error);
      return false;
    }
  }

  function stop() {
    if (disposed) return;
    clearCurrent();
    onState("idle");
  }

  function dispose() {
    if (disposed) return;
    disposed = true;
    clearCurrent();
    onState("idle");
  }

  async function listDevices() {
    if (disposed) return [];
    const mediaDevices = requireMediaDevices();
    if (typeof mediaDevices.enumerateDevices !== "function") {
      throw new Error("Camera requires HTTPS or localhost");
    }
    const devices = await mediaDevices.enumerateDevices();
    if (disposed) return [];
    return devices.filter((device) => device.kind === "videoinput");
  }

  return { start, stop, dispose, listDevices };
}
