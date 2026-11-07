import { createAsciiArtwork } from './artwork.js';

/** Export paths for the committed ASCII frame: clipboard text and PNG download. */
export async function copyAscii(text) {
  if (!navigator.clipboard?.writeText) throw new Error('Clipboard is unavailable in this browser');
  await navigator.clipboard.writeText(text);
}

/** Still images retain the 2x policy while sharing the video's ink-safe artwork path. */
export async function asciiToPng(frame, options = {}) {
  let artwork;
  try {
    artwork = createAsciiArtwork(frame, { scale: 2, maxEdge: Infinity, evenDimensions: false, ...options });
    artwork.draw(frame.text);
    return await new Promise((resolve, reject) => {
      artwork.canvas.toBlob((blob) => {
        if (blob) resolve(blob);
        else reject(new Error('Unable to render the PNG'));
      }, 'image/png');
    });
  } catch (cause) {
    throw new Error('Unable to render the PNG', { cause });
  } finally {
    if (artwork) {
      artwork.canvas.width = 0;
      artwork.canvas.height = 0;
    }
  }
}
