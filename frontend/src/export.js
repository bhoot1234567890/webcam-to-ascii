/** Export paths for the committed ASCII frame: clipboard text and PNG download. */
export async function copyAscii(text) {
  if (!navigator.clipboard?.writeText) throw new Error('Clipboard is unavailable in this browser');
  await navigator.clipboard.writeText(text);
}

export function asciiToPng({ text, columns, rows, advance, lineHeight, font, color, background = '#09090b', scale = 2 }) {
  return new Promise((resolve, reject) => {
    let context;
    try {
      const canvas = document.createElement('canvas');
      canvas.width = Math.ceil(columns * advance * scale);
      canvas.height = Math.ceil(rows * lineHeight * scale);
      context = canvas.getContext('2d');
      if (!context) throw new Error('Unable to render the PNG');
      context.scale(scale, scale);
      context.font = font;
      context.textBaseline = 'top';
      context.fillStyle = background;
      context.fillRect(0, 0, columns * advance, rows * lineHeight);
      context.fillStyle = color;
      const lines = text.split('\n');
      for (let index = 0; index < lines.length; index++) {
        context.fillText(lines[index], 0, index * lineHeight);
      }
      canvas.toBlob((blob) => {
        if (blob) resolve(blob);
        else reject(new Error('Unable to render the PNG'));
      }, 'image/png');
    } catch {
      reject(new Error('Unable to render the PNG'));
    }
  });
}
