export const FRAMINGS = Object.freeze([
  { id: 'full', label: 'Full', width: null, height: null },
  { id: 'square', label: '1:1', width: 1, height: 1 },
  { id: 'portrait', label: '3:4', width: 3, height: 4 },
  { id: 'story', label: '9:16', width: 9, height: 16 },
  { id: 'classic', label: '4:3', width: 4, height: 3 },
  { id: 'wide', label: '16:9', width: 16, height: 9 },
].map(Object.freeze));

/** Largest centered whole-pixel rectangle with exactly the selected ratio. */
export function computeCropRect(sourceWidth, sourceHeight, frameId = 'full') {
  if (!Number.isSafeInteger(sourceWidth) || sourceWidth <= 0 ||
      !Number.isSafeInteger(sourceHeight) || sourceHeight <= 0) {
    throw new RangeError('Source dimensions must be positive integers');
  }
  const frame = FRAMINGS.find(({ id }) => id === frameId);
  if (!frame) throw new RangeError('Choose a supported frame');
  if (frame.id === 'full') return { x: 0, y: 0, width: sourceWidth, height: sourceHeight };
  // Integer multiples avoid a subtly stretched ratio after Canvas pixel rounding.
  const multiple = Math.floor(Math.min(sourceWidth / frame.width, sourceHeight / frame.height));
  if (multiple < 1) throw new RangeError('Image is too small for this frame — choose Full or a larger image');
  const width = multiple * frame.width;
  const height = multiple * frame.height;
  return { x: Math.floor((sourceWidth - width) / 2), y: Math.floor((sourceHeight - height) / 2), width, height };
}
