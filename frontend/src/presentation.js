const FONT_FAMILY = 'Roboto Mono';

/** Monospace cell metrics: one advance wide, two advances tall. */
export function asciiFontMetrics(fontSizePx = 12) {
  if (!Number.isFinite(fontSizePx) || fontSizePx <= 0) {
    throw new RangeError('ASCII font size must be finite and positive');
  }
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('Unable to measure the ASCII font');
  }
  const font = `400 ${fontSizePx}px "${FONT_FAMILY}"`;
  context.font = font;
  const advance = context.measureText('M').width;
  if (!Number.isFinite(advance) || advance <= 0) {
    throw new Error('Unable to measure the ASCII character advance');
  }
  return { advance, lineHeight: advance * 2, font };
}

/** Load the local font and give every ASCII cell a 1:2 width-to-height ratio. */
export async function configureAsciiText(element, fontSizePx = 12) {
  if (!(element instanceof HTMLElement)) {
    throw new TypeError('Expected an HTML element for ASCII presentation');
  }
  if (!Number.isFinite(fontSizePx) || fontSizePx <= 0) {
    throw new RangeError('ASCII font size must be finite and positive');
  }
  if (!document.fonts) {
    throw new Error('Unable to load the required Roboto Mono font: Font Loading API is unavailable');
  }

  const font = `400 ${fontSizePx}px "${FONT_FAMILY}"`;
  let faces;
  try {
    // WebKit can evaluate a module before its linked CSS registers @font-face.
    await Promise.all([...document.querySelectorAll('link[rel~="stylesheet"]')].map((link) => {
      if (link.sheet) return;
      return new Promise((resolve, reject) => {
        const cleanup = () => {
          link.removeEventListener('load', loaded);
          link.removeEventListener('error', failed);
          window.removeEventListener('load', loaded);
        };
        const loaded = () => { cleanup(); link.sheet ? resolve() : reject(new Error('Unable to load ASCII stylesheet')); };
        const failed = () => { cleanup(); reject(new Error('Unable to load ASCII stylesheet')); };
        link.addEventListener('load', loaded, { once: true });
        link.addEventListener('error', failed, { once: true });
        window.addEventListener('load', loaded, { once: true });
        if (document.readyState === 'complete') loaded();
      });
    }));
    faces = await document.fonts.load(font, 'M@ ');
  } catch (cause) {
    throw new Error('Unable to load the required Roboto Mono font', { cause });
  }
  const requiredFaces = faces.filter((face) => face.family.replace(/["']/g, '') === FONT_FAMILY);
  if (
    requiredFaces.length === 0 ||
    requiredFaces.some((face) => face.status !== 'loaded') ||
    !document.fonts.check(font, 'M@ ')
  ) {
    throw new Error('The required Roboto Mono font is missing or failed to load');
  }
  const { advance } = asciiFontMetrics(fontSizePx);
  Object.assign(element.style, {
    fontFamily: `"${FONT_FAMILY}"`,
    fontWeight: '400',
    fontStyle: 'normal',
    fontSize: `${fontSizePx}px`,
    fontVariantLigatures: 'none',
    fontFeatureSettings: '"liga" 0, "calt" 0',
    fontKerning: 'none',
    letterSpacing: '0px',
    lineHeight: `${advance * 2}px`,
    whiteSpace: 'pre',
    overflowWrap: 'normal',
    wordBreak: 'normal',
  });
}
