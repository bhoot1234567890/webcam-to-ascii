/**
 * Save-time desktop chrome. Pure 1-bit line art: white rules, glyphs and text
 * on black. Ordered dithering stands in for any tone or shade.
 */
export const EXPORT_FRAMES = Object.freeze([
  { id: 'none', label: 'None' },
  { id: 'mac', label: 'Mac' },
  { id: 'win95', label: 'Win 95' },
  { id: 'winxp', label: 'Win XP' },
]);

export const DEFAULT_EXPORT_FRAME = 'win95';

const INK = '#ffffff';
const BACKGROUND = '#000000';
const MONO_FONT = '"Roboto Mono", ui-monospace, monospace';

const BAYER_4 = [
  [0, 8, 2, 10],
  [12, 4, 14, 6],
  [3, 11, 1, 9],
  [15, 7, 13, 5],
];

export function normalizeExportFrame(value, fallback = DEFAULT_EXPORT_FRAME) {
  if (EXPORT_FRAMES.some(({ id }) => id === value)) return value;
  return EXPORT_FRAMES.some(({ id }) => id === fallback) ? fallback : 'none';
}

const fill = (context, color, x, y, width, height) => {
  context.fillStyle = color;
  context.fillRect(x, y, width, height);
};

/** Four white rules. Every visible edge in these frames is one of these. */
function outline(context, x, y, width, height, line = 1) {
  fill(context, INK, x, y, width, line);
  fill(context, INK, x, y + height - line, width, line);
  fill(context, INK, x, y, line, height);
  fill(context, INK, x + width - line, y, line, height);
}

const ditherPatterns = new WeakMap();

/** Ordered dither fill: tiny white cells on black, never a smooth gradient. */
function dither(context, x, y, width, height, coverage, unit = 1) {
  let patterns = ditherPatterns.get(context);
  if (!patterns) {
    patterns = new Map();
    ditherPatterns.set(context, patterns);
  }
  const key = `${coverage}:${unit}`;
  let pattern = patterns.get(key);
  if (!pattern) {
    const size = 4 * unit;
    const tile = document.createElement('canvas');
    tile.width = size;
    tile.height = size;
    const tileContext = tile.getContext('2d');
    tileContext.fillStyle = INK;
    const threshold = Math.max(0, Math.min(16, coverage * 16));
    for (let row = 0; row < 4; row++) {
      for (let column = 0; column < 4; column++) {
        if (BAYER_4[row][column] < threshold) {
          tileContext.fillRect(column * unit, row * unit, unit, unit);
        }
      }
    }
    pattern = context.createPattern(tile, 'repeat');
    patterns.set(key, pattern);
  }
  context.fillStyle = pattern;
  context.fillRect(x, y, width, height);
  context.fillStyle = INK;
}

function roundRectPath(context, x, y, width, height, radius) {
  const r = Math.max(0, Math.min(radius, width / 2, height / 2));
  context.beginPath();
  context.moveTo(x + r, y);
  context.lineTo(x + width - r, y);
  context.quadraticCurveTo(x + width, y, x + width, y + r);
  context.lineTo(x + width, y + height - r);
  context.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  context.lineTo(x + r, y + height);
  context.quadraticCurveTo(x, y + height, x, y + height - r);
  context.lineTo(x, y + r);
  context.quadraticCurveTo(x, y, x + r, y);
  context.closePath();
}

function fitText(context, value, maxWidth) {
  if (maxWidth <= 0) return '';
  if (context.measureText(value).width <= maxWidth) return value;
  const ellipsis = '...';
  let text = value;
  while (text && context.measureText(text + ellipsis).width > maxWidth) text = text.slice(0, -1);
  return text ? text + ellipsis : '';
}

function setText(context, font, align = 'left', baseline = 'middle') {
  context.font = font;
  context.fillStyle = INK;
  context.textAlign = align;
  context.textBaseline = baseline;
}

function glyphBox(context, x, y, width, height, kind, unit) {
  outline(context, x, y, width, height, Math.max(1, unit));
  const inset = Math.max(unit * 3, Math.round(Math.min(width, height) * 0.28));
  context.strokeStyle = INK;
  context.fillStyle = INK;
  context.lineWidth = Math.max(1, unit);
  context.lineCap = 'butt';
  if (kind === 'close') {
    context.beginPath();
    context.moveTo(x + inset, y + inset);
    context.lineTo(x + width - inset, y + height - inset);
    context.moveTo(x + width - inset, y + inset);
    context.lineTo(x + inset, y + height - inset);
    context.stroke();
  } else if (kind === 'minimize') {
    fill(context, INK, x + inset, y + height - inset - unit, width - inset * 2, Math.max(unit, unit * 2));
  } else {
    const box = Math.max(unit * 5, Math.min(width, height) - inset * 2);
    outline(context, x + width - inset - box, y + inset, box, box, Math.max(1, unit));
  }
}

function drawMac(context, source, width, height, scale, title) {
  const u = (value) => value * scale;
  const border = Math.max(1, u(1));
  const pad = u(5);
  const titleHeight = u(22);
  const windowWidth = width + pad * 2 + border * 2;
  const windowHeight = titleHeight + height + pad * 2 + border * 2;
  const margin = u(36);
  const desktopWidth = Math.max(windowWidth + margin * 2, u(380));
  const desktopHeight = Math.max(windowHeight + margin * 2, u(300));
  const windowX = Math.round((desktopWidth - windowWidth) / 2);
  const windowY = Math.round((desktopHeight - windowHeight) / 2);
  const titleX = windowX + border;
  const titleY = windowY + border;
  const titleWidth = windowWidth - border * 2;

  fill(context, BACKGROUND, 0, 0, desktopWidth, desktopHeight);
  dither(context, 0, 0, desktopWidth, desktopHeight, 0.125, scale);
  fill(context, BACKGROUND, windowX, windowY, windowWidth, windowHeight);
  outline(context, windowX, windowY, windowWidth, windowHeight, border);
  fill(context, BACKGROUND, titleX, titleY, titleWidth, titleHeight);
  for (let y = titleY + u(2); y < titleY + titleHeight - u(2); y += u(3)) {
    fill(context, INK, titleX + u(1), y, titleWidth - u(2), Math.max(1, u(1)));
  }

  const controlSize = u(13);
  const controlY = titleY + Math.round((titleHeight - controlSize) / 2);
  const controlLine = Math.max(1, u(1) / 2);
  fill(context, BACKGROUND, titleX + u(5), controlY, controlSize + u(2), controlSize + u(2));
  glyphBox(context, titleX + u(6), controlY, controlSize, controlSize, 'close', controlLine);
  fill(context, BACKGROUND, titleX + titleWidth - u(6) - controlSize * 2 - u(5), controlY,
    controlSize * 2 + u(5), controlSize + u(2));
  glyphBox(context, titleX + titleWidth - u(6) - controlSize, controlY, controlSize, controlSize,
    'minimize', controlLine);
  glyphBox(context, titleX + titleWidth - u(6) - controlSize * 2 - u(4), controlY, controlSize,
    controlSize, 'maximize', controlLine);

  setText(context, `700 ${u(12)}px ${MONO_FONT}`, 'center');
  const text = fitText(context, title, titleWidth - controlSize * 2 - u(28));
  const textWidth = context.measureText(text).width + u(10);
  fill(context, BACKGROUND, titleX + titleWidth / 2 - textWidth / 2, titleY + u(4),
    textWidth, titleHeight - u(8));
  context.fillStyle = INK;
  context.fillText(text, titleX + titleWidth / 2, titleY + titleHeight / 2 + u(1));

  const artX = windowX + border + pad;
  const artY = windowY + border + titleHeight + pad;
  outline(context, artX - u(1), artY - u(1), width + u(2), height + u(2), u(1));
  context.drawImage(source, artX, artY);
}

function drawWin95Chrome(context, source, width, height, scale, title, now) {
  const u = (value) => value * scale;
  const border = u(3);
  const pad = u(3);
  const titleHeight = u(20);
  const windowWidth = width + border * 2 + pad * 2;
  const windowHeight = titleHeight + height + border * 2 + pad * 2;
  const margin = u(44);
  const taskHeight = u(28);
  const desktopWidth = Math.max(windowWidth + margin * 2, u(460));
  const desktopHeight = Math.max(windowHeight + margin * 2 + taskHeight, u(340));
  const workHeight = desktopHeight - taskHeight;
  const windowX = Math.round((desktopWidth - windowWidth) / 2);
  const windowY = Math.round(Math.max(0, (workHeight - windowHeight) / 2));

  fill(context, BACKGROUND, 0, 0, desktopWidth, desktopHeight);
  fill(context, BACKGROUND, windowX, windowY, windowWidth, windowHeight);
  outline(context, windowX, windowY, windowWidth, windowHeight, Math.max(1, u(1)));

  const titleX = windowX + border;
  const titleY = windowY + border;
  const titleWidth = windowWidth - border * 2;
  fill(context, BACKGROUND, titleX, titleY, titleWidth, titleHeight);
  fill(context, INK, titleX, titleY + titleHeight - Math.max(1, u(1)), titleWidth, Math.max(1, u(1)));

  const controls = ['minimize', 'maximize', 'close'];
  const buttonWidth = u(16);
  const buttonHeight = u(14);
  const buttonGap = u(2);
  const controlsWidth = buttonWidth * controls.length + buttonGap * (controls.length - 1);
  const controlsX = titleX + titleWidth - u(3) - controlsWidth;
  setText(context, `700 ${u(11)}px ${MONO_FONT}`);
  context.fillText(fitText(context, title, controlsX - titleX - u(10)),
    titleX + u(6), titleY + titleHeight / 2 + u(1));
  controls.forEach((kind, index) => {
    glyphBox(context, controlsX + index * (buttonWidth + buttonGap),
      titleY + Math.round((titleHeight - buttonHeight) / 2), buttonWidth, buttonHeight, kind,
      Math.max(1, u(1)));
  });

  const contentX = windowX + border;
  const contentY = windowY + border + titleHeight;
  const contentWidth = windowWidth - border * 2;
  const contentHeight = height + pad * 2;
  outline(context, contentX, contentY, contentWidth, contentHeight, Math.max(1, u(1)));
  const artX = contentX + pad;
  const artY = contentY + pad;
  context.drawImage(source, artX, artY);

  drawWin95Taskbar(context, desktopWidth, desktopHeight, scale, now);
}

function drawWin95Taskbar(context, desktopWidth, desktopHeight, scale, now) {
  const u = (value) => value * scale;
  const height = u(28);
  const y = desktopHeight - height;
  fill(context, BACKGROUND, 0, y, desktopWidth, height);
  fill(context, INK, 0, y, desktopWidth, Math.max(1, u(1)));

  const startWidth = u(68);
  const buttonHeight = height - u(4);
  const startY = y + u(2);
  outline(context, u(2), startY, startWidth, buttonHeight, Math.max(1, u(1)));

  const flagX = u(7);
  const flagSize = u(13);
  const flagY = startY + Math.round((buttonHeight - flagSize) / 2);
  const half = Math.floor(flagSize / 2);
  outline(context, flagX, flagY, flagSize, flagSize, Math.max(1, u(1)));
  fill(context, INK, flagX + half, flagY, Math.max(1, u(1)), flagSize);
  fill(context, INK, flagX, flagY + half, flagSize, Math.max(1, u(1)));
  setText(context, `700 ${u(11)}px ${MONO_FONT}`);
  const startTextX = flagX + flagSize + u(6);
  context.fillText(fitText(context, 'Start', startWidth - (startTextX - u(2)) - u(4)),
    startTextX, startY + buttonHeight / 2 + u(1));

  const trayWidth = u(66);
  const taskX = u(2) + startWidth + u(5);
  const taskWidth = Math.max(0, Math.min(u(210), desktopWidth - taskX - trayWidth - u(8)));
  if (taskWidth > u(42)) {
    outline(context, taskX, startY, taskWidth, buttonHeight, Math.max(1, u(1)));
    dither(context, taskX + u(1), startY + u(1), taskWidth - u(2), buttonHeight - u(2), 0.125, scale);
    setText(context, `700 ${u(11)}px ${MONO_FONT}`);
    context.fillText(fitText(context, 'clack - ASCII Camera', taskWidth - u(10)),
      taskX + u(5), startY + buttonHeight / 2 + u(1));
  }

  const trayX = desktopWidth - trayWidth - u(2);
  outline(context, trayX, startY, trayWidth, buttonHeight, Math.max(1, u(1)));
  setText(context, `400 ${u(10)}px ${MONO_FONT}`, 'right');
  context.fillText(now.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }),
    trayX + trayWidth - u(5), startY + buttonHeight / 2 + u(1));
}

function drawXpButton(context, x, y, width, height, kind, unit) {
  const line = Math.max(1, unit);
  fill(context, BACKGROUND, x, y, width, height);
  roundRectPath(context, x, y, width, height, unit * 3);
  context.strokeStyle = INK;
  context.lineWidth = line;
  context.stroke();
  const inset = Math.max(unit * 4, Math.round(width * 0.28));
  context.strokeStyle = INK;
  context.fillStyle = INK;
  context.lineWidth = Math.max(1, unit * 2);
  context.lineCap = 'butt';
  if (kind === 'minimize') {
    context.beginPath();
    context.moveTo(x + inset, y + height - inset);
    context.lineTo(x + width - inset, y + height - inset);
    context.stroke();
  } else if (kind === 'maximize') {
    context.strokeRect(x + inset, y + inset, width - inset * 2, height - inset * 2);
  } else {
    context.beginPath();
    context.moveTo(x + inset, y + inset);
    context.lineTo(x + width - inset, y + height - inset);
    context.moveTo(x + width - inset, y + inset);
    context.lineTo(x + inset, y + height - inset);
    context.stroke();
  }
}

function drawXpTaskbar(context, desktopWidth, desktopHeight, scale, now) {
  const u = (value) => value * scale;
  const height = u(30);
  const y = desktopHeight - height;
  fill(context, BACKGROUND, 0, y, desktopWidth, height);
  fill(context, INK, 0, y, desktopWidth, Math.max(1, u(1)));
  fill(context, INK, 0, y + height - Math.max(1, u(1)), desktopWidth, Math.max(1, u(1)));

  const startX = u(2);
  const startY = y + u(2);
  const startWidth = u(86);
  const startHeight = height - u(4);
  const radius = u(12);
  const startPath = () => {
    context.beginPath();
    context.moveTo(startX, startY);
    context.lineTo(startX + startWidth - radius, startY);
    context.quadraticCurveTo(startX + startWidth, startY, startX + startWidth, startY + radius);
    context.lineTo(startX + startWidth, startY + startHeight - radius);
    context.quadraticCurveTo(startX + startWidth, startY + startHeight,
      startX + startWidth - radius, startY + startHeight);
    context.lineTo(startX, startY + startHeight);
    context.closePath();
  };
  startPath();
  context.strokeStyle = INK;
  context.lineWidth = Math.max(1, u(1));
  context.stroke();

  const flagX = startX + u(8);
  const flagSize = u(13);
  const flagY = startY + Math.round((startHeight - flagSize) / 2);
  const half = Math.floor(flagSize / 2);
  outline(context, flagX, flagY, flagSize, flagSize, Math.max(1, u(1)));
  fill(context, INK, flagX + half, flagY, Math.max(1, u(1)), flagSize);
  fill(context, INK, flagX, flagY + half, flagSize, Math.max(1, u(1)));
  setText(context, `700 ${u(13)}px ${MONO_FONT}`);
  context.fillText('start', flagX + flagSize + u(5), startY + startHeight / 2 + u(1));

  const trayWidth = u(84);
  const taskX = startX + startWidth + u(4);
  const taskWidth = Math.max(0, Math.min(u(210), desktopWidth - taskX - trayWidth - u(8)));
  if (taskWidth > u(50)) {
    fill(context, BACKGROUND, taskX, y + u(3), taskWidth, height - u(6));
    outline(context, taskX, y + u(3), taskWidth, height - u(6), Math.max(1, u(1)));
    dither(context, taskX + u(1), y + u(4), taskWidth - u(2), height - u(8), 0.125, scale);
    setText(context, `400 ${u(11)}px ${MONO_FONT}`);
    context.fillText(fitText(context, 'clack - ASCII Camera', taskWidth - u(12)),
      taskX + u(7), y + height / 2 + u(1));
  }

  const trayX = desktopWidth - trayWidth - u(2);
  fill(context, INK, trayX, y + u(1), Math.max(1, u(1)), height - u(2));
  setText(context, `400 ${u(11)}px ${MONO_FONT}`, 'right');
  context.fillText(now.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }),
    trayX + trayWidth - u(7), y + height / 2 + u(1));
}

function drawXp(context, source, width, height, scale, title, now) {
  const u = (value) => value * scale;
  const border = u(3);
  const pad = u(4);
  const titleHeight = u(28);
  const windowWidth = width + border * 2 + pad * 2;
  const windowHeight = titleHeight + height + border * 2 + pad * 2;
  const margin = u(44);
  const taskHeight = u(30);
  const desktopWidth = Math.max(windowWidth + margin * 2, u(520));
  const desktopHeight = Math.max(windowHeight + margin * 2 + taskHeight, u(320));
  const workHeight = desktopHeight - taskHeight;
  const windowX = Math.round((desktopWidth - windowWidth) / 2);
  const windowY = Math.round(Math.max(0, (workHeight - windowHeight) / 2));
  const radius = u(8);

  fill(context, BACKGROUND, 0, 0, desktopWidth, desktopHeight);
  fill(context, BACKGROUND, windowX, windowY, windowWidth, windowHeight);
  context.save();
  roundRectPath(context, windowX, windowY, windowWidth, windowHeight, radius);
  context.clip();
  fill(context, BACKGROUND, windowX, windowY + titleHeight, windowWidth, windowHeight - titleHeight);
  fill(context, INK, windowX, windowY + titleHeight, windowWidth, Math.max(1, u(1)));
  context.restore();

  const controls = ['minimize', 'maximize', 'close'];
  const buttonSize = u(21);
  const gap = u(2);
  const controlsWidth = buttonSize * controls.length + gap * (controls.length - 1);
  const controlsX = windowX + windowWidth - u(4) - controlsWidth;
  const buttonY = windowY + u(4);
  controls.forEach((kind, index) => {
    drawXpButton(context, controlsX + index * (buttonSize + gap), buttonY, buttonSize, buttonSize,
      kind, scale);
  });

  setText(context, `700 ${u(12)}px ${MONO_FONT}`);
  context.fillText(fitText(context, title, controlsX - windowX - u(14)),
    windowX + u(10), windowY + titleHeight / 2 + u(1));

  const artX = windowX + border + pad;
  const artY = windowY + titleHeight + pad;
  outline(context, artX - u(1), artY - u(1), width + u(2), height + u(2), Math.max(1, u(1)));
  context.drawImage(source, artX, artY);

  roundRectPath(context, windowX, windowY, windowWidth, windowHeight, radius);
  context.strokeStyle = INK;
  context.lineWidth = Math.max(1, u(2));
  context.stroke();

  drawXpTaskbar(context, desktopWidth, desktopHeight, scale, now);
}

/** Draw the chosen desktop around the already-rendered artwork canvas. */
export function createExportFrame(source, style, options = {}) {
  const nextStyle = normalizeExportFrame(style, 'none');
  if (nextStyle === 'none') return source;
  if (!source || !Number.isSafeInteger(source.width) || !Number.isSafeInteger(source.height) ||
      source.width <= 0 || source.height <= 0) {
    throw new TypeError('The saved artwork is not a valid canvas');
  }
  const { title = 'clack - ASCII Camera', now = new Date() } = options;
  const canvas = document.createElement('canvas');
  try {
    const context = canvas.getContext('2d', { alpha: false });
    if (!context) throw new Error('Unable to create the save frame');
    const scale = Math.min(4, Math.max(1, Math.ceil(Math.max(source.width, source.height) / 1100)));
    const width = source.width;
    const height = source.height;
    const metrics = {
      win95: { margin: 44, task: 28 },
      winxp: { margin: 44, task: 30 },
      mac: { margin: 36, task: 0 },
    }[nextStyle];
    const titleHeight = nextStyle === 'mac' ? 22 : nextStyle === 'winxp' ? 28 : 20;
    const border = nextStyle === 'mac' ? 1 : 3;
    const pad = nextStyle === 'mac' ? 5 : nextStyle === 'winxp' ? 4 : 3;
    const u = (value) => value * scale;
    const windowWidth = width + u(border * 2 + pad * 2);
    const windowHeight = u(titleHeight) + height + u(border * 2 + pad * 2);
    const minWidth = nextStyle === 'mac' ? 380 : nextStyle === 'winxp' ? 520 : 460;
    const minHeight = nextStyle === 'mac' ? 300 : nextStyle === 'winxp' ? 320 : 340;
    const desktopWidth = Math.max(windowWidth + u(metrics.margin * 2), u(minWidth));
    const desktopHeight = Math.max(windowHeight + u(metrics.margin * 2 + metrics.task), u(minHeight));
    canvas.width = desktopWidth;
    canvas.height = desktopHeight;
    if (nextStyle === 'mac') drawMac(context, source, width, height, scale, title.split(' - ')[0]);
    else if (nextStyle === 'winxp') drawXp(context, source, width, height, scale, title, now);
    else drawWin95Chrome(context, source, width, height, scale, title, now);
    return canvas;
  } catch (error) {
    canvas.width = 0;
    canvas.height = 0;
    throw error;
  }
}
