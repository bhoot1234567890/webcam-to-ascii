/** Fit-only presentation; the renderer's grid, text, and cell proportions stay intact. */
export function createOutputView(viewport, output, fitButton) {
  const stage = output.parentElement;
  const guides = stage.querySelector('.composition-grid');
  let width = 0;
  let height = 0;
  let artworkWidth = 0;
  let artworkHeight = 0;

  function layout() {
    const style = getComputedStyle(viewport);
    // clientWidth/Height round to integers; retain fractional CSS sizes so Fit cannot clip a subpixel.
    const innerWidth = parseFloat(style.width) - parseFloat(style.borderLeftWidth) - parseFloat(style.borderRightWidth);
    const innerHeight = parseFloat(style.height) - parseFloat(style.borderTopWidth) - parseFloat(style.borderBottomWidth);
    const availableWidth = Math.min(viewport.clientWidth, innerWidth) - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
    const availableHeight = Math.min(viewport.clientHeight, innerHeight) - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom);
    // The halo is presentation-only and occupies screen pixels, not extra ASCII cells.
    const gutter = document.body.dataset.glow === 'on' && width && height ? 8 : 0;
    const scale = width && height
      ? Math.max(0, Math.min(1, (availableWidth - gutter * 2) / width, (availableHeight - gutter * 2) / height)) : 1;
    stage.style.width = `${width * scale + gutter * 2}px`;
    stage.style.height = `${height * scale + gutter * 2}px`;
    output.style.left = `${gutter}px`;
    output.style.top = `${gutter}px`;
    output.style.transform = `scale(${scale})`;
    if (guides) {
      guides.style.left = `${gutter}px`;
      guides.style.top = `${gutter}px`;
      guides.style.width = `${artworkWidth * scale}px`;
      guides.style.height = `${artworkHeight * scale}px`;
    }
    viewport.dataset.view = 'fit';
  }

  const fit = () => {
    viewport.scrollTop = 0;
    viewport.scrollLeft = 0;
    layout();
  };
  fitButton.setAttribute('aria-pressed', 'true');
  fitButton.addEventListener('click', fit);
  const observer = new ResizeObserver(layout);
  observer.observe(viewport);
  fit();

  return {
    update() {
      // Layout dimensions exclude the transform; measure only when the grid changes.
      const style = getComputedStyle(output);
      artworkWidth = parseFloat(style.width);
      artworkHeight = parseFloat(style.height);
      width = Math.ceil(artworkWidth);
      height = Math.ceil(artworkHeight) + 2; // Preserve bottom ink overhang.
      layout();
    },
    dispose() {
      observer.disconnect();
      fitButton.removeEventListener('click', fit);
    },
  };
}
