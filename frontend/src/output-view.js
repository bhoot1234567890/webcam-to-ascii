/** Fit presentation only; the renderer's grid, text, and cell proportions stay intact. */
export function createOutputView(viewport, output, fitButton, detailButton) {
  const stage = output.parentElement;
  let mode = matchMedia('(max-width: 1000px)').matches ? 'fit' : 'detail';
  let width = 0;
  let height = 0;

  function layout() {
    const style = getComputedStyle(viewport);
    // clientWidth/Height round to integers; retain fractional CSS sizes so Fit cannot clip a subpixel.
    const innerWidth = parseFloat(style.width) - parseFloat(style.borderLeftWidth) - parseFloat(style.borderRightWidth);
    const innerHeight = parseFloat(style.height) - parseFloat(style.borderTopWidth) - parseFloat(style.borderBottomWidth);
    const availableWidth = Math.min(viewport.clientWidth, innerWidth) - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
    const availableHeight = Math.min(viewport.clientHeight, innerHeight) - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom);
    const scale = mode === 'fit' && width && height
      ? Math.min(1, availableWidth / width, availableHeight / height) : 1;
    stage.style.width = `${width * scale}px`;
    stage.style.height = `${height * scale}px`;
    output.style.transform = `scale(${scale})`;
  }

  function select(next) {
    mode = next;
    fitButton.setAttribute('aria-pressed', String(mode === 'fit'));
    detailButton.setAttribute('aria-pressed', String(mode === 'detail'));
    viewport.dataset.view = mode;
    viewport.scrollTop = 0;
    viewport.scrollLeft = 0;
    layout();
  }

  const fit = () => select('fit');
  const detail = () => select('detail');
  fitButton.addEventListener('click', fit);
  detailButton.addEventListener('click', detail);
  const observer = new ResizeObserver(layout);
  observer.observe(viewport);
  select(mode);

  return {
    update() {
      // Layout dimensions exclude the transform; measure only when the grid changes.
      const style = getComputedStyle(output);
      width = Math.ceil(parseFloat(style.width));
      height = Math.ceil(parseFloat(style.height)) + 2; // Preserve bottom ink overhang.
      layout();
    },
    dispose() {
      observer.disconnect();
      fitButton.removeEventListener('click', fit);
      detailButton.removeEventListener('click', detail);
    },
  };
}
