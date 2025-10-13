document.addEventListener("DOMContentLoaded", function () {
  // Atualiza label do input de ovos
  const ovosInput = document.getElementById('file-ovos');
  const ovosLabel = document.querySelector('label[for="file-ovos"].file-label');
  if (ovosInput && ovosLabel) {
    ovosInput.addEventListener('change', function () {
      if (this.files && this.files[0]) {
        ovosLabel.innerText = this.files[0].name;
      }
    });
  }

  // Atualiza label do input de calibração
  const calibInput = document.getElementById('file-calib');
  const calibLabel = document.querySelector('label[for="file-calib"].file-label');
  if (calibInput && calibLabel) {
    calibInput.addEventListener('change', function () {
      if (this.files && this.files[0]) {
        calibLabel.innerText = this.files[0].name;
      }
    });
  }

  // Atualização dinâmica dos valores numéricos dos ajustes
  document.querySelectorAll('[data-value-for]').forEach(function (span) {
    const targetId = span.getAttribute('data-value-for');
    if (!targetId) return;
    const input = document.getElementById(targetId);
    if (!input) return;
    const decimals = parseInt(input.dataset.decimals || '2', 10);
    const format = function (value) {
      if (Number.isFinite(value)) {
        return value.toFixed(decimals);
      }
      return '--';
    };
    const update = function () {
      const val = parseFloat(input.value);
      span.textContent = format(val);
    };
    update();
    input.addEventListener('input', update);
    input.addEventListener('change', update);
  });

  // Interação de destaque das cores
  const highlightOverlay = document.getElementById('highlight-overlay');
  const highlightContainer = document.querySelector('.image-highlight-wrapper');
  let currentOverlayElement = null;

  const showOverlay = (b64, el) => {
    if (!highlightOverlay || !b64) return;
    currentOverlayElement = el;
    const dataUrl = `data:image/png;base64,${b64}`;
    if (highlightOverlay.src !== dataUrl) {
      highlightOverlay.src = dataUrl;
    }
    highlightOverlay.classList.add('active');
  };

  const clearOverlay = (el) => {
    if (!highlightOverlay) return;
    if (el && currentOverlayElement && el !== currentOverlayElement) {
      return;
    }
    currentOverlayElement = null;
    highlightOverlay.classList.remove('active');
    highlightOverlay.removeAttribute('src');
  };

  const bindOverlayListeners = (element) => {
    if (!element) return;
    const overlayData = element.getAttribute('data-overlay');
    if (!overlayData) return;
    const show = () => showOverlay(overlayData, element);
    const hide = () => clearOverlay(element);
    element.addEventListener('mouseenter', show);
    element.addEventListener('focus', show);
    element.addEventListener('mouseleave', hide);
    element.addEventListener('blur', hide);
  };

  document.querySelectorAll('.cor-item[data-overlay]').forEach(bindOverlayListeners);
  document.querySelectorAll('.media-chip[data-overlay]').forEach(bindOverlayListeners);

  if (highlightContainer) {
    highlightContainer.addEventListener('mouseleave', () => {
      clearOverlay(currentOverlayElement);
    });
  }
});
