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
});
