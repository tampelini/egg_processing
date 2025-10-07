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

  // Controle do fator de claridade (fator_v_backup)
  const fatorInput = document.getElementById("fator-v-backup");
  const fatorValue = document.getElementById("fator-v-backup-value");

  if (fatorInput && fatorValue) {
    // valor inicial
    fatorValue.innerText = fatorInput.value;

    // atualiza dinamicamente
    fatorInput.addEventListener("input", function () {
      fatorValue.innerText = this.value;
    });
  }
});
