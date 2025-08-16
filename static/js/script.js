document.addEventListener("DOMContentLoaded", function () {
  // Atualiza label do input de ovos
  const ovosInput = document.getElementById('file-ovos');
  const ovosLabel = document.querySelector('label[for="file-ovos"].file-label');
  if (ovosInput && ovosLabel) {
    ovosInput.addEventListener('change', function () {
      if (this.files && this.files[0]) ovosLabel.innerText = this.files[0].name;
    });
  }

  // Atualiza label do input de calibração
  const calibInput = document.getElementById('file-calib');
  const calibLabel = document.querySelector('label[for="file-calib"].file-label');
  if (calibInput && calibLabel) {
    calibInput.addEventListener('change', function () {
      if (this.files && this.files[0]) calibLabel.innerText = this.files[0].name;
    });
  }
});
