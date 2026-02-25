# 🥚 Egg Classify – Sistema de Processamento e Calibração de Imagens

Este projeto é um sistema web desenvolvido em **Flask** para **classificação e calibração de imagens de ovos**.  
Ele possui uma interface simples em HTML/CSS que permite ao usuário:

- **Processar imagens** de ovos usando algoritmos definidos no `processamento.py`.
- **Calibrar imagens** com o auxílio de uma paleta **ColorChecker**, detectando marcadores **ArUco** e salvando a configuração em `config/colorcheck_config.json`.

---

## 📂 Estrutura do Projeto

```bash
egg_classify/
│
├── app.py                 # Código principal Flask
├── processamento.py       # Script de processamento de imagens
├── calibrate.py           # Script de calibração com ColorChecker
│
├── templates/             # Templates HTML (renderizados pelo Flask)
│   ├── index.html         # Página inicial com as duas opções (Processar e Calibrar)
│   ├── result.html        # Página de resultados do processamento
│   ├── calibration_result.html  # Página de resultados da calibração
│
├── static/
│   ├── css/
│   │   └── style.css      # Arquivo de estilos
│   ├── uploads/           # Imagens enviadas pelos usuários
│   └── calibrated/        # Imagens calibradas (resultado do ColorChecker)
│
├── config/
│   └── colorcheck_config.json   # Arquivo salvo após calibração
│
├── data/                  # Exemplos e arquivos auxiliares
│   ├── *.csv              # Saídas de referência
│   └── *.json             # Configurações
│
└── README.md              # Documentação do projeto


## 🔬 Algoritmo atual de calibração e detecção (passo a passo)

### 1) Fluxo de calibração manual (rota `/calibrar`)

1. **Upload da imagem** e eventual conversão HEIC/HEIF para JPG.
2. Se marcado, o sistema pode:
   - **usar imagem como referência** para recriar `config/ref_colors.csv`;
   - **ignorar calibração existente** e recalcular;
   - **apagar calibrações anteriores** (`color_calibration.json`, `ref_colors.csv`, `measured_*.csv`).
3. Se houver calibração salva e o usuário não ignorar, o sistema só aplica a matriz já existente.
4. Caso contrário, o sistema executa a calibração completa com detecção da paleta, warp, amostragem e solve da matriz 3x4.
5. A interface exibe artefatos de debug (detecção ArUco, warp, labels, imagem calibrada) e status textual para diagnóstico.

---

### 2) Detecção da paleta (ArUco)

A detecção usa 4 marcadores ArUco esperados nos cantos da paleta (IDs **10, 11, 12, 13**):

1. **Pré-processamento de detecção em cópia da imagem** (nitidez via unsharp) para melhorar reconhecimento dos marcadores.
2. Tentativa de detecção ArUco na cópia nítida.
3. Fallback para detecção na imagem original caso a primeira tentativa não encontre marcadores suficientes.
4. Ordenação dos cantos no padrão TL/TR/BR/BL usando os IDs esperados.
5. Se IDs esperados não forem encontrados, o fluxo para antes da amostragem e retorna motivo.

> Observação importante: o pré-processamento de nitidez é usado **somente para detectar a paleta**. O restante do pipeline segue com a imagem original.

---

### 3) Warp, amostragem e solução da calibração

Após detectar a paleta:

1. Faz o **warp em perspectiva** para tamanho fixo (`WARP_W x WARP_H`).
2. Define os 24 centros dos patches (12x2) por:
   - `config/patch_centers.csv` (se existir), ou
   - grade paramétrica (`MARGIN_X/Y`, `GAP_X/Y`).
3. Lê média RGB de cada patch em uma janela (`SAMPLE_WIN`).
4. Carrega `config/ref_colors.csv` como alvo real de referência.
5. Resolve a matriz afim 3x4 com regularização (ridge/Tikhonov).
6. Aplica a matriz na imagem inteira e salva resultado em `static/calibrated/`.
7. Persiste parâmetros em `config/color_calibration.json` para reutilização futura.

---

### 4) Fluxo automático durante o processamento de ovos

No pipeline principal de ovos (`processar_imagem`):

1. O sistema tenta auto-calibrar com a paleta, se habilitado.
2. Se não detectar paleta, processa com a imagem original sem interromper o fluxo.
3. Depois aplica os ajustes manuais da interface na imagem-base:
   - brilho, exposição, contraste, saturação, temperatura e nitidez;
   - **matiz do vermelho (manual)** para deslocar apenas tons vermelhos no HSV.
4. Segmenta ovos na imagem de trabalho e calcula métricas de cor por ovo.

---

### 5) Checklist rápido para melhorar detecção da paleta

- Garanta os 4 ArUco inteiros e nítidos na foto.
- Evite reflexo especular sobre os marcadores/paleta.
- Use iluminação homogênea e foco travado.
- Verifique se os IDs da sua arte de paleta são realmente 10/11/12/13.
- Se necessário, use `patch_centers.csv` para alinhar amostragem à impressão real da paleta.
