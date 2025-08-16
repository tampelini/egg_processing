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
