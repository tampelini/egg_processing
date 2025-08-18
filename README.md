# 🥚 Egg Classify – Sistema de Processamento e Calibração de Imagens

Este projeto é um sistema web desenvolvido em **Flask** para **classificação e calibração de imagens de ovos**.  
Ele possui uma interface simples em HTML/CSS que permite ao usuário:

- **Processar imagens** de ovos usando algoritmos definidos em `app/processamento.py`.
- **Calibrar imagens** com o auxílio de uma paleta **ColorChecker**, detectando marcadores **ArUco** e salvando a configuração em `config/color_calibration.json`.

---

## 📂 Estrutura do Projeto

```bash
egg_processing/
│
├── app/                      # Código principal do Flask
│   ├── __init__.py
│   ├── routes.py             # Rotas da aplicação
│   ├── processamento.py      # Script de processamento de imagens
│   ├── calibrate.py          # Script de calibração com ColorChecker
│   └── templates/            # Templates HTML
│       └── index.html
│
├── config/
│   └── color_calibration.json   # Arquivo salvo após calibração
│
├── palette/                   # Utilitários de paleta de cores
├── static/
│   ├── css/
│   ├── js/
│   ├── uploads/
│   ├── calibrated/
│   └── output/
│
├── requirements.txt
├── wsgi.py
└── README.md
```
