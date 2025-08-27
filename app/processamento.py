import cv2
import numpy as np
import io
import base64
import logging
from skimage.color import rgb2lab, rgb2xyz
import colour

logging.basicConfig(level=logging.INFO)

def rgb_to_cmyk(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    k = 1 - max(r, g, b)
    if k == 1:
        return (0, 0, 0, 100)
    c = (1 - r - k) / (1 - k)
    m = (1 - g - k) / (1 - k)
    y = (1 - b - k) / (1 - k)
    return (round(c * 100), round(m * 100), round(y * 100), round(k * 100))

def rgb_to_hex(rgb):
    return '#{:02X}{:02X}{:02X}'.format(*rgb)

def aplicar_unsharp_mask(imagem, alpha=1.5, beta=-0.85):
    borrada = cv2.GaussianBlur(imagem, (0, 0), 3)
    return cv2.addWeighted(imagem, alpha, borrada, beta, 0)

def destacar_bordas(imagem):
    if len(imagem.shape) == 3:
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    else:
        cinza = imagem.copy()
    bordas = cv2.Canny(cinza, 100, 200)
    kernel = np.ones((3, 3), np.uint8)
    bordas_dilatadas = cv2.dilate(bordas, kernel, iterations=1)
    imagem_bordas = imagem.copy()
    if len(imagem_bordas.shape) == 2:
        imagem_bordas = cv2.cvtColor(imagem_bordas, cv2.COLOR_GRAY2BGR)
    imagem_bordas[bordas_dilatadas > 0] = [0, 0, 255]
    return imagem_bordas

def verificar_predominio_cinza(imagem_cv):
    media_rgb = cv2.mean(imagem_cv)[:3]
    variancia_rgb = np.var(media_rgb)
    print(variancia_rgb)
    return variancia_rgb < 100  # limiar ajustável

def _ler_imagem_cv(imagem):
    """Aceita: caminho (str), bytes/bytearray, file-like com .read(), numpy array (BGR). Retorna BGR uint8."""
    # numpy array já em BGR
    if isinstance(imagem, np.ndarray):
        img = imagem
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    # bytes / bytearray
    if isinstance(imagem, (bytes, bytearray)):
        data = np.frombuffer(imagem, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Falha ao decodificar bytes de imagem.")
        return img

    # file-like (ex.: FileStorage do Flask)
    if hasattr(imagem, "read"):
        raw = imagem.read()
        if hasattr(imagem, "seek"):
            try:
                imagem.seek(0)
            except Exception:
                pass
        data = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Falha ao decodificar stream de imagem.")
        return img

    # caminho de arquivo
    if isinstance(imagem, str):
        img = cv2.imread(imagem, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Falha ao ler a imagem em: {imagem}")
        return img

    raise TypeError("Tipo de entrada de imagem não suportado.")

def processar_imagem(imagem):
    # Lê a imagem em BGR
    imagem_cv = _ler_imagem_cv(imagem)
    imagem_backup = imagem_cv.copy()

    if verificar_predominio_cinza(imagem_cv):
        imagem_cv = destacar_bordas(imagem_cv)
        imagem_cv = aplicar_unsharp_mask(imagem_cv)
        imagem_cv = cv2.GaussianBlur(imagem_cv, (61, 11), 0)

    h_original, w_original = imagem_cv.shape[:2]
    target_h, target_w = 826, 1113
    scale = min(target_w / w_original, target_h / h_original)
    if scale < 1:
        new_w, new_h = int(w_original * scale), int(h_original * scale)
        imagem_cv = cv2.resize(imagem_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)
        imagem_backup = cv2.resize(imagem_backup, (new_w, new_h), interpolation=cv2.INTER_AREA)

    imagem_cv = cv2.GaussianBlur(imagem_cv, (5, 5), 0)
    lab = cv2.cvtColor(imagem_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    imagem_cv = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    cinza = cv2.cvtColor(imagem_cv, cv2.COLOR_BGR2GRAY)
    cinza = cv2.GaussianBlur(cinza, (5, 5), 0)
    binario = cv2.bitwise_not(
        cv2.adaptiveThreshold(cinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 11, 2)
    )

    contornos, _ = cv2.findContours(binario, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = [ctr for ctr in contornos if 100 < cv2.contourArea(ctr) < 50000]
    boxes = [cv2.boundingRect(ctr) for ctr in contornos]

    median_w = float(np.median([w for _, _, w, _ in boxes])) if boxes else 0
    median_h = float(np.median([h for _, _, _, h in boxes])) if boxes else 0

    if median_w > 0 and median_h > 0:
        boxes = [
            b for b in boxes
            if abs(b[2] - median_w)/median_w < 0.5 and abs(b[3] - median_h)/median_h < 0.5
        ]
    else:
        boxes = []

    linhas, tolerancia_y = [], median_h * 0.5 if median_h else 50
    for box in sorted(boxes, key=lambda b: b[1]):
        for linha in linhas:
            if abs(linha[0][1] - box[1]) < tolerancia_y:
                linha.append(box)
                break
        else:
            linhas.append([box])

    for linha in linhas:
        linha.sort(key=lambda b: b[0])

    ovos_info, cnt = [], 1
    for linha in linhas:
        for (x, y, w, h) in linha:
            centro_x, centro_y = x + w // 2, y + h // 2
            raio = min(w, h) // 2
            mask = np.zeros(imagem_cv.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (centro_x, centro_y), raio, 255, -1)

            # média em RGB para usar nas bibliotecas de cor (skimage/colour usam RGB [0..1])
            mean_val_bgr = cv2.mean(imagem_cv, mask=mask)[:3]
            mean_val_rgb = mean_val_bgr[::-1]
            rgb = tuple(int(round(v)) for v in mean_val_rgb)

            hexval = rgb_to_hex(rgb)
            cmyk = rgb_to_cmyk(rgb)

            lab = rgb2lab(np.array([[rgb]]) / 255.0)[0][0]
            L, a_, b_ = lab
            C = (a_**2 + b_**2) ** 0.5
            H = np.degrees(np.arctan2(b_, a_)) % 360

            xyz = rgb2xyz(np.array([[rgb]]) / 255.0)[0][0]
            rgb_norm = np.array(rgb) / 255.0
            aces = colour.RGB_to_RGB(
                rgb_norm,
                input_colourspace=colour.RGB_COLOURSPACES['sRGB'],
                output_colourspace=colour.RGB_COLOURSPACES['ACES2065-1']
            )
            acescg = colour.RGB_to_RGB(
                rgb_norm,
                input_colourspace=colour.RGB_COLOURSPACES['sRGB'],
                output_colourspace=colour.RGB_COLOURSPACES['ACEScg']
            )
            linsrgb = colour.cctf_decoding(rgb_norm)

            ovos_info.append({
                "num": cnt,
                "rgb": tuple(map(int, rgb)),
                "hex": hexval,
                "cmyk": cmyk,
                "lab": (float(L), float(a_), float(b_)),
                "lch": (float(L), float(C), float(H)),
                "xyz": tuple(map(float, xyz)),
                "aces": tuple(map(float, aces)),
                "acescg": tuple(map(float, acescg)),
                "linsrgb": tuple(map(float, linsrgb)),
            })

            # desenho/anotações na imagem de saída (imagem_backup)
            cv2.rectangle(imagem_backup, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(imagem_backup, str(cnt),
                        (x + w//2 - 10, y + h//2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.rectangle(imagem_backup, (x, y-25), (x+25, y),
                          (int(rgb[2]), int(rgb[1]), int(rgb[0])), -1)
            cv2.putText(imagem_backup,
                        f'C:{cmyk[0]} M:{cmyk[1]} Y:{cmyk[2]} K:{cmyk[3]}',
                        (x+30, y-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            orien = 'Horizontal' if w/h > 1.1 else 'Vertical' if w/h < 0.9 else 'Indefinido'
            cv2.putText(imagem_backup, orien, (x, y+h+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            if w/h > 1.1:
                cv2.line(imagem_backup, (x, y), (x+w, y+h), (0, 0, 255), 4)
                cv2.line(imagem_backup, (x+w, y), (x, y+h), (0, 0, 255), 4)

            cnt += 1

    # Codifica PNG diretamente com OpenCV (sem PIL)
    # Observação: cv2.imencode espera BGR; usamos imagem_backup (BGR).
    ok, buf = cv2.imencode(".png", imagem_backup)
    if not ok:
        raise RuntimeError("Falha ao codificar a imagem de saída em PNG.")

    img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return img_b64, ovos_info
