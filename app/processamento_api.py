import base64
import io
import logging
from typing import BinaryIO, Tuple, Union

import cv2
import numpy as np
import requests
from skimage.color import rgb2lab, rgb2xyz

import colour


logging.basicConfig(level=logging.INFO)


# ---------------------------
# Utilidades de cor e imagem
# ---------------------------

def rgb_to_cmyk(rgb: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    """Converte RGB (0..255) para CMYK (%)."""
    r, g, b = [x / 255.0 for x in rgb]
    k = 1 - max(r, g, b)
    if k == 1:
        return (0, 0, 0, 100)
    c = (1 - r - k) / (1 - k)
    m = (1 - g - k) / (1 - k)
    y = (1 - b - k) / (1 - k)
    return (round(c * 100), round(m * 100), round(y * 100), round(k * 100))


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Converte RGB (0..255) para string HEX."""
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def imagem_escura(imagem_cv: np.ndarray, limiar_media: float = 50) -> bool:
    """True se a imagem for escura com base na média do canal V em HSV."""
    hsv = cv2.cvtColor(imagem_cv, cv2.COLOR_BGR2HSV)
    v_medio = hsv[:, :, 2].mean()
    return v_medio < limiar_media


def clarear_imagem(imagem_cv: np.ndarray, fator: float = 1.8) -> np.ndarray:
    """Aumenta brilho multiplicando canal V do HSV pelo fator."""
    hsv = cv2.cvtColor(imagem_cv, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * fator, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def cortar_bordas_proporcional(imagem_cv: np.ndarray, proporcao: float = 0.03) -> np.ndarray:
    """Corta bordas proporcionalmente à resolução (para remover ruído nas extremidades)."""
    h, w = imagem_cv.shape[:2]
    corte_x = int(w * proporcao)
    corte_y = int(h * proporcao)
    if corte_x * 2 >= w or corte_y * 2 >= h:
        return imagem_cv
    return imagem_cv[corte_y:h - corte_y, corte_x:w - corte_x]


def aplicar_unsharp_mask(imagem: np.ndarray, alpha: float = 1.5, beta: float = -0.85) -> np.ndarray:
    """Realça detalhes (unsharp mask)."""
    borrada = cv2.GaussianBlur(imagem, (0, 0), 3)
    return cv2.addWeighted(imagem, alpha, borrada, beta, 0)


def destacar_bordas(imagem: np.ndarray) -> np.ndarray:
    """Destaca bordas em vermelho sobre a imagem."""
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


def verificar_predominio_cinza(imagem_cv: np.ndarray) -> bool:
    """True se a variância das médias RGB for baixa (predomínio de tons cinza)."""
    media_rgb = cv2.mean(imagem_cv)[:3]
    variancia_rgb = np.var(media_rgb)
    return variancia_rgb < 100  # ajustável


# ---------------------------
# Leitura de imagem + Landscape
# ---------------------------

def _ensure_landscape(img_bgr: np.ndarray) -> np.ndarray:
    """Garante orientação landscape apenas pela proporção."""
    h, w = img_bgr.shape[:2]
    if h > w:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    return img_bgr


def _ler_imagem_cv(
    imagem: Union[np.ndarray, bytes, bytearray, BinaryIO, io.BytesIO, str]
) -> np.ndarray:
    """Lê imagem em BGR (uint8) sem usar EXIF."""
    if isinstance(imagem, np.ndarray):
        img = imagem
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return _ensure_landscape(img)

    if isinstance(imagem, (bytes, bytearray)):
        data = np.frombuffer(bytes(imagem), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Falha ao decodificar bytes de imagem.")
        return _ensure_landscape(img)

    if hasattr(imagem, "read"):
        raw = imagem.read()
        try:
            if hasattr(imagem, "seek"):
                imagem.seek(0)
        except Exception:  # noqa: BLE001
            pass
        data = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Falha ao decodificar stream de imagem.")
        return _ensure_landscape(img)

    if isinstance(imagem, str):
        with open(imagem, "rb") as f:
            raw = f.read()
        data = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Falha ao ler a imagem em: {imagem}")
        return _ensure_landscape(img)

    raise TypeError("Tipo de entrada de imagem não suportado.")


# ---------------------------
# Pipeline principal
# ---------------------------

def processar_imagem(imagem: Union[np.ndarray, bytes, bytearray, BinaryIO, io.BytesIO, str]):
    """Processa a imagem detectando ovos e retornando visualização anotada + metadados."""
    fator_elipse = 0.8

    imagem_cv = _ler_imagem_cv(imagem)
    imagem_cv = cortar_bordas_proporcional(imagem_cv)
    imagem_backup = imagem_cv.copy()

    if imagem_escura(imagem_cv, limiar_media=50):
        imagem_cv = clarear_imagem(imagem_cv, fator=1.8)

    if verificar_predominio_cinza(imagem_cv):
        imagem_cv = aplicar_unsharp_mask(imagem_cv)

    h0, w0 = imagem_cv.shape[:2]
    target_h, target_w = 826, 1113
    scale = min(target_w / w0, target_h / h0)
    if scale < 1.0:
        new_size = (int(w0 * scale), int(h0 * scale))
        imagem_cv = cv2.resize(imagem_cv, new_size, interpolation=cv2.INTER_AREA)
        imagem_backup = cv2.resize(imagem_backup, new_size, interpolation=cv2.INTER_AREA)

    imagem_cv = cv2.GaussianBlur(imagem_cv, (5, 5), 0)
    lab = cv2.cvtColor(imagem_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    imagem_cv = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    cinza = cv2.cvtColor(imagem_cv, cv2.COLOR_BGR2GRAY)
    cinza = cv2.GaussianBlur(cinza, (5, 5), 0)
    binario = cv2.bitwise_not(
        cv2.adaptiveThreshold(
            cinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    )

    contornos, _ = cv2.findContours(binario, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = [ctr for ctr in contornos if 200 < cv2.contourArea(ctr) < 50000]
    boxes = [cv2.boundingRect(ctr) for ctr in contornos]

    median_w = float(np.median([w for _, _, w, _ in boxes])) if boxes else 0.0
    median_h = float(np.median([h for _, _, _, h in boxes])) if boxes else 0.0
    if median_w and median_h:
        boxes = [
            b for b in boxes
            if abs(b[2] - median_w) / median_w < 0.5 and abs(b[3] - median_h) / median_h < 0.5
        ]
    else:
        boxes = []

    linhas, tolerancia_y = [], median_h * 0.5 if median_h else 50
    for box in sorted(boxes, key=lambda bb: bb[1]):
        for linha in linhas:
            if abs(linha[0][1] - box[1]) < tolerancia_y:
                linha.append(box)
                break
        else:
            linhas.append([box])
    for linha in linhas:
        linha.sort(key=lambda bb: bb[0])

    backup_rgb = cv2.cvtColor(imagem_backup, cv2.COLOR_BGR2RGB)

    ovos_info, cnt = [], 1
    for linha in linhas:
        for (x, y, w, h) in linha:
            centro_x, centro_y = x + w // 2, y + h // 2
            eixo_maior = int((w // 2) * fator_elipse)
            eixo_menor = int((h // 2) * fator_elipse)

            mask = np.zeros(backup_rgb.shape[:2], dtype=np.uint8)
            cv2.ellipse(mask, (centro_x, centro_y), (eixo_maior, eixo_menor), 0, 0, 360, 255, -1)

            mean_rgb = cv2.mean(backup_rgb, mask=mask)[:3]
            rgb = tuple(int(round(v)) for v in mean_rgb)

            hexval = rgb_to_hex(rgb)
            cmyk = rgb_to_cmyk(rgb)

            lab_vec = rgb2lab(np.array([[rgb]]) / 255.0)[0][0]
            L, a_, b_ = float(lab_vec[0]), float(lab_vec[1]), float(lab_vec[2])
            C = float((a_ ** 2 + b_ ** 2) ** 0.5)
            H = float(np.degrees(np.arctan2(b_, a_)) % 360)

            xyz = rgb2xyz(np.array([[rgb]]) / 255.0)[0][0]
            X, Y, Z = float(xyz[0]), float(xyz[1]), float(xyz[2])

            rgb_norm = np.array(rgb, dtype=np.float64) / 255.0
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

            orien = 'Horizontal' if w / h > 1.1 else ('Vertical' if w / h < 0.9 else 'Indefinido')

            info = {
                "num": cnt,
                "centro": (int(centro_x), int(centro_y)),
                "orientacao": orien,
                "rgb": tuple(map(int, rgb)),
                "hex": hexval,
                "cmyk": tuple(map(int, cmyk)),
                "lab": (L, a_, b_),
                "lch": (L, C, H),
                "xyz": (X, Y, Z),
                "aces": tuple(map(float, aces)),
                "acescg": tuple(map(float, acescg)),
                "linsrgb": tuple(map(float, linsrgb)),
            }

            info["descricao"] = (
                f"Centro: {info['centro']}\n"
                f"Orientação: {info['orientacao']}\n"
                f"RGB: {info['rgb']}\n"
                f"HEX: {info['hex']}\n"
                f"LAB: L={info['lab'][0]:.2f}, a={info['lab'][1]:.2f}, b={info['lab'][2]:.2f}\n"
                f"LCH: L={info['lch'][0]:.2f}, C={info['lch'][1]:.2f}, H={info['lch'][2]:.2f}\n"
                f"XYZ: X={info['xyz'][0]:.3f}, Y={info['xyz'][1]:.3f}, Z={info['xyz'][2]:.3f}\n"
                f"CMYK: {info['cmyk']}\n"
                f"ACES: R={info['aces'][0]:.4f}, G={info['aces'][1]:.4f}, B={info['aces'][2]:.4f}\n"
                f"ACEScg: R={info['acescg'][0]:.4f}, G={info['acescg'][1]:.4f}, B={info['acescg'][2]:.4f}\n"
                f"Linear sRGB: R={info['linsrgb'][0]:.4f}, G={info['linsrgb'][1]:.4f}, B={info['linsrgb'][2]:.4f}"
            )

            ovos_info.append(info)

            cv2.rectangle(imagem_backup, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if w / h > 1.1:
                cv2.line(imagem_backup, (x, y), (x + w, y + h), (0, 0, 255), 4)
                cv2.line(imagem_backup, (x + w, y), (x, y + h), (0, 0, 255), 4)
            cv2.ellipse(imagem_backup, (centro_x, centro_y), (eixo_maior, eixo_menor),
                        0, 0, 360, (255, 0, 255), 2)
            cv2.putText(
                imagem_backup,
                str(cnt),
                (centro_x - 10, centro_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )

            rect_y1 = max(y - 25, 0)
            rect_y2 = max(y, 0)
            cv2.rectangle(
                imagem_backup,
                (x, rect_y1),
                (x + 25, rect_y2),
                (int(rgb[2]), int(rgb[1]), int(rgb[0])),
                -1,
            )
            cv2.putText(
                imagem_backup,
                f'C:{cmyk[0]} M:{cmyk[1]} Y:{cmyk[2]} K:{cmyk[3]}',
                (x + 30, rect_y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                imagem_backup,
                orien,
                (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

            cnt += 1

    ok, buf = cv2.imencode(".png", imagem_backup)
    if not ok:
        raise RuntimeError("Falha ao codificar a imagem de saída em PNG.")
    img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return img_b64, ovos_info


def processar_imagem_por_url(url: str, timeout: int = 20):
    """Baixa a imagem de ``url`` e processa com ``processar_imagem``."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return processar_imagem(io.BytesIO(resp.content))


__all__ = [
    "processar_imagem",
    "processar_imagem_por_url",
]
