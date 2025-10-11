# processamento.py
import cv2
import numpy as np
import io
import base64
import logging
import requests
from skimage.color import rgb2lab, rgb2xyz
import colour

# =====[ Suporte opcional a HEIC/HEIF via pillow-heif ]=====
try:
    import pillow_heif  # pip install pillow-heif
    from PIL import Image
    _PILLOW_HEIF_OK = True
except Exception:
    _PILLOW_HEIF_OK = False
# ==========================================================

logging.basicConfig(level=logging.INFO)

# ---------------------------
# Utilidades de cor e imagem
# ---------------------------

def rgb_to_cmyk(rgb):
    """Converte RGB (0..255) para CMYK (%)"""
    r, g, b = [x / 255.0 for x in rgb]
    k = 1 - max(r, g, b)
    if k == 1:
        return (0, 0, 0, 100)
    c = (1 - r - k) / (1 - k)
    m = (1 - g - k) / (1 - k)
    y = (1 - b - k) / (1 - k)
    return (round(c * 100), round(m * 100), round(y * 100), round(k * 100))

def rgb_to_hex(rgb):
    """Converte RGB (0..255) para string HEX"""
    return '#{:02X}{:02X}{:02X}'.format(*rgb)

def imagem_escura(imagem_cv, limiar_media=50):
    """True se a imagem for escura com base na média do canal V em HSV."""
    hsv = cv2.cvtColor(imagem_cv, cv2.COLOR_BGR2HSV)
    v_medio = hsv[:, :, 2].mean()
    return v_medio < limiar_media

def clarear_imagem(imagem_cv, fator=1.8):
    """Aumenta brilho multiplicando canal V do HSV pelo fator (para detecção)."""
    hsv = cv2.cvtColor(imagem_cv, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * fator, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def ajustar_brilho_hsv(img_bgr: np.ndarray, fator_v: float = 1.0) -> np.ndarray:
    """
    Ajusta apenas o brilho (canal V) no espaço HSV.
    - fator_v < 1.0 escurece; > 1.0 clareia.
    Usado SOMENTE na imagem-base (origem das cores e da renderização final).
    """
    fator_v = float(fator_v)
    if abs(fator_v - 1.0) < 1e-6:
        return img_bgr
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    v = np.clip(v * fator_v, 0, 255).astype(np.uint8)
    hsv[:, :, 2] = v
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def ajustar_contraste(img_bgr: np.ndarray, fator: float = 1.0) -> np.ndarray:
    """Aplica ganho de contraste simples mantendo o ponto médio em 128."""
    fator = float(fator)
    if abs(fator - 1.0) < 1e-6:
        return img_bgr
    img = img_bgr.astype(np.float32)
    img = (img - 127.5) * fator + 127.5
    return np.clip(img, 0, 255).astype(np.uint8)


def ajustar_saturacao(img_bgr: np.ndarray, fator: float = 1.0) -> np.ndarray:
    """Ajusta a saturação no espaço HSV."""
    fator = float(fator)
    if abs(fator - 1.0) < 1e-6:
        return img_bgr
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    s = np.clip(s * fator, 0, 255).astype(np.uint8)
    hsv[:, :, 1] = s
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def ajustar_exposicao(img_bgr: np.ndarray, ev: float = 0.0) -> np.ndarray:
    """Ajusta a exposição considerando EV (usa fator 2 ** EV)."""
    ev = float(ev)
    if abs(ev) < 1e-6:
        return img_bgr
    fator = 2.0 ** ev
    img = img_bgr.astype(np.float32) * fator
    return np.clip(img, 0, 255).astype(np.uint8)


def ajustar_nitidez(img_bgr: np.ndarray, intensidade: float = 0.0) -> np.ndarray:
    """Aplica unsharp mask controlada pela intensidade."""
    intensidade = float(intensidade)
    if abs(intensidade) < 1e-6:
        return img_bgr
    borrada = cv2.GaussianBlur(img_bgr, (0, 0), 3)
    alpha = 1.0 + intensidade
    beta = -intensidade
    return cv2.addWeighted(img_bgr, alpha, borrada, beta, 0)


def ajustar_temperatura(img_bgr: np.ndarray, intensidade: float = 0.0) -> np.ndarray:
    """Ajusta temperatura de cor (positivo aquece, negativo esfria)."""
    intensidade = float(intensidade)
    if abs(intensidade) < 1e-6:
        return img_bgr
    img = img_bgr.astype(np.float32)
    # escala suave (±50% no máximo quando intensidade = ±1)
    r_scale = np.clip(1.0 + 0.5 * intensidade, 0.2, 3.0)
    b_scale = np.clip(1.0 - 0.5 * intensidade, 0.2, 3.0)
    g_scale = np.clip(1.0 + 0.2 * intensidade, 0.2, 3.0)
    img[:, :, 2] *= r_scale
    img[:, :, 0] *= b_scale
    img[:, :, 1] *= g_scale
    return np.clip(img, 0, 255).astype(np.uint8)

def cortar_bordas_proporcional(imagem_cv, proporcao=0.03):
    """Corta bordas proporcionalmente à resolução (para remover ruído nas extremidades)."""
    h, w = imagem_cv.shape[:2]
    corte_x = int(w * proporcao)
    corte_y = int(h * proporcao)
    if corte_x * 2 >= w or corte_y * 2 >= h:
        return imagem_cv
    return imagem_cv[corte_y:h - corte_y, corte_x:w - corte_x]

def aplicar_unsharp_mask(imagem, alpha=1.5, beta=-0.85):
    """Realça detalhes (unsharp mask) — só para detecção."""
    borrada = cv2.GaussianBlur(imagem, (0, 0), 3)
    return cv2.addWeighted(imagem, alpha, borrada, beta, 0)

def destacar_bordas(imagem):
    """(Opcional) Destaca bordas em vermelho sobre a imagem."""
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
    """True se a variância das médias RGB for baixa (predomínio de tons cinza)."""
    media_rgb = cv2.mean(imagem_cv)[:3]
    variancia_rgb = np.var(media_rgb)
    return variancia_rgb < 100  # ajustável

# -----------------------------------------
# Contagem de cores por histograma (RGB puro)
# -----------------------------------------

def extrair_cores_predominantes_hist(img_rgb: np.ndarray,
                                     mask: np.ndarray,
                                     top_n: int | None = None):
    """
    Conta cores EXATAS dentro da máscara e retorna TODAS (ordenadas por frequência).
    - img_rgb: imagem em RGB uint8 (H, W, 3)
    - mask: máscara binária (0/255) do mesmo HxW
    - top_n: se None, retorna todas; se int, limita às top-N.
    """
    if not (isinstance(img_rgb, np.ndarray) and img_rgb.ndim == 3 and img_rgb.shape[2] == 3):
        raise ValueError("img_rgb deve ser um array (H, W, 3).")
    if img_rgb.dtype != np.uint8:
        raise ValueError("img_rgb deve ser dtype=uint8.")
    if mask.shape != img_rgb.shape[:2]:
        raise ValueError("mask e img_rgb devem ter mesmas dimensões HxW.")

    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return []

    roi = img_rgb[ys, xs].reshape(-1, 3)
    unique_rows, counts = np.unique(roi, axis=0, return_counts=True)

    order = np.argsort(-counts)
    if top_n is not None:
        order = order[:top_n]

    total = int(counts.sum())
    resultado = []
    for idx in order:
        r, g, b = map(int, unique_rows[idx])
        cnt = int(counts[idx])
        perc = (100.0 * cnt / total) if total else 0.0
        rgb = (r, g, b)
        resultado.append({
            "rgb": rgb,
            "hex": rgb_to_hex(rgb),
            "count": cnt,
            "perc": perc
        })
    return resultado

# ---------------------------
# Leitura de imagem + Landscape
# ---------------------------

def _ensure_landscape(img_bgr: np.ndarray) -> np.ndarray:
    """Garante orientação 'landscape' apenas pela proporção."""
    h, w = img_bgr.shape[:2]
    if h > w:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    return img_bgr

def _decode_image_bytes(raw: bytes) -> np.ndarray:
    """
    Tenta decodificar bytes de imagem para BGR (uint8).
    1) OpenCV (PNG/JPEG/WebP/etc.)
    2) pillow-heif (HEIC/HEIF) -> PIL -> RGB -> BGR
    """
    data = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    if _PILLOW_HEIF_OK:
        try:
            heif = pillow_heif.read_heif(raw)
            pil_img = Image.frombytes(heif.mode, heif.size, heif.data, "raw")
            rgb = np.array(pil_img.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return bgr
        except Exception:
            pass
    raise ValueError(
        "Falha ao decodificar a imagem (possivelmente HEIC/HEIF). "
        "Instale: pip install pillow-heif pillow"
    )

def _ler_imagem_cv(imagem):
    """Lê imagem em BGR (uint8) sem usar EXIF (ndarray, bytes, file-like, caminho)."""
    if isinstance(imagem, np.ndarray):
        img = imagem
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return _ensure_landscape(img)
    if isinstance(imagem, (bytes, bytearray)):
        img = _decode_image_bytes(bytes(imagem))
        return _ensure_landscape(img)
    if hasattr(imagem, "read"):
        raw = imagem.read()
        try:
            if hasattr(imagem, "seek"):
                imagem.seek(0)
        except Exception:
            pass
        img = _decode_image_bytes(raw)
        return _ensure_landscape(img)
    if isinstance(imagem, str):
        with open(imagem, "rb") as f:
            raw = f.read()
        img = _decode_image_bytes(raw)
        return _ensure_landscape(img)
    raise TypeError("Tipo de entrada de imagem não suportado.")

# ---------------------------
# Pipeline principal
# ---------------------------

def processar_imagem(
    imagem,
    fator_elipse=(0.85, 0.75),
    usar_fitellipse=True,
    fator_v_backup: float = 1.0,
    fator_contraste: float = 1.0,
    fator_saturacao: float = 1.0,
    ev_exposicao: float = 0.0,
    fator_nitidez: float = 0.0,
    fator_temperatura: float = 0.0,
):
    """
    Detecta ovos usando uma cópia tratada para segmentação (imagem_trabalho),
    mas CAPTURA as cores e RENDERIZA a saída a partir da cópia base (imagem_base),
    que é a imagem antes de qualquer filtro + (opcionalmente) o fator_v_backup.

    Retorna: (img_b64, ovos_info)
    """
    fx, fy = map(float, fator_elipse)

    # 1) Ler BGR (sem EXIF) e cortar bordas
    imagem_trabalho = _ler_imagem_cv(imagem)       # BGR
    imagem_trabalho = cortar_bordas_proporcional(imagem_trabalho)

    # 2) CÓPIA CRUA (antes de filtros) -> vai virar nossa BASE (fonte de cores + render final)
    imagem_base = imagem_trabalho.copy()

    # 3) Cópia de trabalho: só para detecção (pode aplicar filtros de ajuda)
    if imagem_escura(imagem_trabalho, limiar_media=50):
        imagem_trabalho = clarear_imagem(imagem_trabalho, fator=1.8)
    if verificar_predominio_cinza(imagem_trabalho):
        imagem_trabalho = aplicar_unsharp_mask(imagem_trabalho)

    # 4) Padronizar tamanho (mantém alinhamento geométrico e aplica MESMO resize nas duas cópias)
    h0, w0 = imagem_trabalho.shape[:2]
    target_h, target_w = 826, 1113
    scale = min(target_w / w0, target_h / h0)
    if scale < 1.0:
        new_size = (int(w0 * scale), int(h0 * scale))
        imagem_trabalho = cv2.resize(imagem_trabalho, new_size, interpolation=cv2.INTER_AREA)
        imagem_base = cv2.resize(imagem_base, new_size, interpolation=cv2.INTER_AREA)

    # 5) Ajustes finos SOMENTE na BASE (origem de cor e render final)
    if abs(float(fator_v_backup) - 1.0) > 1e-6:
        imagem_base = ajustar_brilho_hsv(imagem_base, fator_v=float(fator_v_backup))
    if abs(float(ev_exposicao)) > 1e-6:
        imagem_base = ajustar_exposicao(imagem_base, ev=float(ev_exposicao))
    if abs(float(fator_contraste) - 1.0) > 1e-6:
        imagem_base = ajustar_contraste(imagem_base, fator=float(fator_contraste))
    if abs(float(fator_saturacao) - 1.0) > 1e-6:
        imagem_base = ajustar_saturacao(imagem_base, fator=float(fator_saturacao))
    if abs(float(fator_temperatura)) > 1e-6:
        imagem_base = ajustar_temperatura(imagem_base, intensidade=float(fator_temperatura))
    if abs(float(fator_nitidez)) > 1e-6:
        imagem_base = ajustar_nitidez(imagem_base, intensidade=float(fator_nitidez))

    # 6) Cópia para exibição (vamos desenhar nela e é ela que será retornada)
    imagem_exibicao = imagem_base.copy()

    # 7) (somente para detecção) CLAHE na cópia de trabalho
    imagem_trabalho = cv2.GaussianBlur(imagem_trabalho, (5, 5), 0)
    lab = cv2.cvtColor(imagem_trabalho, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    imagem_trabalho = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    # 8) Segmentação (na cópia de trabalho)
    cinza = cv2.cvtColor(imagem_trabalho, cv2.COLOR_BGR2GRAY)
    cinza = cv2.GaussianBlur(cinza, (5, 5), 0)
    binario = cv2.bitwise_not(
        cv2.adaptiveThreshold(
            cinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    )

    # 9) Contornos e filtragem
    contornos, _ = cv2.findContours(binario, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = [ctr for ctr in contornos if 200 < cv2.contourArea(ctr) < 50000]

    boxes = [cv2.boundingRect(ctr) for ctr in contornos]
    median_w = float(np.median([w for _, _, w, _ in boxes])) if boxes else 0.0
    median_h = float(np.median([h for _, _, _, h in boxes])) if boxes else 0.0
    if median_w and median_h:
        paired = [
            (b, c) for b, c in zip(boxes, contornos)
            if abs(b[2] - median_w) / (median_w + 1e-6) < 0.5
            and abs(b[3] - median_h) / (median_h + 1e-6) < 0.5
        ]
    else:
        paired = []

    # 10) Agrupar em linhas
    linhas, tolerancia_y = [], (median_h * 0.5 if median_h else 50)
    for (x, y, w, h), ctr in sorted(paired, key=lambda it: it[0][1]):
        for linha in linhas:
            if abs(linha[0][0][1] - y) < tolerancia_y:
                linha.append(((x, y, w, h), ctr))
                break
        else:
            linhas.append([((x, y, w, h), ctr)])
    for linha in linhas:
        linha.sort(key=lambda it: it[0][0])

    # 11) Conversão para RGB da BASE (origem da colorimetria)
    backup_rgb = cv2.cvtColor(imagem_base, cv2.COLOR_BGR2RGB)

    # 12) Medidas e anotações
    ovos_info, cnt = [], 1
    for linha in linhas:
        for (x, y, w, h), ctr in linha:
            # Elipse (fitEllipse) define a área de amostragem
            use_fit = usar_fitellipse and len(ctr) >= 5
            if use_fit:
                (cx, cy), (W, H), ang = cv2.fitEllipse(ctr)
                eixo_x = int(max(1, round((W * 0.5) * fx)))
                eixo_y = int(max(1, round((H * 0.5) * fy)))
                centro_x, centro_y = int(round(cx)), int(round(cy))
                angle_deg = float(ang)
            else:
                centro_x, centro_y = x + w // 2, y + h // 2
                eixo_x = int(max(1, round((w * 0.5) * fx)))
                eixo_y = int(max(1, round((h * 0.5) * fy)))
                angle_deg = 0.0

            # Máscara elíptica (na BASE)
            mask = np.zeros(backup_rgb.shape[:2], dtype=np.uint8)
            cv2.ellipse(mask, (centro_x, centro_y), (eixo_x, eixo_y),
                        angle_deg, 0, 360, 255, -1)

            # Cor média e todas as cores (na BASE)
            mean_rgb = cv2.mean(backup_rgb, mask=mask)[:3]
            rgb = tuple(int(round(v)) for v in mean_rgb)

            todas_cores = extrair_cores_predominantes_hist(backup_rgb, mask, top_n=None)

            # Espaços de cor (da média)
            hexval = rgb_to_hex(rgb)
            cmyk = rgb_to_cmyk(rgb)

            lab_vec = rgb2lab(np.array([[rgb]]) / 255.0)[0][0]
            L, a_, b_ = float(lab_vec[0]), float(lab_vec[1]), float(lab_vec[2])
            C = float((a_**2 + b_**2) ** 0.5)
            Hh = float(np.degrees(np.arctan2(b_, a_)) % 360)

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

            orien = 'Horizontal' if eixo_x >= eixo_y * 1.1 else ('Vertical' if eixo_y >= eixo_x * 1.1 else 'Indefinido')

            preview = todas_cores[:8]
            top_cores_preview_text = ", ".join([f"{c['hex']} ({c['perc']:.1f}%) — {c['count']} px" for c in preview])

            info = {
                "num": cnt,
                "centro": (int(centro_x), int(centro_y)),
                "eixo": (int(eixo_x), int(eixo_y)),   # útil p/ overlays
                "orientacao": orien,
                "rgb": tuple(map(int, rgb)),
                "hex": hexval,
                "cmyk": tuple(map(int, cmyk)),
                "lab": (L, a_, b_),
                "lch": (L, C, Hh),
                "xyz": (X, Y, Z),
                "aces": tuple(map(float, aces)),
                "acescg": tuple(map(float, acescg)),
                "linsrgb": tuple(map(float, linsrgb)),
                "todas_cores": todas_cores,
                "top_cores_preview_text": top_cores_preview_text,
                "ellipse_angle_deg": angle_deg
            }

            # ===== Anotações visuais — sempre na imagem_exibicao (que já tem fator_v aplicado) =====
            cv2.rectangle(imagem_exibicao, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if w / (h + 1e-6) > 1.1:
                cv2.line(imagem_exibicao, (x, y), (x + w, y + h), (0, 0, 255), 4)
                cv2.line(imagem_exibicao, (x + w, y), (x, y + h), (0, 0, 255), 4)
            cv2.ellipse(imagem_exibicao, (centro_x, centro_y), (eixo_x, eixo_y),
                        angle_deg, 0, 360, (255, 0, 255), 2)
            cv2.putText(imagem_exibicao, str(cnt),
                        (centro_x - 10, centro_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Amostra de cor (em BGR) da MÉDIA (convertida de RGB medido)
            rect_y1 = max(y - 25, 0); rect_y2 = max(y, 0)
            cv2.rectangle(imagem_exibicao, (x, rect_y1), (x + 25, rect_y2),
                          (int(rgb[2]), int(rgb[1]), int(rgb[0])), -1)
            cv2.putText(imagem_exibicao,
                        f'C:{int(cmyk[0])} M:{int(cmyk[1])} Y:{int(cmyk[2])} K:{int(cmyk[3])}',
                        (x + 30, rect_y1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(imagem_exibicao, orien,
                        (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            ovos_info.append(info)
            cnt += 1

    # 13) Codifica PNG a partir da imagem_exibicao (já com fator_v_backup aplicado)
    ok, buf = cv2.imencode(".png", imagem_exibicao)
    if not ok:
        raise RuntimeError("Falha ao codificar a imagem de saída em PNG.")
    img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return img_b64, ovos_info


def processar_imagem_por_url(
    url: str,
    timeout: int = 20,
    fator_elipse=(0.85, 0.75),
    usar_fitellipse=True,
    fator_v_backup: float = 1.0,
    fator_contraste: float = 1.0,
    fator_saturacao: float = 1.0,
    ev_exposicao: float = 0.0,
    fator_nitidez: float = 0.0,
    fator_temperatura: float = 0.0,
):
    """Baixa a imagem de URL e processa com os mesmos parâmetros."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return processar_imagem(
        io.BytesIO(resp.content),
        fator_elipse=fator_elipse,
        usar_fitellipse=usar_fitellipse,
        fator_v_backup=fator_v_backup,
        fator_contraste=fator_contraste,
        fator_saturacao=fator_saturacao,
        ev_exposicao=ev_exposicao,
        fator_nitidez=fator_nitidez,
        fator_temperatura=fator_temperatura,
    )
