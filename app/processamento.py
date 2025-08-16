import cv2
import numpy as np
from PIL import Image
import pillow_heif
import io
import base64
import logging
from skimage.color import rgb2lab, rgb2xyz
import colour

# Ativa suporte a imagens HEIC
pillow_heif.register_heif_opener()
logging.basicConfig(level=logging.INFO)

def rgb_to_cmyk(rgb):
    """Converte RGB para CMYK, valores em porcentagem."""
    r, g, b = [x / 255.0 for x in rgb]
    k = 1 - max(r, g, b)
    if k == 1:
        return (0, 0, 0, 100)
    c = (1 - r - k) / (1 - k)
    m = (1 - g - k) / (1 - k)
    y = (1 - b - k) / (1 - k)
    return (round(c * 100), round(m * 100), round(y * 100), round(k * 100))

def rgb_to_hex(rgb):
    """Converte RGB para HEX string."""
    return '#{:02X}{:02X}{:02X}'.format(*rgb)

def imagem_escura(imagem_cv, limiar_media=50):
    """Retorna True se a imagem for escura, baseado na média do canal V em HSV."""
    hsv = cv2.cvtColor(imagem_cv, cv2.COLOR_BGR2HSV)
    v_medio = hsv[:,:,2].mean()
    return v_medio < limiar_media

def clarear_imagem(imagem_cv, fator=1.8):
    """Aumenta o brilho multiplicando o canal V do HSV pelo fator."""
    hsv = cv2.cvtColor(imagem_cv, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * fator, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def cortar_bordas_proporcional(imagem_cv, proporcao=0.03):
    """Corta as bordas da imagem proporcionalmente à resolução."""
    h, w = imagem_cv.shape[:2]
    corte_x = int(w * proporcao)
    corte_y = int(h * proporcao)
    if corte_x*2 >= w or corte_y*2 >= h:
        return imagem_cv
    return imagem_cv[corte_y:h-corte_y, corte_x:w-corte_x]

def aplicar_unsharp_mask(imagem, alpha=1.5, beta=-0.85):
    """Realça detalhes aplicando unsharp mask."""
    borrada = cv2.GaussianBlur(imagem, (0, 0), 3)
    return cv2.addWeighted(imagem, alpha, borrada, beta, 0)

def destacar_bordas(imagem):
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

def verificar_predominio_cinza(imagem_cv):
    """Retorna True se a variância das médias RGB for baixa (predomínio de tons cinza)."""
    media_rgb = cv2.mean(imagem_cv)[:3]
    variancia_rgb = np.var(media_rgb)
    return variancia_rgb < 100  # limiar ajustável

def processar_imagem(imagem):
    """Processa uma imagem detectando ovos, extraindo a média de cor em área elíptica, e anotando visualmente."""
    fator_elipse = 0.7  # Proporção do semi-eixo para a máscara elíptica

    # 1. Carregar e preparar imagem
    imagem_pil = Image.open(imagem).convert("RGB")
    imagem_cv = cv2.cvtColor(np.array(imagem_pil), cv2.COLOR_RGB2BGR)
    imagem_cv = cortar_bordas_proporcional(imagem_cv)
    imagem_backup = imagem_cv.copy()

    # 2. Clarear imagem se escura
    if imagem_escura(imagem_cv, limiar_media=50):
        imagem_cv = clarear_imagem(imagem_cv, fator=1.8)

    # 3. Realçar bordas e aplicar unsharp mask se predominância de cinza
    if verificar_predominio_cinza(imagem_cv):
        imagem_cv = destacar_bordas(imagem_cv)
        imagem_cv = aplicar_unsharp_mask(imagem_cv)
        imagem_cv = cv2.GaussianBlur(imagem_cv, (61, 37), 0)

    # 4. Redimensionamento se necessário (mantém proporção)
    h_original, w_original = imagem_cv.shape[:2]
    target_h, target_w = 826, 1113
    scale = min(target_w / w_original, target_h / h_original)
    if scale < 1:
        new_size = (int(w_original * scale), int(h_original * scale))
        imagem_cv = cv2.resize(imagem_cv, new_size, interpolation=cv2.INTER_AREA)
        imagem_backup = cv2.resize(imagem_backup, new_size, interpolation=cv2.INTER_AREA)

    # 5. Realce local de contraste (CLAHE)
    imagem_cv = cv2.GaussianBlur(imagem_cv, (5, 5), 0)
    lab = cv2.cvtColor(imagem_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    imagem_cv = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    # 6. Segmentação dos ovos (threshold adaptativo)
    cinza = cv2.cvtColor(imagem_cv, cv2.COLOR_BGR2GRAY)
    cinza = cv2.GaussianBlur(cinza, (5, 5), 0)
    binario = cv2.bitwise_not(
        cv2.adaptiveThreshold(cinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    )

    # 7. Encontrar contornos e filtrar por área
    contornos, _ = cv2.findContours(binario, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = [ctr for ctr in contornos if 100 < cv2.contourArea(ctr) < 50000]
    boxes = [cv2.boundingRect(ctr) for ctr in contornos]

    # 8. Filtragem por tamanho dos ovos (remove outliers)
    median_w = float(np.median([w for _, _, w, _ in boxes])) if boxes else 0
    median_h = float(np.median([h for _, _, _, h in boxes])) if boxes else 0
    boxes = [
        b for b in boxes
        if median_w and median_h and abs(b[2] - median_w) / median_w < 0.5 and abs(b[3] - median_h) / median_h < 0.5
    ]

    # 9. Agrupar ovos em linhas (para ordenação visual)
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

    # --- OTIMIZAÇÃO PRINCIPAL: converte imagem uma única vez para RGB para cálculo de média ---
    imagem_rgb_backup = cv2.cvtColor(imagem_backup, cv2.COLOR_BGR2RGB)

    # 10. Para cada ovo, calcular média de cor e desenhar anotações
    ovos_info, cnt = [], 1
    for linha in linhas:
        for (x, y, w, h) in linha:
            centro_x, centro_y = x + w // 2, y + h // 2
            eixo_maior = int(w // 2 * fator_elipse)
            eixo_menor = int(h // 2 * fator_elipse)
            mask = np.zeros(imagem_rgb_backup.shape[:2], dtype=np.uint8)
            cv2.ellipse(mask, (centro_x, centro_y), (eixo_maior, eixo_menor), 0, 0, 360, 255, -1)

            # Agora o cálculo da média usa a imagem já convertida
            mean_val = cv2.mean(imagem_rgb_backup, mask=mask)[:3]
            rgb = tuple(map(int, mean_val))
            hexval, cmyk = rgb_to_hex(rgb), rgb_to_cmyk(rgb)
            lab = rgb2lab(np.array([[rgb]]) / 255.0)[0][0]
            L, a_, b_ = lab
            C = (a_ ** 2 + b_ ** 2) ** 0.5
            H = np.degrees(np.arctan2(b_, a_)) % 360
            xyz = rgb2xyz(np.array([[rgb]]) / 255.0)[0][0]
            rgb_norm = np.array(rgb) / 255.0
            aces = colour.RGB_to_RGB(
                rgb_norm,
                input_colourspace=colour.RGB_COLOURSPACES['sRGB'],
                output_colourspace=colour.RGB_COLOURSPACES['ACES2065-1'])
            acescg = colour.RGB_to_RGB(
                rgb_norm,
                input_colourspace=colour.RGB_COLOURSPACES['sRGB'],
                output_colourspace=colour.RGB_COLOURSPACES['ACEScg'])
            linsrgb = colour.cctf_decoding(rgb_norm)

            ovos_info.append({
                "num": cnt, "rgb": rgb, "hex": hexval, "cmyk": cmyk,
                "lab": (L, a_, b_), "lch": (L, C, H),
                "xyz": tuple(xyz), "aces": tuple(aces), "acescg": tuple(acescg),
                "linsrgb": tuple(linsrgb)
            })

            # --- Anotações na imagem ---
            cv2.rectangle(imagem_backup, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if w / h > 1.1:
                cv2.line(imagem_backup, (x, y), (x + w, y + h), (0, 0, 255), 4)
                cv2.line(imagem_backup, (x + w, y), (x, y + h), (0, 0, 255), 4)
            cv2.ellipse(imagem_backup, (centro_x, centro_y), (eixo_maior, eixo_menor), 0, 0, 360, (255, 0, 255), 2)
            cv2.putText(imagem_backup, str(cnt), (centro_x - 10, centro_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            rect_y1 = y - 25 if y - 25 > 0 else 0
            rect_y2 = y if y > 0 else 0
            cv2.rectangle(imagem_backup, (x, rect_y1), (x + 25, rect_y2), tuple(int(c) for c in rgb[::-1]), -1)
            cv2.putText(imagem_backup, f'C:{cmyk[0]} M:{cmyk[1]} Y:{cmyk[2]} K:{cmyk[3]}',
                        (x + 30, rect_y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            orien = 'Horizontal' if w / h > 1.1 else 'Vertical' if w / h < 0.9 else 'Indefinido'
            cv2.putText(imagem_backup, orien, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cnt += 1

    # 11. Converte a imagem anotada para PNG base64
    img_rgb = cv2.cvtColor(imagem_backup, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8'), ovos_info










































































































