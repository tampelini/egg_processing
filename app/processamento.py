# processamento.py
import base64
import csv
import io
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from skimage.color import rgb2lab, rgb2xyz

import colour

# =====[ Suporte opcional a HEIC/HEIF via pillow-heif ]=====
_PILLOW_HEIF_OK = False
_PILLOW_HEIF_ERR = None
try:
    import pillow_heif  # pip install pillow-heif
    from PIL import Image

    pillow_heif.register_heif_opener()
    _PILLOW_HEIF_OK = True
except Exception as exc:  # noqa: BLE001
    _PILLOW_HEIF_ERR = str(exc)
# ==========================================================

logging.basicConfig(level=logging.INFO)

# Diretórios padrão relativos ao projeto
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
DEFAULT_STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
DEFAULT_CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_STATIC_DIR, "output")
DEFAULT_CALIBRATED_DIR = os.path.join(DEFAULT_STATIC_DIR, "calibrated")

# ===================== Configurações da calibração =====================

# IDs esperados dos 4 ArUco (dicionário 4x4_50) nos cantos da paleta
# TL/TR/BL/BR = Top-Left / Top-Right / Bottom-Left / Bottom-Right
EXPECTED_IDS = {"TL": 10, "TR": 11, "BL": 12, "BR": 13}

# Layout da paleta: 12 linhas × 2 colunas (24 patches)
ROWS, COLS = 12, 2

# Tamanho do warp (paleta retificada)
WARP_W, WARP_H = 800, 1600

# Margens e espaçamentos relativos dentro do warp
MARGIN_X, MARGIN_Y = 0.14, 0.05
GAP_X, GAP_Y = 0.06, 0.02

# Janela de amostragem (pixels) para média RGB dentro de cada patch
SAMPLE_WIN = 34


# ===================== Utilidades da calibração =====================


def _empty_auto_calibration(performed: bool = False) -> Dict[str, Optional[str]]:
    return {
        "performed": performed,
        "palette_detected": False,
        "skip_reason": None,
        "error": None,
        "used_image": "original",
        "calibrated_path": None,
        "calibrated_name": None,
        "annotated_path": None,
        "warp_path": None,
        "warp_debug_path": None,
        "warp_labels_path": None,
        "measured_csv_path": None,
    }


def _ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        if d:
            os.makedirs(d, exist_ok=True)


def _save_csv(path: str, rows: List[List]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def _aruco_detector():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "OpenCV ArUco não disponível. Instale opencv-contrib-python."
        )
    aruco = cv2.aruco
    if hasattr(aruco, "ArucoDetector"):
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        params = aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 33
        params.adaptiveThreshWinSizeStep = 10
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        detector = aruco.ArucoDetector(dictionary, params)
        return detector, dictionary
    dictionary = aruco.Dictionary_get(aruco.DICT_4X4_50)
    params = aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return (dictionary, params), dictionary


def _detect_markers(image_bgr: np.ndarray):
    det, dictn = _aruco_detector()
    if isinstance(det, tuple):
        dictionary, params = det
        corners, ids, _ = cv2.aruco.detectMarkers(
            image_bgr, dictionary, parameters=params
        )
    else:
        corners, ids, _ = det.detectMarkers(image_bgr)
    if ids is None or len(ids) < 4:
        raise RuntimeError(
            "Não encontrei marcadores ArUco suficientes (precisa de 4)."
        )
    return corners, ids.flatten().tolist()


def _order_corners_by_expected(corners, ids: List[int]) -> np.ndarray:
    id_to_corner = {idv: c.reshape(-1, 2) for c, idv in zip(corners, ids)}
    try:
        tl = id_to_corner[EXPECTED_IDS["TL"]][0]
        tr = id_to_corner[EXPECTED_IDS["TR"]][1]
        br = id_to_corner[EXPECTED_IDS["BR"]][2]
        bl = id_to_corner[EXPECTED_IDS["BL"]][3]
    except KeyError:
        raise RuntimeError(
            "IDs esperados (10,11,12,13) não foram encontrados."
        )
    quad = np.array([tl, tr, br, bl], dtype=np.float32)
    return quad


def _compute_homography_and_warp(image_bgr: np.ndarray, quad_src: np.ndarray):
    dst = np.array(
        [
            [0, 0],
            [WARP_W - 1, 0],
            [WARP_W - 1, WARP_H - 1],
            [0, WARP_H - 1],
        ],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(quad_src, dst)
    warped = cv2.warpPerspective(image_bgr, H, (WARP_W, WARP_H))
    return warped, H


def _grid_centers_parametric() -> List[Tuple[float, float]]:
    xs, ys = [], []
    cell_w = (1.0 - 2 * MARGIN_X - (COLS - 1) * GAP_X) / COLS
    cell_h = (1.0 - 2 * MARGIN_Y - (ROWS - 1) * GAP_Y) / ROWS
    for r in range(ROWS):
        cy = MARGIN_Y + r * (cell_h + GAP_Y) + cell_h / 2
        ys.append(cy)
    for c in range(COLS):
        cx = MARGIN_X + c * (cell_w + GAP_X) + cell_w / 2
        xs.append(cx)
    centers: List[Tuple[float, float]] = []
    for r in range(ROWS):
        for c in range(COLS):
            centers.append((xs[c], ys[r]))
    return centers


def _load_centers_from_csv(config_dir: str) -> Optional[List[Tuple[float, float]]]:
    path = os.path.join(config_dir, "patch_centers.csv")
    if not os.path.exists(path):
        return None
    out: List[Tuple[float, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            x = float(row["x"])
            y = float(row["y"])
            out.append((x, y))
    if len(out) != ROWS * COLS:
        raise RuntimeError(
            f"patch_centers.csv deve ter {ROWS * COLS} linhas; tem {len(out)}."
        )
    return out


def _centers_px(centers_norm: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
    return [
        (int(nx * WARP_W), int(ny * WARP_H))
        for (nx, ny) in centers_norm
    ]


def _read_rgb_at_points(
    image_bgr: np.ndarray,
    points: List[Tuple[int, int]],
    win: int = SAMPLE_WIN,
) -> np.ndarray:
    half = win // 2
    h, w = image_bgr.shape[:2]
    rgbs = []
    for (x, y) in points:
        x0, x1 = max(0, x - half), min(w, x + half + 1)
        y0, y1 = max(0, y - half), min(h, y + half + 1)
        patch = image_bgr[y0:y1, x0:x1, :]
        if patch.size == 0:
            rgbs.append([0, 0, 0])
        else:
            mean_bgr = patch.reshape(-1, 3).mean(axis=0)
            r, g, b = mean_bgr[2], mean_bgr[1], mean_bgr[0]
            rgbs.append([r, g, b])
    return np.array(rgbs, dtype=np.float32)


def _save_aruco_overlay(
    image_bgr: np.ndarray,
    corners,
    ids: List[int],
    out_path: str,
) -> None:
    vis = image_bgr.copy()
    aruco = cv2.aruco
    ids_arr = np.array(ids).reshape(-1, 1)
    aruco.drawDetectedMarkers(vis, corners, ids_arr)
    try:
        quad = _order_corners_by_expected(corners, ids)
        cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 255, 255), 3)
    except Exception:
        pass
    cv2.imwrite(out_path, vis)


def _draw_sampling_debug(
    warped: np.ndarray,
    centers_px: List[Tuple[int, int]],
    labels: List[str],
    win: int = SAMPLE_WIN,
    out_dbg: Optional[str] = None,
    out_lbl: Optional[str] = None,
) -> None:
    half = win // 2
    dbg = warped.copy()
    for (x, y) in centers_px:
        cv2.rectangle(
            dbg, (x - half, y - half), (x + half, y + half), (0, 255, 0), 2
        )
        cv2.circle(dbg, (x, y), 3, (0, 0, 255), -1)
    if out_dbg:
        cv2.imwrite(out_dbg, dbg)

    lbl = warped.copy()
    for (x, y), text in zip(centers_px, labels):
        cv2.rectangle(
            lbl, (x - half, y - half), (x + half, y + half), (0, 255, 0), 1
        )
        cv2.putText(
            lbl,
            text,
            (x - half, y - half - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            lbl,
            text,
            (x - half, y - half - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    if out_lbl:
        cv2.imwrite(out_lbl, lbl)


def _solve_color_matrix(
    measured: np.ndarray, target: np.ndarray
) -> np.ndarray:
    assert measured.shape == target.shape and measured.shape[1] == 3
    if np.min(np.std(target, axis=0)) < 3:
        raise RuntimeError(
            "Targets muito pouco variados. Forneça ref_colors.csv real."
        )
    if np.min(np.std(measured, axis=0)) < 1:
        raise RuntimeError(
            "Medições quase constantes. Verifique warp/amostragem."
        )

    N = measured.shape[0]
    A = np.hstack(
        [measured.astype(np.float32), np.ones((N, 1), dtype=np.float32)]
    )
    lam = 1e-2
    AtA = A.T @ A + lam * np.eye(4, dtype=np.float32)
    M = np.zeros((3, 4), dtype=np.float32)
    for ch in range(3):
        y = target[:, ch].astype(np.float32)
        x = np.linalg.solve(AtA, A.T @ y)
        M[ch, :] = x

    gains = np.abs(M[:, :3])
    bias = np.abs(M[:, 3])
    if np.any(gains < 0.05) or np.any(gains > 5) or np.any(bias > 60):
        Mf = np.zeros_like(M)
        for ch in range(3):
            x = measured[:, ch]
            y = target[:, ch]
            a, b = np.polyfit(x, y, 1)
            Mf[ch, :3] = [0, 0, 0]
            Mf[ch, ch] = a
            Mf[ch, 3] = b
        M = Mf

    return M


def _apply_color_matrix(image_bgr: np.ndarray, M: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    H, W = rgb.shape[:2]
    flat = rgb.reshape(-1, 3)
    A = np.hstack([flat, np.ones((flat.shape[0], 1), dtype=np.float32)])
    out = A @ M.T
    out = np.clip(out, 0, 255).reshape(H, W, 3).astype(np.uint8)
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return bgr


def _save_calibration(config_dir: str, M: np.ndarray, meta: Dict) -> None:
    data = {"matrix_3x4": M.tolist(), "meta": meta}
    with open(
        os.path.join(config_dir, "color_calibration.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_targets_csv(
    config_dir: str, expected_n: int
) -> Tuple[np.ndarray, List[str]]:
    path = os.path.join(config_dir, "ref_colors.csv")
    if not os.path.exists(path):
        raise RuntimeError(
            "ref_colors.csv não encontrado em config/. Gere a referência antes."
        )
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        fieldset = {k.strip().lower() for k in (rd.fieldnames or [])}
        labels: List[str] = []
        targets: List[List[float]] = []

        if {"label", "row", "col", "r", "g", "b"} <= fieldset:
            for row in rd:
                labels.append(row["label"])
                targets.append(
                    [float(row["R"]), float(row["G"]), float(row["B"])]
                )
        elif {"row", "col", "r", "g", "b"} <= fieldset:
            i = 1
            for row in rd:
                labels.append(f"P{i:02d}")
                targets.append(
                    [float(row["R"]), float(row["G"]), float(row["B"])]
                )
                i += 1
        elif {"r", "g", "b"} <= fieldset:
            i = 1
            for row in rd:
                labels.append(f"P{i:02d}")
                targets.append(
                    [float(row["R"]), float(row["G"]), float(row["B"])]
                )
                i += 1
        else:
            raise RuntimeError(
                "Cabeçalho do ref_colors.csv inválido."
            )

    if len(targets) != expected_n:
        raise RuntimeError(
            f"ref_colors.csv contém {len(targets)} linhas, esperado {expected_n}."
        )
    return np.array(targets, dtype=np.float32), labels


def _auto_calibrar_imagem(
    image_bgr: np.ndarray,
    *,
    config_dir: str,
    static_dir: str,
    output_dir: Optional[str],
    calibrated_dir: Optional[str],
    source_path: Optional[str] = None,
    save_debug: bool = True,
) -> Tuple[np.ndarray, Dict[str, Optional[str]]]:
    info: Dict[str, Optional[str]] = {
        "performed": True,
        "palette_detected": False,
        "skip_reason": None,
        "error": None,
        "used_image": "original",
        "calibrated_path": None,
        "calibrated_name": None,
        "annotated_path": None,
        "warp_path": None,
        "warp_debug_path": None,
        "warp_labels_path": None,
        "measured_csv_path": None,
    }

    base_image = image_bgr.copy()

    try:
        corners, ids = _detect_markers(base_image)
        quad = _order_corners_by_expected(corners, ids)
    except Exception as exc:  # noqa: BLE001
        info["skip_reason"] = str(exc)
        return image_bgr, info

    info["palette_detected"] = True

    try:
        warped, _ = _compute_homography_and_warp(base_image, quad)
        centers_from_csv = _load_centers_from_csv(config_dir)
        centers_source = "patch_centers.csv" if centers_from_csv else "parametric_grid"
        centers_norm = centers_from_csv or _grid_centers_parametric()
        centers_px = _centers_px(centers_norm)
        labels = [f"P{i:02d}" for i in range(1, ROWS * COLS + 1)]

        measured_rgb = _read_rgb_at_points(warped, centers_px, win=SAMPLE_WIN)
        target_rgb, ref_labels = _load_targets_csv(
            config_dir, expected_n=ROWS * COLS
        )
        if target_rgb.shape != measured_rgb.shape:
            raise RuntimeError(
                "Dimensão dos alvos não bate com os patches medidos (ref_colors.csv vs amostragem)."
            )

        M = _solve_color_matrix(measured_rgb, target_rgb)
        corrected = _apply_color_matrix(base_image, M)

        corrected_sem_paleta = corrected.copy()
        mask = np.zeros(corrected_sem_paleta.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
        corrected_sem_paleta[mask == 255] = 0
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)
        return image_bgr, info

    base_name = os.path.basename(source_path) if source_path else None
    if not base_name:
        base_name = f"auto_{uuid.uuid4().hex[:8]}"
    name_wo, _ = os.path.splitext(base_name)

    info["used_image"] = "calibrated"

    if calibrated_dir:
        os.makedirs(calibrated_dir, exist_ok=True)
        calib_name = f"calibrated_{name_wo}.jpg"
        calib_path = os.path.join(calibrated_dir, calib_name)
        cv2.imwrite(
            calib_path, corrected_sem_paleta, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        )
        info["calibrated_path"] = calib_path
        info["calibrated_name"] = calib_name

    if output_dir and save_debug:
        os.makedirs(output_dir, exist_ok=True)
        annotated = os.path.join(output_dir, f"{name_wo}_det_aruco.png")
        warp_name = os.path.join(output_dir, f"{name_wo}_palette_warp.png")
        warp_dbg = os.path.join(output_dir, f"{name_wo}_palette_warp_debug.png")
        warp_lbl = os.path.join(output_dir, f"{name_wo}_palette_warp_labels.png")

        _save_aruco_overlay(base_image, corners, ids, annotated)
        cv2.imwrite(warp_name, warped)
        _draw_sampling_debug(
            warped,
            centers_px,
            labels,
            win=SAMPLE_WIN,
            out_dbg=warp_dbg,
            out_lbl=warp_lbl,
        )

        info["annotated_path"] = annotated
        info["warp_path"] = warp_name
        info["warp_debug_path"] = warp_dbg
        info["warp_labels_path"] = warp_lbl

    if config_dir:
        os.makedirs(config_dir, exist_ok=True)
        measured_csv = os.path.join(config_dir, f"measured_{name_wo}.csv")
        rows = [["label", "row", "col", "R", "G", "B"]]
        idx = 0
        for r in range(ROWS):
            for c in range(COLS):
                R, G, B = measured_rgb[idx].tolist()
                label = ref_labels[idx] if idx < len(ref_labels) else labels[idx]
                rows.append([label, r, c, R, G, B])
                idx += 1
        _save_csv(measured_csv, rows)
        info["measured_csv_path"] = measured_csv

        meta = {
            "image_used": base_name,
            "chart_layout": f"{ROWS}x{COLS}",
            "aruco_expected_ids": EXPECTED_IDS,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "warp_size": [WARP_W, WARP_H],
            "margins_gap": [MARGIN_X, MARGIN_Y, GAP_X, GAP_Y],
            "sample_win": SAMPLE_WIN,
            "ref_colors_csv": os.path.join(config_dir, "ref_colors.csv"),
            "measured_csv": measured_csv,
            "centers_source": centers_source,
        }
        _save_calibration(config_dir, M, meta)

    return corrected_sem_paleta, info

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
            heif = pillow_heif.open_heif(
                raw,
                convert_hdr_to_8bit=True,
                apply_transformations=True,
            )
            pil_img = heif.to_pillow() if hasattr(heif, "to_pillow") else Image.frombytes(
                heif.mode, heif.size, heif.data, "raw"
            )
            rgb = np.array(pil_img.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return bgr
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                "Falha ao decodificar a imagem HEIC/HEIF com pillow-heif: "
                f"{exc}. Tente reenviar como JPEG/PNG ou habilite 'Formato mais compatível' "
                "na câmera do iPhone."
            ) from exc
    msg = [
        "Falha ao decodificar a imagem (possivelmente HEIC/HEIF).",
        "Instale pillow-heif e Pillow: pip install pillow-heif pillow.",
    ]
    if _PILLOW_HEIF_ERR:
        msg.append(f"Motivo original: {_PILLOW_HEIF_ERR}")
    raise ValueError(" ".join(msg))

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

def _split_blob_watershed(binario: np.ndarray, ctr: np.ndarray) -> list:
    """
    Tenta separar um contorno colado (2 ovos viraram 1 blob) usando watershed
    dentro da ROI do contorno.
    Retorna uma lista de contornos (pode ser [ctr] se não conseguiu separar).
    """
    x, y, w, h = cv2.boundingRect(ctr)

    # ROI com margem pra não cortar bordas
    pad = 10
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(binario.shape[1], x + w + pad)
    y1 = min(binario.shape[0], y + h + pad)

    roi = binario[y0:y1, x0:x1].copy()

    # precisa ser branco=255 como foreground
    fg = (roi > 0).astype(np.uint8) * 255
    if fg.sum() < 500:  # muito pequeno
        return [ctr]

    # distancia -> picos no centro dos ovos
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # "sure foreground" (sementes) — ajuste o 0.45–0.65 se precisar
    sure_fg = (dist_norm > 0.55).astype(np.uint8) * 255

    # se não houver sementes suficientes, não dá pra separar
    n_lbl, markers = cv2.connectedComponents(sure_fg)
    if n_lbl <= 2:
        return [ctr]

    # sure background
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sure_bg = cv2.dilate(fg, kernel, iterations=2)

    # unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # markers para watershed
    markers = markers + 1
    markers[unknown > 0] = 0

    # watershed precisa de imagem 3 canais
    color = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color, markers)

    # cada label > 1 vira um objeto separado
    out_contours = []
    for label in range(2, markers.max() + 1):
        mask = (markers == label).astype(np.uint8) * 255
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            area = cv2.contourArea(c)
            if area < 800:  # descarta lixo
                continue
            c[:, 0, 0] += x0
            c[:, 0, 1] += y0
            out_contours.append(c)

    return out_contours if len(out_contours) >= 2 else [ctr]


def _maybe_split_big_blobs(binario: np.ndarray, contornos: list) -> list:
    """
    Para contornos muito grandes (colados), tenta separar via watershed.
    """
    if not contornos:
        return contornos

    areas = np.array([cv2.contourArea(c) for c in contornos], dtype=np.float32)
    med_area = float(np.median(areas)) if len(areas) else 0.0

    out = []
    for c in contornos:
        a = float(cv2.contourArea(c))
        x, y, w, h = cv2.boundingRect(c)

        # critérios de "provável colado" — ajuste se quiser
        too_big_area = (med_area > 0 and a > 1.7 * med_area)
        too_tall = (med_area > 0 and h > 1.7 * np.median([cv2.boundingRect(k)[3] for k in contornos]))

        if too_big_area or too_tall:
            out.extend(_split_blob_watershed(binario, c))
        else:
            out.append(c)

    return out


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
    *,
    auto_calibrar: bool = True,
    config_dir: Optional[str] = None,
    static_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    calibrated_dir: Optional[str] = None,
    retornar_calibracao: bool = False,
    salvar_debug_calibracao: bool = True,
):
    """
    Executa o pipeline completo dos ovos.

    Quando ``auto_calibrar`` estiver habilitado (padrão), o módulo tenta detectar a
    paleta ColorChecker, aplica a matriz de correção de cor e decide automaticamente
    se utiliza a imagem calibrada ou a original. Todo o processo de calibração é
    interno a este arquivo para permitir sua utilização isolada em outros projetos.

    Retorna: (img_b64, ovos_info) ou (img_b64, ovos_info, auto_calibracao) quando
    ``retornar_calibracao=True``.
    """
    fx, fy = map(float, fator_elipse)

    config_dir = config_dir or DEFAULT_CONFIG_DIR
    static_dir = static_dir or DEFAULT_STATIC_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    calibrated_dir = calibrated_dir or DEFAULT_CALIBRATED_DIR

    imagem_cv = _ler_imagem_cv(imagem)
    auto_calibration = _empty_auto_calibration(performed=False)
    imagem_para_processar = imagem_cv.copy()

    if auto_calibrar:
        _ensure_dirs(config_dir, static_dir)
        try:
            imagem_para_processar, info = _auto_calibrar_imagem(
                imagem_cv,
                config_dir=config_dir,
                static_dir=static_dir,
                output_dir=output_dir,
                calibrated_dir=calibrated_dir,
                source_path=imagem if isinstance(imagem, str) else None,
                save_debug=salvar_debug_calibracao,
            )
            auto_calibration = info
        except Exception as exc:  # noqa: BLE001
            auto_calibration = _empty_auto_calibration(performed=True)
            auto_calibration["error"] = str(exc)
            imagem_para_processar = imagem_cv.copy()
    else:
        imagem_para_processar = imagem_cv.copy()

    # 1) Ler BGR (sem EXIF) e cortar bordas
    imagem_trabalho = cortar_bordas_proporcional(imagem_para_processar.copy())

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

    # 8) Segmentação (na cópia de trabalho) — mais estável e com limpeza
    cinza = cv2.cvtColor(imagem_trabalho, cv2.COLOR_BGR2GRAY)

    # um blur um pouco mais forte ajuda a matar textura/ruído
    cinza = cv2.GaussianBlur(cinza, (9, 9), 0)

    # adaptiveThreshold com janela maior (11 é pequeno demais e gera "granulado")
    binario = cv2.bitwise_not(
        cv2.adaptiveThreshold(
            cinza, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,  # ↑ maior: 21, 31, 41
            9  # ↑ maior: 3, 5, 7
        )
    )

       # --- limpeza morfológica (ordem importa!) ---
    # 1) OPEN remove pontinhos (ruído) sem destruir ovos
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binario = cv2.morphologyEx(binario, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # 2) CLOSE fecha buracos / rachaduras no ovo
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binario = cv2.morphologyEx(binario, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    conts, _ = cv2.findContours(binario, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = sorted([cv2.contourArea(c) for c in conts], reverse=True)[:5]
    print("blobs:", len(conts), "top_areas:", areas)


    # 9) Contornos e filtragem — robusto (sem mediana)
    contornos, _ = cv2.findContours(binario, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = binario.shape[:2]
    area_min = 2200  # ajuste fino conforme seu setup
    area_max = int((W * H) * 0.20)  # evita pegar "fundo inteiro"

    candidatos = []

    for ctr in contornos:
        area = cv2.contourArea(ctr)
        if area < area_min or area > area_max:
            continue

        x, y, w, h = cv2.boundingRect(ctr)

        # descarta coisas muito pequenas
        if w < 40 or h < 40:
            continue

        # ovos normalmente não são extremamente finos/esticados
        aspect = w / (h + 1e-6)
        if aspect < 0.45 or aspect > 2.2:
            continue

        # circularidade: remove rabiscos/contornos muito irregulares
        per = cv2.arcLength(ctr, True)
        circ = (4 * np.pi * area) / ((per * per) + 1e-6)  # 0..1 (quanto mais perto de 1, mais "redondo")
        print(circ)
        if circ < 0.02:  # ajuste 0.15–0.30 (coloquei 0.2 para pegar o ovo do meio com menor circunferencia)
            continue

        candidatos.append(((x, y, w, h), ctr, area))

    # ordena por área (maiores primeiro) e limita (pra não pegar sujeira)
    candidatos.sort(key=lambda it: it[2], reverse=True)

    TOP_N = 40  # se você tem ~15-25 ovos, deixe 30-60
    paired = [(b, c) for (b, c, _) in candidatos[:TOP_N]]

    # 10) Agrupar em linhas (sem median_h, tolerância adaptativa)
    linhas = []

    # ordena por Y (de cima para baixo)
    for (x, y, w, h), ctr in sorted(paired, key=lambda it: it[0][1]):

        # tolerância vertical baseada no próprio ovo
        # evita depender de estatística global frágil
        tolerancia_y = max(40, int(h * 0.6))  # ajuste fino: 0.5–0.7

        for linha in linhas:
            # compara com o Y do primeiro elemento da linha
            y_ref = linha[0][0][1]
            if abs(y_ref - y) < tolerancia_y:
                linha.append(((x, y, w, h), ctr))
                break
        else:
            # cria nova linha
            linhas.append([((x, y, w, h), ctr)])

    # dentro de cada linha, ordenar da esquerda para a direita
    for linha in linhas:
        linha.sort(key=lambda it: it[0][0])

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
    if retornar_calibracao:
        return img_b64, ovos_info, auto_calibration
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
    *,
    auto_calibrar: bool = True,
    config_dir: Optional[str] = None,
    static_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    calibrated_dir: Optional[str] = None,
    retornar_calibracao: bool = False,
    salvar_debug_calibracao: bool = True,
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
        auto_calibrar=auto_calibrar,
        config_dir=config_dir,
        static_dir=static_dir,
        output_dir=output_dir,
        calibrated_dir=calibrated_dir,
        retornar_calibracao=retornar_calibracao,
        salvar_debug_calibracao=salvar_debug_calibracao,
    )
