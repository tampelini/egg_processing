# calibrate.py
from __future__ import annotations
import os
import json
import csv
from dataclasses import dataclass
from typing import Tuple, List, Dict
from datetime import datetime

import numpy as np
import cv2

# ===================== Configurações gerais =====================

# Extensões aceitas
ALLOWED_EXT = {"jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"}

# IDs esperados dos 4 ArUco (dicionário 4x4_50) nos cantos da paleta
# TL/TR/BL/BR = Top-Left / Top-Right / Bottom-Left / Bottom-Right
EXPECTED_IDS = {"TL": 10, "TR": 11, "BL": 12, "BR": 13}

# Layout da sua paleta: 12 linhas × 2 colunas (24 patches)
ROWS, COLS = 12, 2

# Tamanho do warp (paleta retificada)
WARP_W, WARP_H = 800, 1600  # alto > largo, combina com a paleta em coluna dupla

# Margens e espaçamentos relativos dentro do warp (ajuste fino conforme sua arte)
MARGIN_X, MARGIN_Y = 0.14, 0.05
GAP_X, GAP_Y = 0.06, 0.02

# Janela de amostragem (pixels) para média RGB dentro de cada patch
SAMPLE_WIN = 34

@dataclass
class CalibrationResult:
    annotated_name: str | None
    calibrated_name: str
    matrix: np.ndarray  # 3x4 (inclui bias)
    ref_csv: str
    measured_csv: str
    warp_name: str
    warp_debug_name: str
    warp_labels_name: str
    orig_overlay_name: str


# ===================== Utilidades de diretório =====================

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def _ensure_output_dir(static_dir: str):
    out_dir = os.path.join(static_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def _save_csv(path: str, rows: List[List]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


# ===================== ArUco: detecção e ordenação =====================

def _aruco_detector():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco não disponível. Instale opencv-contrib-python.")
    aruco = cv2.aruco
    if hasattr(aruco, "ArucoDetector"):
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        params = aruco.DetectorParameters()
        # ajustes úteis em cenário real
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 33
        params.adaptiveThreshWinSizeStep = 10
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        detector = aruco.ArucoDetector(dictionary, params)
        return detector, dictionary
    else:
        dictionary = aruco.Dictionary_get(aruco.DICT_4X4_50)
        params = aruco.DetectorParameters_create()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        return (dictionary, params), dictionary

def _detect_markers(image_bgr):
    det, dictn = _aruco_detector()
    if isinstance(det, tuple):
        dictionary, params = det
        corners, ids, _ = cv2.aruco.detectMarkers(image_bgr, dictionary, parameters=params)
    else:
        corners, ids, _ = det.detectMarkers(image_bgr)
    if ids is None or len(ids) < 4:
        raise RuntimeError("Não encontrei marcadores ArUco suficientes (precisa de 4).")
    return corners, ids.flatten().tolist()

def _order_corners_by_expected(corners, ids: List[int]) -> np.ndarray:
    id_to_corner = {idv: c.reshape(-1, 2) for c, idv in zip(corners, ids)}
    try:
        tl = id_to_corner[EXPECTED_IDS["TL"]][0]
        tr = id_to_corner[EXPECTED_IDS["TR"]][1]
        br = id_to_corner[EXPECTED_IDS["BR"]][2]
        bl = id_to_corner[EXPECTED_IDS["BL"]][3]
    except KeyError:
        raise RuntimeError("IDs esperados (10,11,12,13) não foram encontrados.")
    quad = np.array([tl, tr, br, bl], dtype=np.float32)
    return quad


# ===================== Warp e amostragem =====================

def _compute_homography_and_warp(image_bgr, quad_src: np.ndarray):
    dst = np.array([[0, 0],
                    [WARP_W - 1, 0],
                    [WARP_W - 1, WARP_H - 1],
                    [0, WARP_H - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(quad_src, dst)
    warped = cv2.warpPerspective(image_bgr, H, (WARP_W, WARP_H))
    return warped, H

def _grid_centers_parametric():
    """Centros (x,y) normalizados 0..1 para 12×2 dentro do warp, ordem por linhas."""
    xs, ys = [], []
    cell_w = (1.0 - 2 * MARGIN_X - (COLS - 1) * GAP_X) / COLS
    cell_h = (1.0 - 2 * MARGIN_Y - (ROWS - 1) * GAP_Y) / ROWS
    for r in range(ROWS):
        cy = MARGIN_Y + r * (cell_h + GAP_Y) + cell_h / 2
        ys.append(cy)
    for c in range(COLS):
        cx = MARGIN_X + c * (cell_w + GAP_X) + cell_w / 2
        xs.append(cx)
    centers = []
    for r in range(ROWS):
        for c in range(COLS):
            centers.append((xs[c], ys[r]))  # normalizados
    return centers  # 24

def _load_centers_from_csv(config_dir: str) -> List[Tuple[float, float]] | None:
    """Lê config/patch_centers.csv com cabeçalho: x,y (normalizados 0..1)."""
    path = os.path.join(config_dir, "patch_centers.csv")
    if not os.path.exists(path):
        return None
    out = []
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            x = float(row["x"])
            y = float(row["y"])
            out.append((x, y))
    if len(out) != ROWS * COLS:
        raise RuntimeError(f"patch_centers.csv deve ter {ROWS*COLS} linhas; tem {len(out)}.")
    return out

def _centers_px(centers_norm: List[Tuple[float, float]]):
    return [(int(nx * WARP_W), int(ny * WARP_H)) for (nx, ny) in centers_norm]

def _read_rgb_at_points(image_bgr, points: List[Tuple[int, int]], win: int = SAMPLE_WIN):
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


# ===================== Debug overlays =====================

def _save_aruco_overlay(image_bgr, corners, ids, out_path):
    vis = image_bgr.copy()
    aruco = cv2.aruco
    ids_arr = np.array(ids).reshape(-1, 1)
    aruco.drawDetectedMarkers(vis, corners, ids_arr)
    # desenha polígono do quad se possível
    try:
        quad = _order_corners_by_expected(corners, ids)
        cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 255, 255), 3)
    except Exception:
        pass
    cv2.imwrite(out_path, vis)

def _draw_sampling_debug(warped, centers_px, labels, win=SAMPLE_WIN, out_dbg=None, out_lbl=None):
    half = win // 2
    dbg = warped.copy()
    for (x, y) in centers_px:
        cv2.rectangle(dbg, (x - half, y - half), (x + half, y + half), (0, 255, 0), 2)
        cv2.circle(dbg, (x, y), 3, (0, 0, 255), -1)
    if out_dbg:
        cv2.imwrite(out_dbg, dbg)

    lbl = warped.copy()
    for (x, y), text in zip(centers_px, labels):
        cv2.rectangle(lbl, (x - half, y - half), (x + half, y + half), (0, 255, 0), 1)
        cv2.putText(lbl, text, (x - half, y - half - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(lbl, text, (x - half, y - half - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    if out_lbl:
        cv2.imwrite(out_lbl, lbl)


# ===================== Solver de calibração (robusto) =====================

def _solve_color_matrix(measured: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Resolve transformação afim 3x4 com regularização; fallback para regressão por canal."""
    assert measured.shape == target.shape and measured.shape[1] == 3

    # 1) variância mínima para evitar degenerescência
    if np.min(np.std(target, axis=0)) < 3:
        raise RuntimeError("Targets muito pouco variados. Forneça ref_colors.csv real (não constante).")
    if np.min(np.std(measured, axis=0)) < 1:
        raise RuntimeError("Medições quase constantes. Verifique warp/amostragem (SAMPLE_WIN, margens/gaps).")

    # 2) Tikhonov (ridge) para estabilizar
    N = measured.shape[0]
    A = np.hstack([measured.astype(np.float32), np.ones((N, 1), dtype=np.float32)])  # Nx4
    lam = 1e-2
    AtA = A.T @ A + lam * np.eye(4, dtype=np.float32)
    M = np.zeros((3, 4), dtype=np.float32)
    for ch in range(3):
        y = target[:, ch].astype(np.float32)
        x = np.linalg.solve(AtA, A.T @ y)
        M[ch, :] = x

    # 3) sanity checks — ganhos/bias dentro de limites razoáveis
    gains = np.abs(M[:, :3])
    bias = np.abs(M[:, 3])
    if np.any(gains < 0.05) or np.any(gains > 5) or np.any(bias > 60):
        # fallback: regressão por canal (y = a*x + b) — simples e estável
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
    A = np.hstack([flat, np.ones((flat.shape[0], 1), dtype=np.float32)])  # Nx4
    out = A @ M.T
    out = np.clip(out, 0, 255).reshape(H, W, 3).astype(np.uint8)
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return bgr


# ===================== Persistência da calibração =====================

def has_saved_calibration(config_dir: str) -> bool:
    return os.path.exists(os.path.join(config_dir, "color_calibration.json"))

def _save_calibration(config_dir: str, M: np.ndarray, meta: Dict):
    data = {"matrix_3x4": M.tolist(), "meta": meta}
    with open(os.path.join(config_dir, "color_calibration.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _load_calibration(config_dir: str) -> np.ndarray:
    with open(os.path.join(config_dir, "color_calibration.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    M = np.array(data["matrix_3x4"], dtype=np.float32)
    return M

def apply_saved_calibration(image_path: str, config_dir: str, save_path: str):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError("Não consegui abrir a imagem para aplicar a calibração.")
    M = _load_calibration(config_dir)
    corrected = _apply_color_matrix(image_bgr, M)
    cv2.imwrite(str(save_path), corrected, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return save_path


# ===================== Referência e targets =====================

def _require_targets_csv(config_dir: str) -> str:
    """Exige que exista config/ref_colors.csv com RGBs reais da paleta."""
    path = os.path.join(config_dir, "ref_colors.csv")
    if not os.path.exists(path):
        raise RuntimeError(
            "ref_colors.csv não encontrado em config/. "
            "Gere a referência marcando 'Usar esta imagem como referência' "
            "ou forneça o CSV manualmente."
        )
    return path

def _load_targets_csv(config_dir: str, expected_n: int):
    """Lê ref_colors.csv aceitando formatos:
       1) label,row,col,R,G,B
       2) row,col,R,G,B
       3) R,G,B
    Retorna (target_rgb: Nx3 float32, labels: List[str]).
    """
    path = os.path.join(config_dir, "ref_colors.csv")
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        fieldset = {k.strip().lower() for k in (rd.fieldnames or [])}

        labels, targets = [], []

        if {"label", "row", "col", "r", "g", "b"} <= fieldset:
            for row in rd:
                labels.append(row["label"])
                targets.append([float(row["R"]), float(row["G"]), float(row["B"])])
        elif {"row", "col", "r", "g", "b"} <= fieldset:
            i = 1
            for row in rd:
                labels.append(f"P{i:02d}")
                targets.append([float(row["R"]), float(row["G"]), float(row["B"])])
                i += 1
        elif {"r", "g", "b"} <= fieldset:
            i = 1
            for row in rd:
                labels.append(f"P{i:02d}")
                targets.append([float(row["R"]), float(row["G"]), float(row["B"])])
                i += 1
        else:
            raise RuntimeError(
                "Cabeçalho do ref_colors.csv inválido. Use label,row,col,R,G,B "
                "ou row,col,R,G,B ou R,G,B."
            )

    if len(targets) != expected_n:
        raise RuntimeError(f"ref_colors.csv contém {len(targets)} linhas, esperado {expected_n}.")
    return np.array(targets, dtype=np.float32), labels

def build_reference_from_image(image_path: str, config_dir: str, static_dir: str):
    """Gera/atualiza config/ref_colors.csv a partir de uma foto da paleta."""
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError("Não consegui abrir a imagem para criar a referência.")

    corners, ids = _detect_markers(image_bgr)
    quad = _order_corners_by_expected(corners, ids)
    warped, _ = _compute_homography_and_warp(image_bgr, quad)

    centers_norm = _load_centers_from_csv(config_dir) or _grid_centers_parametric()
    centers_px = _centers_px(centers_norm)
    measured_rgb = _read_rgb_at_points(warped, centers_px, win=SAMPLE_WIN)

    # grava ref_colors.csv com labels P01..P24
    path = os.path.join(config_dir, "ref_colors.csv")
    rows = [["label", "row", "col", "R", "G", "B"]]
    idx = 0
    for r in range(ROWS):
        for c in range(COLS):
            R, G, B = measured_rgb[idx].tolist()
            rows.append([f"P{idx + 1:02d}", r, c, int(round(R)), int(round(G)), int(round(B))])
            idx += 1
    _save_csv(path, rows)

    # também salvo warp e debug para inspeção
    out_dir = _ensure_output_dir(static_dir)
    cv2.imwrite(os.path.join(out_dir, "ref_palette_warp.png"), warped)
    _draw_sampling_debug(
        warped, centers_px, [f"P{i:02d}" for i in range(1, ROWS * COLS + 1)],
        win=SAMPLE_WIN,
        out_dbg=os.path.join(out_dir, "ref_palette_warp_debug.png"),
        out_lbl=os.path.join(out_dir, "ref_palette_warp_labels.png"),
    )
    return path


# ===================== Pipeline principal (primeira vez) =====================

def build_calibration_from_image(image_path: str, config_dir: str, static_dir: str, calibrated_dir: str) -> Dict:
    """
    1) Detecta paleta via ArUco (4 cantos com IDs esperados)
    2) Warpa a paleta e lê RGB dos 24 patches
    3) Carrega targets de ref_colors.csv
    4) Resolve M (3x4) e aplica na imagem inteira
    5) Salva calibrated_<nome>.jpg e arquivos de debug
    6) Persiste color_calibration.json para reuso
    """
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError("Não consegui abrir a imagem.")

    # 1) Detecta ArUco e overlay na original
    corners, ids = _detect_markers(image_bgr)
    output_dir = _ensure_output_dir(static_dir)
    orig_overlay_name = "det_aruco.png"
    _save_aruco_overlay(image_bgr, corners, ids, os.path.join(output_dir, orig_overlay_name))

    quad = _order_corners_by_expected(corners, ids)
    # overlay do quadrilátero também
    vis = image_bgr.copy()
    cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 255, 255), 3)
    cv2.imwrite(os.path.join(output_dir, "orig_quad_overlay.png"), vis)

    # 2) Warp da paleta
    warped, H = _compute_homography_and_warp(image_bgr, quad)
    warp_name = "palette_warp.png"
    cv2.imwrite(os.path.join(output_dir, warp_name), warped)

    # 3) Centros (CSV se existir; senão grid paramétrico)
    centers_norm = _load_centers_from_csv(config_dir) or _grid_centers_parametric()
    centers_px = _centers_px(centers_norm)

    # Debug da amostragem
    labels = [f"P{i:02d}" for i in range(1, ROWS * COLS + 1)]
    warp_debug_name = "palette_warp_debug.png"
    warp_labels_name = "palette_warp_labels.png"
    _draw_sampling_debug(
        warped, centers_px, labels,
        win=SAMPLE_WIN,
        out_dbg=os.path.join(output_dir, warp_debug_name),
        out_lbl=os.path.join(output_dir, warp_labels_name),
    )

    # 4) Meça RGB dos patches
    measured_rgb = _read_rgb_at_points(warped, centers_px, win=SAMPLE_WIN)  # (24,3)

    # 5) Carrega targets reais
    _ = _require_targets_csv(config_dir)  # valida existência
    target_rgb, ref_labels = _load_targets_csv(config_dir, expected_n=ROWS * COLS)

    if target_rgb.shape != measured_rgb.shape:
        raise RuntimeError("Dimensão dos alvos não bate com os patches medidos (ref_colors.csv vs amostragem).")

    # 6) Resolve M e aplica
    M = _solve_color_matrix(measured_rgb, target_rgb)
    corrected = _apply_color_matrix(image_bgr, M)

    # 7) Salva imagem calibrada e CSV das medições
    base = os.path.basename(image_path)
    name_wo, _ = os.path.splitext(base)
    os.makedirs(calibrated_dir, exist_ok=True)
    calib_name = f"calibrated_{name_wo}.jpg"
    cv2.imwrite(os.path.join(calibrated_dir, calib_name), corrected, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    measured_csv = os.path.join(config_dir, f"measured_{name_wo}.csv")
    rows = [["label", "row", "col", "R", "G", "B"]]
    idx = 0
    for r in range(ROWS):
        for c in range(COLS):
            R, G, B = measured_rgb[idx].tolist()
            rows.append([labels[idx], r, c, R, G, B])
            idx += 1
    _save_csv(measured_csv, rows)

    # 8) Persiste parâmetros para reuso
    meta = {
        "image_used": base,
        "chart_layout": f"{ROWS}x{COLS}",
        "aruco_expected_ids": EXPECTED_IDS,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "warp_size": [WARP_W, WARP_H],
        "margins_gap": [MARGIN_X, MARGIN_Y, GAP_X, GAP_Y],
        "sample_win": SAMPLE_WIN,
        "ref_colors_csv": os.path.join(config_dir, "ref_colors.csv"),
        "measured_csv": measured_csv,
        "centers_source": "patch_centers.csv" if _load_centers_from_csv(config_dir) else "parametric_grid",
    }
    _save_calibration(config_dir, M, meta)

    return {
        "annotated_name": orig_overlay_name,           # det_aruco.png
        "orig_overlay_name": "orig_quad_overlay.png",  # quad na original
        "calibrated_name": calib_name,
        "warp_name": warp_name,
        "warp_debug_name": warp_debug_name,
        "warp_labels_name": warp_labels_name,
    }
