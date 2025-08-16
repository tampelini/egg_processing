# validate_strip.py
import os
import sys
import cv2
import cv2.aruco as aruco
import numpy as np

EXPECTED_IDS = {10, 11, 12, 13}  # TL, TR, BL, BR (em qualquer ordem no papel)

def order_corners_tl_tr_br_bl(pts):
    # pts: np.array shape (4,2)
    s  = pts.sum(axis=1)           # x+y
    d  = np.diff(pts, axis=1).ravel()  # x - y
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def detect_markers(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dict_ = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    try:
        detector = aruco.ArucoDetector(dict_, aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
    except Exception:
        corners, ids, _ = aruco.detectMarkers(gray, dict_)
    return corners, ids

def annotate(image_bgr, corners, ids):
    out = image_bgr.copy()
    if ids is not None and len(ids) > 0:
        aruco.drawDetectedMarkers(out, corners, ids)
        for quad, mid in zip(corners, ids.ravel()):
            c = quad.reshape(-1,2).mean(axis=0).astype(int)
            cv2.putText(out, f"ID {int(mid)}", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
    return out

def quality_report(corners, ids, w, h):
    if ids is None:
        return {"ok": False, "reason": "Nenhum marcador detectado."}
    got = set(int(i) for i in ids.ravel().tolist())
    missing = list(EXPECTED_IDS - got)
    if missing:
        return {"ok": False, "reason": f"IDs faltando: {missing}. Detectados: {sorted(got)}"}

    # compute areas (em px²) e tamanhos dos 4 markers
    sizes = []
    centers = []
    by_id = {}
    for quad, mid in zip(corners, ids.ravel()):
        q = quad.reshape(-1,2)
        area = cv2.contourArea(q.astype(np.float32))
        sizes.append((int(mid), area))
        centers.append(q.mean(axis=0))
        by_id[int(mid)] = q

    sizes.sort(key=lambda x: x[0])
    # regra simples: todos precisam ter área suficiente e variação < 50%
    areas = np.array([a for (_, a) in sizes], dtype=np.float32)
    area_min = max(50.0, 0.00002 * w * h)  # ~0.002% da imagem como mínimo
    if np.any(areas < area_min):
        return {"ok": False, "reason": f"Marcador muito pequeno/recortado (mín {area_min:.1f} px²). Áreas: {areas.round(1).tolist()}"}
    if areas.max() / max(areas.min(), 1.0) > 1.5:
        # variação muito grande pode indicar corte/angulo extremo
        warn = f"Variação de tamanho alta (max/min={areas.max()/areas.min():.2f})."
    else:
        warn = None

    return {"ok": True, "reason": warn or "OK", "areas": areas.round(1).tolist()}

def try_warp(image_bgr, corners, ids, out_path="warped_strip.png", W=720, H=2000):
    # usa centros dos IDs esperados como cantos do strip (TL,TR,BR,BL) – robusto p/ inspeção
    centers = {}
    for quad, mid in zip(corners, ids.ravel()):
        centers[int(mid)] = quad.reshape(-1,2).mean(axis=0)

    if not EXPECTED_IDS.issubset(centers.keys()):
        return False

    pts = np.array([centers[10], centers[11], centers[13], centers[12]], dtype=np.float32)  # TL,TR,BR,BL
    pts = order_corners_tl_tr_br_bl(pts)
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
    Hmat = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image_bgr, Hmat, (W, H))
    cv2.imwrite(out_path, warped)
    return True

def main():
    if len(sys.argv) < 2:
        print("Uso: python validate_strip.py <imagem1> [imagem2 ...]")
        sys.exit(1)

    for img_path in sys.argv[1:]:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERRO] Não abriu: {img_path}")
            continue

        corners, ids = detect_markers(img)
        ann = annotate(img, corners, ids)

        base, ext = os.path.splitext(img_path)
        out_annot = base + "_annotated.png"
        cv2.imwrite(out_annot, ann)

        rep = quality_report(corners, ids, img.shape[1], img.shape[0])
        print(f"\nArquivo: {img_path}")
        print("Relatório:", rep)

        # tenta warp (opcional; útil para ver se os cantos fazem sentido)
        warped_ok = try_warp(img, corners, ids, out_path=base + "_warped.png")
        if warped_ok:
            print("Warp salvo em:", base + "_warped.png")
        print("Anotado salvo em:", out_annot)

if __name__ == "__main__":
    main()
