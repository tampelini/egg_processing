# app/routes.py
import json
import logging
import os
import uuid
from flask import Blueprint, render_template, request
from werkzeug.utils import secure_filename

# === pipeline de ovos ===
from .processamento import processar_imagem

# === pipeline de calibração ===
from .calibrate import (
    has_saved_calibration,
    build_reference_from_image,
    build_calibration_from_image,
    apply_saved_calibration,
)

bp = Blueprint('routes', __name__)

# --- caminhos base ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))      # .../egg_processing/app
PROJECT_ROOT = os.path.dirname(APP_DIR)                   # .../egg_processing
STATIC_DIR = os.path.join(PROJECT_ROOT, 'static')
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
UPLOADS_DIR = os.path.join(STATIC_DIR, 'uploads')
OUTPUT_DIR = os.path.join(STATIC_DIR, 'output')
CALIBRATED_DIR = os.path.join(STATIC_DIR, 'calibrated')
for d in (STATIC_DIR, CONFIG_DIR, UPLOADS_DIR, OUTPUT_DIR, CALIBRATED_DIR):
    os.makedirs(d, exist_ok=True)

DEFAULT_AJUSTES = {
    "fator_v_backup": 1.0,
    "fator_contraste": 1.0,
    "fator_saturacao": 1.0,
    "ev_exposicao": 0.0,
    "fator_nitidez": 0.0,
    "fator_temperatura": 0.0,
}


def _convert_heic_if_needed(path: str) -> str:
    """Converte arquivos HEIC/HEIF em JPEG para compatibilidade ampla.

    Retorna o caminho (possivelmente alterado) que deve ser usado no restante
    do fluxo. Se o arquivo não for HEIC/HEIF, apenas devolve ``path``.
    """

    ext = os.path.splitext(path)[1].lower()
    if ext not in {".heic", ".heif"}:
        return path

    try:
        import pillow_heif  # type: ignore
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Suporte a HEIC requer as dependências pillow-heif e Pillow instaladas."
        ) from exc

    try:
        heif = pillow_heif.open_heif(
            path,
            convert_hdr_to_8bit=True,
            apply_transformations=True,
        )
        pil_img = heif.to_pillow() if hasattr(heif, "to_pillow") else Image.frombytes(
            heif.mode, heif.size, heif.data, "raw"
        )
        rgb = pil_img.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Não foi possível abrir o arquivo HEIC/HEIF: {exc}. "
            "Tente reenviar como JPEG/PNG ou ativar o modo 'Formato mais compatível' no iPhone."
        ) from exc

    new_path = os.path.splitext(path)[0] + ".jpg"
    try:
        rgb.save(new_path, format="JPEG", quality=95)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Falha ao salvar conversão HEIC→JPEG: {exc}") from exc

    logging.info("Arquivo HEIC convertido para JPEG: %s -> %s", path, new_path)
    return new_path

def _to_url(path_abs: str) -> str:
    """Converte caminho absoluto para URL relativa ao root do app (prefixada com '/')."""
    rel = os.path.relpath(path_abs, start=PROJECT_ROOT).replace("\\", "/")
    return "/" + rel

# ---------------------- normalização dos dicts de ovos ---------------------- #
def _normalize_ovo(o: dict) -> dict:
    """
    Cria aliases em minúsculas para chaves comuns (HEX->hex, RGB->rgb, etc.)
    sem perder as chaves originais. Também normaliza centro/orientação/num.
    """
    o = dict(o)  # cópia rasa
    # cores/campos numéricos
    if 'HEX' in o and 'hex' not in o: o['hex'] = o['HEX']
    if 'RGB' in o and 'rgb' not in o: o['rgb'] = o['RGB']
    if 'LAB' in o and 'lab' not in o: o['lab'] = o['LAB']
    if 'LCH' in o and 'lch' not in o: o['lch'] = o['LCH']
    if 'XYZ' in o and 'xyz' not in o: o['xyz'] = o['XYZ']
    if 'ACES' in o and 'aces' not in o: o['aces'] = o['ACES']
    if 'ACEScg' in o and 'acescg' not in o: o['acescg'] = o['ACEScg']
    if 'Linear_sRGB' in o and 'linsrgb' not in o: o['linsrgb'] = o['Linear_sRGB']
    if 'CMYK' in o and 'cmyk' not in o: o['cmyk'] = o['CMYK']
    # posição/orientação
    if 'orientacao' in o and 'orientation' not in o: o['orientation'] = o['orientacao']
    if 'center' in o and 'centro' not in o: o['centro'] = o['center']
    # identificador
    if 'num' not in o: o['num'] = None
    return o
# --------------------------------------------------------------------------- #

@bp.route('/', methods=['GET', 'POST'])
def index():
    """
    Opção 1 (Processar ovos):
      - Form com name="imagem" envia POST para "/"
      - Aceita também o campo opcional name="fator_v_backup" (float; 0.2..1.5)
      - Chama processar_imagem(file, fator_v_backup=...) e devolve (imagem em base64, ovos_info)
    """
    imagem_b64 = None
    ovos_info = None
    ajustes = DEFAULT_AJUSTES.copy()
    error_message = None

    auto_calibration = None

    if request.method == 'POST' and 'imagem' in request.files:
        file = request.files['imagem']
        if file and file.filename:
            def _parse_float(nome, default, minimo=None, maximo=None):
                raw = (request.form.get(nome) or '').strip()
                try:
                    valor = float(raw) if raw else default
                except Exception:
                    valor = default
                if minimo is not None:
                    valor = max(minimo, valor)
                if maximo is not None:
                    valor = min(maximo, valor)
                ajustes[nome] = valor
                return valor

            fator_v_backup = _parse_float('fator_v_backup', 1.0, 0.2, 3.0)
            fator_contraste = _parse_float('fator_contraste', 1.0, 0.2, 3.0)
            fator_saturacao = _parse_float('fator_saturacao', 1.0, 0.0, 3.0)
            ev_exposicao = _parse_float('ev_exposicao', 0.0, -5.0, 5.0)
            fator_nitidez = _parse_float('fator_nitidez', 0.0, -1.0, 2.0)
            fator_temperatura = _parse_float('fator_temperatura', 0.0, -1.0, 1.0)

            filename = secure_filename(file.filename)
            name_wo, ext = os.path.splitext(filename)
            unique_suffix = uuid.uuid4().hex[:8]
            unique_name = f"{name_wo}_{unique_suffix}{ext}" if ext else f"{name_wo}_{unique_suffix}"
            upload_path = os.path.join(UPLOADS_DIR, unique_name)
            file.save(upload_path)

            try:
                upload_path = _convert_heic_if_needed(upload_path)
            except Exception as exc:  # noqa: BLE001
                logging.exception("Falha na conversão HEIC")
                error_message = str(exc)
                imagem_b64 = None
                ovos_info = None
                auto_calibration = None
                return render_template(
                    'index.html',
                    imagem=imagem_b64,
                    ovos_info=ovos_info,
                    ajustes=ajustes,
                    auto_calibration=auto_calibration,
                    error_message=error_message,
                    img_url=None, ann_url=None, warp_url=None,
                    warp_debug_url=None, warp_labels_url=None,
                    calibrated_url=None, cfg=None, cfg_path=None,
                    calib_exists=has_saved_calibration(CONFIG_DIR),
                )

            try:
                imagem_b64, ovos_info, auto_calibration = processar_imagem(
                    upload_path,
                    fator_elipse=(0.85, 0.75),
                    usar_fitellipse=True,
                    fator_v_backup=fator_v_backup,
                    fator_contraste=fator_contraste,
                    fator_saturacao=fator_saturacao,
                    ev_exposicao=ev_exposicao,
                    fator_nitidez=fator_nitidez,
                    fator_temperatura=fator_temperatura,
                    auto_calibrar=True,
                    config_dir=CONFIG_DIR,
                    static_dir=STATIC_DIR,
                    output_dir=OUTPUT_DIR,
                    calibrated_dir=CALIBRATED_DIR,
                    retornar_calibracao=True,
                )
            except Exception as exc:  # noqa: BLE001
                logging.exception("Falha no processamento da imagem")
                error_message = (
                    "Não foi possível processar a imagem enviada. "
                    f"Detalhes: {exc}. Tente reenviar em JPEG/PNG ou usar a opção "
                    "'Formato mais compatível' na câmera do iPhone."
                )
                imagem_b64 = None
                ovos_info = None
                auto_calibration = None
            if auto_calibration:
                calib_path = auto_calibration.get("calibrated_path")
                if calib_path and os.path.exists(calib_path):
                    auto_calibration["calibrated_url"] = _to_url(calib_path)
                else:
                    auto_calibration["calibrated_url"] = None
            # normaliza cada ovo (se veio lista de dicts)
            if ovos_info:
                ovos_info = [_normalize_ovo(o) for o in ovos_info]

    calib_exists = has_saved_calibration(CONFIG_DIR)

    return render_template(
        'index.html',
        # processamento
        imagem=imagem_b64,
        ovos_info=ovos_info,
        ajustes=ajustes,
        auto_calibration=auto_calibration,
        error_message=error_message,
        # calibração (vazio por padrão)
        img_url=None, ann_url=None, warp_url=None,
        warp_debug_url=None, warp_labels_url=None,
        calibrated_url=None, cfg=None, cfg_path=None,
        # status
        calib_exists=calib_exists
    )

@bp.route('/calibrar', methods=['POST'])
def calibrar():
    """
    Opção 2 (Calibrar ColorChecker):
      - Form com name="image" + checkboxes (as_reference, ignore_saved, purge_saved)
      - Executa fluxo do calibrate.py e exibe resultados na mesma página
    """
    f = request.files.get('image')
    if not f or not f.filename:
        # volta ao index mantendo tudo vazio
        return render_template(
            'index.html',
            ajustes=DEFAULT_AJUSTES.copy(),
            auto_calibration=None,
            imagem=None,
            ovos_info=None,
            error_message=None,
            calib_exists=has_saved_calibration(CONFIG_DIR),
            img_url=None, ann_url=None, warp_url=None,
            warp_debug_url=None, warp_labels_url=None,
            calibrated_url=None, cfg=None, cfg_path=None,
        )

    filename = secure_filename(f.filename)
    upload_path = os.path.join(UPLOADS_DIR, filename)
    f.save(upload_path)

    try:
        upload_path = _convert_heic_if_needed(upload_path)
    except Exception as exc:  # noqa: BLE001
        logging.exception("Falha na conversão HEIC para calibração")
        return render_template(
            'index.html',
            ajustes=DEFAULT_AJUSTES.copy(),
            auto_calibration=None,
            imagem=None,
            ovos_info=None,
            error_message=str(exc),
            calib_exists=has_saved_calibration(CONFIG_DIR),
            img_url=None, ann_url=None, warp_url=None,
            warp_debug_url=None, warp_labels_url=None,
            calibrated_url=None, cfg=None, cfg_path=None,
        )

    use_as_ref   = bool(request.form.get('as_reference'))
    ignore_saved = bool(request.form.get('ignore_saved'))
    purge_saved  = bool(request.form.get('purge_saved'))

    # reset opcional
    if purge_saved:
        try:
            for name in ("color_calibration.json", "ref_colors.csv"):
                p = os.path.join(CONFIG_DIR, name)
                if os.path.exists(p):
                    os.remove(p)
            for n in os.listdir(CONFIG_DIR):
                if n.startswith("measured_") and n.endswith(".csv"):
                    os.remove(os.path.join(CONFIG_DIR, n))
        except Exception:
            pass

    # criar/atualizar referência
    if use_as_ref:
        try:
            build_reference_from_image(upload_path, CONFIG_DIR, STATIC_DIR)
        except Exception:
            pass

    have_saved = has_saved_calibration(CONFIG_DIR)

    context = {
        "img_url": _to_url(upload_path),
        "ann_url": None,
        "warp_url": None,
        "warp_debug_url": None,
        "warp_labels_url": None,
        "calibrated_url": None,
        "cfg": None,
        "cfg_path": None,
        # resultados de processamento desligados nesta renderização
        "imagem": None,
        "ovos_info": None,
        "ajustes": DEFAULT_AJUSTES.copy(),
        "auto_calibration": None,
        "error_message": None,
        "calib_exists": have_saved,
    }

    try:
        if have_saved and not ignore_saved:
            # aplica calibração existente direto
            name_wo, _ = os.path.splitext(filename)
            out_img = os.path.join(CALIBRATED_DIR, f"calibrated_{name_wo}.jpg")
            apply_saved_calibration(upload_path, CONFIG_DIR, out_img)
            context["calibrated_url"] = _to_url(out_img)
        else:
            # constrói calibração a partir desta imagem
            out = build_calibration_from_image(upload_path, CONFIG_DIR, STATIC_DIR, CALIBRATED_DIR)

            def _out_url(name):
                return _to_url(os.path.join(OUTPUT_DIR, name))

            if out.get("annotated_name"):
                context["ann_url"] = _out_url(out["annotated_name"])
            if out.get("warp_name"):
                context["warp_url"] = _out_url(out["warp_name"])
            if out.get("warp_debug_name"):
                context["warp_debug_url"] = _out_url(out["warp_debug_name"])
            if out.get("warp_labels_name"):
                context["warp_labels_url"] = _out_url(out["warp_labels_name"])
            if out.get("calibrated_name"):
                context["calibrated_url"] = _to_url(os.path.join(CALIBRATED_DIR, out["calibrated_name"]))

        # carrega JSON de calibração (se houver)
        colors_json_path = os.path.join(CONFIG_DIR, "color_calibration.json")
        if os.path.exists(colors_json_path):
            with open(colors_json_path, "r", encoding="utf-8") as fjson:
                context["cfg"] = json.load(fjson)
                context["cfg_path"] = "config/color_calibration.json"

    except Exception as e:
        context["cfg"] = {"error": str(e)}
        context["cfg_path"] = "Erro durante a calibração"

    return render_template('index.html', **context)
