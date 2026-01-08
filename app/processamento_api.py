"""Camada fina para expor o pipeline calibrado com retorno compatível com a API."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import processamento as _processamento

_ProcessamentoBaseReturn = Tuple[str, List[Dict[str, Any]]]


def _strip_calibration(result: Iterable[Any]) -> _ProcessamentoBaseReturn:
    """Garante que apenas (imagem_base64, ovos_info) seja retornado."""
    if not isinstance(result, tuple):  # pragma: no cover - proteção adicional
        raise TypeError("Resultado inesperado do processamento: esperado tupla.")
    if len(result) < 2:
        raise ValueError("Resultado do processamento precisa ter ao menos 2 elementos.")
    return result[0], result[1]


def processar_imagem(
    imagem,
    fator_elipse=(0.85, 0.75),
    usar_fitellipse: bool = True,
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
) -> _ProcessamentoBaseReturn:
    """Processa a imagem executando a mesma calibração utilizada no app principal."""
    result = _processamento.processar_imagem(
        imagem,
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
    return _strip_calibration(result)


def processar_imagem_por_url(
    url: str,
    timeout: int = 20,
    fator_elipse=(0.85, 0.75),
    usar_fitellipse: bool = True,
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
) -> _ProcessamentoBaseReturn:
    """Baixa a imagem da URL e processa utilizando o pipeline completo."""
    result = _processamento.processar_imagem_por_url(
        url,
        timeout=timeout,
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
    return _strip_calibration(result)
