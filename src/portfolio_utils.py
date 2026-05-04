"""
portfolio_utils.py — Utilitários de análise de portfólio.

Funções puras e reutilizáveis para cálculo de métricas de risco/retorno.
Todas as funções assumem retornos log-diários como input padrão.
"""
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DIAS_ANO_CRIPTO = 365  # cripto opera 24/7, sem fins de semana


# ────────────────────────────────────────────────────────────
# Risco e retorno individuais
# ────────────────────────────────────────────────────────────

def calcular_matriz_covariancia(
    retornos: pd.DataFrame,
    anualizar: bool = True,
) -> pd.DataFrame:
    """Matriz de covariância dos retornos, opcionalmente anualizada.

    Args:
        retornos: DataFrame de log-retornos diários (linhas=datas, colunas=ativos).
        anualizar: Se True, multiplica por DIAS_ANO_CRIPTO.

    Returns:
        DataFrame n×n de covariâncias.

    Raises:
        ValueError: Se retornos estiver vazio ou contiver apenas NaNs.
    """
    if retornos.empty:
        raise ValueError("retornos está vazio.")
    if retornos.isnull().all().all():
        raise ValueError("retornos contém apenas valores nulos.")

    cov = retornos.cov()
    if anualizar:
        cov = cov * DIAS_ANO_CRIPTO
    return cov


def calcular_volatilidades(
    retornos: pd.DataFrame,
    anualizar: bool = True,
) -> pd.Series:
    """Volatilidade por ativo, opcionalmente anualizada.

    Args:
        retornos: DataFrame de log-retornos diários.
        anualizar: Se True, multiplica por sqrt(DIAS_ANO_CRIPTO).

    Returns:
        Series com volatilidade por ativo.
    """
    if retornos.empty:
        raise ValueError("retornos está vazio.")

    vol = retornos.std()
    if anualizar:
        vol = vol * np.sqrt(DIAS_ANO_CRIPTO)
    return vol


# ────────────────────────────────────────────────────────────
# Métricas de portfólio
# ────────────────────────────────────────────────────────────

def calcular_sharpe(
    retornos_portfolio: pd.Series,
    risk_free: float = 0.0,
) -> float:
    """Sharpe anualizado do portfólio.

    Args:
        retornos_portfolio: Series de retornos diários do portfólio.
        risk_free: Taxa livre de risco anualizada (default 0).

    Returns:
        Sharpe ratio anualizado.

    Raises:
        ValueError: Se volatilidade for zero.
    """
    if retornos_portfolio.empty:
        raise ValueError("retornos_portfolio está vazio.")

    retorno_anual = retornos_portfolio.mean() * DIAS_ANO_CRIPTO
    vol_anual     = retornos_portfolio.std()  * np.sqrt(DIAS_ANO_CRIPTO)

    if np.isclose(vol_anual, 0):
        raise ValueError("Volatilidade zero — Sharpe indefinido.")

    return (retorno_anual - risk_free) / vol_anual


def calcular_drawdown_maximo(retornos_portfolio: pd.Series) -> float:
    """Máximo drawdown a partir da curva de retorno acumulado.

    Args:
        retornos_portfolio: Series de retornos diários do portfólio.

    Returns:
        Máximo drawdown como número negativo (ex: -0.35 = -35%).
    """
    if retornos_portfolio.empty:
        raise ValueError("retornos_portfolio está vazio.")

    acumulado = (1 + retornos_portfolio).cumprod()
    pico      = acumulado.cummax()
    drawdown  = (acumulado - pico) / pico
    return float(drawdown.min())


def estatisticas_portfolio(retornos_portfolio: pd.Series) -> dict:
    """Resumo completo de métricas do portfólio.

    Args:
        retornos_portfolio: Series de retornos diários do portfólio.

    Returns:
        Dict com: retorno_anual_%, volatilidade_%, sharpe, max_drawdown_%,
        n_observacoes.
    """
    retorno_anual = retornos_portfolio.mean() * DIAS_ANO_CRIPTO
    vol_anual     = retornos_portfolio.std()  * np.sqrt(DIAS_ANO_CRIPTO)

    return {
        "retorno_anual_%":  round(retorno_anual * 100, 2),
        "volatilidade_%":   round(vol_anual     * 100, 2),
        "sharpe":           round(calcular_sharpe(retornos_portfolio), 4),
        "max_drawdown_%":   round(calcular_drawdown_maximo(retornos_portfolio) * 100, 2),
        "n_observacoes":    int(retornos_portfolio.count()),
    }


def aplicar_pesos(
    retornos: pd.DataFrame,
    pesos: pd.Series,
) -> pd.Series:
    """Retorno diário do portfólio dado um vetor de pesos.

    Args:
        retornos: DataFrame de log-retornos diários.
        pesos: Series com pesos por ativo (deve somar 1).

    Returns:
        Series de retornos do portfólio.

    Raises:
        ValueError: Se pesos não somarem 1, houver NaNs ou índices desalinhados.
    """
    if not np.isclose(pesos.sum(), 1.0, atol=1e-6):
        raise ValueError(
            f"Pesos devem somar 1.0 (soma atual: {pesos.sum():.6f})."
        )
    if pesos.isnull().any():
        raise ValueError("pesos contém valores nulos.")
    if retornos.isnull().any().any():
        raise ValueError("retornos contém valores nulos.")
    if not set(pesos.index).issubset(set(retornos.columns)):
        ausentes = set(pesos.index) - set(retornos.columns)
        raise ValueError(
            f"Ativos em pesos não encontrados em retornos: {ausentes}"
        )

    return (retornos[pesos.index] * pesos).sum(axis=1)
