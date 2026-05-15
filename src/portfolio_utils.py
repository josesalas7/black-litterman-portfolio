"""
portfolio_utils.py — Utilitários de análise de portfólio.

Funções puras e reutilizáveis para cálculo de métricas de risco/retorno.
Todas as funções assumem retornos log-diários como input padrão.
"""
import logging
import numpy as np
import pandas as pd

from src.config import DIAS_ANO_CRIPTO  # re-exportado para backward compat

log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Risco e retorno individuais
# ────────────────────────────────────────────────────────────

def calcular_matriz_covariancia(
    retornos: pd.DataFrame,
    anualizar: bool = True,
) -> pd.DataFrame:
    """Matriz de covariância amostral dos retornos, opcionalmente anualizada.

    Estimador clássico (não regularizado). Pode ser singular quando
    n_ativos >= n_períodos. Prefira `calcular_matriz_covariancia_ledoitwolf()`
    para janelas curtas ou universos com muitos ativos.

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

    n_obs, n_ativos = retornos.shape
    log.info("Calculando matriz de covariância amostral (%d ativos, %d obs)...", n_ativos, n_obs)

    cov = retornos.cov()
    if anualizar:
        cov = cov * DIAS_ANO_CRIPTO

    det = np.linalg.det(cov.values)
    vols = np.sqrt(np.diag(cov.values))
    log.debug("Covariância amostral: shape=%s, det=%.6e", cov.shape, det)
    log.debug("Volatilidades anualizadas: %s", dict(zip(retornos.columns, vols.round(4))))
    if det < 1e-12:
        log.warning("Matriz de covariância quasi-singular (det=%.2e). Use Ledoit-Wolf.", det)

    log.info("Matriz de covariância amostral calculada com sucesso.")
    return cov


def calcular_matriz_covariancia_ledoitwolf(
    retornos: pd.DataFrame,
    anualizar: bool = True,
) -> pd.DataFrame:
    """Estimador Ledoit-Wolf de covariância com regularização automática.

    Resolve o problema de matriz singular (det ≈ 0) que ocorre quando
    n_ativos >= n_períodos. O shrinkage é calculado analiticamente, sem
    necessidade de validação cruzada.

    Referência: Ledoit & Wolf (2004). "A well-conditioned estimator for
    large-dimensional covariance matrices."

    Args:
        retornos: DataFrame de log-retornos diários (linhas=datas, colunas=ativos).
        anualizar: Se True, multiplica por DIAS_ANO_CRIPTO.

    Returns:
        DataFrame n×n de covariâncias regularizadas (matriz positiva definida).

    Raises:
        ValueError: Se retornos estiver vazio ou contiver apenas NaNs.
    """
    from sklearn.covariance import LedoitWolf

    if retornos.empty:
        raise ValueError("retornos está vazio.")
    if retornos.isnull().all().all():
        raise ValueError("retornos contém apenas valores nulos.")

    dados = retornos.dropna()
    n_obs, n_ativos = dados.shape
    log.info(
        "Calculando matriz de covariância Ledoit-Wolf (%d ativos, %d obs)...",
        n_ativos, n_obs,
    )

    lw = LedoitWolf().fit(dados.values)
    cov = pd.DataFrame(lw.covariance_, index=retornos.columns, columns=retornos.columns)
    if anualizar:
        cov = cov * DIAS_ANO_CRIPTO

    det = np.linalg.det(cov.values)
    vols = np.sqrt(np.diag(cov.values))
    log.debug(
        "Ledoit-Wolf: shrinkage=%.4f, shape=%s, det=%.6e",
        lw.shrinkage_, cov.shape, det,
    )
    log.debug("Volatilidades anualizadas (LW): %s", dict(zip(retornos.columns, vols.round(4))))
    log.info("Ledoit-Wolf calculado com sucesso (shrinkage=%.4f).", lw.shrinkage_)
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


def metricas_equity_curve(
    equity: pd.Series,
    benchmark: pd.Series | None = None,
) -> dict:
    """Métricas de risco/retorno a partir de uma equity curve (base 100).

    Args:
        equity:    Series com equity curve indexada por data (começa em 100).
        benchmark: Series opcional com equity curve do benchmark (mesma base).

    Returns:
        Dict com retorno_total_%, retorno_anualizado_%, volatilidade_%,
        sharpe, max_drawdown_%, e — se benchmark fornecido — alpha e beta.
    """
    ret = equity.pct_change().dropna()
    n = len(ret)
    if n == 0:
        return {}

    ret_anual = ret.mean() * DIAS_ANO_CRIPTO
    vol_anual = ret.std() * np.sqrt(DIAS_ANO_CRIPTO)
    sharpe = ret_anual / vol_anual if vol_anual > 0 else np.nan

    cum   = (1 + ret).cumprod()
    pico  = cum.cummax()
    max_dd = float(((cum - pico) / pico).min())

    ret_total = float(equity.iloc[-1] / equity.iloc[0] - 1)

    resultado = {
        "retorno_total_%":      round(ret_total  * 100, 2),
        "retorno_anualizado_%": round(ret_anual  * 100, 2),
        "volatilidade_%":       round(vol_anual  * 100, 2),
        "sharpe":               round(sharpe,       4) if not np.isnan(sharpe) else float("nan"),
        "max_drawdown_%":       round(max_dd     * 100, 2),
        "n_dias":               n,
    }

    if benchmark is not None:
        ret_b = benchmark.pct_change().dropna()
        ret_b = ret_b.reindex(ret.index).dropna()
        ret_alinhado = ret.reindex(ret_b.index).dropna()
        if len(ret_b) > 1:
            cov_matrix = np.cov(ret_alinhado.values, ret_b.values)
            beta  = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else np.nan
            alpha = (ret_alinhado.mean() - beta * ret_b.mean()) * DIAS_ANO_CRIPTO
            resultado["beta"]      = round(float(beta),  4)
            resultado["alpha_%"]   = round(float(alpha) * 100, 2)

    return resultado


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
