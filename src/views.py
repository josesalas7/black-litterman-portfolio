"""
views.py — Geração de views quantitativas para o modelo Black-Litterman.

Views são opiniões do gestor sobre retornos esperados de ativos ou spreads
entre grupos de ativos. Aqui elas são derivadas de indicadores técnicos
(RSI e momentum), tornando o processo sistemático e reproduzível.

Estrutura de saída (tripla de Theil-Mixed):
    P: matriz k×n  — quais ativos cada view afeta e com qual peso
    Q: vetor k     — retorno esperado de cada view
    Omega: matriz k×k diagonal — incerteza de cada view (variância)

onde k = número de views ativas e n = número de ativos.

ESCALA TEMPORAL DAS VIEWS
--------------------------
O modelo usa Σ anualizada (× DIAS_ANO_CRIPTO = 365) e, portanto,
Π = λΣw também está em escala anual. Para consistência, Q deve
estar em escala anual (ex: Q = 0.05 → view de +5% ao ano).

Se você tiver views em retorno diário, converta com:
    r_anual = (1 + r_diario) ** DIAS_ANO_CRIPTO - 1

Se você tiver views em retorno anual e quiser verificar a escala diária:
    r_diario = (1 + r_anual) ** (1 / DIAS_ANO_CRIPTO) - 1

Use `view_anual_para_diaria()` e `view_diaria_para_anual()` abaixo.
"""
import logging
import numpy as np
import pandas as pd

from src.portfolio_utils import calcular_matriz_covariancia_ledoitwolf, DIAS_ANO_CRIPTO

log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Conversores de escala temporal
# ────────────────────────────────────────────────────────────

def view_anual_para_diaria(r_anual: float) -> float:
    """Converte retorno anual de uma view para escala diária (cripto: 365 dias).

    Útil para inspecionar a magnitude da view na mesma escala dos retornos brutos.
    NÃO use para passar ao modelo — o modelo espera escala anual.

    Args:
        r_anual: Retorno anual da view (ex: 0.05 = 5% a.a.).

    Returns:
        Retorno diário equivalente.
    """
    return (1 + r_anual) ** (1 / DIAS_ANO_CRIPTO) - 1


def view_diaria_para_anual(r_diario: float) -> float:
    """Converte retorno diário de uma view para escala anual (cripto: 365 dias).

    Args:
        r_diario: Retorno diário da view.

    Returns:
        Retorno anual equivalente.
    """
    return (1 + r_diario) ** DIAS_ANO_CRIPTO - 1


# ────────────────────────────────────────────────────────────
# Indicadores técnicos
# ────────────────────────────────────────────────────────────

def calcular_rsi(precos: pd.Series, periodo: int = 14) -> pd.Series:
    """RSI (Relative Strength Index) usando média exponencial (EWM).

    Args:
        precos: Series de preços de fechamento diários.
        periodo: Janela do RSI (padrão 14 dias).

    Returns:
        Series com valores RSI entre 0 e 100.

    Notes:
        RSI > 70 → sobrecomprado (sinal de venda / view negativa)
        RSI < 30 → sobrevendido (sinal de compra / view positiva)
    """
    if precos.empty:
        raise ValueError("precos está vazio.")
    if periodo < 2:
        raise ValueError(f"periodo deve ser >= 2 (recebido: {periodo}).")

    delta  = precos.diff()
    ganhos = delta.clip(lower=0)
    perdas = (-delta).clip(lower=0)

    # EWM com com=periodo-1 equivale à média suavizada padrão do RSI
    media_ganhos = ganhos.ewm(com=periodo - 1, min_periods=periodo).mean()
    media_perdas = perdas.ewm(com=periodo - 1, min_periods=periodo).mean()

    rs = media_ganhos / media_perdas.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calcular_momentum(precos: pd.Series, periodo: int = 30) -> pd.Series:
    """Retorno acumulado nos últimos N dias (momentum simples).

    Args:
        precos: Series de preços de fechamento diários.
        periodo: Janela de lookback em dias (padrão 30).

    Returns:
        Series com retorno percentual do período.
    """
    if precos.empty:
        raise ValueError("precos está vazio.")
    if periodo < 1:
        raise ValueError(f"periodo deve ser >= 1 (recebido: {periodo}).")

    return precos.pct_change(periodo)


# ────────────────────────────────────────────────────────────
# Geração de views
# ────────────────────────────────────────────────────────────

def gerar_views_rsi(
    precos: pd.DataFrame,
    data_referencia: pd.Timestamp,
    threshold_compra: float = 30.0,
    threshold_venda: float = 70.0,
    retorno_esperado_view: float = 0.05,
    tau: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gera views ABSOLUTAS baseadas em RSI na data de referência.

    Lógica:
        RSI < threshold_compra → view positiva (+retorno_esperado_view anual)
        RSI > threshold_venda  → view negativa (-retorno_esperado_view anual)
        Caso contrário         → ativo não recebe view

    Args:
        precos: DataFrame de preços diários (colunas = tickers).
        data_referencia: Data para calcular o RSI.
        threshold_compra: RSI abaixo deste valor gera view positiva.
        threshold_venda: RSI acima deste valor gera view negativa.
        retorno_esperado_view: Magnitude da view em retorno ANUAL (ex: 0.05 = 5% a.a.).
            Deve estar na mesma escala que Π (anualizado). Veja nota de escala
            temporal no cabeçalho do módulo.
        tau: Parâmetro de escala (usado na heurística de Omega).

    Returns:
        Tupla (P, Q, Omega):
            P:     np.ndarray (k, n) — matriz de mapeamento
            Q:     np.ndarray (k,)   — retornos esperados das views
            Omega: np.ndarray (k, k) — matriz de incerteza diagonal

    Raises:
        ValueError: Se não houver nenhuma view ativa na data de referência.
    """
    ativos = list(precos.columns)
    n = len(ativos)

    dados_ate_ref = precos.loc[:data_referencia]

    P_linhas, Q_vals = [], []

    for i, ativo in enumerate(ativos):
        rsi_serie = calcular_rsi(dados_ate_ref[ativo].dropna())
        if rsi_serie.empty or rsi_serie.isna().all():
            continue

        rsi_atual = rsi_serie.iloc[-1]
        if np.isnan(rsi_atual):
            continue

        if rsi_atual < threshold_compra:
            linha = np.zeros(n)
            linha[i] = 1.0
            P_linhas.append(linha)
            Q_vals.append(retorno_esperado_view)
            log.info(f"View RSI positiva: {ativo} (RSI={rsi_atual:.1f})")

        elif rsi_atual > threshold_venda:
            linha = np.zeros(n)
            linha[i] = 1.0
            P_linhas.append(linha)
            Q_vals.append(-retorno_esperado_view)
            log.info(f"View RSI negativa: {ativo} (RSI={rsi_atual:.1f})")

    if not P_linhas:
        raise ValueError(
            f"Nenhuma view RSI ativa em {data_referencia.date()} "
            f"(thresholds: compra<{threshold_compra}, venda>{threshold_venda})."
        )

    P = np.array(P_linhas)
    Q = np.array(Q_vals)

    retornos_temp = np.log(precos / precos.shift(1)).dropna()
    cov_anual = calcular_matriz_covariancia_ledoitwolf(retornos_temp).values
    Omega = _calcular_omega_idzorek(P, cov_anual, tau)

    log.info(f"{len(Q_vals)} view(s) RSI gerada(s) para {data_referencia.date()}")
    return P, Q, Omega


def gerar_views_momentum(
    precos: pd.DataFrame,
    data_referencia: pd.Timestamp,
    top_n: int = 3,
    bottom_n: int = 3,
    spread_esperado: float = 0.10,
    periodo_momentum: int = 30,
    tau: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gera views RELATIVAS baseadas em momentum (long top / short bottom).

    View: "os top_n ativos por momentum vão render spread_esperado a mais
    que os bottom_n ativos por momentum no próximo período."

    Args:
        precos: DataFrame de preços diários.
        data_referencia: Data de referência para calcular momentum.
        top_n: Número de ativos com maior momentum (lado long da view).
        bottom_n: Número de ativos com menor momentum (lado short da view).
        spread_esperado: Excesso de retorno ANUAL esperado do spread (ex: 0.10 = 10% a.a.).
            Deve estar na mesma escala que Π (anualizado). O momentum de N dias é
            usado apenas para ranking, não como magnitude da view.
        periodo_momentum: Janela de momentum em dias (só para ranking).
        tau: Parâmetro de escala para Omega.

    Returns:
        Tupla (P, Q, Omega) com uma única view relativa.

    Raises:
        ValueError: Se top_n + bottom_n > número de ativos.
    """
    ativos = list(precos.columns)
    n = len(ativos)

    if top_n + bottom_n > n:
        raise ValueError(
            f"top_n ({top_n}) + bottom_n ({bottom_n}) > n_ativos ({n})."
        )

    dados_ate_ref = precos.loc[:data_referencia]
    mom = calcular_momentum(dados_ate_ref, periodo=periodo_momentum).iloc[-1]
    mom = mom.dropna().sort_values(ascending=False)

    top_ativos    = list(mom.index[:top_n])
    bottom_ativos = list(mom.index[-bottom_n:])

    log.info(f"Momentum top: {top_ativos} | bottom: {bottom_ativos}")

    # View: long top com peso 1/top_n, short bottom com peso -1/bottom_n
    linha = np.zeros(n)
    for ativo in top_ativos:
        linha[ativos.index(ativo)] = 1.0 / top_n
    for ativo in bottom_ativos:
        linha[ativos.index(ativo)] = -1.0 / bottom_n

    P = linha.reshape(1, n)
    Q = np.array([spread_esperado])

    retornos_temp = np.log(precos / precos.shift(1)).dropna()
    cov_anual = calcular_matriz_covariancia_ledoitwolf(retornos_temp).values
    Omega = _calcular_omega_idzorek(P, cov_anual, tau)

    return P, Q, Omega


# ────────────────────────────────────────────────────────────
# Heurística de Omega (Idzorek)
# ────────────────────────────────────────────────────────────

def _calcular_omega_idzorek(
    P: np.ndarray,
    cov: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Calcula Omega via heurística de Idzorek.

    Omega_ii = tau * (P_i @ Sigma @ P_i')

    Incerteza de cada view é proporcional à variância implícita pelo
    portfólio que a view representa. Views sobre ativos voláteis têm
    automaticamente maior incerteza.

    Args:
        P:   Matriz k×n de views.
        cov: Matriz n×n de covariância anualizada.
        tau: Parâmetro de escala.

    Returns:
        Matriz diagonal k×k.
    """
    k = P.shape[0]
    omega_diag = np.array([
        tau * float(P[i] @ cov @ P[i].T)
        for i in range(k)
    ])
    return np.diag(omega_diag)
