"""
utils_dash.py — Funções auxiliares puras para o dashboard Black-Litterman.

Sem chamadas a `streamlit` — todas as funções são testáveis de forma isolada.
Responsabilidades:
  - Conversão de inputs do gestor (retorno periódico → anual, confiança → Ω)
  - Mini-backtest estático dos últimos N dias
  - Construção dos gráficos Plotly
  - Tabela de métricas comparativa
"""
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import DIAS_ANO_CRIPTO

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 1. Conversão de confiança → Ω  (método Idzorek, 2005)
# ─────────────────────────────────────────────────────────────

def idzorek_omega(
    P: np.ndarray,
    Sigma: np.ndarray,
    confiancas: np.ndarray,
    tau: float = 0.05,
) -> np.ndarray:
    """Converte vetor de confianças em matriz Ω diagonal via Idzorek.

    Fórmula para cada view k:
        ω_kk = (1/c_k − 1) · τ · P_k Σ P_k'

    Propriedades:
        c → 1  ⟹  ω → 0  (view virtualmente certa, domina o prior)
        c → 0  ⟹  ω → ∞  (view ignorada, prior inalterado)

    Σ deve estar na mesma escala de Q (anual): use a covariância
    anualizada retornada por `BlackLitterman.cov`.

    Args:
        P:          Matriz de views (k_views × n_ativos).
        Sigma:      Covariância dos retornos, escala anual (n × n).
        confiancas: Array de confianças em (0, 1) — ex: 0.50 para 50%.
        tau:        Parâmetro τ do BL (default 0.05).

    Returns:
        Matriz diagonal Ω (k × k).

    Raises:
        ValueError: Se shapes incompatíveis ou confianças fora de (0, 1).
    """
    if P.ndim != 2:
        raise ValueError(f"P deve ser 2-D (recebido ndim={P.ndim}).")

    k, n = P.shape

    if Sigma.ndim != 2 or Sigma.shape != (n, n):
        raise ValueError(
            f"Sigma deve ter shape ({n},{n}), recebido {Sigma.shape}."
        )
    if confiancas.shape != (k,):
        raise ValueError(
            f"confiancas deve ter shape ({k},), recebido {confiancas.shape}."
        )
    if not np.all((confiancas > 0) & (confiancas < 1)):
        raise ValueError(
            f"confiancas deve estar em (0, 1) exclusivo. "
            f"Recebido min={confiancas.min():.4f}, max={confiancas.max():.4f}."
        )

    omegas = []
    for i in range(k):
        p_i = P[i]
        variancia_view = float(p_i @ Sigma @ p_i)
        omega_ii = (1.0 / confiancas[i] - 1.0) * tau * variancia_view
        log.debug(
            "Idzorek view[%d]: c=%.2f  var_view=%.6f  omega_ii=%.6f",
            i, confiancas[i], variancia_view, omega_ii,
        )
        omegas.append(omega_ii)

    return np.diag(omegas)


# ─────────────────────────────────────────────────────────────
# 2. Construção de P e Q a partir dos inputs do gestor
# ─────────────────────────────────────────────────────────────

def construir_view_absoluta(
    ativo: str,
    retorno_pct: float,
    horizonte_dias: int,
    universo: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Constrói P e Q a partir de inputs do gestor.

    CONVERSÃO DE ESCALA (crítico):
        O gestor informa retorno periódico (ex: "+5% em 30 dias").
        O modelo BL usa Σ anualizada → Q deve estar em escala ANUAL.

        Passo 1 – retorno diário equivalente:
            r_diario = (1 + retorno_pct/100)^(1/horizonte_dias) − 1

        Passo 2 – anualizar:
            Q = (1 + r_diario)^DIAS_ANO_CRIPTO − 1

        Exemplo: +5% em 30 dias → r_diario ≈ 0.1626%/dia → Q ≈ 80.98% a.a.

    Args:
        ativo:         Nome do ativo (ex: "BTC"). Deve estar em universo.
        retorno_pct:   Variação esperada em % no horizonte (pode ser negativa).
        horizonte_dias: Período da view em dias (≥ 1).
        universo:      Lista ordenada de ativos (ordem define as colunas de P).

    Returns:
        Tupla (P, Q):
            P — shape (1, n_ativos): linha com 1 na posição do ativo.
            Q — shape (1,): retorno ANUAL equivalente.

    Raises:
        ValueError: Se ativo não estiver no universo ou horizonte_dias < 1.
    """
    if ativo not in universo:
        raise ValueError(
            f"Ativo '{ativo}' não encontrado no universo: {universo}."
        )
    if horizonte_dias < 1:
        raise ValueError(
            f"horizonte_dias deve ser >= 1 (recebido: {horizonte_dias})."
        )

    n = len(universo)
    P = np.zeros((1, n))
    P[0, universo.index(ativo)] = 1.0

    retorno_periodo = retorno_pct / 100.0
    retorno_diario  = (1.0 + retorno_periodo) ** (1.0 / horizonte_dias) - 1.0
    retorno_anual   = (1.0 + retorno_diario) ** DIAS_ANO_CRIPTO - 1.0

    Q = np.array([retorno_anual])

    log.info(
        "View absoluta: %s %.2f%% em %dd → r_diario=%.4f%% → Q_anual=%.2f%%",
        ativo, retorno_pct, horizonte_dias,
        retorno_diario * 100, retorno_anual * 100,
    )
    return P, Q


# ─────────────────────────────────────────────────────────────
# 3. Mini-backtest estático (últimos N dias)
# ─────────────────────────────────────────────────────────────

def mini_backtest_view(
    retornos: pd.DataFrame,
    w_bl: pd.Series,
    w_mkt: pd.Series,
    dias: int = 90,
) -> pd.DataFrame:
    """Equity curves dos últimos N dias para três estratégias.

    Calcula performance histórica ESTÁTICA — os pesos são fixos durante
    todo o período (sem rebalanceamento). Útil para ver como a view
    se comportaria se aplicada N dias atrás.

    Args:
        retornos: DataFrame de retornos diários (linhas=datas, colunas=ativos).
        w_bl:     Pesos BL com view (podem ter menos ativos que retornos).
        w_mkt:    Pesos market-cap benchmark.
        dias:     Janela do backtest em dias (default 90).

    Returns:
        DataFrame com colunas ['BL_com_view', 'Sem_view', 'EW'],
        index = datas, valores = equity normalizada (começa em 100).

    Raises:
        ValueError: Se não houver ativos em comum ou dias < 1.
    """
    if dias < 1:
        raise ValueError(f"dias deve ser >= 1 (recebido: {dias}).")

    ret = retornos.tail(dias)
    if ret.empty:
        raise ValueError("retornos está vazio após tail().")

    # Ativos disponíveis em todos os três conjuntos
    ativos = ret.columns.intersection(w_bl.index).intersection(w_mkt.index)
    if ativos.empty:
        raise ValueError(
            "Nenhum ativo em comum entre retornos, w_bl e w_mkt."
        )

    def _ret_portfolio(pesos: pd.Series) -> pd.Series:
        p = pesos[ativos].clip(lower=0)
        soma = p.sum()
        if soma <= 0:
            raise ValueError("Pesos somam zero ou negativo após clip.")
        p = p / soma
        return (ret[ativos] * p).sum(axis=1)

    n_ew = len(ativos)
    w_ew = pd.Series(1.0 / n_ew, index=ativos)

    ret_bl  = _ret_portfolio(w_bl)
    ret_mkt = _ret_portfolio(w_mkt)
    ret_ew  = _ret_portfolio(w_ew)

    equity = pd.DataFrame({
        "BL_com_view": (1 + ret_bl).cumprod() * 100,
        "Sem_view":    (1 + ret_mkt).cumprod() * 100,
        "EW":          (1 + ret_ew).cumprod() * 100,
    })

    log.info(
        "Mini-backtest (%d dias): BL=%.1f  Bench=%.1f  EW=%.1f",
        dias,
        equity["BL_com_view"].iloc[-1],
        equity["Sem_view"].iloc[-1],
        equity["EW"].iloc[-1],
    )
    return equity


# ─────────────────────────────────────────────────────────────
# 4. Métricas comparativas
# ─────────────────────────────────────────────────────────────

def tabela_metricas_comparativa(equity: pd.DataFrame) -> pd.DataFrame:
    """Métricas de risco/retorno para cada estratégia do mini-backtest.

    Args:
        equity: DataFrame com equity curves (começa em 100).
                Colunas = estratégias, linhas = datas.

    Returns:
        DataFrame transposto (estratégias × métricas).
    """
    retornos_diarios = equity.pct_change().dropna()

    nomes_display = {
        "BL_com_view": "BL com view",
        "Sem_view":    "Benchmark (sem view)",
        "EW":          "Equal-Weight",
    }

    rows: dict[str, dict] = {}
    for col in equity.columns:
        ret = retornos_diarios[col]
        ret_anual = ret.mean() * DIAS_ANO_CRIPTO
        vol_anual = ret.std() * np.sqrt(DIAS_ANO_CRIPTO)

        sharpe = ret_anual / vol_anual if vol_anual > 0 else np.nan

        neg = ret[ret < 0]
        downside = neg.std() * np.sqrt(DIAS_ANO_CRIPTO) if len(neg) > 1 else np.nan
        sortino = ret_anual / downside if (downside and downside > 0) else np.nan

        cum  = (1 + ret).cumprod()
        pico = cum.cummax()
        max_dd = float(((cum - pico) / pico).min())

        ret_total = float(equity[col].iloc[-1] / equity[col].iloc[0] - 1) * 100

        nome = nomes_display.get(col, col)
        rows[nome] = {
            "Retorno total (%)":  round(ret_total, 2),
            "Retorno anual. (%)": round(ret_anual * 100, 2),
            "Vol anual. (%)":     round(vol_anual * 100, 2),
            "Sharpe":             round(sharpe, 3) if not np.isnan(sharpe) else float("nan"),
            "Sortino":            round(sortino, 3) if not np.isnan(sortino) else float("nan"),
            "Max DD (%)":         round(max_dd * 100, 2),
        }

    return pd.DataFrame(rows).T


# ─────────────────────────────────────────────────────────────
# 5. Gráficos Plotly
# ─────────────────────────────────────────────────────────────

def plot_pesos_comparacao(
    w_mkt: pd.Series,
    w_bl: pd.Series,
    titulo: str = "Pesos do Portfólio",
) -> go.Figure:
    """Barras agrupadas: market-cap (cinza) vs BL com view (azul).

    Args:
        w_mkt:  Pesos market-cap (benchmark).
        w_bl:   Pesos BL com view (posterior).
        titulo: Título do gráfico.

    Returns:
        Figura Plotly com barras agrupadas.
    """
    ativos = list(w_mkt.index)
    pct_mkt = (w_mkt * 100).values
    pct_bl  = (w_bl.reindex(ativos).fillna(0) * 100).values
    tilt_pp = pct_bl - pct_mkt

    hover_bl = [
        f"{a}: {b:.2f}% (tilt: {t:+.2f}pp)"
        for a, b, t in zip(ativos, pct_bl, tilt_pp)
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Market-Cap (benchmark)",
        x=ativos,
        y=pct_mkt,
        marker_color="lightgray",
        marker_line_color="gray",
        marker_line_width=1,
        hovertemplate="%{x}: %{y:.2f}%<extra>Market-Cap</extra>",
    ))

    fig.add_trace(go.Bar(
        name="BL com view",
        x=ativos,
        y=pct_bl,
        marker_color="steelblue",
        marker_line_color="navy",
        marker_line_width=1,
        customdata=hover_bl,
        hovertemplate="%{customdata}<extra>BL</extra>",
    ))

    fig.update_layout(
        title=titulo,
        xaxis_title="Ativo",
        yaxis_title="Peso (%)",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        margin=dict(t=80),
        plot_bgcolor="white",
        yaxis=dict(gridcolor="#eee"),
    )
    return fig


def plot_mini_backtest(
    equity: pd.DataFrame,
    titulo: str = "Mini-backtest",
) -> go.Figure:
    """Equity curves das três estratégias do mini-backtest.

    Args:
        equity: DataFrame com colunas BL_com_view, Sem_view, EW (base 100).
        titulo: Título do gráfico (inclui o período por convenção).

    Returns:
        Figura Plotly com linhas.
    """
    estilos = {
        "BL_com_view": dict(color="steelblue",  dash="solid", width=2.5),
        "Sem_view":    dict(color="#888888",     dash="dash",  width=1.8),
        "EW":          dict(color="seagreen",    dash="dot",   width=1.8),
    }
    nomes = {
        "BL_com_view": "BL com view",
        "Sem_view":    "Benchmark (sem view)",
        "EW":          "Equal-Weight",
    }

    fig = go.Figure()
    for col in equity.columns:
        est  = estilos.get(col, dict(color="black", dash="solid", width=1))
        nome = nomes.get(col, col)
        fig.add_trace(go.Scatter(
            x=equity.index,
            y=equity[col].round(2),
            name=nome,
            mode="lines",
            line=est,
            hovertemplate=f"<b>{nome}</b><br>%{{x|%d/%m/%Y}}: %{{y:.1f}}<extra></extra>",
        ))

    fig.add_hline(y=100, line_dash="dot", line_color="black", line_width=0.8, opacity=0.4)

    fig.update_layout(
        title=titulo,
        xaxis_title="Data",
        yaxis_title="Equity (base 100)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        margin=dict(t=80),
        hovermode="x unified",
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#eee"),
        yaxis=dict(gridcolor="#eee"),
    )
    return fig
