"""
app.py — Dashboard Streamlit para inserção de views Black-Litterman.

Uso (a partir do root do projeto):
    streamlit run dashboard/app.py

Fluxo:
    1. App carrega dados (cached 1h).
    2. Sidebar: gestor preenche ativo, retorno esperado, horizonte, confiança.
    3. "Aplicar View": calcula BL posterior e exibe comparação vs benchmark.
    4. "Limpar View": volta ao estado inicial (só benchmark).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.black_litterman import BlackLitterman
from src.config import (
    ARQUIVO_MARKET_CAP,
    ARQUIVO_PRECOS,
    ARQUIVO_RETORNOS,
    DIAS_ANO_CRIPTO,
    PESO_MAXIMO,
    RISK_AVERSION,
    TAU,
    UNIVERSO,
)
from dashboard.utils_dash import (
    construir_view_absoluta,
    idzorek_omega,
    mini_backtest_view,
    plot_mini_backtest,
    plot_pesos_comparacao,
    tabela_metricas_comparativa,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

UNIVERSO_LIST: list[str] = list(UNIVERSO.keys())
JANELA_SIGMA  = 365   # dias de retornos usados para estimar Σ
DIAS_BACKTEST = 90    # janela do mini-backtest

# ─────────────────────────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BL Views Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Black-Litterman — Inserção de Views em Tempo Real")
st.caption(
    "Insira uma opinião sobre um ativo na barra lateral e veja como ela "
    "altera os pesos ótimos em relação ao equilíbrio de mercado."
)

# ─────────────────────────────────────────────────────────────
# Carregamento de dados (cacheado 1h)
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Carregando dados de mercado...")
def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, str]:
    """Carrega retornos, preços e market cap do disco.

    Returns:
        Tupla (retornos, precos, market_caps, data_market_cap).

    Raises:
        st.stop(): Se os arquivos não existirem.
    """
    try:
        retornos = pd.read_parquet(ARQUIVO_RETORNOS)
        precos   = pd.read_parquet(ARQUIVO_PRECOS)
        mc_df    = pd.read_parquet(ARQUIVO_MARKET_CAP)
    except OSError as e:
        st.error(
            f"**Erro ao carregar dados:** {e}\n\n"
            "Execute o pipeline de coleta primeiro:\n"
            "```\npython -m src.fetch_prices\npython -m src.fetch_market_cap\n```"
        )
        st.stop()

    mc_df = mc_df.set_index("ticker") if "ticker" in mc_df.columns else mc_df
    market_caps = mc_df["market_cap_usd"]

    data_mc = (
        mc_df["atualizado_em"].iloc[0][:10]
        if "atualizado_em" in mc_df.columns
        else "desconhecida"
    )

    # Ativos presentes em todos os datasets
    ativos = (
        retornos.columns
        .intersection(precos.columns)
        .intersection(market_caps.index)
    )
    retornos    = retornos[ativos]
    precos      = precos[ativos]
    market_caps = market_caps[ativos]

    log.info(
        "Dados carregados: %d ativos, retornos %s→%s, market_cap em %s",
        len(ativos),
        retornos.index.min().date(),
        retornos.index.max().date(),
        data_mc,
    )
    return retornos, precos, market_caps, data_mc


# ─────────────────────────────────────────────────────────────
# Cálculos BL
# ─────────────────────────────────────────────────────────────

def _janela_limpa(retornos: pd.DataFrame) -> pd.DataFrame:
    """Últimos JANELA_SIGMA dias sem NaN (exclui ativos com histórico curto)."""
    janela = retornos.tail(JANELA_SIGMA)
    limpa  = janela.dropna(axis=1, how="any")
    excluidos = sorted(set(janela.columns) - set(limpa.columns))
    if excluidos:
        log.info("Ativos excluídos por histórico insuficiente: %s", excluidos)
    return limpa


@st.cache_data(ttl=3600, show_spinner="Calculando equilíbrio de mercado...")
def calcular_benchmark(
    _retornos: pd.DataFrame,
    _market_caps: pd.Series,
) -> dict:
    """BL sem views (equilíbrio puro).

    O argumento começa com _ para sinalizar ao Streamlit que não deve ser
    hashado — DataFrames grandes são hasheados pelo conteúdo, não pela
    referência.

    Returns:
        Dict com pesos_mercado, pesos_otimos, retornos_combinados,
        cov (DataFrame), ativos_usados.
    """
    janela = _janela_limpa(_retornos)
    mc = _market_caps.reindex(janela.columns).dropna()
    ret = janela[mc.index]

    bl = BlackLitterman(ret, mc, risk_aversion=RISK_AVERSION, tau=TAU)
    resultado = bl.executar(peso_maximo=PESO_MAXIMO)
    resultado["cov"]         = bl.cov
    resultado["ativos_usados"] = bl.ativos
    return resultado


def calcular_com_view(
    retornos: pd.DataFrame,
    market_caps: pd.Series,
    ativo: str,
    retorno_pct: float,
    horizonte_dias: int,
    confianca_frac: float,
) -> dict:
    """BL com view absoluta do gestor.

    Args:
        retornos:       DataFrame de retornos diários.
        market_caps:    Series de market cap.
        ativo:          Ticker selecionado (ex: "BTC").
        retorno_pct:    Retorno esperado em % no horizonte.
        horizonte_dias: Período da view em dias.
        confianca_frac: Confiança em [0.01, 0.99].

    Returns:
        Dict com pesos_mercado, pesos_otimos, retornos_combinados,
        cov, ativos_usados.

    Raises:
        ValueError: Se o ativo não tiver dados suficientes na janela.
    """
    janela = _janela_limpa(retornos)
    mc     = market_caps.reindex(janela.columns).dropna()
    ret    = janela[mc.index]

    if ativo not in ret.columns:
        raise ValueError(
            f"Ativo '{ativo}' não tem dados suficientes na janela atual "
            f"({JANELA_SIGMA} dias). Tente um ativo com histórico mais longo."
        )

    bl = BlackLitterman(ret, mc, risk_aversion=RISK_AVERSION, tau=TAU)

    # P e Q na dimensão dos ativos disponíveis (pode ser menor que o universo)
    P, Q = construir_view_absoluta(ativo, retorno_pct, horizonte_dias, bl.ativos)
    Omega = idzorek_omega(P, bl.cov.values, np.array([confianca_frac]), tau=TAU)

    resultado = bl.executar(P=P, Q=Q, Omega=Omega, peso_maximo=PESO_MAXIMO)
    resultado["cov"]           = bl.cov
    resultado["ativos_usados"] = bl.ativos
    result_view = {
        "P": P, "Q": Q, "Omega": Omega,
        "confianca_frac": confianca_frac,
        "ativo": ativo,
        "retorno_pct": retorno_pct,
        "horizonte_dias": horizonte_dias,
        "Q_anual_pct": float(Q[0] * 100),
    }
    resultado["view_info"] = result_view
    return resultado


# ─────────────────────────────────────────────────────────────
# Inicialização do session_state
# ─────────────────────────────────────────────────────────────

if "view_aplicada" not in st.session_state:
    st.session_state.view_aplicada = False
if "resultado_bl" not in st.session_state:
    st.session_state.resultado_bl = None
if "erro_view" not in st.session_state:
    st.session_state.erro_view = None

# ─────────────────────────────────────────────────────────────
# Carregamento
# ─────────────────────────────────────────────────────────────

retornos, precos, market_caps, data_mc = carregar_dados()

with st.spinner("Calculando portfólio de equilíbrio..."):
    resultado_benchmark = calcular_benchmark(retornos, market_caps)

w_mkt        = resultado_benchmark["pesos_mercado"]
ativos_usados = resultado_benchmark["ativos_usados"]

# ─────────────────────────────────────────────────────────────
# Sidebar — formulário de view
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Nova View Absoluta")
    st.caption(
        "Expresse uma opinião sobre o retorno esperado de um ativo "
        "em um horizonte de tempo. O modelo ajustará os pesos ótimos "
        "de acordo com essa view e a confiança atribuída."
    )

    st.divider()

    ativo = st.selectbox(
        "Ativo",
        options=UNIVERSO_LIST,
        help="Selecione o ativo sobre o qual você tem uma opinião.",
    )

    retorno_pct = st.number_input(
        "Retorno esperado (%)",
        min_value=-99.0,
        max_value=1000.0,
        value=10.0,
        step=1.0,
        format="%.1f",
        help=(
            "Variação esperada do ativo no horizonte indicado. "
            "Positivo = alta; negativo = queda."
        ),
    )

    horizonte_dias = st.number_input(
        "Horizonte (dias)",
        min_value=1,
        max_value=365,
        value=30,
        step=1,
        help="Período de validade da view em dias de calendário.",
    )

    confianca = st.slider(
        "Confiança (%)",
        min_value=1,
        max_value=99,
        value=50,
        step=1,
        help=(
            "100% → view domina o prior (pesos mudam muito).\n"
            "1%   → view quase ignorada (pesos próximos ao equilíbrio)."
        ),
    )

    # Conversão prévia para feedback imediato
    r_periodo = retorno_pct / 100
    r_diario  = (1 + r_periodo) ** (1 / max(horizonte_dias, 1)) - 1
    r_anual   = (1 + r_diario) ** DIAS_ANO_CRIPTO - 1
    st.caption(
        f"→ Equivale a **{r_anual * 100:.1f}% a.a.** "
        f"({r_diario * 100:.3f}%/dia) — escala usada no modelo."
    )

    st.divider()

    col_aplicar, col_limpar = st.columns(2)
    with col_aplicar:
        btn_aplicar = st.button("Aplicar View", type="primary", use_container_width=True)
    with col_limpar:
        btn_limpar = st.button("Limpar", use_container_width=True)

    if btn_limpar:
        st.session_state.view_aplicada = False
        st.session_state.resultado_bl  = None
        st.session_state.erro_view     = None
        st.rerun()

    if btn_aplicar:
        confianca_frac = confianca / 100.0
        with st.spinner("Calculando..."):
            try:
                resultado_view = calcular_com_view(
                    retornos, market_caps,
                    ativo, retorno_pct, horizonte_dias, confianca_frac,
                )
                st.session_state.resultado_bl  = resultado_view
                st.session_state.view_aplicada = True
                st.session_state.erro_view     = None
            except (ValueError, RuntimeError) as exc:
                st.session_state.erro_view     = str(exc)
                st.session_state.view_aplicada = False

    if st.session_state.erro_view:
        st.error(st.session_state.erro_view)

    st.divider()
    st.caption(f"Market cap de referência: **{data_mc}**")
    st.caption(f"λ (risk aversion) = {RISK_AVERSION} | τ = {TAU}")

# ─────────────────────────────────────────────────────────────
# Área principal
# ─────────────────────────────────────────────────────────────

# Seleciona resultado a exibir
tem_view = st.session_state.view_aplicada and st.session_state.resultado_bl is not None
resultado_atual = st.session_state.resultado_bl if tem_view else resultado_benchmark

w_atual  = resultado_atual["pesos_otimos"]
mu_atual = resultado_atual["retornos_combinados"]
cov_atual = resultado_atual["cov"]

# Cabeçalho de status
if tem_view:
    info = resultado_atual["view_info"]
    st.success(
        f"**View aplicada:** {info['ativo']} {info['retorno_pct']:+.1f}% "
        f"em {info['horizonte_dias']}d → **{info['Q_anual_pct']:.1f}% a.a.** | "
        f"Confiança: **{info['confianca_frac']*100:.0f}%**"
    )
else:
    st.info("Nenhuma view aplicada — exibindo portfólio de equilíbrio (market-cap).")

# ── Bloco A: métricas ex-ante ──────────────────────────────

st.subheader("Métricas ex-ante do portfólio")

w_arr  = w_atual.values
mu_arr = mu_atual.reindex(w_atual.index).fillna(0).values
cov_arr = cov_atual.reindex(index=w_atual.index, columns=w_atual.index).values

ret_esperado = float(w_arr @ mu_arr) * 100           # anual %
vol_esperada = float(np.sqrt(w_arr @ cov_arr @ w_arr)) * 100  # anual %
sharpe_esperado = (ret_esperado / vol_esperada) if vol_esperada > 0 else 0.0
tilt_max = float((w_atual - w_mkt.reindex(w_atual.index).fillna(0)).abs().max()) * 100

# Benchmark para deltas
w_b   = resultado_benchmark["pesos_otimos"].values
mu_b  = resultado_benchmark["retornos_combinados"].reindex(resultado_benchmark["pesos_otimos"].index).fillna(0).values
cov_b = resultado_benchmark["cov"].values
ret_b  = float(w_b @ mu_b) * 100
vol_b  = float(np.sqrt(w_b @ cov_b @ w_b)) * 100
sharpe_b = (ret_b / vol_b) if vol_b > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Retorno anualizado (ex-ante)",
        f"{ret_esperado:.1f}%",
        delta=f"{ret_esperado - ret_b:+.1f}pp" if tem_view else None,
    )
with col2:
    st.metric(
        "Volatilidade anualizada",
        f"{vol_esperada:.1f}%",
        delta=f"{vol_esperada - vol_b:+.1f}pp" if tem_view else None,
        delta_color="inverse",
    )
with col3:
    st.metric(
        "Sharpe esperado",
        f"{sharpe_esperado:.3f}",
        delta=f"{sharpe_esperado - sharpe_b:+.3f}" if tem_view else None,
    )
with col4:
    st.metric(
        "Tilt máximo vs benchmark",
        f"{tilt_max:.1f}pp",
        help="Maior desvio absoluto de peso em relação ao market-cap.",
    )

st.divider()

# ── Bloco B: comparação de pesos ──────────────────────────

st.subheader("Alocação: Market-Cap vs Black-Litterman")

titulo_grafico = (
    f"Pesos — view: {info['ativo']} {info['retorno_pct']:+.1f}% em {info['horizonte_dias']}d "
    f"({info['Q_anual_pct']:.1f}% a.a.), confiança {info['confianca_frac']*100:.0f}%"
    if tem_view
    else "Pesos — Equilíbrio de Mercado (sem view)"
)
fig_pesos = plot_pesos_comparacao(w_mkt, w_atual, titulo=titulo_grafico)
st.plotly_chart(fig_pesos, use_container_width=True)

# Tabela de pesos detalhada (expansível)
with st.expander("Ver tabela de pesos completa"):
    df_pesos = pd.DataFrame({
        "Market-Cap (%)": (w_mkt * 100).round(2),
        "BL (%)":         (w_atual.reindex(w_mkt.index).fillna(0) * 100).round(2),
        "Tilt (pp)":      ((w_atual.reindex(w_mkt.index).fillna(0) - w_mkt) * 100).round(2),
    })
    st.dataframe(df_pesos.sort_values("Tilt (pp)", ascending=False), use_container_width=True)

st.divider()

# ── Bloco C: mini-backtest ─────────────────────────────────

st.subheader(f"Mini-backtest — últimos {DIAS_BACKTEST} dias")
st.caption(
    "Performance histórica estática: pesos fixos durante todo o período, "
    "sem rebalanceamento. Mostra como a view teria se comportado."
)

try:
    equity = mini_backtest_view(retornos, w_atual, w_mkt, dias=DIAS_BACKTEST)
    fig_bt = plot_mini_backtest(
        equity,
        titulo=f"Equity Curve — últimos {DIAS_BACKTEST} dias (base 100)",
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    # Tabela de métricas
    df_met = tabela_metricas_comparativa(equity)
    st.dataframe(
        df_met.style.format({
            "Retorno total (%)":  "{:.2f}%",
            "Retorno anual. (%)": "{:.2f}%",
            "Vol anual. (%)":     "{:.2f}%",
            "Sharpe":             "{:.3f}",
            "Sortino":            "{:.3f}",
            "Max DD (%)":         "{:.2f}%",
        }),
        use_container_width=True,
    )

except ValueError as exc:
    st.warning(f"Mini-backtest indisponível: {exc}")

# ── Rodapé ────────────────────────────────────────────────

st.divider()
st.caption(
    "**Aviso:** resultados ex-ante baseados em retornos históricos não garantem performance futura. "
    f"Modelo BL: λ={RISK_AVERSION}, τ={TAU}, Σ Ledoit-Wolf ({JANELA_SIGMA}d), "
    f"long-only, peso máx. {PESO_MAXIMO:.0%}. "
    f"Market cap: {data_mc}. "
    "Para limpar o cache: `st.cache_data.clear()` no Python ou Ctrl+R no browser."
)
