"""
visualizacao.py — Funções de visualização para resultados de backtest.

Todas as funções aceitam dict[str, ResultadoBacktest] e retornam plt.Figure.
Podem salvar em PNG (300 dpi) se salvar_em for fornecido.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.backtest import ResultadoBacktest
from src.portfolio_utils import DIAS_ANO_CRIPTO

log = logging.getLogger(__name__)

# Paleta consistente para todas as funções
PALETA   = sns.color_palette("tab10")
FIGSIZE  = (12, 5)
DPI_SAVE = 300

sns.set_theme(style="whitegrid", palette="tab10")


# ────────────────────────────────────────────────────────────
# Helpers internos
# ────────────────────────────────────────────────────────────

def _salvar(fig: plt.Figure, caminho: Path | None) -> None:
    if caminho is not None:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(caminho, dpi=DPI_SAVE, bbox_inches="tight")
        log.info("Figura salva em: %s", caminho)


def _drawdown_serie(retornos: pd.Series) -> pd.Series:
    cum  = (1 + retornos).cumprod()
    pico = cum.cummax()
    return (cum - pico) / pico


# ────────────────────────────────────────────────────────────
# 1. Equity Curves
# ────────────────────────────────────────────────────────────

def plotar_equity_curves(
    resultados: dict[str, ResultadoBacktest],
    titulo: str = "Equity Curves — Comparativo de Estratégias",
    log_scale: bool = False,
    salvar_em: Path | None = None,
) -> plt.Figure:
    """Retornos acumulados de cada estratégia em base 1.0.

    Args:
        resultados: Dict {nome: ResultadoBacktest}.
        titulo: Título do gráfico.
        log_scale: Se True, usa escala logarítmica no eixo y.
        salvar_em: Caminho para salvar PNG (opcional).

    Returns:
        plt.Figure
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for i, (nome, res) in enumerate(resultados.items()):
        eq = res.retornos_acumulados
        ax.plot(eq.index, eq.values, label=nome, linewidth=1.5, color=PALETA[i % len(PALETA)])

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_title(titulo, fontsize=13)
    ax.set_ylabel("Valor da carteira (base 1.0)")
    ax.set_xlabel("Data")
    if log_scale:
        ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    plt.tight_layout()

    _salvar(fig, salvar_em)
    return fig


# ────────────────────────────────────────────────────────────
# 2. Drawdowns
# ────────────────────────────────────────────────────────────

def plotar_drawdowns(
    resultados: dict[str, ResultadoBacktest],
    titulo: str = "Drawdown — Comparativo de Estratégias",
    salvar_em: Path | None = None,
) -> plt.Figure:
    """Drawdown ao longo do tempo com área preenchida em vermelho.

    Args:
        resultados: Dict {nome: ResultadoBacktest}.
        titulo: Título do gráfico.
        salvar_em: Caminho para salvar PNG (opcional).

    Returns:
        plt.Figure
    """
    n   = len(resultados)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (nome, res) in zip(axes, resultados.items()):
        dd = _drawdown_serie(res.retornos_diarios.dropna())
        ax.fill_between(dd.index, dd.values, 0, alpha=0.6, color="crimson", label=nome)
        ax.plot(dd.index, dd.values, color="darkred", linewidth=0.8)
        ax.set_ylabel("Drawdown")
        ax.set_title(nome, fontsize=10)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_ylim(min(dd.min() * 1.1, -0.01), 0.01)

    fig.suptitle(titulo, fontsize=13, y=1.01)
    plt.tight_layout()

    _salvar(fig, salvar_em)
    return fig


# ────────────────────────────────────────────────────────────
# 3. Rolling Sharpe
# ────────────────────────────────────────────────────────────

def plotar_rolling_sharpe(
    resultados: dict[str, ResultadoBacktest],
    janela_dias: int = 90,
    titulo: str = "Sharpe Ratio (janela móvel 90 dias)",
    salvar_em: Path | None = None,
) -> plt.Figure:
    """Sharpe ratio em janela móvel.

    Args:
        resultados: Dict {nome: ResultadoBacktest}.
        janela_dias: Tamanho da janela em dias.
        titulo: Título do gráfico.
        salvar_em: Caminho para salvar PNG (opcional).

    Returns:
        plt.Figure
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for i, (nome, res) in enumerate(resultados.items()):
        ret = res.retornos_diarios.dropna()
        rolling_mean = ret.rolling(janela_dias).mean() * DIAS_ANO_CRIPTO
        rolling_std  = ret.rolling(janela_dias).std()  * np.sqrt(DIAS_ANO_CRIPTO)
        sharpe_roll  = rolling_mean / rolling_std.replace(0, np.nan)
        ax.plot(sharpe_roll.index, sharpe_roll.values, label=nome, linewidth=1.3,
                color=PALETA[i % len(PALETA)])

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(titulo, fontsize=13)
    ax.set_ylabel("Sharpe (anualizado)")
    ax.set_xlabel("Data")
    ax.legend(fontsize=9)
    plt.tight_layout()

    _salvar(fig, salvar_em)
    return fig


# ────────────────────────────────────────────────────────────
# 4. Heatmap de pesos
# ────────────────────────────────────────────────────────────

def plotar_heatmap_pesos(
    resultado: ResultadoBacktest,
    titulo: str | None = None,
    salvar_em: Path | None = None,
) -> plt.Figure:
    """Heatmap de pesos ao longo do tempo.

    Eixo x = data de rebalanceamento, eixo y = ativo.
    Cores mais escuras = maior alocação.

    Args:
        resultado: Um único ResultadoBacktest.
        titulo: Título do gráfico.
        salvar_em: Caminho para salvar PNG (opcional).

    Returns:
        plt.Figure
    """
    pesos = resultado.historico_pesos.T  # ativos nas linhas, datas nas colunas
    n_datas = pesos.shape[1]

    fig, ax = plt.subplots(figsize=(max(10, n_datas * 0.15), 5))

    sns.heatmap(
        pesos,
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        vmax=pesos.values.max(),
        cbar_kws={"label": "Peso"},
        linewidths=0.2,
        annot=False,
    )

    # Formata datas no eixo x (mostra um subconjunto para não poluir)
    step = max(1, n_datas // 12)
    ax.set_xticks(range(0, n_datas, step))
    ax.set_xticklabels(
        [str(pesos.columns[i].date()) for i in range(0, n_datas, step)],
        rotation=45, ha="right", fontsize=8,
    )

    ttl = titulo or f"Alocação ao longo do tempo — {resultado.nome_estrategia}"
    ax.set_title(ttl, fontsize=13)
    ax.set_ylabel("Ativo")
    ax.set_xlabel("Data de rebalanceamento")
    plt.tight_layout()

    _salvar(fig, salvar_em)
    return fig


# ────────────────────────────────────────────────────────────
# 5. Distribuição de retornos
# ────────────────────────────────────────────────────────────

def plotar_distribuicao_retornos(
    resultados: dict[str, ResultadoBacktest],
    titulo: str = "Distribuição dos Retornos Diários",
    salvar_em: Path | None = None,
) -> plt.Figure:
    """Histograma + KDE dos retornos diários por estratégia.

    Args:
        resultados: Dict {nome: ResultadoBacktest}.
        titulo: Título do gráfico.
        salvar_em: Caminho para salvar PNG (opcional).

    Returns:
        plt.Figure
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for i, (nome, res) in enumerate(resultados.items()):
        ret = res.retornos_diarios.dropna()
        sns.kdeplot(ret, ax=ax, label=nome, linewidth=1.5,
                    color=PALETA[i % len(PALETA)])

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(titulo, fontsize=13)
    ax.set_xlabel("Retorno diário")
    ax.set_ylabel("Densidade")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    plt.tight_layout()

    _salvar(fig, salvar_em)
    return fig


# ────────────────────────────────────────────────────────────
# 6. Tabela de métricas
# ────────────────────────────────────────────────────────────

def tabela_metricas(
    resultados: dict[str, ResultadoBacktest],
    cdi_diario: pd.Series | None = None,
) -> pd.DataFrame:
    """DataFrame comparativo com todas as métricas.

    Linhas = métricas, colunas = estratégias.
    Sharpe/Sortino usam CDI se fornecido, caso contrário rf=0.

    Args:
        resultados: Dict {nome: ResultadoBacktest}.
        cdi_diario: Taxa CDI diária (opcional) para métricas risk-adjusted.

    Returns:
        DataFrame com métricas comparativas.
    """
    dados = {nome: res.metricas(cdi_diario) for nome, res in resultados.items()}
    df = pd.DataFrame(dados)
    return df


# ────────────────────────────────────────────────────────────
# 7. Figura 3×1: equity, drawdown, vol rolante
# ────────────────────────────────────────────────────────────

def plotar_backtest_completo_3x1(
    resultado_var: ResultadoBacktest,
    resultado_benchmark: ResultadoBacktest,
    janela_vol: int = 30,
    salvar_em: Path | None = None,
) -> plt.Figure:
    """Figura 3×1 com equity curve (GROSS + NET), drawdown e vol rolante 30d.

    Painéis:
        (a) Equity curves: BL-VAR Gross, BL-VAR Net, Benchmark (BL Neutro)
        (b) Drawdown das mesmas três séries
        (c) Volatilidade rolante 30d anualizada

    Args:
        resultado_var: ResultadoBacktest da estratégia BL-VAR.
        resultado_benchmark: ResultadoBacktest do benchmark (BL Neutro).
        janela_vol: Janela para vol rolante em dias (padrão 30).
        salvar_em: Caminho para salvar PNG (opcional).

    Returns:
        plt.Figure
    """
    ret_gross = resultado_var.retornos_diarios.dropna()
    ret_net   = (resultado_var.retornos_net if resultado_var.retornos_net is not None
                 else ret_gross).dropna()
    ret_bench = resultado_benchmark.retornos_diarios.dropna()

    eq_gross = (1 + ret_gross).cumprod()
    eq_net   = (1 + ret_net).cumprod()
    eq_bench = (1 + ret_bench).reindex(eq_gross.index).cumprod()

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.patch.set_facecolor("white")

    cores = {"gross": "#1f77b4", "net": "#ff7f0e", "bench": "#2ca02c"}

    # (a) Equity curves
    ax0 = axes[0]
    ax0.plot(eq_gross.index, eq_gross.values, label="BL-VAR Gross",
             color=cores["gross"], linewidth=1.6)
    ax0.plot(eq_net.index, eq_net.values, label="BL-VAR Net",
             color=cores["net"], linewidth=1.6, linestyle="--")
    ax0.plot(eq_bench.index, eq_bench.values, label="BL Neutro (benchmark)",
             color=cores["bench"], linewidth=1.4, linestyle=":")
    ax0.axhline(1.0, color="black", linewidth=0.7, linestyle="-", alpha=0.3)
    ax0.set_ylabel("Valor da carteira (base 1,0)")
    ax0.set_title("Equity Curve — BL-VAR vs Benchmark", fontsize=12)
    ax0.legend(fontsize=9)
    ax0.set_facecolor("white")
    ax0.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))

    # (b) Drawdown
    ax1 = axes[1]
    for ret, nome, cor in [
        (ret_gross, "BL-VAR Gross", cores["gross"]),
        (ret_net,   "BL-VAR Net",   cores["net"]),
        (ret_bench, "BL Neutro",    cores["bench"]),
    ]:
        dd = _drawdown_serie(ret)
        ax1.plot(dd.index, dd.values, label=nome, color=cor, linewidth=1.3)
    ax1.fill_between(
        _drawdown_serie(ret_net).index,
        _drawdown_serie(ret_net).values,
        0, alpha=0.15, color=cores["net"],
    )
    ax1.axhline(0, color="black", linewidth=0.7, alpha=0.3)
    ax1.set_ylabel("Drawdown")
    ax1.set_title("Drawdown", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.set_facecolor("white")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    # (c) Volatilidade rolante 30d
    ax2 = axes[2]
    for ret, nome, cor in [
        (ret_gross, "BL-VAR Gross", cores["gross"]),
        (ret_net,   "BL-VAR Net",   cores["net"]),
        (ret_bench, "BL Neutro",    cores["bench"]),
    ]:
        vol_roll = ret.rolling(janela_vol).std() * np.sqrt(DIAS_ANO_CRIPTO)
        ax2.plot(vol_roll.index, vol_roll.values, label=nome, color=cor, linewidth=1.3)
    ax2.set_ylabel(f"Volatilidade rolante {janela_vol}d (anualizada)")
    ax2.set_title(f"Volatilidade Rolante {janela_vol}d", fontsize=12)
    ax2.set_xlabel("Data")
    ax2.legend(fontsize=9)
    ax2.set_facecolor("white")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    fig.suptitle(
        f"BL-VAR — Backtest Completo ({ret_gross.index[0].date()} → {ret_gross.index[-1].date()})",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    _salvar(fig, salvar_em)
    return fig


def plotar_equity_curve_individual(
    resultado_var: ResultadoBacktest,
    resultado_benchmark: ResultadoBacktest,
    salvar_em: Path | None = None,
) -> plt.Figure:
    """Equity curve individual: BL-VAR Gross, Net e Benchmark.

    Args:
        resultado_var: ResultadoBacktest da estratégia BL-VAR.
        resultado_benchmark: ResultadoBacktest do benchmark.
        salvar_em: Caminho para salvar PNG (opcional).

    Returns:
        plt.Figure
    """
    ret_gross = resultado_var.retornos_diarios.dropna()
    ret_net   = (resultado_var.retornos_net if resultado_var.retornos_net is not None
                 else ret_gross).dropna()
    ret_bench = resultado_benchmark.retornos_diarios.dropna()

    eq_gross = (1 + ret_gross).cumprod()
    eq_net   = (1 + ret_net).cumprod()
    eq_bench = (1 + ret_bench).reindex(eq_gross.index).cumprod()

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(eq_gross.index, eq_gross.values, label="BL-VAR Gross", linewidth=1.6, color="#1f77b4")
    ax.plot(eq_net.index, eq_net.values, label="BL-VAR Net", linewidth=1.6,
            linestyle="--", color="#ff7f0e")
    ax.plot(eq_bench.index, eq_bench.values, label="BL Neutro (benchmark)", linewidth=1.4,
            linestyle=":", color="#2ca02c")
    ax.axhline(1.0, color="black", linewidth=0.7, linestyle="-", alpha=0.3)
    ax.set_title("Equity Curve — BL-VAR vs Benchmark", fontsize=13)
    ax.set_ylabel("Valor da carteira (base 1,0)")
    ax.set_xlabel("Data")
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    plt.tight_layout()
    _salvar(fig, salvar_em)
    return fig


def plotar_drawdown_individual(
    resultado_var: ResultadoBacktest,
    resultado_benchmark: ResultadoBacktest,
    salvar_em: Path | None = None,
) -> plt.Figure:
    """Drawdown individual: BL-VAR Gross, Net e Benchmark.

    Args:
        resultado_var: ResultadoBacktest da estratégia BL-VAR.
        resultado_benchmark: ResultadoBacktest do benchmark.
        salvar_em: Caminho para salvar PNG (opcional).

    Returns:
        plt.Figure
    """
    ret_gross = resultado_var.retornos_diarios.dropna()
    ret_net   = (resultado_var.retornos_net if resultado_var.retornos_net is not None
                 else ret_gross).dropna()
    ret_bench = resultado_benchmark.retornos_diarios.dropna()

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for ret, nome, cor in [
        (ret_gross, "BL-VAR Gross", "#1f77b4"),
        (ret_net,   "BL-VAR Net",   "#ff7f0e"),
        (ret_bench, "BL Neutro",    "#2ca02c"),
    ]:
        dd = _drawdown_serie(ret)
        ax.plot(dd.index, dd.values, label=nome, color=cor, linewidth=1.4)
    ax.fill_between(
        _drawdown_serie(ret_net).index,
        _drawdown_serie(ret_net).values,
        0, alpha=0.15, color="#ff7f0e",
    )
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.3)
    ax.set_title("Drawdown — BL-VAR vs Benchmark", fontsize=13)
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Data")
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    plt.tight_layout()
    _salvar(fig, salvar_em)
    return fig


def plotar_vol_rolante(
    resultado_var: ResultadoBacktest,
    resultado_benchmark: ResultadoBacktest,
    janela_vol: int = 30,
    salvar_em: Path | None = None,
) -> plt.Figure:
    """Volatilidade rolante 30d anualizada: BL-VAR Gross, Net e Benchmark.

    Args:
        resultado_var: ResultadoBacktest da estratégia BL-VAR.
        resultado_benchmark: ResultadoBacktest do benchmark.
        janela_vol: Janela em dias (padrão 30).
        salvar_em: Caminho para salvar PNG (opcional).

    Returns:
        plt.Figure
    """
    ret_gross = resultado_var.retornos_diarios.dropna()
    ret_net   = (resultado_var.retornos_net if resultado_var.retornos_net is not None
                 else ret_gross).dropna()
    ret_bench = resultado_benchmark.retornos_diarios.dropna()

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for ret, nome, cor in [
        (ret_gross, "BL-VAR Gross", "#1f77b4"),
        (ret_net,   "BL-VAR Net",   "#ff7f0e"),
        (ret_bench, "BL Neutro",    "#2ca02c"),
    ]:
        vol_roll = ret.rolling(janela_vol).std() * np.sqrt(DIAS_ANO_CRIPTO)
        ax.plot(vol_roll.index, vol_roll.values, label=nome, color=cor, linewidth=1.4)

    ax.set_title(f"Volatilidade Rolante {janela_vol}d (anualizada)", fontsize=13)
    ax.set_ylabel(f"Vol {janela_vol}d anualizada")
    ax.set_xlabel("Data")
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    plt.tight_layout()
    _salvar(fig, salvar_em)
    return fig
