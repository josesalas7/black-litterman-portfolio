"""
run_backtest.py — Pipeline end-to-end do backtest walk-forward.

Lê dados, instancia as 7 estratégias, executa o backtest comparativo,
gera todos os plots em data/figuras/ e salva métricas em CSV.

Uso:
    python -m src.run_backtest
"""
import logging
import sys
from pathlib import Path

import pandas as pd

# Garante que src/ seja encontrado mesmo rodando com -m fora da pasta do projeto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ARQUIVO_RETORNOS,
    ARQUIVO_PRECOS,
    ARQUIVO_MARKET_CAP,
    ARQUIVO_METRICAS_BACKTEST,
    DIR_FIGURAS,
)
from src.backtest import WalkForwardBacktest
from src.estrategias import (
    EqualWeight,
    MarketCapWeight,
    MarkowitzPuro,
    BlackLittermanNeutro,
    BlackLittermanRSI,
    BlackLittermanMomentum,
    BlackLittermanCombinado,
)
from src.visualizacao import (
    plotar_equity_curves,
    plotar_drawdowns,
    plotar_rolling_sharpe,
    plotar_heatmap_pesos,
    plotar_distribuicao_retornos,
    tabela_metricas,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
# Parâmetros do backtest
# ────────────────────────────────────────────────────────────
LOOKBACK_DAYS     = 365
HOLDING_DAYS      = 7
RISK_AVERSION     = 2.5
TAU               = 0.05
PESO_MAXIMO       = 0.40


def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Carrega retornos, preços e market caps do disco."""
    log.info("Carregando dados...")
    retornos = pd.read_parquet(ARQUIVO_RETORNOS)
    precos   = pd.read_parquet(ARQUIVO_PRECOS)

    mc_df       = pd.read_parquet(ARQUIVO_MARKET_CAP)
    mc_df       = mc_df.set_index("ticker") if "ticker" in mc_df.columns else mc_df
    market_caps = mc_df["market_cap_usd"]

    # Alinha ativos presentes em todos os datasets
    ativos = retornos.columns.intersection(precos.columns).intersection(market_caps.index)
    retornos    = retornos[ativos]
    precos      = precos[ativos]
    market_caps = market_caps[ativos]

    log.info(f"Ativos: {list(ativos)}")
    log.info(f"Período retornos: {retornos.index.min().date()} → {retornos.index.max().date()}")
    return retornos, precos, market_caps


def definir_estrategias(market_caps: pd.Series) -> dict:
    """Instancia as 7 estratégias com os parâmetros configurados."""
    kw_bl = dict(
        market_caps=market_caps,
        risk_aversion=RISK_AVERSION,
        tau=TAU,
        peso_maximo=PESO_MAXIMO,
    )
    return {
        "Equal-Weight":       EqualWeight(),
        "Market-Cap":         MarketCapWeight(market_caps),
        "Markowitz Puro":     MarkowitzPuro(RISK_AVERSION, PESO_MAXIMO),
        "BL Neutro":          BlackLittermanNeutro(**kw_bl),
        "BL + RSI":           BlackLittermanRSI(**kw_bl),
        "BL + Momentum":      BlackLittermanMomentum(**kw_bl),
        "BL + RSI + Momentum": BlackLittermanCombinado(**kw_bl),
    }


def gerar_plots(resultados: dict, estrategias_bl: list[str]) -> None:
    """Gera e salva todos os plots em DIR_FIGURAS."""
    log.info("Gerando visualizações...")

    plotar_equity_curves(
        resultados,
        salvar_em=DIR_FIGURAS / "equity_curves.png",
    )
    plotar_equity_curves(
        resultados,
        titulo="Equity Curves (escala log)",
        log_scale=True,
        salvar_em=DIR_FIGURAS / "equity_curves_log.png",
    )
    plotar_drawdowns(
        resultados,
        salvar_em=DIR_FIGURAS / "drawdowns.png",
    )
    plotar_rolling_sharpe(
        resultados,
        janela_dias=90,
        salvar_em=DIR_FIGURAS / "rolling_sharpe.png",
    )
    plotar_distribuicao_retornos(
        resultados,
        salvar_em=DIR_FIGURAS / "distribuicao_retornos.png",
    )

    # Heatmap de pesos apenas para estratégias BL
    for nome in estrategias_bl:
        if nome in resultados:
            plotar_heatmap_pesos(
                resultados[nome],
                salvar_em=DIR_FIGURAS / f"heatmap_pesos_{nome.replace(' ', '_').replace('+', 'mais')}.png",
            )

    import matplotlib.pyplot as plt
    plt.close("all")
    log.info(f"Plots salvos em: {DIR_FIGURAS}")


def main() -> None:
    log.info("=" * 60)
    log.info("BACKTEST WALK-FORWARD — BLACK-LITTERMAN CRYPTO")
    log.info("=" * 60)
    log.info(f"Lookback: {LOOKBACK_DAYS} dias | Holding: {HOLDING_DAYS} dias")

    # 1. Carregar dados
    retornos, precos, market_caps = carregar_dados()

    # 2. Configurar backtest
    backtest = WalkForwardBacktest(
        retornos=retornos,
        precos=precos,
        market_caps_atuais=market_caps,
        lookback_days=LOOKBACK_DAYS,
        holding_period_days=HOLDING_DAYS,
    )
    datas = backtest.gerar_datas_rebalanceamento()
    log.info(f"Período de teste: {datas[0].date()} → {datas[-1].date()}")
    log.info(f"Total de rebalanceamentos: {len(datas)}")

    # 3. Definir estratégias
    estrategias = definir_estrategias(market_caps)

    # 4. Executar backtest comparativo
    log.info("Iniciando execução das estratégias...")
    resultados = backtest.comparar_estrategias(estrategias)

    # 5. Tabela de métricas
    df_metricas = tabela_metricas(resultados)

    log.info("\n" + "=" * 60)
    log.info("MÉTRICAS COMPARATIVAS")
    log.info("=" * 60)
    try:
        print(df_metricas.to_markdown())
    except ImportError:
        print(df_metricas.to_string())

    # 6. Salvar CSV
    df_metricas.to_csv(ARQUIVO_METRICAS_BACKTEST)
    log.info(f"Métricas salvas em: {ARQUIVO_METRICAS_BACKTEST}")

    # 7. Gerar plots
    estrategias_bl = ["BL Neutro", "BL + RSI", "BL + Momentum", "BL + RSI + Momentum"]
    gerar_plots(resultados, estrategias_bl)

    log.info("=" * 60)
    log.info("BACKTEST CONCLUÍDO")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
