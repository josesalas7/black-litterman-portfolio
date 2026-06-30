"""
run_backtest.py — Pipeline end-to-end do backtest walk-forward.

Lê dados, instancia as 8 estratégias (incluindo BL-VAR), executa o backtest
comparativo com custos (20 bps), busca CDI do Banco Central, gera métricas
completas (GROSS/NET, CDI-based Sharpe/Sortino, excesso vs CDI) e salva
todas as figuras em data/figuras/.

Uso:
    python -m src.run_backtest
"""
import logging
import sys
from pathlib import Path

import pandas as pd

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
    BlackLittermanVAR,
)
from src.data.cdi import fetch_cdi
from src.visualizacao import (
    plotar_equity_curves,
    plotar_drawdowns,
    plotar_rolling_sharpe,
    plotar_heatmap_pesos,
    plotar_distribuicao_retornos,
    tabela_metricas,
    plotar_backtest_completo_3x1,
    plotar_equity_curve_individual,
    plotar_drawdown_individual,
    plotar_vol_rolante,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("sklearn").setLevel(logging.WARNING)

log = logging.getLogger(__name__)

LOOKBACK_DAYS = 365
HOLDING_DAYS  = 7
RISK_AVERSION = 2.5
TAU           = 0.05
PESO_MAXIMO   = 0.40
CUSTO_BPS     = 20.0   # 20 basis points por rebalanceamento
JANELA_VAR    = 90     # janela de estimação do VAR(1)


def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Carrega retornos, preços e market caps do disco."""
    log.info("=" * 55)
    log.info("[Etapa 0] Carregando dados do disco...")
    log.info("=" * 55)

    retornos = pd.read_parquet(ARQUIVO_RETORNOS)
    precos   = pd.read_parquet(ARQUIVO_PRECOS)

    mc_df       = pd.read_parquet(ARQUIVO_MARKET_CAP)
    mc_df       = mc_df.set_index("ticker") if "ticker" in mc_df.columns else mc_df
    market_caps = mc_df["market_cap_usd"]

    ativos = retornos.columns.intersection(precos.columns).intersection(market_caps.index)
    retornos    = retornos[ativos]
    precos      = precos[ativos]
    market_caps = market_caps[ativos]

    log.info(
        "[Etapa 0] Dados carregados: %d ativos, %d observacoes.",
        len(ativos), retornos.shape[0],
    )
    log.info(
        "[Etapa 0] Periodo: %s -> %s",
        retornos.index.min().date(), retornos.index.max().date(),
    )
    log.info("[Etapa 0] Ativos: %s", list(ativos))
    return retornos, precos, market_caps


def buscar_cdi(retornos: pd.DataFrame) -> pd.Series:
    """Busca CDI diário via SGS/BCB para o período do backtest."""
    log.info("[Etapa 1] Buscando CDI do Banco Central (SGS série 12)...")
    from datetime import date
    data_inicio = retornos.index.min().date()
    data_fim    = retornos.index.max().date()
    cdi = fetch_cdi(data_inicio=data_inicio, data_fim=data_fim)
    log.info(
        "[Etapa 1] CDI carregado: %d observacoes, média=%.6f a.d. (%.2f%% a.a.)",
        len(cdi), cdi.mean(), ((1 + cdi.mean()) ** 365 - 1) * 100,
    )
    return cdi


def definir_estrategias(market_caps: pd.Series) -> dict:
    """Instancia as 8 estratégias com os parâmetros configurados."""
    kw_bl = dict(
        market_caps=market_caps,
        risk_aversion=RISK_AVERSION,
        tau=TAU,
        peso_maximo=PESO_MAXIMO,
    )
    return {
        "Equal-Weight":          EqualWeight(),
        "Market-Cap":            MarketCapWeight(market_caps),
        "Markowitz Puro":        MarkowitzPuro(RISK_AVERSION, PESO_MAXIMO),
        "BL Neutro":             BlackLittermanNeutro(**kw_bl),
        "BL + RSI":              BlackLittermanRSI(**kw_bl),
        "BL + Momentum":         BlackLittermanMomentum(**kw_bl),
        "BL + RSI + Momentum":   BlackLittermanCombinado(**kw_bl),
        "BL-VAR":                BlackLittermanVAR(
            janela_var=JANELA_VAR, **kw_bl,
        ),
    }


def verificar_benchmark_neutro(resultados: dict, market_caps: pd.Series) -> None:
    """Sanity check: BL sem views deve aproximar os pesos de mercado."""
    if "BL Neutro" not in resultados:
        log.warning("'BL Neutro' ausente nos resultados — sanity check pulado.")
        return

    pesos_neutro = resultados["BL Neutro"].historico_pesos.mean()
    ativos = pesos_neutro.index.intersection(market_caps.index)
    if ativos.empty:
        return

    pesos_mc   = market_caps[ativos] / market_caps[ativos].sum()
    pesos_neutro = pesos_neutro[ativos]
    correlacao = pesos_neutro.corr(pesos_mc)
    mae        = (pesos_neutro - pesos_mc).abs().mean()

    log.info("=" * 55)
    log.info("SANITY CHECK — BL Neutro vs Market Cap")
    log.info("  Correlacao de pesos: %.4f  (esperado > 0.80)", correlacao)
    log.info("  MAE medio:           %.4f", mae)

    if correlacao < 0.80:
        log.warning(
            "Correlacao baixa (%.4f) — possivel bug no modelo.", correlacao,
        )
    else:
        log.info("Sanity check OK: BL Neutro aproxima pesos de mercado.")
    log.info("=" * 55)


def gerar_plots(
    resultados: dict,
    estrategias_bl: list[str],
    cdi_diario: pd.Series | None,
) -> None:
    """Gera e salva todos os plots em DIR_FIGURAS."""
    import matplotlib.pyplot as plt

    log.info("Gerando visualizações...")

    # Figuras comparativas (todas as estratégias)
    plotar_equity_curves(resultados, salvar_em=DIR_FIGURAS / "equity_curves.png")
    plotar_equity_curves(
        resultados,
        titulo="Equity Curves (escala log)",
        log_scale=True,
        salvar_em=DIR_FIGURAS / "equity_curves_log.png",
    )
    plotar_drawdowns(resultados, salvar_em=DIR_FIGURAS / "drawdowns.png")
    plotar_rolling_sharpe(resultados, janela_dias=90, salvar_em=DIR_FIGURAS / "rolling_sharpe.png")
    plotar_distribuicao_retornos(resultados, salvar_em=DIR_FIGURAS / "distribuicao_retornos.png")

    # Heatmap de pesos para estratégias BL
    for nome in estrategias_bl:
        if nome in resultados:
            plotar_heatmap_pesos(
                resultados[nome],
                salvar_em=DIR_FIGURAS / f"heatmap_pesos_{nome.replace(' ', '_').replace('+', 'mais')}.png",
            )

    # Figuras BL-VAR específicas (checklist)
    if "BL-VAR" in resultados and "BL Neutro" in resultados:
        res_var   = resultados["BL-VAR"]
        res_bench = resultados["BL Neutro"]

        plotar_backtest_completo_3x1(
            res_var, res_bench,
            janela_vol=30,
            salvar_em=DIR_FIGURAS / "backtest_completo_3x1.png",
        )
        plotar_equity_curve_individual(
            res_var, res_bench,
            salvar_em=DIR_FIGURAS / "equity_curve.png",
        )
        plotar_drawdown_individual(
            res_var, res_bench,
            salvar_em=DIR_FIGURAS / "drawdown.png",
        )
        plotar_vol_rolante(
            res_var, res_bench,
            janela_vol=30,
            salvar_em=DIR_FIGURAS / "vol_rolante.png",
        )

    plt.close("all")
    log.info("Plots salvos em: %s", DIR_FIGURAS)


def main() -> None:
    log.info("=" * 60)
    log.info("BACKTEST WALK-FORWARD — BLACK-LITTERMAN CRYPTO")
    log.info("=" * 60)
    log.info(
        "Lookback: %d dias | Holding: %d dias | Custo: %.0f bps",
        LOOKBACK_DAYS, HOLDING_DAYS, CUSTO_BPS,
    )

    # 1. Carregar dados
    retornos, precos, market_caps = carregar_dados()

    # 2. Buscar CDI
    cdi_diario = buscar_cdi(retornos)

    # 3. Configurar backtest
    backtest = WalkForwardBacktest(
        retornos=retornos,
        precos=precos,
        market_caps_atuais=market_caps,
        lookback_days=LOOKBACK_DAYS,
        holding_period_days=HOLDING_DAYS,
    )
    datas = backtest.gerar_datas_rebalanceamento()
    log.info("Periodo de teste: %s -> %s", datas[0].date(), datas[-1].date())
    log.info("Total de rebalanceamentos: %d", len(datas))

    # 4. Definir estratégias
    estrategias = definir_estrategias(market_caps)

    # 5. Executar backtest comparativo (com custo_bps=20)
    log.info("Iniciando execução das %d estratégias...", len(estrategias))
    resultados = backtest.comparar_estrategias(estrategias, custo_bps=CUSTO_BPS)

    # 6. Sanity check
    verificar_benchmark_neutro(resultados, market_caps)

    # 7. Tabela de métricas completa (CDI-based Sharpe/Sortino)
    df_metricas = tabela_metricas(resultados, cdi_diario=cdi_diario)

    log.info("\n" + "=" * 60)
    log.info("METRICAS COMPARATIVAS (Sharpe/Sortino com CDI)")
    log.info("=" * 60)
    try:
        log.info("\n%s", df_metricas.to_markdown())
    except ImportError:
        log.info("\n%s", df_metricas.to_string())

    # 8. Salvar CSV
    df_metricas.to_csv(ARQUIVO_METRICAS_BACKTEST)
    log.info("Metricas salvas em: %s", ARQUIVO_METRICAS_BACKTEST)

    # 9. Gerar plots
    estrategias_bl = [
        "BL Neutro", "BL + RSI", "BL + Momentum",
        "BL + RSI + Momentum", "BL-VAR",
    ]
    gerar_plots(resultados, estrategias_bl, cdi_diario)

    log.info("=" * 60)
    log.info("BACKTEST CONCLUÍDO")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
