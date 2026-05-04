"""
fetch_prices.py — Coleta preços históricos via Yahoo Finance (yfinance).

Sem necessidade de API key, sem restrição geográfica.
Yahoo Finance retorna dados diários de fechamento (Close) para todos
os principais pares cripto no formato TICKER-USD.
"""
import logging
import numpy as np
import pandas as pd
import yfinance as yf

from src.config import (
    TICKERS_YAHOO,
    DATA_INICIO,
    DATA_FIM,
    ARQUIVO_PRECOS,
    ARQUIVO_RETORNOS,
)

log = logging.getLogger(__name__)


def coletar_todos_pares() -> pd.DataFrame:
    """Baixa preços de fechamento diários para todos os ativos do universo."""
    tickers_str = " ".join(TICKERS_YAHOO.values())

    log.info("Baixando %d ativos do Yahoo Finance...", len(TICKERS_YAHOO))
    raw = yf.download(
        tickers=tickers_str,
        start=DATA_INICIO.date(),
        end=DATA_FIM.date(),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    # yfinance retorna MultiIndex quando há múltiplos tickers
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]]

    # Renomeia colunas de TICKER-USD para BTC, ETH, etc.
    ticker_inverso = {v: k for k, v in TICKERS_YAHOO.items()}
    close = close.rename(columns=ticker_inverso)

    # Remove colunas completamente vazias (ticker não encontrado no Yahoo)
    antes = set(close.columns)
    close = close.dropna(axis=1, how="all")
    ausentes = antes - set(close.columns)
    if ausentes:
        log.warning("Sem dados no Yahoo Finance para: %s", ausentes)

    close = close.sort_index()
    log.info("Coleta finalizada: %d dias x %d ativos", close.shape[0], close.shape[1])
    return close


def calcular_retornos(df_precos: pd.DataFrame) -> pd.DataFrame:
    """Calcula log-retornos diários a partir dos preços de fechamento."""
    n_ativos = df_precos.shape[1]
    log.info("Calculando log-retornos diários (%d ativos)...", n_ativos)

    retornos = np.log(df_precos / df_precos.shift(1))
    retornos = retornos.dropna(how="all")

    n_obs  = retornos.shape[0]
    n_nans = retornos.isnull().sum().sum()
    log.debug(
        "Log-retornos: shape=(%d, %d), NaNs=%d",
        n_obs, n_ativos, n_nans,
    )
    log.debug(
        "Retorno médio diário por ativo (%%): %s",
        dict((retornos.mean() * 100).round(4)),
    )
    log.debug(
        "Volatilidade diária por ativo (%%): %s",
        dict((retornos.std() * 100).round(4)),
    )
    log.info(
        "Log-retornos calculados: %d dias x %d ativos.",
        n_obs, n_ativos,
    )
    return retornos


def main():
    log.info("=" * 60)
    log.info("INICIO DA COLETA DE PRECOS (Yahoo Finance)")
    log.info("Periodo: %s -> %s", DATA_INICIO.date(), DATA_FIM.date())
    log.info("Ativos: %d", len(TICKERS_YAHOO))
    log.info("=" * 60)

    df_precos = coletar_todos_pares()

    if df_precos.empty:
        raise RuntimeError("Nenhum dado coletado — arquivos existentes não foram sobrescritos.")

    df_retornos = calcular_retornos(df_precos)

    df_precos.to_parquet(ARQUIVO_PRECOS)
    df_retornos.to_parquet(ARQUIVO_RETORNOS)

    log.info("Precos salvos em: %s", ARQUIVO_PRECOS)
    log.info("Retornos salvos em: %s", ARQUIVO_RETORNOS)
    log.info("Preview dos precos (ultimos 5 dias):\n%s", df_precos.tail().to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
