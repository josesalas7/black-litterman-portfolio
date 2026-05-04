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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def coletar_todos_pares() -> pd.DataFrame:
    tickers_str = " ".join(TICKERS_YAHOO.values())

    log.info(f"Baixando {len(TICKERS_YAHOO)} ativos do Yahoo Finance...")
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
        log.warning(f"⚠️  Sem dados no Yahoo Finance para: {ausentes}")

    close = close.sort_index()
    log.info(f"\n✅ Coleta finalizada: {close.shape[0]} dias × {close.shape[1]} ativos")
    return close


def calcular_retornos(df_precos: pd.DataFrame) -> pd.DataFrame:
    retornos = np.log(df_precos / df_precos.shift(1))
    return retornos.dropna(how="all")


def main():
    log.info("=" * 60)
    log.info("INÍCIO DA COLETA DE PREÇOS (Yahoo Finance)")
    log.info(f"Período: {DATA_INICIO.date()} → {DATA_FIM.date()}")
    log.info(f"Ativos: {len(TICKERS_YAHOO)}")
    log.info("=" * 60)

    df_precos = coletar_todos_pares()

    if df_precos.empty:
        raise RuntimeError("Nenhum dado coletado — arquivos existentes não foram sobrescritos.")

    df_retornos = calcular_retornos(df_precos)

    df_precos.to_parquet(ARQUIVO_PRECOS)
    df_retornos.to_parquet(ARQUIVO_RETORNOS)

    log.info(f"💾 Preços salvos em: {ARQUIVO_PRECOS}")
    log.info(f"💾 Retornos salvos em: {ARQUIVO_RETORNOS}")

    log.info("\n📊 Preview dos preços (últimos 5 dias):")
    print(df_precos.tail())


if __name__ == "__main__":
    main()
