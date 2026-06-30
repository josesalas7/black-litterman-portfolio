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


def _baixar_ticker_individual(yahoo_ticker: str, max_tentativas: int = 3) -> pd.Series | None:
    """Baixa um ticker isolado com retry. Usado para resgatar ativos perdidos
    em batch por race condition no cache SQLite do yfinance ('database is locked').
    """
    import time
    for tentativa in range(1, max_tentativas + 1):
        try:
            df = yf.download(
                tickers=yahoo_ticker,
                start=DATA_INICIO.date(),
                end=DATA_FIM.date(),
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if df.empty:
                log.warning("[retry %d/%d] %s veio vazio.", tentativa, max_tentativas, yahoo_ticker)
            else:
                col = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
                if hasattr(col, "columns"):  # ainda DataFrame (multi-coluna)
                    col = col.iloc[:, 0]
                return col.dropna()
        except Exception as exc:
            log.warning("[retry %d/%d] %s falhou: %s", tentativa, max_tentativas, yahoo_ticker, exc)
        time.sleep(0.5 * tentativa)  # backoff linear
    return None


def coletar_todos_pares() -> pd.DataFrame:
    """Baixa preços de fechamento diários para todos os ativos do universo.

    Robustez contra race condition do cache SQLite do yfinance: se algum
    ticker faltar no batch inicial, refaz a coleta individualmente com
    retry. Se ao final algum ticker ainda estiver ausente, levanta erro
    explícito em vez de seguir silenciosamente com universo incompleto.
    """
    tickers_str = " ".join(TICKERS_YAHOO.values())

    log.info("Baixando %d ativos do Yahoo Finance (batch)...", len(TICKERS_YAHOO))
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

    # Remove colunas completamente vazias e identifica faltantes
    close = close.dropna(axis=1, how="all")
    esperados = set(TICKERS_YAHOO.keys())
    faltantes = esperados - set(close.columns)

    if faltantes:
        log.warning(
            "Batch trouxe %d/%d ativos. Refazendo individualmente: %s",
            len(close.columns), len(esperados), sorted(faltantes),
        )
        for tk_curto in sorted(faltantes):
            yahoo_t = TICKERS_YAHOO[tk_curto]
            serie = _baixar_ticker_individual(yahoo_t)
            if serie is not None and not serie.empty:
                close[tk_curto] = serie
                log.info("Resgate OK: %s (%d obs)", tk_curto, len(serie))

    # Recheca faltantes após o resgate
    faltantes_final = esperados - set(close.columns)
    if faltantes_final:
        raise RuntimeError(
            f"Falha ao coletar {len(faltantes_final)} ativo(s) após retries: "
            f"{sorted(faltantes_final)}. Verifique conexão/cache yfinance."
        )

    close = close.sort_index()
    log.info("Coleta finalizada: %d dias x %d ativos (universo completo)", close.shape[0], close.shape[1])
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
