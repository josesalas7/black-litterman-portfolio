"""
fetch_market_cap.py — Coleta market cap atual via CoinGecko.

Os pesos da carteira de mercado para o Black-Litterman são proporcionais
ao market cap de cada ativo:

    w_mercado_BTC = market_cap_BTC / soma(market_caps)

API: CoinGecko (free tier, sem API key, rate limit ~30 req/min)
"""
import logging
import time
import requests
import pandas as pd

from src.config import IDS_COINGECKO, ARQUIVO_MARKET_CAP

log = logging.getLogger(__name__)

_URL_BASE = "https://api.coingecko.com/api/v3"


def coletar_market_cap_atual() -> pd.DataFrame:
    """Coleta market cap atual de todos os ativos do universo via CoinGecko.

    Returns:
        DataFrame com colunas: id_coingecko, ticker, nome, preco_usd,
        market_cap_usd, volume_24h_usd, supply_circulante, rank_market_cap,
        atualizado_em, peso_mercado.
    """
    ids_str = ",".join(IDS_COINGECKO)
    params = {
        "vs_currency": "usd",
        "ids": ids_str,
        "order": "market_cap_desc",
        "per_page": len(IDS_COINGECKO),
        "page": 1,
        "sparkline": False,
    }

    log.info("Consultando CoinGecko para %d ativos...", len(IDS_COINGECKO))

    try:
        response = requests.get(f"{_URL_BASE}/coins/markets", params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        log.error("Erro na requisicao CoinGecko: %s", e)
        return pd.DataFrame()

    dados = response.json()

    df = pd.DataFrame([
        {
            "id_coingecko":     item["id"],
            "ticker":           item["symbol"].upper(),
            "nome":             item["name"],
            "preco_usd":        item["current_price"],
            "market_cap_usd":   item["market_cap"],
            "volume_24h_usd":   item["total_volume"],
            "supply_circulante": item["circulating_supply"],
            "rank_market_cap":  item["market_cap_rank"],
            "atualizado_em":    item["last_updated"],
        }
        for item in dados
    ])

    market_cap_total = df["market_cap_usd"].sum()
    df["peso_mercado"] = df["market_cap_usd"] / market_cap_total
    df = df.sort_values("market_cap_usd", ascending=False).reset_index(drop=True)

    return df


def coletar_market_cap_historico(dias: int = 730) -> pd.DataFrame:
    """[Opcional] Coleta market cap histórico por ativo.

    Útil para backtest sem lookahead bias: usar o market cap da data de
    rebalanceamento, não o atual. Free tier do CoinGecko limita a 365 dias
    em algumas chamadas.

    Args:
        dias: Número de dias de histórico.

    Returns:
        DataFrame com datas no índice e tickers nas colunas.
    """
    log.info("Coletando market cap historico (%d dias)...", dias)
    historico: dict[str, pd.Series] = {}

    for cg_id in IDS_COINGECKO:
        log.info("  -> %s", cg_id)
        try:
            response = requests.get(
                f"{_URL_BASE}/coins/{cg_id}/market_chart",
                params={"vs_currency": "usd", "days": dias, "interval": "daily"},
                timeout=30,
            )
            response.raise_for_status()
            dados = response.json()
        except requests.RequestException as e:
            log.warning("Erro em %s: %s", cg_id, e)
            continue

        market_caps = dados.get("market_caps", [])
        if not market_caps:
            continue

        df_cg = pd.DataFrame(market_caps, columns=["timestamp", "market_cap"])
        df_cg["timestamp"] = pd.to_datetime(df_cg["timestamp"], unit="ms").dt.normalize()
        historico[cg_id.upper()] = df_cg.set_index("timestamp")["market_cap"]

        time.sleep(2.5)  # respeita rate limit (~30 req/min)

    return pd.DataFrame(historico).sort_index()


def main():
    log.info("=" * 60)
    log.info("COLETA DE MARKET CAP")
    log.info("=" * 60)

    df = coletar_market_cap_atual()

    if df.empty:
        log.error("Falha na coleta — verifique sua conexao.")
        return

    df.to_parquet(ARQUIVO_MARKET_CAP)
    log.info("Market cap salvo em: %s", ARQUIVO_MARKET_CAP)

    preview = df[["ticker", "nome", "market_cap_usd", "peso_mercado"]].copy()
    preview["market_cap_usd"] = preview["market_cap_usd"].apply(lambda x: f"${x/1e9:.2f}B")
    preview["peso_mercado"]   = preview["peso_mercado"].apply(lambda x: f"{x*100:.2f}%")
    log.info("Carteira de mercado (peso por market cap):\n%s", preview.to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
