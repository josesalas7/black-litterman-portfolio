"""
fetch_market_cap.py — Coleta market cap atual via CoinGecko.

POR QUE PRECISAMOS DISSO?
A premissa central do Black-Litterman é que a "carteira de mercado"
representa o consenso de equilíbrio. Os pesos dessa carteira são
proporcionais ao market cap de cada ativo.

Ou seja:
    w_mercado_BTC = market_cap_BTC / soma(market_caps)

Sem essa informação, não há ponto de partida pro modelo.

API USADA: CoinGecko (free tier)
- Não precisa de API key pra free tier
- Rate limit: 10-30 chamadas/min (suficiente pra nosso uso)
"""
import logging
import time
import requests
import pandas as pd

from src.config import IDS_COINGECKO, ARQUIVO_MARKET_CAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Endpoint do CoinGecko
URL_BASE = "https://api.coingecko.com/api/v3"


def coletar_market_cap_atual() -> pd.DataFrame:
    """
    Coleta market cap atual de todos os ativos do universo.

    Endpoint: /coins/markets
    Retorna: market cap, preço, volume e supply circulante.
    """
    # CoinGecko aceita lista de IDs separada por vírgula
    ids_str = ",".join(IDS_COINGECKO)

    params = {
        "vs_currency": "usd",
        "ids": ids_str,
        "order": "market_cap_desc",
        "per_page": len(IDS_COINGECKO),
        "page": 1,
        "sparkline": False,
    }

    log.info(f"Consultando CoinGecko para {len(IDS_COINGECKO)} ativos...")

    try:
        response = requests.get(f"{URL_BASE}/coins/markets", params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        log.error(f"Erro na requisição: {e}")
        return pd.DataFrame()

    dados = response.json()

    # Monta DataFrame com colunas relevantes
    df = pd.DataFrame([
        {
            "id_coingecko": item["id"],
            "ticker": item["symbol"].upper(),
            "nome": item["name"],
            "preco_usd": item["current_price"],
            "market_cap_usd": item["market_cap"],
            "volume_24h_usd": item["total_volume"],
            "supply_circulante": item["circulating_supply"],
            "rank_market_cap": item["market_cap_rank"],
            "atualizado_em": item["last_updated"],
        }
        for item in dados
    ])

    # Calcula peso na carteira de mercado (essencial pro Black-Litterman!)
    market_cap_total = df["market_cap_usd"].sum()
    df["peso_mercado"] = df["market_cap_usd"] / market_cap_total

    # Ordena por market cap (maior primeiro)
    df = df.sort_values("market_cap_usd", ascending=False).reset_index(drop=True)

    return df


def coletar_market_cap_historico(dias: int = 730) -> pd.DataFrame:
    """
    [Opcional] Coleta market cap histórico — útil pra backtest correto.

    Pra um backtest robusto, você não pode usar o market cap ATUAL pra
    calcular pesos de mercado em períodos passados (lookahead bias!).
    O ideal é usar o market cap *daquela data*.

    Endpoint: /coins/{id}/market_chart?days=730
    Retorna timeseries com [timestamp, price, market_cap, volume].

    NOTA: free tier limita a 365 dias de histórico em algumas chamadas.
    Pra mais histórico você pode precisar do tier pago ou usar a CoinGecko Pro.
    """
    log.info(f"Coletando market cap histórico ({dias} dias)...")

    historico_completo = {}

    for cg_id in IDS_COINGECKO:
        log.info(f"  → {cg_id}")
        try:
            response = requests.get(
                f"{URL_BASE}/coins/{cg_id}/market_chart",
                params={"vs_currency": "usd", "days": dias, "interval": "daily"},
                timeout=30,
            )
            response.raise_for_status()
            dados = response.json()
        except requests.RequestException as e:
            log.warning(f"    Erro: {e}")
            continue

        # market_caps é lista de [timestamp_ms, valor]
        market_caps = dados.get("market_caps", [])
        if not market_caps:
            continue

        df_cg = pd.DataFrame(market_caps, columns=["timestamp", "market_cap"])
        df_cg["timestamp"] = pd.to_datetime(df_cg["timestamp"], unit="ms").dt.normalize()
        df_cg = df_cg.set_index("timestamp")["market_cap"]

        # Usa o ticker da Binance equivalente
        ticker = cg_id.upper()  # simplificação — você pode fazer mapping reverso
        historico_completo[ticker] = df_cg

        # Respeita rate limit do CoinGecko free (~30 chamadas/min)
        time.sleep(2.5)

    df_hist = pd.DataFrame(historico_completo).sort_index()
    return df_hist


def main():
    """Pipeline: coleta market cap atual e salva."""
    log.info("=" * 60)
    log.info("COLETA DE MARKET CAP")
    log.info("=" * 60)

    df = coletar_market_cap_atual()

    if df.empty:
        log.error("Falha na coleta — verifique sua conexão.")
        return

    # Salva
    df.to_parquet(ARQUIVO_MARKET_CAP)
    log.info(f"💾 Market cap salvo em: {ARQUIVO_MARKET_CAP}")

    # Preview
    log.info("\n📊 Carteira de mercado (peso por market cap):")
    preview = df[["ticker", "nome", "market_cap_usd", "peso_mercado"]].copy()
    preview["market_cap_usd"] = preview["market_cap_usd"].apply(lambda x: f"${x/1e9:.2f}B")
    preview["peso_mercado"] = preview["peso_mercado"].apply(lambda x: f"{x*100:.2f}%")
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
