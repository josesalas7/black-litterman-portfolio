"""
config.py — Configurações centrais do projeto.

Centraliza universo de ativos, parâmetros temporais e caminhos.
Qualquer mudança no escopo do projeto começa aqui.
"""
import os
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ============================================================
# UNIVERSO DE ATIVOS
# ============================================================
# Cada ativo tem: ticker Yahoo Finance, ID CoinGecko e categoria setorial.
# UNI foi removido (375 dias faltando no Yahoo). LINK adicionado como
# representante da categoria Oracle/Infrastructure.

UNIVERSO: dict[str, dict] = {
    "BTC":    {"yahoo": "BTC-USD",    "coingecko": "bitcoin",      "categoria": "Store of value"},
    "ETH":    {"yahoo": "ETH-USD",    "coingecko": "ethereum",     "categoria": "Smart contracts L1"},
    "SOL":    {"yahoo": "SOL-USD",    "coingecko": "solana",       "categoria": "Smart contracts L1"},
    "ADA":    {"yahoo": "ADA-USD",    "coingecko": "cardano",      "categoria": "Smart contracts L1"},
    "DOT":    {"yahoo": "DOT-USD",    "coingecko": "polkadot",     "categoria": "Interop L0"},
    "TRX":    {"yahoo": "TRX-USD",    "coingecko": "tron",         "categoria": "Smart contracts L1"},
    "XRP":    {"yahoo": "XRP-USD",    "coingecko": "ripple",       "categoria": "Payments"},
    "AAVE":   {"yahoo": "AAVE-USD",   "coingecko": "aave",         "categoria": "DeFi lending"},
    "PENDLE": {"yahoo": "PENDLE-USD", "coingecko": "pendle",       "categoria": "DeFi yield"},
    "ONDO":   {"yahoo": "ONDO-USD",   "coingecko": "ondo-finance", "categoria": "RWA"},
    "LINK":   {"yahoo": "LINK-USD",   "coingecko": "chainlink",    "categoria": "Oracle/Infrastructure"},
}

# Listas derivadas — não editar diretamente, alterar UNIVERSO acima
TICKERS_YAHOO:   dict[str, str] = {t: v["yahoo"]      for t, v in UNIVERSO.items()}
IDS_COINGECKO:   list[str]      = [v["coingecko"]     for v in UNIVERSO.values()]
PARES_BINANCE:   list[str]      = [f"{t}/USDT"        for t in UNIVERSO]
CATEGORIAS:      dict[str, str] = {t: v["categoria"]  for t, v in UNIVERSO.items()}

# ============================================================
# PARÂMETROS TEMPORAIS
# ============================================================

TIMEFRAME = "1d"
ANOS_HISTORICO = 2

DATA_FIM    = datetime.now()
DATA_INICIO = DATA_FIM - timedelta(days=365 * ANOS_HISTORICO)

# ============================================================
# PARÂMETROS DA BINANCE / CCXT (mantido para compatibilidade)
# ============================================================

EXCHANGE_ID        = "binance"
BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
LIMITE_POR_REQ     = 1000
PAUSA_REQ          = 0.2

# ============================================================
# CAMINHOS DE ARQUIVOS
# ============================================================

RAIZ     = Path(__file__).resolve().parent.parent
DIR_DADOS = RAIZ / "data"
DIR_DADOS.mkdir(exist_ok=True)

ARQUIVO_PRECOS              = DIR_DADOS / "precos_diarios.parquet"
ARQUIVO_RETORNOS            = DIR_DADOS / "retornos_diarios.parquet"
ARQUIVO_MARKET_CAP          = DIR_DADOS / "market_cap.parquet"
ARQUIVO_RELATORIO_QUALIDADE = DIR_DADOS / "relatorio_qualidade.txt"
ARQUIVO_METRICAS_BACKTEST   = DIR_DADOS / "metricas_backtest.csv"

DIR_FIGURAS = DIR_DADOS / "figuras"
DIR_FIGURAS.mkdir(exist_ok=True)
