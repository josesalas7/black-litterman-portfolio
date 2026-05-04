"""
run.py — Ponto de entrada para coleta e validação de dados.

Pipeline:
  1. Coleta de precos historicos (Yahoo Finance)
  2. Coleta de market cap (CoinGecko)
  3. Validacao de qualidade dos dados
"""
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

from src.fetch_prices import main as fetch_prices
from src.fetch_market_cap import main as fetch_market_cap
from src.data_quality import main as data_quality


def main():
    log.info("=" * 60)
    log.info("PIPELINE BLACK-LITTERMAN CRYPTO")
    log.info("=" * 60)

    log.info("[1/3] Coletando precos historicos (Yahoo Finance)...")
    try:
        fetch_prices()
        log.info("[1/3] Precos coletados com sucesso.")
    except Exception as e:
        log.error("[1/3] Erro na coleta de precos: %s", e)

    log.info("[2/3] Coletando market cap (CoinGecko)...")
    try:
        fetch_market_cap()
        log.info("[2/3] Market cap coletado com sucesso.")
    except Exception as e:
        log.error("[2/3] Erro na coleta de market cap: %s", e)

    log.info("[3/3] Validando qualidade dos dados...")
    try:
        data_quality()
        log.info("[3/3] Validacao concluida.")
    except Exception as e:
        log.error("[3/3] Erro na validacao: %s", e)

    log.info("=" * 60)
    log.info("PIPELINE CONCLUIDA")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
