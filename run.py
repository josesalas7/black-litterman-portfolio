"""
run.py — Ponto de entrada único do projeto.

Roda a pipeline completa:
  1. Coleta de preços (Binance)
  2. Coleta de market cap (CoinGecko)
  3. Validação de qualidade dos dados
"""
import sys
from pathlib import Path

# Garante que 'src' seja encontrado independente de onde o script é chamado
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────
# Imports dos módulos do projeto
# ─────────────────────────────────────────────
from src.fetch_prices import main as fetch_prices
from src.fetch_market_cap import main as fetch_market_cap
from src.data_quality import main as data_quality


SEPARATOR = "\n" + "=" * 60 + "\n"


def main():
    print(SEPARATOR)
    print("  PIPELINE BLACK-LITTERMAN CRYPTO")
    print(SEPARATOR)

    # ── Etapa 1: Preços ──────────────────────────────────────
    print("\n[1/3] COLETANDO PRECOS HISTORICOS (Binance)...\n")
    try:
        fetch_prices()
        print("\n✅  Preços coletados com sucesso.")
    except Exception as e:
        print(f"\n❌  Erro na coleta de preços: {e}")

    # ── Etapa 2: Market Cap ──────────────────────────────────
    print(SEPARATOR)
    print("[2/3] COLETANDO MARKET CAP (CoinGecko)...\n")
    try:
        fetch_market_cap()
        print("\n✅  Market cap coletado com sucesso.")
    except Exception as e:
        print(f"\n❌  Erro na coleta de market cap: {e}")

    # ── Etapa 3: Qualidade ───────────────────────────────────
    print(SEPARATOR)
    print("[3/3] VALIDANDO QUALIDADE DOS DADOS...\n")
    try:
        data_quality()
        print("\n✅  Validação concluída.")
    except Exception as e:
        print(f"\n❌  Erro na validação: {e}")

    print(SEPARATOR)
    print("  PIPELINE CONCLUÍDA")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
