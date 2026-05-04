# Black-Litterman Crypto Portfolio Optimization

Projeto de otimização de portfólio para criptomoedas usando o modelo Black-Litterman.
Desenvolvido em parceria entre **Insper Quantitative Finance** e **Vault Capital**.

## Estrutura do projeto

```
black_litterman_crypto/
├── src/
│   ├── config.py           # Universo de ativos, parâmetros e caminhos
│   ├── fetch_prices.py     # Coleta de preços via Yahoo Finance
│   ├── fetch_market_cap.py # Coleta de market cap via CoinGecko
│   ├── data_quality.py     # Validação dos dados coletados
│   ├── portfolio_utils.py  # Métricas de risco/retorno
│   ├── views.py            # Geração de views (RSI, momentum)
│   └── black_litterman.py  # Modelo Black-Litterman completo
├── data/
│   ├── precos_diarios.parquet
│   ├── retornos_diarios.parquet
│   └── market_cap.parquet
├── notebooks/
│   └── 01_exploracao.ipynb # Análise exploratória e validação end-to-end
├── tests/
│   ├── test_portfolio_utils.py
│   └── test_black_litterman.py
├── requirements.txt
└── README.md
```

## Como usar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Coletar dados

```bash
python black_litterman_crypto/run.py
```

Ou etapa por etapa:

```bash
cd black_litterman_crypto

# Preços históricos (2 anos diários via Yahoo Finance)
python -m src.fetch_prices

# Market cap atual (CoinGecko)
python -m src.fetch_market_cap

# Relatório de qualidade dos dados
python -m src.data_quality
```

### 3. Rodar o modelo Black-Litterman

```python
import pandas as pd
from src.black_litterman import BlackLitterman
from src.config import ARQUIVO_RETORNOS, ARQUIVO_MARKET_CAP

# Carregar dados
retornos = pd.read_parquet(ARQUIVO_RETORNOS)
mc_df    = pd.read_parquet(ARQUIVO_MARKET_CAP).set_index('ticker')
market_caps = mc_df['market_cap_usd']

# Inicializar modelo
bl = BlackLitterman(
    retornos=retornos,
    market_caps=market_caps,
    risk_aversion=2.5,
    tau=0.05,
)

# Sem views — retorna portfólio de equilíbrio (≈ pesos de mercado)
resultado = bl.executar()

# Com views RSI
from src.views import gerar_views_rsi
P, Q, Omega = gerar_views_rsi(
    precos=pd.read_parquet(ARQUIVO_RETORNOS),  # use precos aqui
    data_referencia=retornos.index[-1],
)
resultado_views = bl.executar(P=P, Q=Q, Omega=Omega)

print(resultado_views['pesos_otimos'])
print(resultado_views['estatisticas'])
```

### 4. Rodar o backtest walk-forward

```bash
python -m src.run_backtest
```

Gera plots em `data/figuras/` e métricas em `data/metricas_backtest.csv`.

### 5. Análise exploratória e backtest nos notebooks

```bash
jupyter notebook notebooks/01_exploracao.ipynb   # modelo estático
jupyter notebook notebooks/02_backtest.ipynb     # backtest comparativo
```

### 6. Rodar os testes

```bash
python -m pytest tests/ -v
```

## Universo de Ativos

11 criptomoedas com histórico ≥ 2 anos e alta liquidez:

| Ticker | Nome        | Categoria               |
|--------|-------------|-------------------------|
| BTC    | Bitcoin     | Store of value          |
| ETH    | Ethereum    | Smart contracts L1      |
| SOL    | Solana      | Smart contracts L1      |
| ADA    | Cardano     | Smart contracts L1      |
| TRX    | Tron        | Smart contracts L1      |
| DOT    | Polkadot    | Interop L0              |
| XRP    | Ripple      | Payments                |
| AAVE   | Aave        | DeFi lending            |
| PENDLE | Pendle      | DeFi yield              |
| ONDO   | Ondo Finance| RWA                     |
| LINK   | Chainlink   | Oracle/Infrastructure   |

> UNI foi removido por apresentar 375 dias faltando no Yahoo Finance.
> LINK adicionado para cobrir a categoria Oracle/Infrastructure.

## Backtest Walk-Forward

### Metodologia

- **Tipo:** walk-forward sem lookahead (janela de treino usa apenas dados anteriores a t)
- **Lookback:** 365 dias de negociação
- **Holding period:** 7 dias (rebalanceamento semanal)
- **Custos de transação:** não incluídos nesta fase
- **Universo:** estático (11 ativos fixos)

### Estratégias comparadas

| Estratégia | Descrição |
|---|---|
| Equal-Weight | 1/N — benchmark trivial |
| Market-Cap | Pesos por market cap — benchmark passivo |
| Markowitz Puro | MV clássico com μ histórico — mostra instabilidade |
| BL Neutro | Black-Litterman sem views — sanity check |
| BL + RSI | BL + views absolutas via RSI(14) |
| BL + Momentum | BL + view relativa top3 vs bottom3 (momentum 30d) |
| BL + RSI + Momentum | BL + views combinadas |

### Como rodar

```bash
python -m src.run_backtest
```

Saídas em `data/figuras/` (PNG) e `data/metricas_backtest.csv`.
Análise completa em [notebooks/02_backtest.ipynb](notebooks/02_backtest.ipynb).

## Modelo Black-Litterman — Fluxo

```
Market caps → w_mkt = cap_i / Σcap
                ↓
    Π = λ · Σ · w_mkt   (retornos implícitos de equilíbrio)
                ↓
    Views (P, Q, Ω) via RSI ou momentum
                ↓
    E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ · [(τΣ)⁻¹Π + P'Ω⁻¹Q]
                ↓
    max  w'E[R] - (λ/2)·w'Σw   s.t. Σw=1, w≥0, w≤0.4
                ↓
    Pesos ótimos
```
