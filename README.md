# Black-Litterman Portfolio Optimization — Crypto

Implementação do modelo **Black-Litterman (1992)** para otimização de portfólio de criptomoedas, com backtest walk-forward comparativo entre 7 estratégias de alocação.

---

## Visão Geral

O modelo Black-Litterman parte do equilíbrio de mercado (pesos proporcionais ao market cap) e combina via inferência bayesiana com views do gestor, gerando retornos esperados mais estáveis e economicamente fundamentados do que a média histórica (Markowitz puro).

**Pipeline principal:**

```
Yahoo Finance → Preços diários
CoinGecko     → Market caps
      ↓
Ledoit-Wolf   → Σ regularizada (evita singularidade)
CAPM reverso  → Π = λΣw_mkt  (retornos implícitos de equilíbrio)
Views (RSI/Momentum) → Q, P, Ω
BL bayesiano  → E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹Π + P'Ω⁻¹Q]
Otimização MV → pesos ótimos
Walk-forward  → backtest sem lookahead
```

---

## Universo de Ativos (11 criptomoedas)

| Ticker  | Nome        | Categoria            |
|---------|-------------|----------------------|
| BTC     | Bitcoin     | Store of value       |
| ETH     | Ethereum    | Smart contracts L1   |
| SOL     | Solana      | Smart contracts L1   |
| ADA     | Cardano     | Smart contracts L1   |
| DOT     | Polkadot    | Interop L0           |
| TRX     | Tron        | Smart contracts L1   |
| XRP     | Ripple      | Payments             |
| AAVE    | Aave        | DeFi lending         |
| PENDLE  | Pendle      | DeFi yield           |
| ONDO    | Ondo Finance| RWA                  |
| LINK    | Chainlink   | Oracle/Infrastructure|

---

## Estratégias Comparadas

| Estratégia              | Descrição                                           |
|-------------------------|-----------------------------------------------------|
| Equal-Weight            | 1/N — benchmark trivial                             |
| Market-Cap              | Pesos proporcionais ao market cap — índice passivo  |
| Markowitz Puro          | Média-variância clássico (instável — mostra o problema) |
| BL Neutro               | BL sem views — deve recuperar pesos de mercado (sanity check) |
| BL + RSI                | BL com views absolutas baseadas em RSI(14)          |
| BL + Momentum           | BL com view relativa baseada em momentum(30 dias)   |
| BL + RSI + Momentum     | BL com views combinadas                             |

---

## Estrutura do Projeto

```
black-litterman-portfolio/
├── src/
│   ├── config.py           # Universo de ativos, parâmetros, caminhos
│   ├── fetch_prices.py     # Coleta preços (Yahoo Finance / yfinance)
│   ├── fetch_market_cap.py # Coleta market caps (CoinGecko)
│   ├── data_quality.py     # Validação e relatório de qualidade dos dados
│   ├── portfolio_utils.py  # Covariância (amostral + Ledoit-Wolf), Sharpe, drawdown
│   ├── black_litterman.py  # Modelo BL: pesos de mercado, Π, fórmula bayesiana, otimização
│   ├── views.py            # Geração de views: RSI, Momentum; conversores de escala temporal
│   ├── estrategias.py      # 7 estratégias de alocação (herdam EstrategiaBase)
│   ├── backtest.py         # Walk-forward backtest, ResultadoBacktest, métricas
│   ├── visualizacao.py     # Equity curves, drawdowns, rolling Sharpe, heatmaps
│   └── run_backtest.py     # Pipeline end-to-end (entry point)
├── tests/                  # Pytest: 70+ casos (BL, portfolio_utils, backtest)
├── notebooks/
│   ├── 01_exploracao.ipynb # EDA dos dados coletados
│   └── 02_backtest.ipynb   # Pipeline interativo do backtest
├── run.py                  # Coleta dados (preços + market caps + qualidade)
├── requirements.txt
└── .gitignore
```

---

## Como Rodar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Coletar dados

```bash
python run.py
```

Salva `data/precos_diarios.parquet`, `data/retornos_diarios.parquet`, `data/market_cap.parquet` e `data/relatorio_qualidade.txt`.

### 3. Executar backtest

```bash
python -m src.run_backtest
```

Gera `data/metricas_backtest.csv` e figuras em `data/figuras/`.

### 4. Rodar testes

```bash
pytest tests/
```

---

## Parâmetros do Backtest

| Parâmetro        | Valor padrão | Descrição                            |
|------------------|-------------|--------------------------------------|
| `LOOKBACK_DAYS`  | 365         | Janela de treino (dias)              |
| `HOLDING_DAYS`   | 7           | Período de holding entre rebalanceamentos |
| `RISK_AVERSION`  | 2.5         | λ — aversão ao risco                 |
| `TAU`            | 0.05        | Escala da incerteza no prior         |
| `PESO_MAXIMO`    | 0.40        | Limite máximo por ativo              |

---

## Decisões de Design

- **Ledoit-Wolf por padrão:** evita matrizes de covariância singulares quando o número de ativos se aproxima do número de observações na janela de treino.
- **Escala temporal das views:** o modelo usa Σ anualizada (×365, cripto opera 24/7) e Π também em escala anual. As views Q devem estar em retorno anual (ex: `0.05` = 5% a.a.). Utilitários de conversão em `src/views.py`.
- **Market cap estático:** proxy do equilíbrio atual. Limitação conhecida — o ideal seria usar market cap histórico por data de rebalanceamento para eliminar lookahead bias completo.
- **Sanity check BL Neutro:** ao rodar sem views, os pesos otimizados devem ter correlação > 0.80 com os pesos de market cap. O pipeline loga este diagnóstico automaticamente.
- **Fallback:** se a estratégia falhar em um período, o backtest usa equal-weight como fallback e loga o aviso.

---

## Referências

- Black, F. & Litterman, R. (1992). *Global Portfolio Optimization.* Financial Analysts Journal.
- Ledoit, O. & Wolf, M. (2004). *A well-conditioned estimator for large-dimensional covariance matrices.* Journal of Multivariate Analysis.
- Idzorek, T. M. (2004). *A Step-By-Step Guide to the Black-Litterman Model.*

---

*Projeto desenvolvido pelo IQF para a Vault.*
