# Black-Litterman Portfolio Optimization — Crypto

Implementação do modelo **Black-Litterman (1992)** para otimização de portfólio de criptomoedas, com backtest rolante e dashboard interativo para inserção de views em tempo real.

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
Views gestor  → Q (anualizado: r_T × 365/T), P, Ω = diag(P·τΣ·Pᵀ)
BL bayesiano  → E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹Π + P'Ω⁻¹Q]
Otimização MV → pesos ótimos (long-only, peso máx. 40%)
Backtest      → janela rolante 365d, rebalanceamento 30d
```

---

## Universo de Ativos (11 criptomoedas)

| Ticker  | Nome         | Categoria             |
|---------|--------------|-----------------------|
| BTC     | Bitcoin      | Store of value        |
| ETH     | Ethereum     | Smart contracts L1    |
| SOL     | Solana       | Smart contracts L1    |
| ADA     | Cardano      | Smart contracts L1    |
| DOT     | Polkadot     | Interop L0            |
| TRX     | Tron         | Smart contracts L1    |
| XRP     | Ripple       | Payments              |
| AAVE    | Aave         | DeFi lending          |
| PENDLE  | Pendle       | DeFi yield            |
| ONDO    | Ondo Finance | RWA                   |
| LINK    | Chainlink    | Oracle/Infrastructure |

---

## Estrutura do Projeto

```
black-litterman-portfolio/
├── src/
│   ├── __init__.py         # Exporta BlackLitterman e backtest_rolante_bl
│   ├── config.py           # Universo de ativos, parâmetros, caminhos
│   ├── fetch_prices.py     # Coleta preços (Yahoo Finance / yfinance)
│   ├── fetch_market_cap.py # Coleta market caps (CoinGecko)
│   ├── data_quality.py     # Validação e relatório de qualidade dos dados
│   ├── portfolio_utils.py  # Covariância (amostral + Ledoit-Wolf), Sharpe,
│   │                       #   drawdown, metricas_equity_curve
│   ├── black_litterman.py  # Classe BlackLitterman: Π, fórmula BL, otimização
│   ├── views.py            # Geração de views: RSI, Momentum; conversores temporais
│   ├── estrategias.py      # 7 estratégias de alocação (herdam EstrategiaBase)
│   ├── backtest.py         # WalkForwardBacktest + backtest_rolante_bl standalone
│   ├── visualizacao.py     # Equity curves, drawdowns, rolling Sharpe, heatmaps
│   └── run_backtest.py     # Pipeline end-to-end (entry point)
├── dashboard/
│   ├── app.py              # Dashboard Streamlit (views em tempo real + backtest rolante)
│   └── utils_dash.py       # Funções auxiliares: idzorek_omega, omega_padrao,
│                           #   construir_view_absoluta, mini_backtest_view, plots
├── tests/                  # Pytest (BL, portfolio_utils, backtest, testes de sanidade)
├── report/
│   └── relatorio_bl.tex    # Relatório LaTeX (compilar no Overleaf)
├── run.py                  # Coleta dados (preços + market caps + qualidade)
└── requirements.txt
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

Salva:
- `data/precos_diarios.parquet`
- `data/retornos_diarios.parquet`
- `data/market_cap.parquet`
- `data/relatorio_qualidade.txt`

### 3. Rodar o dashboard

```bash
streamlit run dashboard/app.py
```

O dashboard exibe:
- **Inserção de views em tempo real**: selecione ativo, retorno esperado, horizonte e confiança
- **Métricas ex-ante**: retorno, vol e Sharpe esperados vs benchmark
- **Comparação de pesos**: market-cap vs BL com view (gráfico de barras)
- **Mini-backtest estático**: últimos 90 dias com pesos fixos (3 estratégias)
- **Backtest histórico rolante**: BL neutro vs benchmark (equity curve + métricas)

### 4. Executar backtest completo (7 estratégias)

```bash
python -m src.run_backtest
```

Gera `data/metricas_backtest.csv` e figuras em `data/figuras/`.

### 5. Usar o backtest rolante standalone

```python
from src import backtest_rolante_bl
import pandas as pd

retornos    = pd.read_parquet("data/retornos_diarios.parquet")
market_caps = pd.read_parquet("data/market_cap.parquet").set_index("ticker")["market_cap_usd"]

resultado = backtest_rolante_bl(
    retornos, market_caps,
    janela=252, holding=30,
    tau=0.05, risk_aversion=2.5, peso_maximo=0.40,
)

print(resultado["metricas_portfolio"])
equity_bl    = resultado["equity_portfolio"]   # Series indexada por data
equity_bench = resultado["equity_benchmark"]
```

### 6. Rodar os testes

```bash
pytest tests/
```

Para rodar apenas os testes de sanidade do BL:

```bash
pytest tests/test_black_litterman.py::TestSanidadePosterior -v
```

### 7. Compilar o relatório LaTeX

Faça upload de `report/relatorio_bl.tex` no [Overleaf](https://www.overleaf.com) e compile.
As figuras referenciadas (`data/figuras/backtest_rolante.png`) precisam ser geradas antes
com `python -m src.run_backtest`.

---

## Parâmetros do Modelo

| Parâmetro        | Valor padrão | Descrição                                      |
|------------------|-------------|------------------------------------------------|
| `RISK_AVERSION`  | 2.5         | λ — aversão ao risco do investidor representativo |
| `TAU`            | 0.05        | τ — escala da incerteza no prior               |
| `PESO_MAXIMO`    | 0.40        | Limite máximo por ativo na otimização          |
| `JANELA_SIGMA`   | 365         | Janela para estimação de Σ (dias)              |
| `HOLDING_PERIOD` | 30          | Dias entre rebalanceamentos (backtest rolante) |

---

## Conversão de Escala das Views

O modelo usa Σ **anualizada** (×365, cripto opera 24/7). Para consistência,
Q deve estar em retorno anual. A conversão de uma view de horizonte $T$ dias:

```
Q_anual = r_periodo × (365 / T)      ← escalonamento linear (implementado)
```

**Por que não compounding geométrico?**
O compounding geométrico `(1+r)^(365/T) − 1` produz valores extremos para
horizontes curtos: 5% em 30 dias → ~81% a.a. (geométrico) vs 60.8% a.a. (linear).
O escalonamento linear é consistente com como Σ é anualizada (linear, não composta).

---

## Omega Simplificado

Para o backtest rolante (sem confiança do gestor), usa-se:

```
Ω = diag(P · (τΣ) · Pᵀ)
```

A incerteza de cada view escala com a variância dos ativos envolvidos.
Para o dashboard (com confiança do gestor), usa-se o método Idzorek:

```
Ω_ii = (1/c_i − 1) × τ × P_i Σ P_i'
```

Disponível em `dashboard.utils_dash.idzorek_omega` e `omega_padrao`.

---

## Testes de Sanidade do BL

Dois testes verificam a fórmula central da equação BL:

| Cenário | Ω | Resultado esperado |
|---------|---|-------------------|
| Confiança 100% | Ω → 0 | posterior do ativo ≈ Q (exato) |
| Confiança 0% | Ω → ∞ | posterior ≈ Π (equilíbrio) |

---

## Dependências

```
pandas>=2.0.0       # DataFrames e séries temporais
numpy>=1.24.0       # Álgebra linear
scikit-learn>=1.3.0 # Ledoit-Wolf
scipy>=1.11.0       # SLSQP (otimização)
yfinance>=0.2.0     # Preços (Yahoo Finance)
requests>=2.31.0    # Market caps (CoinGecko)
pyarrow>=14.0.0     # Parquet I/O
python-dotenv>=1.0.0
plotly>=5.0.0       # Gráficos interativos (dashboard)
streamlit>=1.30.0   # Dashboard web
matplotlib>=3.7.0   # Gráficos estáticos (backtest)
seaborn>=0.12.0
pytest>=7.0.0
tabulate>=0.9.0
```

---

## Referências

- Black, F. & Litterman, R. (1992). *Global Portfolio Optimization.* Financial Analysts Journal.
- Ledoit, O. & Wolf, M. (2004). *A well-conditioned estimator for large-dimensional covariance matrices.* Journal of Multivariate Analysis.
- Idzorek, T. M. (2004). *A Step-By-Step Guide to the Black-Litterman Model.*

---

*Projeto desenvolvido pelo IQF para a Vault Capital.*
