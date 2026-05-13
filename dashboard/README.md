# Black-Litterman Views Dashboard

Dashboard interativo para inserção de views qualitativas/quantitativas no modelo
Black-Litterman e visualização imediata do impacto nos pesos do portfólio.

---

## Instalação

```bash
# A partir do root do projeto
pip install streamlit plotly
# ou, se existir requirements.txt:
pip install -r requirements.txt
```

Dependências mínimas além do core do projeto:

| Pacote | Versão mínima |
|--------|--------------|
| streamlit | ≥ 1.35 |
| plotly | ≥ 5.20 |

---

## Pré-requisito: dados coletados

O dashboard lê os arquivos `data/precos_diarios.parquet` e `data/market_cap.parquet`.
Se ainda não foram gerados:

```bash
python -m src.fetch_prices
python -m src.fetch_market_cap
```

---

## Como rodar (passo a passo)

> **Importante:** use `streamlit run`, nunca `python app.py`.
> Rodar com `python` diretamente gera erros `missing ScriptRunContext` — é o modo de execução errado.

**1. Abra um terminal no root do projeto**

No VS Code: `Ctrl + J`. Verifique que o caminho termina em `black-litterman-portfolio`.
Se precisar navegar manualmente:
```
cd "C:\Users\luiso\OneDrive\Documentos\black-litterman-portfolio"
```

**2. Confirme que está no lugar certo**
```bash
dir dashboard\app.py   # Windows
ls dashboard/app.py    # Mac/Linux
```
Deve aparecer o arquivo. Se não aparecer, volte ao passo 1.

**3. Rode o dashboard**
```bash
streamlit run dashboard\app.py
```

**4. Aguarde a mensagem de sucesso**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
```
O browser abre automaticamente. Se não abrir, cole `http://localhost:8501` manualmente.

**5. Para parar o servidor:** `Ctrl + C` no terminal.

---

> Se aparecer erro de dados faltando (`OSError: parquet`), gere os arquivos primeiro:
> ```bash
> python -m src.fetch_prices
> python -m src.fetch_market_cap
> ```

---

## Fluxo de uso

```
1. App carrega → mostra portfólio de equilíbrio (market-cap)
                 sem nenhuma view aplicada.

2. Sidebar → preencha:
   - Ativo: qual cripto você tem opinião
   - Retorno esperado (%): variação esperada no horizonte
   - Horizonte (dias): período de validade da view
   - Confiança (%): o quanto você confia na view vs. o prior de mercado

3. Clique "Aplicar View" → o modelo BL recalcula os pesos ótimos
   incorporando sua view.

4. Visualize:
   - Bloco A: métricas ex-ante (retorno/vol/Sharpe esperados)
   - Bloco B: comparação de pesos market-cap vs BL com view
   - Bloco C: mini-backtest dos últimos 90 dias

5. "Limpar" → volta ao estado inicial sem view.
```

---

## O que é a "Confiança" (método Idzorek)?

O modelo Black-Litterman combina duas fontes de informação:
1. **Prior de mercado (Π)**: retornos implícitos derivados dos pesos de mercado.
2. **View do gestor (Q)**: sua opinião sobre o retorno esperado de um ativo.

A confiança controla *quanto peso* a view tem na média posterior:

| Confiança | Significado prático |
|-----------|-------------------|
| **99%** | View domina — pesos mudam muito em direção ao ativo da view |
| **50%** | Peso igual entre prior e view |
| **1%** | View quase ignorada — pesos ficam próximos ao equilíbrio |

**Formula usada (Idzorek, 2005):**
```
ω_kk = (1/c − 1) · τ · P_k Σ P_k'
```
onde `c` é a confiança em (0,1), `τ = 0.05`, e `Σ` é a covariância anualizada.

---

## Conversão de escala (importante para interpretar Q)

O modelo BL usa **Σ anualizada** (×365), portanto Q deve estar em escala anual.
Ao inserir "+5% em 30 dias", o app converte automaticamente:

```
Q_diário = (1 + 5%)^(1/30) − 1 ≈ 0.163%/dia
Q_anual  = (1 + 0.163%)^365 − 1 ≈ 81% a.a.
```

Isso aparece no sidebar como: *"→ Equivale a 81.0% a.a."*

Esse é o valor que entra no vetor Q do modelo — não os 5% originais.

---

## Parâmetros do modelo

| Parâmetro | Valor | Onde alterar |
|-----------|-------|-------------|
| λ (risk aversion) | 2.5 | `src/config.py` → `RISK_AVERSION` |
| τ (tau) | 0.05 | `src/config.py` → `TAU` |
| Peso máximo por ativo | 40% | `src/config.py` → `PESO_MAXIMO` |
| Janela Σ | 365 dias | `dashboard/app.py` → `JANELA_SIGMA` |
| Mini-backtest | 90 dias | `dashboard/app.py` → `DIAS_BACKTEST` |

---

## Limpeza de cache

O Streamlit cacheia os dados por 1 hora (`ttl=3600`).
Para forçar recarregamento imediato:

```python
# No Python/REPL:
import streamlit as st
st.cache_data.clear()
```

Ou simplesmente **Ctrl + R** no browser (rerun sem limpar o cache do servidor).
Para limpar o cache do servidor: reinicie o processo `streamlit`.

---

## Estrutura dos arquivos

```
dashboard/
├── app.py          # aplicação principal Streamlit
├── utils_dash.py   # funções puras (testáveis sem Streamlit)
└── README.md       # este arquivo

tests/
└── test_dashboard.py  # testes unitários para utils_dash.py
```
