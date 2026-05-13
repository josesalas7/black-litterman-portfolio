# Relatório de Auditoria Técnica — Black-Litterman Crypto Portfolio
**Data:** 2026-05-12  
**Escopo:** `src/`, `tests/`, `notebooks/`  
**Resultado final:** 53 testes passando, 0 falhas

---

## Bug 1 — Inversão de matrizes sem proteção contra singularidade

**Arquivo:** `src/black_litterman.py`, função `combinar_views()` (linhas ~195–205)  
**Severidade:** Alta — pode causar `LinAlgError` silencioso ou resultados NaN em janelas de alta correlação

**Antes:**
```python
tauSigma_inv = np.linalg.inv(tauSigma)
Omega_inv    = np.linalg.inv(Omega)
M_inv        = np.linalg.inv(tauSigma_inv + P.T @ Omega_inv @ P)
```

**Depois:** Adicionada `inverter_matriz_robusta()` que loga `cond(M)` e faz fallback para `np.linalg.pinv` se `cond > 1e10`:
```python
tauSigma_inv = inverter_matriz_robusta(tauSigma, "tau*Sigma")
Omega_inv    = inverter_matriz_robusta(Omega, "Omega")
M_inv        = inverter_matriz_robusta(tauSigma_inv + P.T @ Omega_inv @ P, "M_bl")
```

**Testes adicionados:** `TestInvertirMatrizRobusta` (5 casos) em `tests/test_black_litterman.py`

---

## Bug 2 — `AttributeError` em índice não-datetime no construtor BlackLitterman

**Arquivo:** `src/black_litterman.py`, `__init__`, linhas ~108–113  
**Severidade:** Média — crashava o construtor sempre que o DataFrame de retornos não tinha DatetimeIndex (comum em testes sintéticos)

**Antes:**
```python
log.debug("Periodo dos retornos: %s -> %s",
    self.retornos.index.min().date(),  # AttributeError para índice int
    self.retornos.index.max().date(),
)
```

**Depois:** Guard com `hasattr(index, "date")`:
```python
if hasattr(self.retornos.index, "date"):
    log.debug("Periodo dos retornos: %s -> %s", ...)
else:
    log.debug("Periodo dos retornos: [%s, %s]", ...)
```

**Efeito observado:** Todos os 11 testes do `TestPesosMercado`, `TestRetornosImplicitos`, `TestCombinarViews`, `TestOtimizacao` que antes terminavam em ERROR agora passam.

---

## Bug 3 — `DIAS_ANO_CRIPTO` hardcoded em `data_quality.py` como `fator_anual = 365`

**Arquivo:** `src/data_quality.py`, função `estatisticas_basicas()`, linha ~63  
**Severidade:** Baixa — funciona, mas diverge da constante central; manutenção propensa a drift

**Antes:**
```python
fator_anual = 365
```

**Depois:**  
1. `DIAS_ANO_CRIPTO = 365` adicionado a `src/config.py` como fonte canônica.  
2. `src/portfolio_utils.py` importa de `config` (re-exporta para backward compat).  
3. `src/data_quality.py` importa de `config` — variável local removida.

---

## Melhoria 1 — Aviso de escala das views (Q)

**Arquivo:** `src/black_litterman.py`, `combinar_views()`  
**Motivo:** Prevenção de inconsistência de escala (views em retorno diário vs. anual)

Adicionado:
```python
if np.abs(Q).max() > 0.1:
    log.warning("Q contém valores > 10%% ao dia (max=%.4f). "
                "Verifique se Q está em escala diária em vez de anual ...", ...)
```

O modelo usa Σ anualizada; Q deve estar em escala **anual** para consistência com Π = λΣw. O aviso captura casos onde a escala foi inadvertidamente trocada.

---

## Melhoria 2 — Logging de ativos excluídos por histórico insuficiente

**Arquivo:** `src/backtest.py`, método `executar_estrategia()`, linha ~245  

**Antes:** Ativos com NaN na janela de treino eram silenciosamente descartados via `dropna`.

**Depois:**
```python
excluidos = sorted(set(treino.columns) - set(treino_limpo.columns))
if excluidos:
    log.info("[%s] %s — Ativos excluídos por histórico insuficiente: %s",
             estrategia.nome, data_t.date(), excluidos)
```

Garante rastreabilidade: saber quais datas PENDLE/ONDO ficaram fora do universo.

---

## Melhoria 3 — Função `relatorio_disponibilidade()` em `data_quality.py`

**Arquivo:** `src/data_quality.py`  
Adicionada função que retorna DataFrame com: `data_inicio`, `data_fim`, `n_dias`, `n_nan`, `pct_nan`, `dias_ate_hoje` por ativo, ordenado por `data_inicio`.

Emite `log.warning` para ativos com menos de 365 dias de histórico (PENDLE, ONDO).

**Testes adicionados:** `TestRelatioDisponibilidade` (7 casos) + `TestEstatisticasBasicas` (2 casos) em `tests/test_data_quality.py`.

---

## Melhoria 4 — Notebooks: robustez e reprodutibilidade

**Arquivos:** `notebooks/01_exploracao.ipynb`, `notebooks/02_backtest.ipynb`

| Item | Antes | Depois |
|---|---|---|
| ROOT detection | `Path().resolve().parent` (falha fora de notebooks/) | `Path.cwd().parent if cwd == "notebooks" else cwd` |
| Sementes | ausente | `np.random.seed(42); random.seed(42)` |
| Magic command | ausente | `%matplotlib inline` no topo |
| Erro de dados | `OSError` sem mensagem útil | `try/except` com instrução para regenerar dados |
| FIG_DIR | ausente | `FIG_DIR = ROOT / "data" / "figuras"; FIG_DIR.mkdir(exist_ok=True)` |

**Nota sobre parquets corrompidos:** Os arquivos em `data/` retornam `OSError: Repetition level histogram size mismatch` — incompatibilidade de versão pyarrow. Execute `python -m src.fetch_prices && python -m src.fetch_market_cap` para regenerar.

---

## Melhoria 5 — Otimização long-only: verificação e documentação

**Arquivo:** `src/black_litterman.py`, `otimizar()`  

O código já estava **correto**: `permitir_short=False` define `lb=0.0`, garantindo `w_i ∈ [0, peso_maximo]`. Pós-otimização, `clip(lower=0)` elimina resíduos numéricos negativos da SLSQP.

Atualizada a docstring para documentar explicitamente o default long-only e o raciocínio (custo de funding em cripto).

---

## Comportamento esperado vs. anterior

| Métrica | Antes (simulado) | Depois | Causa da diferença |
|---|---|---|---|
| `cond(tau*Sigma)` logado | não logado | logado em DEBUG | Bug 1 corrigido |
| `inverter_matriz_robusta` em singular | `LinAlgError` | retorna pinv + warning | Bug 1 corrigido |
| BL com índice inteiro | `AttributeError` (crash) | funciona normalmente | Bug 2 corrigido |
| Ativos excluídos por NaN | silencioso | logado em INFO | Melhoria 2 |

---

## Testes executados

```
pytest tests/ -v
53 passed in 6.50s
```

Arquivos de teste:
- `tests/test_backtest.py` — 14 testes (pré-existentes, todos passando)
- `tests/test_black_litterman.py` — 17 testes (12 pré-existentes + 5 novos para `inverter_matriz_robusta`)
- `tests/test_data_quality.py` — 9 testes (todos novos)
- `tests/test_portfolio_utils.py` — 13 testes (pré-existentes, todos passando)
