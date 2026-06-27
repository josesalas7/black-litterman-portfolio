"""
backtest.py — Infraestrutura de backtest walk-forward.

Walk-forward: em cada data de rebalanceamento t, o modelo usa APENAS
dados anteriores a t (sem lookahead). Pesos são mantidos constantes
pelo holding_period até o próximo rebalanceamento.

Design decisions documentadas:
- Market caps tratados como constantes ao longo do backtest (proxy estático).
  Limitação conhecida: idealmente usaríamos market cap histórico por data.
- Ativos com NaN na janela de treino são descartados naquele período.
  Se um ativo reaparecer na próxima janela sem NaN, volta ao portfólio.
- Último período de holding pode ser menor que holding_period_days se o
  backtest terminar antes — tratado naturalmente pelo slice de índice.
- Em caso de falha da estratégia num período, usa equal-weight como fallback.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.portfolio_utils import DIAS_ANO_CRIPTO

if TYPE_CHECKING:
    from src.estrategias import EstrategiaBase

log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Resultado
# ────────────────────────────────────────────────────────────

@dataclass
class ResultadoBacktest:
    """Container imutável dos resultados de um backtest."""

    nome_estrategia: str
    historico_pesos: pd.DataFrame        # linhas=datas_rebal, colunas=ativos
    retornos_diarios: pd.Series          # retorno diário GROSS do portfólio
    datas_rebalanceamento: pd.DatetimeIndex
    retornos_net: pd.Series | None = field(default=None)         # retornos após custos
    custos_diarios: pd.Series | None = field(default=None)       # custo por dia
    historico_turnover_real: pd.Series | None = field(default=None)  # turnover market-drifted

    @property
    def retornos_acumulados(self) -> pd.Series:
        """Equity curve GROSS com base em 1.0."""
        return (1 + self.retornos_diarios).cumprod()

    @property
    def retornos_acumulados_net(self) -> pd.Series:
        """Equity curve NET com base em 1.0 (igual à GROSS se sem custos)."""
        ret = self.retornos_net if self.retornos_net is not None else self.retornos_diarios
        return (1 + ret).cumprod()

    def metricas(self, cdi_diario: pd.Series | None = None) -> dict:
        """Métricas completas de risco/retorno com suporte a CDI e custos.

        Args:
            cdi_diario: Series com taxa CDI diária (decimal) indexada por data.
                Se fornecida, Sharpe e Sortino são calculados com CDI como
                risk-free; se None, usa rf=0.

        Returns:
            Dict com métricas GROSS e NET, CDI-based e excesso vs CDI.
        """
        ret = self.retornos_diarios.dropna()
        if ret.empty:
            return {}

        ret_anual = ret.mean() * DIAS_ANO_CRIPTO
        vol_anual = ret.std() * np.sqrt(DIAS_ANO_CRIPTO)

        cum   = (1 + ret).cumprod()
        pico  = cum.cummax()
        max_dd = float(((cum - pico) / pico).min())
        ret_total = float(cum.iloc[-1] - 1)
        calmar = ret_anual / abs(max_dd) if max_dd < 0 else np.nan

        # Sharpe / Sortino com CDI como risk-free (ou rf=0 se CDI ausente)
        if cdi_diario is not None and not cdi_diario.empty:
            # CDI alinhado por data: 0 em fins de semana/feriados (CDI acumula apenas
            # em dias úteis; ffill inflaria a taxa ao anualizar por 365 dias)
            cdi_alinhado = cdi_diario.reindex(ret.index).fillna(0.0)
            excesso_diario = ret - cdi_alinhado
            ret_excesso_anual = excesso_diario.mean() * DIAS_ANO_CRIPTO
            sharpe = ret_excesso_anual / vol_anual if vol_anual > 0 else np.nan
            downside = excesso_diario[excesso_diario < 0].std() * np.sqrt(DIAS_ANO_CRIPTO)
            sortino = ret_excesso_anual / downside if downside > 0 else np.nan
            # CDI anual: média dos dias úteis × DIAS_ANO_CRIPTO (consistente com
            # anualização de retornos)
            cdi_anual = float(cdi_alinhado.mean() * DIAS_ANO_CRIPTO)
            excesso_vs_cdi = ret_anual - cdi_anual
        else:
            sharpe = ret_anual / vol_anual if vol_anual > 0 else np.nan
            downside = ret[ret < 0].std() * np.sqrt(DIAS_ANO_CRIPTO)
            sortino = ret_anual / downside if downside > 0 else np.nan
            excesso_vs_cdi = ret_anual  # sem benchmark rf

        # Métricas NET
        if self.retornos_net is not None:
            ret_n = self.retornos_net.dropna()
            ret_anual_net = ret_n.mean() * DIAS_ANO_CRIPTO
            cum_n = (1 + ret_n).cumprod()
            pico_n = cum_n.cummax()
            max_dd_net = float(((cum_n - pico_n) / pico_n).min())
        else:
            ret_anual_net = ret_anual
            max_dd_net = max_dd

        # Custo total acumulado (em % do notional)
        custo_total = float(self.custos_diarios.sum()) if self.custos_diarios is not None else 0.0

        # Turnover médio (market-drifted se disponível, senão target-to-target)
        turn = self.turnover()
        turnover_medio = float(turn.mean()) if not turn.empty else np.nan

        return {
            "retorno_total_%":          round(ret_total * 100, 2),
            "retorno_anualizado_%":     round(ret_anual * 100, 2),
            "retorno_anualizado_net_%": round(ret_anual_net * 100, 2),
            "volatilidade_%":           round(vol_anual * 100, 2),
            "sharpe":                   round(sharpe, 4) if not np.isnan(sharpe) else float("nan"),
            "sortino":                  round(sortino, 4) if not np.isnan(sortino) else float("nan"),
            "max_drawdown_%":           round(max_dd * 100, 2),
            "max_drawdown_net_%":       round(max_dd_net * 100, 2),
            "calmar":                   round(calmar, 4) if not np.isnan(calmar) else float("nan"),
            "excesso_vs_cdi_%":         round(excesso_vs_cdi * 100, 2),
            "turnover_medio":           round(turnover_medio, 4),
            "custo_total_%":            round(custo_total * 100, 4),
        }

    def turnover(self) -> pd.Series:
        """Turnover entre rebalanceamentos.

        Usa market-drifted turnover se disponível (calculado em executar_estrategia),
        caso contrário cai de volta no target-to-target (Σ|w_t - w_{t-1}|/2).
        """
        if self.historico_turnover_real is not None and not self.historico_turnover_real.empty:
            return self.historico_turnover_real
        if self.historico_pesos.shape[0] < 2:
            return pd.Series(dtype=float)
        delta = self.historico_pesos.diff().abs().sum(axis=1) / 2
        return delta.dropna()


# ────────────────────────────────────────────────────────────
# Backtest
# ────────────────────────────────────────────────────────────

class WalkForwardBacktest:
    """Backtest walk-forward para estratégias de alocação.

    Em cada data de rebalanceamento t:
        1. Usa janela [t - lookback, t) para calcular μ, Σ e views
        2. Otimiza pesos com a estratégia escolhida
        3. Aplica os pesos no período [t, t + holding_period)
        4. Mede o retorno realizado
        5. Avança holding_period dias e repete

    Args:
        retornos: DataFrame de log-retornos diários.
        precos: DataFrame de preços diários (para views baseadas em preço).
        market_caps_atuais: Series de market cap (proxy estático).
        lookback_days: Tamanho da janela de treino em dias de negociação.
        holding_period_days: Número de dias entre rebalanceamentos.
        data_inicio_teste: Primeira data de rebalanceamento (default: primeiro
            dia com lookback completo disponível).
        data_fim_teste: Última data do período de teste (default: última data
            disponível nos retornos).
    """

    def __init__(
        self,
        retornos: pd.DataFrame,
        precos: pd.DataFrame,
        market_caps_atuais: pd.Series,
        lookback_days: int = 365,
        holding_period_days: int = 7,
        data_inicio_teste: pd.Timestamp | None = None,
        data_fim_teste: pd.Timestamp | None = None,
    ) -> None:
        if retornos.empty:
            raise ValueError("retornos está vazio.")
        if lookback_days < 30:
            raise ValueError(f"lookback_days deve ser >= 30 (recebido: {lookback_days}).")
        if holding_period_days < 1:
            raise ValueError(f"holding_period_days deve ser >= 1 (recebido: {holding_period_days}).")

        self.retornos          = retornos.sort_index()
        self.precos            = precos.sort_index()
        self.market_caps       = market_caps_atuais
        self.lookback_days     = lookback_days
        self.holding_period    = holding_period_days

        todas_datas = self.retornos.index

        # Primeira data com lookback completo
        primeiro_idx = lookback_days  # precisa de lookback dias ANTES
        if primeiro_idx >= len(todas_datas):
            raise ValueError(
                f"Série de retornos ({len(todas_datas)} dias) menor que lookback ({lookback_days})."
            )

        self.data_inicio_teste = (
            data_inicio_teste if data_inicio_teste is not None
            else todas_datas[primeiro_idx]
        )
        self.data_fim_teste = (
            data_fim_teste if data_fim_teste is not None
            else todas_datas[-1]
        )

        if self.data_inicio_teste > self.data_fim_teste:
            raise ValueError("data_inicio_teste > data_fim_teste.")

        log.info(
            f"Backtest configurado: {self.data_inicio_teste.date()} → "
            f"{self.data_fim_teste.date()} | lookback={lookback_days} | "
            f"holding={holding_period_days}"
        )

    def gerar_datas_rebalanceamento(self) -> pd.DatetimeIndex:
        """Datas de rebalanceamento espaçadas por holding_period_days.

        Usa as datas reais do índice (não datas de calendário), garantindo
        que cada data de rebalanceamento existe nos dados.
        """
        todas = self.retornos.index
        datas_validas = todas[
            (todas >= self.data_inicio_teste) & (todas <= self.data_fim_teste)
        ]
        return datas_validas[:: self.holding_period]

    def janela_treino(self, data_rebal: pd.Timestamp) -> pd.DataFrame:
        """Retorna [data_rebal - lookback, data_rebal) — exclui data_rebal.

        CRÍTICO: data_rebal NÃO está incluída para evitar lookahead bias.
        """
        antes = self.retornos[self.retornos.index < data_rebal]
        if len(antes) < self.lookback_days:
            log.warning(
                f"Janela de treino em {data_rebal.date()} tem apenas "
                f"{len(antes)} dias (esperado {self.lookback_days})."
            )
        return antes.iloc[-self.lookback_days :]

    def janela_holding(self, data_rebal: pd.Timestamp) -> pd.DataFrame:
        """Retorna [data_rebal, data_rebal + holding_period).

        Pode ser menor que holding_period no último período do backtest.
        """
        idx = self.retornos.index.get_loc(data_rebal)
        fim = min(idx + self.holding_period, len(self.retornos))
        return self.retornos.iloc[idx:fim]

    def executar_estrategia(
        self,
        estrategia: EstrategiaBase,
        custo_bps: float = 20.0,
    ) -> ResultadoBacktest:
        """Executa o walk-forward completo para uma estratégia.

        Calcula turnover contra pesos DERIVADOS do mercado (não pesos-alvo anteriores),
        pois os retornos de mercado deslocam os pesos entre rebalanceamentos.
        Gera séries GROSS (sem custo) e NET (após custo de transação).

        Args:
            estrategia: Instância de EstrategiaBase com calcular_pesos().
            custo_bps: Custo de transação em basis points (padrão 20 bps = 0.20%).

        Returns:
            ResultadoBacktest com pesos históricos, retornos GROSS/NET e custos.
        """
        datas_rebal = self.gerar_datas_rebalanceamento()
        n_rebal     = len(datas_rebal)

        historico_pesos: dict[pd.Timestamp, pd.Series] = {}
        lista_retornos: list[pd.Series] = []
        custos_rebalanceamento: dict[pd.Timestamp, float] = {}
        historico_turnover: dict[pd.Timestamp, float] = {}

        prev_pesos: pd.Series | None = None
        prev_data: pd.Timestamp | None = None

        for i, data_t in enumerate(datas_rebal):
            log.info(
                "[%s] Rebalanceamento %d/%d: %s",
                estrategia.nome, i + 1, n_rebal, data_t.date(),
            )

            treino = self.janela_treino(data_t)

            # Remove ativos com NaN na janela (histórico insuficiente ou gaps).
            # Decisão baseada APENAS em dados anteriores a data_t — sem lookahead.
            treino_limpo = treino.dropna(axis=1, how="any")

            excluidos = sorted(set(treino.columns) - set(treino_limpo.columns))
            if excluidos:
                log.info(
                    "[%s] %s — Ativos excluídos por histórico insuficiente: %s",
                    estrategia.nome, data_t.date(), excluidos,
                )

            if treino_limpo.empty:
                log.warning("Treino vazio em %s — pulando.", data_t.date())
                continue

            precos_treino = self.precos[self.precos.index < data_t][treino_limpo.columns]

            # Calcula pesos — fallback para equal-weight em caso de erro
            try:
                pesos = estrategia.calcular_pesos(treino_limpo, precos_treino)
            except Exception as exc:
                log.warning(
                    "[%s] Falha em %s: %s. Usando equal-weight.",
                    estrategia.nome, data_t.date(), exc,
                )
                n = treino_limpo.shape[1]
                pesos = pd.Series(1.0 / n, index=treino_limpo.columns)

            # ── Turnover market-drifted e custo de transação ──────────────
            if prev_pesos is not None and prev_data is not None and custo_bps > 0:
                w_drifted = self._calcular_pesos_drifted(prev_pesos, prev_data, data_t)
                todos_ativos = pesos.index.union(w_drifted.index)
                w_new = pesos.reindex(todos_ativos, fill_value=0.0)
                w_old = w_drifted.reindex(todos_ativos, fill_value=0.0)
                turnover = float((w_new - w_old).abs().sum() / 2)
                custo = turnover * custo_bps / 10_000
                historico_turnover[data_t] = turnover
                custos_rebalanceamento[data_t] = custo
                log.debug(
                    "[%s] Turnover em %s: %.4f → custo=%.2f bps (%.4f%%)",
                    estrategia.nome, data_t.date(), turnover, custo * 10_000, custo * 100,
                )

            prev_pesos = pesos
            prev_data  = data_t

            historico_pesos[data_t] = pesos
            log.debug(
                "[%s] Pesos em %s: %s",
                estrategia.nome,
                data_t.date(),
                {k: round(v, 4) for k, v in pesos.sort_values(ascending=False).items()},
            )

            # Retornos realizados no período de holding
            holding = self.janela_holding(data_t)
            if holding.empty:
                continue

            # Alinha ativos e renormaliza pesos (por segurança)
            ativos = holding.columns.intersection(pesos.index)
            if ativos.empty:
                log.warning("Nenhum ativo em comum entre holding e pesos em %s.", data_t.date())
                continue

            p = pesos[ativos]
            p = p / p.sum()  # renormaliza após possível remoção de ativos

            ret_portfolio = (holding[ativos] * p).sum(axis=1)
            lista_retornos.append(ret_portfolio)

        if not lista_retornos:
            raise RuntimeError(f"Nenhum retorno gerado para estratégia '{estrategia.nome}'.")

        retornos_gross = pd.concat(lista_retornos).sort_index()
        retornos_gross = retornos_gross[~retornos_gross.index.duplicated(keep="first")]

        # Série NET: deduz custo no primeiro dia de cada período de holding
        custos_serie = pd.Series(0.0, index=retornos_gross.index, dtype=float)
        for data_rebal, custo_val in custos_rebalanceamento.items():
            if data_rebal in custos_serie.index:
                custos_serie[data_rebal] = custo_val
        retornos_net = retornos_gross - custos_serie

        df_pesos = pd.DataFrame(historico_pesos).T
        df_pesos.index.name = "data_rebalanceamento"

        turnover_serie = pd.Series(historico_turnover, dtype=float)
        if not turnover_serie.empty:
            turnover_serie.index.name = "data_rebalanceamento"

        return ResultadoBacktest(
            nome_estrategia=estrategia.nome,
            historico_pesos=df_pesos,
            retornos_diarios=retornos_gross,
            datas_rebalanceamento=pd.DatetimeIndex(list(historico_pesos.keys())),
            retornos_net=retornos_net,
            custos_diarios=custos_serie,
            historico_turnover_real=turnover_serie,
        )

    def _calcular_pesos_drifted(
        self,
        pesos_prev: pd.Series,
        data_prev: pd.Timestamp,
        data_curr: pd.Timestamp,
    ) -> pd.Series:
        """Pesos do portfólio anterior deslocados pelos retornos de mercado.

        Os retornos de mercado entre dois rebalanceamentos deslocam os pesos
        mesmo sem qualquer ação do gestor. Esta função computa o vetor de pesos
        atual implícito pela valorização/desvalorização de cada posição.

        Args:
            pesos_prev: Pesos-alvo do rebalanceamento anterior.
            data_prev: Data do rebalanceamento anterior.
            data_curr: Data do novo rebalanceamento.

        Returns:
            Series com pesos drifted (soma = 1).
        """
        prev_ativos = pesos_prev.index
        try:
            precos_prev = self.precos.loc[data_prev, prev_ativos]
            precos_curr = self.precos.loc[data_curr, prev_ativos]
        except KeyError:
            return pesos_prev

        valid = precos_prev.notna() & precos_curr.notna() & (precos_prev > 0)
        if not valid.any():
            return pesos_prev

        r = precos_curr[valid] / precos_prev[valid] - 1
        valores = pesos_prev[valid] * (1 + r)
        total   = valores.sum()
        if total <= 0:
            return pesos_prev

        return valores / total

    def comparar_estrategias(
        self,
        estrategias: dict[str, EstrategiaBase],
        custo_bps: float = 20.0,
    ) -> dict[str, ResultadoBacktest]:
        """Roda múltiplas estratégias no mesmo backtest.

        Args:
            estrategias: Dict {nome: instância_estrategia}.
            custo_bps: Custo de transação em basis points para todas as estratégias.

        Returns:
            Dict {nome: ResultadoBacktest}.
        """
        resultados: dict[str, ResultadoBacktest] = {}
        n_total = len(estrategias)
        for idx, (nome, est) in enumerate(estrategias.items(), start=1):
            log.info("=" * 55)
            log.info("Estrategia %d/%d: %s", idx, n_total, nome)
            log.info("=" * 55)
            resultados[nome] = self.executar_estrategia(est, custo_bps=custo_bps)
            metricas = resultados[nome].metricas()
            log.info(
                "[%s] Concluida — Retorno=%.1f%%, Vol=%.1f%%, Sharpe=%.2f, MDD=%.1f%%",
                nome,
                metricas.get("retorno_anualizado_%", float("nan")),
                metricas.get("volatilidade_%", float("nan")),
                metricas.get("sharpe", float("nan")),
                metricas.get("max_drawdown_%", float("nan")),
            )
        return resultados


# ────────────────────────────────────────────────────────────
# Backtest rolante standalone — independente da classe BL
# ────────────────────────────────────────────────────────────

def backtest_rolante_bl(
    retornos: pd.DataFrame,
    market_caps: pd.Series,
    janela: int = 252,
    holding: int = 30,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    peso_maximo: float = 0.40,
) -> dict:
    """Backtest rolante do Black-Litterman em modo neutro (sem views).

    For loop temporal: a cada t, pega janela [t-janela, t], roda BL sem views,
    guarda os pesos, computa retorno para os próximos `holding` dias.
    Benchmark: pesos de mercado fixos no início do período de teste (buy & hold).

    Args:
        retornos:      DataFrame de log-retornos diários (linhas=datas, colunas=ativos).
        market_caps:   Series de market cap por ativo (para pesos de equilíbrio).
        janela:        Tamanho da janela de treino em dias (default 252).
        holding:       Número de dias entre rebalanceamentos (default 30).
        tau:           Parâmetro τ do BL (default 0.05).
        risk_aversion: λ do BL (default 2.5).
        peso_maximo:   Limite superior por ativo na otimização (default 0.40).

    Returns:
        Dict com:
            'historico_pesos':    DataFrame (datas_rebal × ativos)
            'equity_portfolio':   Series equity curve BL (base 100)
            'equity_benchmark':   Series equity curve benchmark (base 100)
            'metricas_portfolio': dict de métricas do BL vs benchmark
            'metricas_benchmark': dict de métricas do benchmark
    """
    from src.black_litterman import BlackLitterman  # import local evita ciclo
    from src.portfolio_utils import metricas_equity_curve

    retornos = retornos.sort_index()
    todas_datas = retornos.index

    if len(todas_datas) <= janela:
        raise ValueError(
            f"Série de retornos ({len(todas_datas)} dias) menor que janela ({janela})."
        )

    # Datas de rebalanceamento: início em todas_datas[janela], passo = holding
    datas_rebal = todas_datas[janela::holding]

    historico_pesos: dict[pd.Timestamp, pd.Series] = {}
    lista_retornos_bl: list[pd.Series] = []

    for data_t in datas_rebal:
        # Janela de treino: [t-janela, t) — sem lookahead
        treino = retornos[retornos.index < data_t].iloc[-janela:]
        treino_limpo = treino.dropna(axis=1, how="any")

        if treino_limpo.shape[1] == 0:
            log.warning("Treino vazio em %s — pulando.", data_t.date())
            continue

        mc = market_caps.reindex(treino_limpo.columns).dropna()
        ret_bl_ativos = treino_limpo[mc.index]

        if ret_bl_ativos.shape[1] == 0:
            log.warning("Nenhum ativo com market cap em %s — pulando.", data_t.date())
            continue

        try:
            bl = BlackLitterman(ret_bl_ativos, mc, risk_aversion=risk_aversion, tau=tau)
            resultado = bl.executar(peso_maximo=peso_maximo)
            pesos = resultado["pesos_otimos"]
        except Exception as exc:
            log.warning("BL falhou em %s: %s — usando equal-weight.", data_t.date(), exc)
            n = ret_bl_ativos.shape[1]
            pesos = pd.Series(1.0 / n, index=ret_bl_ativos.columns)

        historico_pesos[data_t] = pesos

        # Retornos realizados no período de holding
        idx = retornos.index.get_loc(data_t)
        fim = min(idx + holding, len(retornos))
        holding_ret = retornos.iloc[idx:fim]

        ativos = holding_ret.columns.intersection(pesos.index)
        p = pesos[ativos]
        p = p / p.sum()
        lista_retornos_bl.append((holding_ret[ativos] * p).sum(axis=1))

    if not lista_retornos_bl:
        raise RuntimeError("Nenhum retorno gerado no backtest rolante.")

    ret_bl_serie = pd.concat(lista_retornos_bl).sort_index()
    ret_bl_serie = ret_bl_serie[~ret_bl_serie.index.duplicated(keep="first")]

    # Benchmark: market cap fixo no início do período de teste
    data_inicio_teste = ret_bl_serie.index[0]
    ret_periodo_teste = retornos[retornos.index >= data_inicio_teste]
    ativos_bench = retornos.columns.intersection(market_caps.index)
    mc_bench = market_caps[ativos_bench]
    w_bench = mc_bench / mc_bench.sum()
    ret_benchmark_serie = (ret_periodo_teste[ativos_bench] * w_bench).sum(axis=1)
    ret_benchmark_serie = ret_benchmark_serie.reindex(ret_bl_serie.index).fillna(0)

    equity_bl   = (1 + ret_bl_serie).cumprod() * 100
    equity_bench = (1 + ret_benchmark_serie).cumprod() * 100

    metricas_bl    = metricas_equity_curve(equity_bl,    benchmark=equity_bench)
    metricas_bench = metricas_equity_curve(equity_bench)

    log.info(
        "Backtest rolante concluído: %d rebalanceamentos, %d dias de retorno. "
        "BL: ret=%.1f%%, sharpe=%.2f | Benchmark: ret=%.1f%%, sharpe=%.2f",
        len(historico_pesos),
        len(ret_bl_serie),
        metricas_bl.get("retorno_anualizado_%", float("nan")),
        metricas_bl.get("sharpe", float("nan")),
        metricas_bench.get("retorno_anualizado_%", float("nan")),
        metricas_bench.get("sharpe", float("nan")),
    )

    df_pesos = pd.DataFrame(historico_pesos).T
    df_pesos.index.name = "data_rebalanceamento"

    return {
        "historico_pesos":    df_pesos,
        "equity_portfolio":   equity_bl,
        "equity_benchmark":   equity_bench,
        "metricas_portfolio": metricas_bl,
        "metricas_benchmark": metricas_bench,
    }
