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
    retornos_diarios: pd.Series          # retorno diário do portfólio
    datas_rebalanceamento: pd.DatetimeIndex

    @property
    def retornos_acumulados(self) -> pd.Series:
        """Equity curve com base em 1.0."""
        return (1 + self.retornos_diarios).cumprod()

    def metricas(self) -> dict:
        """Métricas completas de risco/retorno.

        Returns:
            Dict com: retorno_total_%, retorno_anualizado_%, volatilidade_%,
            sharpe, sortino, max_drawdown_%, calmar, turnover_medio.
        """
        ret = self.retornos_diarios.dropna()
        if ret.empty:
            return {}

        n = len(ret)
        ret_anual = ret.mean() * DIAS_ANO_CRIPTO
        vol_anual = ret.std() * np.sqrt(DIAS_ANO_CRIPTO)

        sharpe = ret_anual / vol_anual if vol_anual > 0 else np.nan

        downside = ret[ret < 0].std() * np.sqrt(DIAS_ANO_CRIPTO)
        sortino = ret_anual / downside if downside > 0 else np.nan

        cum   = (1 + ret).cumprod()
        pico  = cum.cummax()
        dd    = (cum - pico) / pico
        max_dd = float(dd.min())

        calmar = ret_anual / abs(max_dd) if max_dd < 0 else np.nan

        ret_total = float(cum.iloc[-1] - 1)

        turn = self.turnover()
        turnover_medio = float(turn.mean()) if not turn.empty else np.nan

        return {
            "retorno_total_%":        round(ret_total  * 100, 2),
            "retorno_anualizado_%":   round(ret_anual  * 100, 2),
            "volatilidade_%":         round(vol_anual  * 100, 2),
            "sharpe":                 round(sharpe,       4),
            "sortino":                round(sortino,      4),
            "max_drawdown_%":         round(max_dd     * 100, 2),
            "calmar":                 round(calmar,       4),
            "turnover_medio":         round(turnover_medio, 4),
        }

    def turnover(self) -> pd.Series:
        """Turnover entre rebalanceamentos consecutivos.

        turnover_t = Σ|w_t - w_{t-1}| / 2

        Valores entre 0 (sem troca) e 1 (portfólio totalmente renovado).
        """
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
    ) -> ResultadoBacktest:
        """Executa o walk-forward completo para uma estratégia.

        Args:
            estrategia: Instância de EstrategiaBase com calcular_pesos().

        Returns:
            ResultadoBacktest com pesos históricos e retornos diários.
        """
        datas_rebal = self.gerar_datas_rebalanceamento()
        n_rebal     = len(datas_rebal)

        historico_pesos: dict[pd.Timestamp, pd.Series] = {}
        lista_retornos: list[pd.Series] = []

        for i, data_t in enumerate(datas_rebal):
            log.info(
                "[%s] Rebalanceamento %d/%d: %s",
                estrategia.nome, i + 1, n_rebal, data_t.date(),
            )

            treino = self.janela_treino(data_t)

            # Remove ativos com qualquer NaN na janela de treino
            treino_limpo = treino.dropna(axis=1, how="any")
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

        retornos_serie = pd.concat(lista_retornos).sort_index()
        # Remove datas duplicadas (não deveria ocorrer, mas por segurança)
        retornos_serie = retornos_serie[~retornos_serie.index.duplicated(keep="first")]

        df_pesos = pd.DataFrame(historico_pesos).T
        df_pesos.index.name = "data_rebalanceamento"

        return ResultadoBacktest(
            nome_estrategia=estrategia.nome,
            historico_pesos=df_pesos,
            retornos_diarios=retornos_serie,
            datas_rebalanceamento=pd.DatetimeIndex(list(historico_pesos.keys())),
        )

    def comparar_estrategias(
        self,
        estrategias: dict[str, EstrategiaBase],
    ) -> dict[str, ResultadoBacktest]:
        """Roda múltiplas estratégias no mesmo backtest.

        Args:
            estrategias: Dict {nome: instância_estrategia}.

        Returns:
            Dict {nome: ResultadoBacktest}.
        """
        resultados: dict[str, ResultadoBacktest] = {}
        n_total = len(estrategias)
        for idx, (nome, est) in enumerate(estrategias.items(), start=1):
            log.info("=" * 55)
            log.info("Estrategia %d/%d: %s", idx, n_total, nome)
            log.info("=" * 55)
            resultados[nome] = self.executar_estrategia(est)
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
