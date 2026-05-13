"""
black_litterman.py — Implementação do modelo Black-Litterman (1992).

O modelo combina o equilíbrio de mercado (CAPM implícito) com as opiniões
do gestor (views) via inferência bayesiana, produzindo um vetor de retornos
esperados mais estável e economicamente fundamentado do que a média histórica.

Fluxo completo:
    1. Pesos de mercado  →  w_mkt = market_cap / Σ market_cap
    2. Retornos implícitos →  Π = λ Σ w_mkt   (engenharia reversa do CAPM)
    3. Combinar views     →  E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹Π + P'Ω⁻¹Q]
    4. Otimização MV      →  max w'μ - (λ/2) w'Σw   s.t. restrições

Referências:
    Black, F. & Litterman, R. (1992). Global Portfolio Optimization.
    Financial Analysts Journal, 48(5), 28-43.
"""
import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.portfolio_utils import (
    calcular_matriz_covariancia,
    calcular_matriz_covariancia_ledoitwolf,
    estatisticas_portfolio,
    aplicar_pesos,
)

log = logging.getLogger(__name__)


def inverter_matriz_robusta(M: np.ndarray, nome: str = "M") -> np.ndarray:
    """Inverte matriz com fallback para pseudo-inversa se mal-condicionada.

    Loga cond(M). Se cond > 1e10, emite warning e usa np.linalg.pinv em
    vez de np.linalg.inv, evitando erros numéricos em matrizes singulares
    ou quase-singulares (comum em janelas curtas de cripto com alta correlação).

    Args:
        M: Matriz quadrada a inverter.
        nome: Identificador para logging (ex: "Sigma", "Omega").

    Returns:
        Inversa (ou pseudo-inversa) de M.

    Raises:
        ValueError: Se M não for quadrada.
    """
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"Matriz {nome} deve ser quadrada (shape={M.shape}).")

    cond = np.linalg.cond(M)
    log.debug("cond(%s) = %.2e", nome, cond)

    if cond > 1e10:
        log.warning(
            "Matriz %s mal-condicionada (cond=%.2e > 1e10) — usando pseudo-inversa.",
            nome, cond,
        )
        return np.linalg.pinv(M)

    return np.linalg.inv(M)


class BlackLitterman:
    """Modelo Black-Litterman para otimização de portfólio de criptomoedas.

    Args:
        retornos:         DataFrame de log-retornos diários (linhas=datas, colunas=ativos).
        market_caps:      Series com market cap por ativo (mesmo índice que colunas de retornos).
        risk_aversion:    λ — aversão ao risco do investidor representativo (típico: 2–4).
        tau:              Escala da incerteza nos retornos de equilíbrio (típico: 0.025–0.05).
        usar_ledoit_wolf: Se True (padrão), usa Ledoit-Wolf para a covariância.
    """

    def __init__(
        self,
        retornos: pd.DataFrame,
        market_caps: pd.Series,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        usar_ledoit_wolf: bool = True,
    ) -> None:
        self._validar_inputs(retornos, market_caps)

        # Alinha ativos comuns entre retornos e market_caps
        ativos_comuns = retornos.columns.intersection(market_caps.index)
        if len(ativos_comuns) == 0:
            raise ValueError("Nenhum ativo em comum entre retornos e market_caps.")
        if len(ativos_comuns) < len(retornos.columns):
            ausentes = set(retornos.columns) - set(ativos_comuns)
            log.warning("Ativos sem market cap removidos: %s", ausentes)

        self.retornos      = retornos[ativos_comuns].copy()
        self.market_caps   = market_caps[ativos_comuns].copy()
        self.risk_aversion = risk_aversion
        self.tau           = tau
        self.ativos        = list(ativos_comuns)
        self.n             = len(self.ativos)

        n_obs = self.retornos.shape[0]
        log.info(
            "BlackLitterman: %d ativos, %d observacoes, lambda=%.2f, tau=%.3f",
            self.n, n_obs, risk_aversion, tau,
        )
        log.debug("Ativos: %s", self.ativos)
        if hasattr(self.retornos.index, "date"):
            log.debug(
                "Periodo dos retornos: %s -> %s",
                self.retornos.index.min().date(),
                self.retornos.index.max().date(),
            )
        else:
            log.debug(
                "Periodo dos retornos: [%s, %s]",
                self.retornos.index.min(),
                self.retornos.index.max(),
            )

        # Covariância anualizada — Ledoit-Wolf por padrão (evita matrizes singulares)
        if usar_ledoit_wolf:
            self.cov = calcular_matriz_covariancia_ledoitwolf(self.retornos, anualizar=True)
            estimador = "Ledoit-Wolf"
        else:
            self.cov = calcular_matriz_covariancia(self.retornos, anualizar=True)
            estimador = "amostral"

        log.info("Covariância calculada com estimador: %s", estimador)

    # ────────────────────────────────────────────────────────
    # Etapa 1 — Pesos de mercado
    # ────────────────────────────────────────────────────────

    def calcular_pesos_mercado(self) -> pd.Series:
        """Pesos proporcionais ao market cap (carteira de equilíbrio).

        Returns:
            Series com pesos normalizados (soma = 1).
        """
        log.info("[Etapa 1] Calculando pesos de mercado (w = market_cap / sum)...")

        w = self.market_caps / self.market_caps.sum()

        log.debug("Pesos de mercado por ativo:")
        for ativo, peso in w.sort_values(ascending=False).items():
            log.debug("  %-8s  %.4f  (%.1f%%)", ativo, peso, peso * 100)

        log.info(
            "[Etapa 1] Pesos de mercado calculados. "
            "Top-3: %s",
            dict(w.sort_values(ascending=False).head(3).round(4)),
        )
        return w

    # ────────────────────────────────────────────────────────
    # Etapa 2 — Retornos implícitos de equilíbrio
    # ────────────────────────────────────────────────────────

    def calcular_retornos_implicitos(self) -> pd.Series:
        """Π = λ Σ w_mkt  (engenharia reversa do CAPM).

        Pergunta: "Dado que o mercado está em equilíbrio com esses pesos,
        qual o vetor de retornos esperados que sustenta essa carteira?"

        Returns:
            Series com retornos implícitos anualizados por ativo.
        """
        log.info(
            "[Etapa 2] Calculando risk premium implícito (Pi = lambda * Sigma * w_mkt, "
            "lambda=%.2f)...",
            self.risk_aversion,
        )

        w_mkt = self.calcular_pesos_mercado()
        pi    = self.risk_aversion * self.cov.values @ w_mkt.values
        pi_series = pd.Series(pi, index=self.ativos, name="retornos_implicitos")

        log.debug("Retornos implícitos Pi (anualizados):")
        for ativo, val in pi_series.sort_values(ascending=False).items():
            log.debug("  %-8s  %.4f  (%.1f%% a.a.)", ativo, val, val * 100)

        log.info(
            "[Etapa 2] Risk premium calculado. "
            "Intervalo: [%.4f, %.4f]",
            pi_series.min(), pi_series.max(),
        )
        return pi_series

    # ────────────────────────────────────────────────────────
    # Etapa 3 — Combinar views (fórmula bayesiana)
    # ────────────────────────────────────────────────────────

    def combinar_views(
        self,
        P: np.ndarray,
        Q: np.ndarray,
        Omega: np.ndarray,
    ) -> pd.Series:
        """Fórmula central do Black-Litterman.

        E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ · [(τΣ)⁻¹Π + P'Ω⁻¹Q]

        ESCALA: Q deve estar em retorno ANUAL (mesma escala de Π = λΣw).
        Σ é anualizada (× 365), portanto Π é anualizada. Q deve seguir
        a mesma escala para que a combinação bayesiana seja coerente.
        Use `views.view_diaria_para_anual()` se tiver views em escala diária.

        Args:
            P:     Matriz k×n de mapeamento views→ativos.
            Q:     Vetor k de retornos anuais esperados das views
                   (ex: 0.05 = 5% a.a.).
            Omega: Matriz k×k diagonal de incerteza das views (escala anual²).

        Returns:
            Series com retornos esperados combinados por ativo (escala anual).

        Raises:
            ValueError: Se dimensões de P, Q, Omega forem inconsistentes.
        """
        self._validar_views(P, Q, Omega)

        if np.abs(Q).max() > 0.1:
            log.warning(
                "Q contém valores > 10%% ao dia (max=%.4f). "
                "Verifique se Q está em escala diária em vez de anual — "
                "o modelo espera retorno ANUAL consistente com Sigma anualizada.",
                np.abs(Q).max(),
            )

        k = len(Q)
        log.info(
            "[Etapa 3] Combinando views com prior BL (k=%d views, tau=%.3f)...",
            k, self.tau,
        )

        log.debug("Views Q (retorno anual esperado por view):")
        for i, q in enumerate(Q):
            log.debug("  view[%d]  Q=%.4f (%.1f%% a.a.)", i, q, q * 100)

        log.debug("Omega diagonal (incerteza por view):")
        for i in range(k):
            log.debug("  view[%d]  Omega_ii=%.6f", i, Omega[i, i])

        log.debug("Matriz P (shape=%s):\n%s", P.shape, P.round(4))

        pi    = self.calcular_retornos_implicitos().values
        Sigma = self.cov.values
        tau   = self.tau

        tauSigma     = tau * Sigma
        tauSigma_inv = inverter_matriz_robusta(tauSigma, "tau*Sigma")
        Omega_inv    = inverter_matriz_robusta(Omega, "Omega")

        M_inv = inverter_matriz_robusta(tauSigma_inv + P.T @ Omega_inv @ P, "M_bl")
        rhs   = tauSigma_inv @ pi + P.T @ Omega_inv @ Q
        mu_bl = M_inv @ rhs

        result = pd.Series(mu_bl, index=self.ativos, name="retornos_combinados")

        log.debug("Retornos combinados (posterior BL) vs prior (Pi):")
        for ativo in self.ativos:
            pi_val  = pd.Series(pi, index=self.ativos)[ativo]
            bl_val  = result[ativo]
            delta   = bl_val - pi_val
            log.debug(
                "  %-8s  Pi=%.4f  BL=%.4f  delta=%+.4f",
                ativo, pi_val, bl_val, delta,
            )

        log.info(
            "[Etapa 3] Views combinadas. Desvio medio do prior: %.4f",
            (result.values - pi).mean(),
        )
        return result

    # ────────────────────────────────────────────────────────
    # Etapa 4 — Otimização média-variância
    # ────────────────────────────────────────────────────────

    def otimizar(
        self,
        retornos_esperados: pd.Series,
        cov: pd.DataFrame,
        permitir_short: bool = False,
        peso_maximo: float = 0.40,
    ) -> pd.Series:
        """Otimização média-variância com restrições.

        Problema:
            max  w'μ - (λ/2) w'Σw
            s.t. Σw = 1
                 w_i ≥ 0          (long-only, default — ver permitir_short)
                 w_i ≤ peso_maximo

        Default é long-only (permitir_short=False): cada peso fica entre
        [0, peso_maximo], o que reflete portfólios institucionais de cripto
        onde short via perp tem custo de funding elevado.

        Args:
            retornos_esperados: Series μ com retornos esperados por ativo.
            cov:                DataFrame Σ de covariância.
            permitir_short:     Se True, remove bound inferior (lb=-1).
                                Default False (long-only).
            peso_maximo:        Limite superior por ativo (0 < peso_maximo ≤ 1).

        Returns:
            Series com pesos ótimos normalizados (soma = 1, ≥ 0 por padrão).

        Raises:
            RuntimeError: Se a otimização não convergir.
        """
        mu    = retornos_esperados.values
        Sigma = cov.values
        lam   = self.risk_aversion
        n     = len(mu)

        log.info(
            "[Etapa 4] Iniciando otimização media-variância "
            "(n=%d ativos, lambda=%.2f, peso_max=%.0f%%)...",
            n, lam, peso_maximo * 100,
        )
        log.debug(
            "Retornos esperados mu (anualizados): %s",
            dict(zip(self.ativos, mu.round(4))),
        )

        def objetivo(w):
            return -(w @ mu - (lam / 2) * w @ Sigma @ w)

        restricoes = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        lb = -1.0 if permitir_short else 0.0
        limites = [(lb, peso_maximo)] * n
        w0 = np.ones(n) / n

        resultado = minimize(
            objetivo,
            w0,
            method="SLSQP",
            bounds=limites,
            constraints=restricoes,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        if not resultado.success:
            raise RuntimeError(
                f"Otimizacao nao convergiu: {resultado.message}"
            )

        pesos = pd.Series(resultado.x, index=self.ativos, name="pesos_otimos")
        pesos = pesos.clip(lower=0).div(pesos.clip(lower=0).sum())

        log.debug("Pesos otimos por ativo:")
        for ativo, peso in pesos.sort_values(ascending=False).items():
            log.debug("  %-8s  %.4f  (%.1f%%)", ativo, peso, peso * 100)

        n_ativos_ativos = (pesos > 1e-4).sum()
        log.info(
            "[Etapa 4] Otimizacao concluida em %d iteracoes. "
            "%d/%d ativos com peso > 0.01%%.",
            resultado.nit, n_ativos_ativos, n,
        )
        return pesos

    # ────────────────────────────────────────────────────────
    # Pipeline completo
    # ────────────────────────────────────────────────────────

    def executar(
        self,
        P: np.ndarray | None = None,
        Q: np.ndarray | None = None,
        Omega: np.ndarray | None = None,
        **kwargs_otimizacao,
    ) -> dict:
        """Executa o pipeline completo do modelo.

        Se P/Q/Omega forem None, retorna o portfólio de equilíbrio (sem views),
        que deve aproximar os pesos de mercado.

        Args:
            P:     Matriz k×n de views (opcional).
            Q:     Vetor k de retornos esperados das views (opcional).
            Omega: Matriz k×k de incerteza das views (opcional).
            **kwargs_otimizacao: Passados para `otimizar()` (ex: peso_maximo=0.3).

        Returns:
            Dict com:
                'pesos_mercado':        Series
                'retornos_implicitos':  Series
                'retornos_combinados':  Series
                'pesos_otimos':         Series
                'estatisticas':         dict
        """
        log.info("=" * 55)
        log.info("PIPELINE BLACK-LITTERMAN (%d ativos)", self.n)
        log.info("=" * 55)

        # Etapas 1 e 2
        w_mkt = self.calcular_pesos_mercado()
        pi    = self.calcular_retornos_implicitos()

        # Etapa 3: combinar views ou usar equilíbrio puro
        tem_views = P is not None and Q is not None and Omega is not None
        if tem_views:
            mu_bl = self.combinar_views(P, Q, Omega)
            log.info("[3/4] %d view(s) incorporada(s) ao prior.", len(Q))
        else:
            mu_bl = pi.copy()
            mu_bl.name = "retornos_combinados"
            log.info("[3/4] Sem views — usando retornos de equilibrio (Pi).")

        # Etapa 4: otimização
        pesos = self.otimizar(mu_bl, self.cov, **kwargs_otimizacao)

        # Estatísticas do portfólio otimizado
        ret_portfolio = aplicar_pesos(self.retornos, pesos)
        stats = estatisticas_portfolio(ret_portfolio)

        log.info(
            "Pipeline concluido. Retorno anual=%.1f%%, Vol=%.1f%%, Sharpe=%.2f, MDD=%.1f%%",
            stats["retorno_anual_%"],
            stats["volatilidade_%"],
            stats["sharpe"],
            stats["max_drawdown_%"],
        )
        log.info("=" * 55)

        return {
            "pesos_mercado":       w_mkt,
            "retornos_implicitos": pi,
            "retornos_combinados": mu_bl,
            "pesos_otimos":        pesos,
            "estatisticas":        stats,
        }

    # ────────────────────────────────────────────────────────
    # Validações internas
    # ────────────────────────────────────────────────────────

    @staticmethod
    def _validar_inputs(retornos: pd.DataFrame, market_caps: pd.Series) -> None:
        if retornos.empty:
            raise ValueError("retornos está vazio.")
        if market_caps.empty:
            raise ValueError("market_caps está vazio.")
        if (market_caps <= 0).any():
            raise ValueError("market_caps contém valores não-positivos.")

    def _validar_views(
        self,
        P: np.ndarray,
        Q: np.ndarray,
        Omega: np.ndarray,
    ) -> None:
        k, n = P.shape
        if n != self.n:
            raise ValueError(
                f"P tem {n} colunas mas o modelo tem {self.n} ativos."
            )
        if Q.shape != (k,):
            raise ValueError(
                f"Q deve ter shape ({k},), recebido {Q.shape}."
            )
        if Omega.shape != (k, k):
            raise ValueError(
                f"Omega deve ter shape ({k},{k}), recebido {Omega.shape}."
            )
        if k == 0:
            raise ValueError("P/Q/Omega estão vazios (k=0 views).")
