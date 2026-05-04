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
    estatisticas_portfolio,
    aplicar_pesos,
)

log = logging.getLogger(__name__)


class BlackLitterman:
    """Modelo Black-Litterman para otimização de portfólio de criptomoedas.

    Args:
        retornos:      DataFrame de log-retornos diários (linhas=datas, colunas=ativos).
        market_caps:   Series com market cap por ativo (mesmo índice que colunas de retornos).
        risk_aversion: λ — aversão ao risco do investidor representativo (típico: 2–4).
        tau:           Escala da incerteza nos retornos de equilíbrio (típico: 0.025–0.05).
    """

    def __init__(
        self,
        retornos: pd.DataFrame,
        market_caps: pd.Series,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
    ) -> None:
        self._validar_inputs(retornos, market_caps)

        # Alinha ativos comuns entre retornos e market_caps
        ativos_comuns = retornos.columns.intersection(market_caps.index)
        if len(ativos_comuns) == 0:
            raise ValueError("Nenhum ativo em comum entre retornos e market_caps.")
        if len(ativos_comuns) < len(retornos.columns):
            ausentes = set(retornos.columns) - set(ativos_comuns)
            log.warning(f"Ativos sem market cap, removidos: {ausentes}")

        self.retornos      = retornos[ativos_comuns].copy()
        self.market_caps   = market_caps[ativos_comuns].copy()
        self.risk_aversion = risk_aversion
        self.tau           = tau
        self.ativos        = list(ativos_comuns)
        self.n             = len(self.ativos)

        # Covariância anualizada — calculada uma vez e reutilizada
        self.cov = calcular_matriz_covariancia(self.retornos, anualizar=True)

        log.info(
            f"BlackLitterman inicializado: {self.n} ativos, "
            f"λ={risk_aversion}, τ={tau}"
        )

    # ────────────────────────────────────────────────────────
    # Etapa 1 — Pesos de mercado
    # ────────────────────────────────────────────────────────

    def calcular_pesos_mercado(self) -> pd.Series:
        """Pesos proporcionais ao market cap (carteira de equilíbrio).

        Returns:
            Series com pesos normalizados (soma = 1).
        """
        w = self.market_caps / self.market_caps.sum()
        log.debug(f"Pesos de mercado:\n{w.round(4)}")
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
        w_mkt = self.calcular_pesos_mercado()
        pi    = self.risk_aversion * self.cov.values @ w_mkt.values
        pi_series = pd.Series(pi, index=self.ativos, name="retornos_implicitos")
        log.debug(f"Retornos implícitos (Π):\n{pi_series.round(4)}")
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

        Args:
            P:     Matriz k×n de mapeamento views→ativos.
            Q:     Vetor k de retornos esperados das views.
            Omega: Matriz k×k diagonal de incerteza das views.

        Returns:
            Series com retornos esperados combinados por ativo.

        Raises:
            ValueError: Se dimensões de P, Q, Omega forem inconsistentes.
        """
        self._validar_views(P, Q, Omega)

        pi    = self.calcular_retornos_implicitos().values
        Sigma = self.cov.values
        tau   = self.tau

        tauSigma     = tau * Sigma
        tauSigma_inv = np.linalg.inv(tauSigma)
        Omega_inv    = np.linalg.inv(Omega)

        # Parte esquerda: [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹
        M_inv = np.linalg.inv(tauSigma_inv + P.T @ Omega_inv @ P)

        # Parte direita: (τΣ)⁻¹Π + P'Ω⁻¹Q
        rhs = tauSigma_inv @ pi + P.T @ Omega_inv @ Q

        mu_bl = M_inv @ rhs

        result = pd.Series(mu_bl, index=self.ativos, name="retornos_combinados")
        log.info(f"Retornos combinados (BL):\n{result.round(4)}")
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
                 w_i ≥ 0          (se não permitir short)
                 w_i ≤ peso_maximo

        Args:
            retornos_esperados: Series μ com retornos esperados por ativo.
            cov:                DataFrame Σ de covariância.
            permitir_short:     Se True, remove restrição de não-negatividade.
            peso_maximo:        Limite superior por ativo (0 < peso_maximo ≤ 1).

        Returns:
            Series com pesos ótimos normalizados.

        Raises:
            RuntimeError: Se a otimização não convergir.
        """
        mu    = retornos_esperados.values
        Sigma = cov.values
        lam   = self.risk_aversion
        n     = len(mu)

        def objetivo(w):
            return -(w @ mu - (lam / 2) * w @ Sigma @ w)

        restricoes = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

        lb = -1.0 if permitir_short else 0.0
        limites = [(lb, peso_maximo)] * n

        w0 = np.ones(n) / n  # chute inicial igualmente ponderado

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
                f"Otimização não convergiu: {resultado.message}"
            )

        pesos = pd.Series(resultado.x, index=self.ativos, name="pesos_otimos")
        pesos = pesos.clip(lower=0).div(pesos.clip(lower=0).sum())  # normaliza resíduos numéricos
        log.info(f"Pesos ótimos:\n{pesos.round(4)}")
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
        log.info("EXECUTANDO MODELO BLACK-LITTERMAN")
        log.info("=" * 55)

        # Etapas 1 e 2
        w_mkt = self.calcular_pesos_mercado()
        log.info(f"[1/4] Pesos de mercado calculados ({self.n} ativos)")

        pi = self.calcular_retornos_implicitos()
        log.info("[2/4] Retornos implícitos de equilíbrio calculados")

        # Etapa 3: combinar views ou usar equilíbrio puro
        tem_views = P is not None and Q is not None and Omega is not None
        if tem_views:
            mu_bl = self.combinar_views(P, Q, Omega)
            log.info(f"[3/4] Views combinadas ({len(Q)} view(s) ativa(s))")
        else:
            mu_bl = pi.copy()
            mu_bl.name = "retornos_combinados"
            log.info("[3/4] Sem views — usando retornos de equilíbrio")

        # Etapa 4: otimização
        pesos = self.otimizar(mu_bl, self.cov, **kwargs_otimizacao)
        log.info("[4/4] Otimização concluída")

        # Estatísticas do portfólio otimizado
        ret_portfolio = aplicar_pesos(self.retornos, pesos)
        stats = estatisticas_portfolio(ret_portfolio)

        log.info(f"Estatísticas: {stats}")
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
