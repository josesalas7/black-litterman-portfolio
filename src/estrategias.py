"""
estrategias.py — Estratégias de alocação para backtest walk-forward.

Todas herdam de EstrategiaBase e implementam calcular_pesos().
Interface uniforme permite que WalkForwardBacktest as trate de forma
intercambiável.

Hierarquia:
    EstrategiaBase (ABC)
    ├── EqualWeight          — benchmark trivial
    ├── MarketCapWeight      — benchmark passivo
    ├── MarkowitzPuro        — média-variância clássico (mostra instabilidade)
    ├── BlackLittermanNeutro — BL sem views (sanity check)
    ├── BlackLittermanRSI    — BL + views absolutas via RSI(14)
    ├── BlackLittermanMomentum — BL + view relativa via momentum(30)
    └── BlackLittermanCombinado — BL + RSI + Momentum combinados
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import scipy.linalg
from scipy.optimize import minimize

from src.black_litterman import BlackLitterman
from src.portfolio_utils import (
    calcular_matriz_covariancia_ledoitwolf,
    DIAS_ANO_CRIPTO,
)
from src.views import gerar_views_rsi, gerar_views_momentum, gerar_views_var

log = logging.getLogger(__name__)


class EstrategiaBase(ABC):
    """Interface comum a todas as estratégias de alocação."""

    nome: str

    @abstractmethod
    def calcular_pesos(
        self,
        retornos_treino: pd.DataFrame,
        precos_treino: pd.DataFrame,
    ) -> pd.Series:
        """Calcula pesos a partir da janela de treino.

        Args:
            retornos_treino: Log-retornos diários do período de lookback.
            precos_treino: Preços de fechamento do período de lookback.

        Returns:
            Series com pesos por ativo (soma = 1, sem short, sem NaN).
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nome='{self.nome}')"


class EqualWeight(EstrategiaBase):
    """1/N — pesos iguais para todos os ativos.

    Benchmark trivial. Surpreendentemente robusto fora da amostra
    em muitos estudos empíricos (DeMiguel et al., 2009).
    """

    nome = "Equal-Weight"

    def calcular_pesos(
        self,
        retornos_treino: pd.DataFrame,
        precos_treino: pd.DataFrame,
    ) -> pd.Series:
        n = retornos_treino.shape[1]
        log.info("[%s] Calculando pesos: 1/%d por ativo.", self.nome, n)
        pesos = pd.Series(1.0 / n, index=retornos_treino.columns)
        log.debug("[%s] Pesos: %s", self.nome, dict(pesos.round(4)))
        return pesos


class MarketCapWeight(EstrategiaBase):
    """Pesos proporcionais ao market cap.

    Benchmark passivo natural — representa o que um investidor indexado
    ao universo de criptomoedas obteria. Usa market caps estáticos
    (proxy do equilíbrio atual).

    Args:
        market_caps: Series com market cap por ativo.
    """

    nome = "Market-Cap"

    def __init__(self, market_caps: pd.Series) -> None:
        self.market_caps = market_caps

    def calcular_pesos(
        self,
        retornos_treino: pd.DataFrame,
        precos_treino: pd.DataFrame,
    ) -> pd.Series:
        log.info("[%s] Calculando pesos por market cap...", self.nome)
        mc = self.market_caps.reindex(retornos_treino.columns).dropna()
        if mc.empty:
            raise ValueError("market_caps não tem ativos em comum com retornos_treino.")
        pesos = mc / mc.sum()
        log.debug("[%s] Pesos: %s", self.nome, dict(pesos.round(4)))
        return pesos


class MarkowitzPuro(EstrategiaBase):
    """Média-variância clássico com retornos históricos.

    Usa a média histórica diretamente como estimativa de μ — abordagem
    instável, propensa a concentração extrema e sensível a outliers.
    Incluído para demonstrar o problema que o Black-Litterman resolve.

    Args:
        risk_aversion: λ na função objetivo max w'μ - (λ/2)w'Σw.
        peso_maximo: Limite superior por ativo.
    """

    nome = "Markowitz Puro"

    def __init__(
        self,
        risk_aversion: float = 2.5,
        peso_maximo: float = 0.40,
    ) -> None:
        self.risk_aversion = risk_aversion
        self.peso_maximo   = peso_maximo

    def calcular_pesos(
        self,
        retornos_treino: pd.DataFrame,
        precos_treino: pd.DataFrame,
    ) -> pd.Series:
        n_obs, n_ativos = retornos_treino.shape
        log.info(
            "[%s] Calculando pesos (MV clássico, %d ativos, %d obs, lambda=%.2f)...",
            self.nome, n_ativos, n_obs, self.risk_aversion,
        )

        mu  = retornos_treino.mean() * DIAS_ANO_CRIPTO
        cov = calcular_matriz_covariancia_ledoitwolf(retornos_treino).values
        n   = len(mu)
        lam = self.risk_aversion

        log.debug(
            "[%s] Retornos esperados mu (anualizados): %s",
            self.nome, dict(mu.round(4)),
        )

        def objetivo(w: np.ndarray) -> float:
            return -(w @ mu.values - (lam / 2) * w @ cov @ w)

        resultado = minimize(
            objetivo,
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=[(0.0, self.peso_maximo)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        if not resultado.success:
            log.warning("[%s] Otimizacao nao convergiu — usando equal-weight.", self.nome)
            return pd.Series(1.0 / n, index=retornos_treino.columns)

        pesos = pd.Series(resultado.x, index=retornos_treino.columns)
        pesos = pesos.clip(lower=0) / pesos.clip(lower=0).sum()

        log.debug("[%s] Pesos otimos: %s", self.nome, dict(pesos.round(4)))
        log.info("[%s] Otimizacao concluida em %d iteracoes.", self.nome, resultado.nit)
        return pesos


class BlackLittermanNeutro(EstrategiaBase):
    """Black-Litterman sem views (equilíbrio puro).

    Sanity check: sem views, a otimização deve recuperar pesos próximos
    ao market cap (propriedade fundamental do modelo).

    Args:
        market_caps: Series com market cap por ativo.
        risk_aversion: λ da função objetivo.
        tau: Parâmetro de escala da incerteza no equilíbrio.
        peso_maximo: Limite superior por ativo.
    """

    nome = "BL Neutro"

    def __init__(
        self,
        market_caps: pd.Series,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        peso_maximo: float = 0.40,
    ) -> None:
        self.market_caps   = market_caps
        self.risk_aversion = risk_aversion
        self.tau           = tau
        self.peso_maximo   = peso_maximo

    def calcular_pesos(
        self,
        retornos_treino: pd.DataFrame,
        precos_treino: pd.DataFrame,
    ) -> pd.Series:
        log.info("[%s] Calculando pesos (sem views — equilíbrio puro)...", self.nome)
        mc  = self.market_caps.reindex(retornos_treino.columns).dropna()
        ret = retornos_treino[mc.index]

        bl    = BlackLitterman(ret, mc, self.risk_aversion, self.tau)
        pesos = bl.executar(peso_maximo=self.peso_maximo)["pesos_otimos"]
        log.info("[%s] Pesos calculados.", self.nome)
        return pesos


class BlackLittermanRSI(EstrategiaBase):
    """Black-Litterman com views absolutas geradas via RSI(14).

    RSI < threshold_compra → view positiva (+retorno_view anual)
    RSI > threshold_venda  → view negativa (-retorno_view anual)
    Sem sinal              → fallback para BL Neutro

    Args:
        market_caps: Series com market cap por ativo.
        periodo_rsi: Período do RSI (padrão 14).
        threshold_compra: RSI abaixo deste valor gera view positiva.
        threshold_venda: RSI acima deste valor gera view negativa.
        retorno_view: Magnitude da view em retorno anual.
        risk_aversion: λ da função objetivo.
        tau: Parâmetro de escala.
        peso_maximo: Limite superior por ativo.
    """

    nome = "BL + RSI"

    def __init__(
        self,
        market_caps: pd.Series,
        periodo_rsi: int = 14,
        threshold_compra: float = 30.0,
        threshold_venda: float = 70.0,
        retorno_view: float = 0.05,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        peso_maximo: float = 0.40,
    ) -> None:
        self.market_caps       = market_caps
        self.periodo_rsi       = periodo_rsi
        self.threshold_compra  = threshold_compra
        self.threshold_venda   = threshold_venda
        self.retorno_view      = retorno_view
        self.risk_aversion     = risk_aversion
        self.tau               = tau
        self.peso_maximo       = peso_maximo

    def calcular_pesos(
        self,
        retornos_treino: pd.DataFrame,
        precos_treino: pd.DataFrame,
    ) -> pd.Series:
        log.info("[%s] Calculando pesos com views RSI...", self.nome)
        mc   = self.market_caps.reindex(retornos_treino.columns).dropna()
        ret  = retornos_treino[mc.index]
        prec = precos_treino[mc.index]

        bl       = BlackLitterman(ret, mc, self.risk_aversion, self.tau)
        data_ref = prec.index[-1]

        try:
            P, Q, Omega = gerar_views_rsi(
                precos=prec,
                data_referencia=data_ref,
                threshold_compra=self.threshold_compra,
                threshold_venda=self.threshold_venda,
                retorno_esperado_view=self.retorno_view,
                tau=self.tau,
            )
            log.info("[%s] %d view(s) RSI ativa(s) em %s.", self.nome, len(Q), data_ref.date())
            pesos = bl.executar(P=P, Q=Q, Omega=Omega, peso_maximo=self.peso_maximo)["pesos_otimos"]

        except ValueError:
            log.info("[%s] Sem views RSI em %s — usando equilíbrio.", self.nome, data_ref.date())
            pesos = bl.executar(peso_maximo=self.peso_maximo)["pesos_otimos"]

        log.info("[%s] Pesos calculados.", self.nome)
        return pesos


class BlackLittermanMomentum(EstrategiaBase):
    """Black-Litterman com view relativa baseada em momentum (30 dias).

    View: top_n ativos vão render spread_esperado mais que bottom_n.
    Sempre gera exatamente 1 view (nunca cai em fallback).

    Args:
        market_caps: Series com market cap por ativo.
        top_n: Ativos com maior momentum (lado long).
        bottom_n: Ativos com menor momentum (lado short da view).
        spread_esperado: Spread anual esperado entre top e bottom.
        periodo_momentum: Janela de lookback do momentum.
        risk_aversion: λ da função objetivo.
        tau: Parâmetro de escala.
        peso_maximo: Limite superior por ativo.
    """

    nome = "BL + Momentum"

    def __init__(
        self,
        market_caps: pd.Series,
        top_n: int = 3,
        bottom_n: int = 3,
        spread_esperado: float = 0.10,
        periodo_momentum: int = 30,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        peso_maximo: float = 0.40,
    ) -> None:
        self.market_caps      = market_caps
        self.top_n            = top_n
        self.bottom_n         = bottom_n
        self.spread_esperado  = spread_esperado
        self.periodo_momentum = periodo_momentum
        self.risk_aversion    = risk_aversion
        self.tau              = tau
        self.peso_maximo      = peso_maximo

    def calcular_pesos(
        self,
        retornos_treino: pd.DataFrame,
        precos_treino: pd.DataFrame,
    ) -> pd.Series:
        log.info("[%s] Calculando pesos com view de momentum...", self.nome)
        mc   = self.market_caps.reindex(retornos_treino.columns).dropna()
        ret  = retornos_treino[mc.index]
        prec = precos_treino[mc.index]

        bl       = BlackLitterman(ret, mc, self.risk_aversion, self.tau)
        data_ref = prec.index[-1]

        P, Q, Omega = gerar_views_momentum(
            precos=prec,
            data_referencia=data_ref,
            top_n=self.top_n,
            bottom_n=self.bottom_n,
            spread_esperado=self.spread_esperado,
            periodo_momentum=self.periodo_momentum,
            tau=self.tau,
        )
        pesos = bl.executar(P=P, Q=Q, Omega=Omega, peso_maximo=self.peso_maximo)["pesos_otimos"]
        log.info("[%s] Pesos calculados.", self.nome)
        return pesos


class BlackLittermanCombinado(EstrategiaBase):
    """Black-Litterman com views de RSI e Momentum combinadas.

    Empilha as matrizes P, vetores Q e diagonais de Omega das duas fontes:
        P_final = [P_rsi; P_mom]
        Q_final = [Q_rsi, Q_mom]
        Omega_final = block_diag(Omega_rsi, Omega_mom)

    Se o RSI não gerar views, usa apenas o Momentum.

    Args:
        market_caps: Series com market cap por ativo.
        periodo_rsi: Período do RSI.
        threshold_compra: Threshold inferior do RSI.
        threshold_venda: Threshold superior do RSI.
        retorno_view_rsi: Magnitude da view RSI.
        top_n: Ativos long no momentum.
        bottom_n: Ativos short no momentum.
        spread_momentum: Spread anual esperado da view de momentum.
        periodo_momentum: Janela de lookback do momentum.
        risk_aversion: λ da função objetivo.
        tau: Parâmetro de escala.
        peso_maximo: Limite superior por ativo.
    """

    nome = "BL + RSI + Momentum"

    def __init__(
        self,
        market_caps: pd.Series,
        periodo_rsi: int = 14,
        threshold_compra: float = 30.0,
        threshold_venda: float = 70.0,
        retorno_view_rsi: float = 0.05,
        top_n: int = 3,
        bottom_n: int = 3,
        spread_momentum: float = 0.10,
        periodo_momentum: int = 30,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        peso_maximo: float = 0.40,
    ) -> None:
        self.market_caps      = market_caps
        self.periodo_rsi      = periodo_rsi
        self.threshold_compra = threshold_compra
        self.threshold_venda  = threshold_venda
        self.retorno_view_rsi = retorno_view_rsi
        self.top_n            = top_n
        self.bottom_n         = bottom_n
        self.spread_momentum  = spread_momentum
        self.periodo_momentum = periodo_momentum
        self.risk_aversion    = risk_aversion
        self.tau              = tau
        self.peso_maximo      = peso_maximo

    def calcular_pesos(
        self,
        retornos_treino: pd.DataFrame,
        precos_treino: pd.DataFrame,
    ) -> pd.Series:
        log.info("[%s] Calculando pesos com views RSI + Momentum...", self.nome)
        mc   = self.market_caps.reindex(retornos_treino.columns).dropna()
        ret  = retornos_treino[mc.index]
        prec = precos_treino[mc.index]

        bl       = BlackLitterman(ret, mc, self.risk_aversion, self.tau)
        data_ref = prec.index[-1]

        # Views de momentum (sempre disponível)
        P_mom, Q_mom, Om_mom = gerar_views_momentum(
            precos=prec,
            data_referencia=data_ref,
            top_n=self.top_n,
            bottom_n=self.bottom_n,
            spread_esperado=self.spread_momentum,
            periodo_momentum=self.periodo_momentum,
            tau=self.tau,
        )

        P_final  = P_mom
        Q_final  = Q_mom
        Om_final = Om_mom
        n_views_rsi = 0

        # Views de RSI (opcionais — combina se disponível)
        try:
            P_rsi, Q_rsi, Om_rsi = gerar_views_rsi(
                precos=prec,
                data_referencia=data_ref,
                threshold_compra=self.threshold_compra,
                threshold_venda=self.threshold_venda,
                retorno_esperado_view=self.retorno_view_rsi,
                tau=self.tau,
            )
            P_final   = np.vstack([P_rsi, P_mom])
            Q_final   = np.concatenate([Q_rsi, Q_mom])
            Om_final  = scipy.linalg.block_diag(Om_rsi, Om_mom)
            n_views_rsi = len(Q_rsi)

        except ValueError:
            log.info("[%s] Sem views RSI em %s — usando só momentum.", self.nome, data_ref.date())

        log.info(
            "[%s] Total de views: %d RSI + 1 momentum = %d.",
            self.nome, n_views_rsi, n_views_rsi + 1,
        )
        log.debug(
            "[%s] Q combinado: %s",
            self.nome, Q_final.round(4).tolist(),
        )

        pesos = bl.executar(
            P=P_final, Q=Q_final, Omega=Om_final, peso_maximo=self.peso_maximo
        )["pesos_otimos"]
        log.info("[%s] Pesos calculados.", self.nome)
        return pesos


class BlackLittermanVAR(EstrategiaBase):
    """Black-Litterman com views absolutas geradas via VAR(1) nos log-retornos.

    Estima VAR(1) em janela móvel dos retornos de treino, faz forecast 1 passo
    à frente para cada ativo e injeta como views absolutas no modelo BL.
    P = identidade, Q = retornos anualizados previstos, Omega = Idzorek.

    Fallback automático para AR(1) por ativo se VAR(1) for singular/instável;
    fallback final para BL Neutro se ambos falharem.

    Args:
        market_caps: Series com market cap por ativo.
        janela_var: Observações recentes usadas no VAR (padrão 90).
        usar_mse_omega: Se True, usa MSE dos resíduos como Omega.
        risk_aversion: λ da função objetivo.
        tau: Parâmetro de escala para Omega e equilíbrio BL.
        peso_maximo: Limite superior por ativo.
    """

    nome = "BL-VAR"

    def __init__(
        self,
        market_caps: pd.Series,
        janela_var: int = 90,
        usar_mse_omega: bool = False,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        peso_maximo: float = 0.40,
    ) -> None:
        self.market_caps    = market_caps
        self.janela_var     = janela_var
        self.usar_mse_omega = usar_mse_omega
        self.risk_aversion  = risk_aversion
        self.tau            = tau
        self.peso_maximo    = peso_maximo

    def calcular_pesos(
        self,
        retornos_treino: pd.DataFrame,
        precos_treino: pd.DataFrame,
    ) -> pd.Series:
        log.info("[%s] Calculando pesos com views VAR(1)...", self.nome)
        mc  = self.market_caps.reindex(retornos_treino.columns).dropna()
        ret = retornos_treino[mc.index]

        bl       = BlackLitterman(ret, mc, self.risk_aversion, self.tau)
        data_ref = ret.index[-1]

        try:
            P, Q, Omega = gerar_views_var(
                retornos=ret,
                data_referencia=data_ref,
                janela=self.janela_var,
                tau=self.tau,
                usar_mse_omega=self.usar_mse_omega,
            )
            log.info(
                "[%s] %d views VAR geradas para %s.", self.nome, len(Q), data_ref.date(),
            )
            pesos = bl.executar(P=P, Q=Q, Omega=Omega, peso_maximo=self.peso_maximo)["pesos_otimos"]

        except ValueError as exc:
            log.warning(
                "[%s] VAR falhou (%s) — usando equilíbrio BL Neutro.", self.nome, exc,
            )
            pesos = bl.executar(peso_maximo=self.peso_maximo)["pesos_otimos"]

        log.info("[%s] Pesos calculados.", self.nome)
        return pesos
