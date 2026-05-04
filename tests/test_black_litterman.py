"""Testes unitários para black_litterman.py."""
import numpy as np
import pandas as pd
import pytest

from src.black_litterman import BlackLitterman


@pytest.fixture
def dados_sinteticos():
    """Retornos e market caps sintéticos para 4 ativos."""
    np.random.seed(0)
    n_dias, n_ativos = 500, 4
    ativos = ["BTC", "ETH", "SOL", "ADA"]

    retornos = pd.DataFrame(
        np.random.multivariate_normal(
            mean=[0.001, 0.0008, 0.0006, 0.0004],
            cov=np.array([
                [0.0004, 0.0002, 0.00015, 0.0001],
                [0.0002, 0.0005, 0.00018, 0.00012],
                [0.00015, 0.00018, 0.0006, 0.00014],
                [0.0001, 0.00012, 0.00014, 0.0007],
            ]),
            size=n_dias,
        ),
        columns=ativos,
    )

    market_caps = pd.Series(
        [500e9, 200e9, 50e9, 20e9],
        index=ativos,
        name="market_cap_usd",
    )

    return retornos, market_caps


@pytest.fixture
def bl(dados_sinteticos):
    retornos, market_caps = dados_sinteticos
    return BlackLitterman(retornos, market_caps, risk_aversion=2.5, tau=0.05)


class TestPesosMercado:
    def test_soma_um(self, bl):
        w = bl.calcular_pesos_mercado()
        assert w.sum() == pytest.approx(1.0, abs=1e-9)

    def test_ordem_preservada(self, bl, dados_sinteticos):
        _, mc = dados_sinteticos
        w = bl.calcular_pesos_mercado()
        assert list(w.index) == list(mc.index)

    def test_maior_cap_tem_maior_peso(self, bl):
        w = bl.calcular_pesos_mercado()
        assert w["BTC"] > w["ETH"] > w["SOL"] > w["ADA"]


class TestRetornosImplicitos:
    def test_retorna_series_com_ativos_corretos(self, bl):
        pi = bl.calcular_retornos_implicitos()
        assert set(pi.index) == set(bl.ativos)

    def test_sem_views_otimizacao_aproxima_pesos_mercado(self, bl):
        """Propriedade fundamental do BL: sem views e sem limite por ativo, pesos ótimos = pesos de mercado."""
        # peso_maximo=1.0 remove a restrição de concentração para validar a propriedade pura
        resultado = bl.executar(peso_maximo=1.0)
        w_mkt   = resultado["pesos_mercado"]
        w_otimo = resultado["pesos_otimos"]

        for ativo in bl.ativos:
            assert abs(w_mkt[ativo] - w_otimo[ativo]) < 0.02, (
                f"{ativo}: mercado={w_mkt[ativo]:.3f}, ótimo={w_otimo[ativo]:.3f}"
            )


class TestCombinarViews:
    def test_formula_bayesiana_com_view_unica(self, bl):
        """Com uma view forte e certa (Omega→0), E[R] deve convergir para Q."""
        n = bl.n
        # View: BTC vai render 20% a.a.
        P = np.zeros((1, n))
        P[0, 0] = 1.0
        Q = np.array([0.20])
        Omega = np.array([[1e-10]])  # certeza quase total

        mu_bl = bl.combinar_views(P, Q, Omega)
        assert mu_bl["BTC"] == pytest.approx(0.20, abs=0.01)

    def test_dimensoes_incorretas_levantam_erro(self, bl):
        n = bl.n
        P_errada = np.zeros((2, n + 1))  # colunas erradas
        Q        = np.array([0.1, 0.1])
        Omega    = np.eye(2) * 0.01

        with pytest.raises(ValueError, match="colunas"):
            bl.combinar_views(P_errada, Q, Omega)

    def test_resultado_tem_todos_ativos(self, bl):
        n = bl.n
        P     = np.eye(1, n)
        Q     = np.array([0.10])
        Omega = np.eye(1) * 0.001

        mu_bl = bl.combinar_views(P, Q, Omega)
        assert set(mu_bl.index) == set(bl.ativos)


class TestOtimizacao:
    def test_pesos_somam_1(self, bl):
        resultado = bl.executar()
        assert resultado["pesos_otimos"].sum() == pytest.approx(1.0, abs=1e-6)

    def test_pesos_nao_negativos_por_padrao(self, bl):
        resultado = bl.executar()
        assert (resultado["pesos_otimos"] >= -1e-9).all()

    def test_peso_maximo_respeitado(self, bl):
        resultado = bl.executar(peso_maximo=0.30)
        assert resultado["pesos_otimos"].max() <= 0.30 + 1e-6


class TestValidacaoInputs:
    def test_market_cap_negativo_levanta_erro(self, dados_sinteticos):
        retornos, mc = dados_sinteticos
        mc_negativo = mc.copy()
        mc_negativo.iloc[0] = -1
        with pytest.raises(ValueError, match="não-positivos"):
            BlackLitterman(retornos, mc_negativo)

    def test_retornos_vazios_levanta_erro(self, dados_sinteticos):
        _, mc = dados_sinteticos
        with pytest.raises(ValueError, match="vazio"):
            BlackLitterman(pd.DataFrame(), mc)
