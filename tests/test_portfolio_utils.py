"""Testes unitários para portfolio_utils.py."""
import numpy as np
import pandas as pd
import pytest

from src.portfolio_utils import (
    calcular_matriz_covariancia,
    calcular_volatilidades,
    calcular_sharpe,
    calcular_drawdown_maximo,
    estatisticas_portfolio,
    aplicar_pesos,
    DIAS_ANO_CRIPTO,
)


@pytest.fixture
def retornos_simples():
    """DataFrame de retornos diários sintéticos (3 ativos, 100 dias)."""
    np.random.seed(42)
    dados = np.random.normal(0.001, 0.02, size=(100, 3))
    return pd.DataFrame(dados, columns=["BTC", "ETH", "SOL"])


@pytest.fixture
def pesos_iguais():
    return pd.Series({"BTC": 1/3, "ETH": 1/3, "SOL": 1/3})


class TestMatrizCovariancia:
    def test_shape_correto(self, retornos_simples):
        cov = calcular_matriz_covariancia(retornos_simples)
        assert cov.shape == (3, 3)

    def test_simetrica(self, retornos_simples):
        cov = calcular_matriz_covariancia(retornos_simples)
        np.testing.assert_array_almost_equal(cov.values, cov.values.T)

    def test_anualizar_multiplica_por_365(self, retornos_simples):
        cov_diaria  = calcular_matriz_covariancia(retornos_simples, anualizar=False)
        cov_anual   = calcular_matriz_covariancia(retornos_simples, anualizar=True)
        np.testing.assert_array_almost_equal(
            cov_anual.values, cov_diaria.values * DIAS_ANO_CRIPTO
        )

    def test_vazio_levanta_erro(self):
        with pytest.raises(ValueError, match="vazio"):
            calcular_matriz_covariancia(pd.DataFrame())


class TestAplicarPesos:
    def test_retorno_correto(self, retornos_simples, pesos_iguais):
        ret_portfolio = aplicar_pesos(retornos_simples, pesos_iguais)
        esperado = retornos_simples.mean(axis=1)
        pd.testing.assert_series_equal(ret_portfolio, esperado, check_names=False)

    def test_pesos_nao_somam_1_levanta_erro(self, retornos_simples):
        pesos_errados = pd.Series({"BTC": 0.5, "ETH": 0.5, "SOL": 0.5})
        with pytest.raises(ValueError, match="somar 1"):
            aplicar_pesos(retornos_simples, pesos_errados)

    def test_pesos_somam_1(self, retornos_simples, pesos_iguais):
        ret = aplicar_pesos(retornos_simples, pesos_iguais)
        assert len(ret) == len(retornos_simples)

    def test_ativo_ausente_levanta_erro(self, retornos_simples):
        pesos_errados = pd.Series({"BTC": 0.5, "ETH": 0.5, "INEXISTENTE": 0.0})
        with pytest.raises(ValueError, match="Ativos em pesos não encontrados"):
            aplicar_pesos(retornos_simples, pesos_errados)


class TestSharpe:
    def test_sharpe_positivo_para_retornos_positivos(self):
        np.random.seed(1)
        retornos = pd.Series(np.random.normal(0.003, 0.02, 365))
        sharpe = calcular_sharpe(retornos)
        assert sharpe > 0

    def test_vol_zero_levanta_erro(self):
        retornos = pd.Series([0.001] * 100)
        with pytest.raises(ValueError, match="Volatilidade zero"):
            calcular_sharpe(retornos)


class TestDrawdown:
    def test_sem_queda_drawdown_zero(self):
        retornos = pd.Series([0.01] * 50)
        dd = calcular_drawdown_maximo(retornos)
        assert dd == pytest.approx(0.0, abs=1e-6)

    def test_queda_total_drawdown_negativo(self):
        retornos = pd.Series([0.1, -0.5, 0.01])
        dd = calcular_drawdown_maximo(retornos)
        assert dd < 0
