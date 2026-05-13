"""Testes unitários para dashboard/utils_dash.py."""
import numpy as np
import pandas as pd
import pytest

from dashboard.utils_dash import (
    construir_view_absoluta,
    idzorek_omega,
    mini_backtest_view,
    tabela_metricas_comparativa,
    plot_pesos_comparacao,
    plot_mini_backtest,
)
from src.config import DIAS_ANO_CRIPTO


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

UNIVERSO_TEST = ["BTC", "ETH", "SOL", "ADA"]


@pytest.fixture
def sigma_diagonal():
    """Covariância diagonal simples (vol anual ≈ 63%/ativo)."""
    return np.diag([0.40, 0.35, 0.50, 0.45])


@pytest.fixture
def retornos_sinteticos():
    """200 dias de retornos diários para 4 ativos."""
    np.random.seed(42)
    idx  = pd.date_range("2024-01-01", periods=200, freq="D")
    data = np.random.normal(0.001, 0.025, (200, 4))
    return pd.DataFrame(data, index=idx, columns=UNIVERSO_TEST)


@pytest.fixture
def pesos_iguais():
    return pd.Series(0.25, index=UNIVERSO_TEST)


# ─────────────────────────────────────────────────────────────
# idzorek_omega
# ─────────────────────────────────────────────────────────────

class TestIdzorekOmega:
    def test_retorna_matriz_quadrada(self, sigma_diagonal):
        P = np.array([[1.0, 0.0, 0.0, 0.0]])
        c = np.array([0.5])
        Om = idzorek_omega(P, sigma_diagonal, c)
        assert Om.shape == (1, 1)

    def test_confianca_alta_gera_omega_pequeno(self, sigma_diagonal):
        P = np.array([[1.0, 0.0, 0.0, 0.0]])
        Om_alta = idzorek_omega(P, sigma_diagonal, np.array([0.99]))
        Om_baixa = idzorek_omega(P, sigma_diagonal, np.array([0.01]))
        assert Om_alta[0, 0] < Om_baixa[0, 0]

    def test_confianca_50_retorna_valor_finito(self, sigma_diagonal):
        P = np.array([[1.0, 0.0, 0.0, 0.0]])
        Om = idzorek_omega(P, sigma_diagonal, np.array([0.5]))
        assert np.isfinite(Om[0, 0])
        assert Om[0, 0] > 0

    def test_multiple_views(self, sigma_diagonal):
        P = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])
        c = np.array([0.7, 0.3])
        Om = idzorek_omega(P, sigma_diagonal, c)
        assert Om.shape == (2, 2)
        assert Om[0, 1] == pytest.approx(0.0)  # diagonal → off-diag = 0

    def test_confianca_zero_levanta_erro(self, sigma_diagonal):
        P = np.array([[1.0, 0.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="\\(0, 1\\) exclusivo"):
            idzorek_omega(P, sigma_diagonal, np.array([0.0]))

    def test_confianca_um_levanta_erro(self, sigma_diagonal):
        P = np.array([[1.0, 0.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="\\(0, 1\\) exclusivo"):
            idzorek_omega(P, sigma_diagonal, np.array([1.0]))

    def test_shape_sigma_incompativel_levanta_erro(self):
        P = np.array([[1.0, 0.0, 0.0]])  # 3 ativos
        Sigma_errada = np.eye(4)          # 4 ativos
        with pytest.raises(ValueError, match="shape"):
            idzorek_omega(P, Sigma_errada, np.array([0.5]))

    def test_tau_afeta_omega(self, sigma_diagonal):
        P = np.array([[1.0, 0.0, 0.0, 0.0]])
        c = np.array([0.5])
        Om1 = idzorek_omega(P, sigma_diagonal, c, tau=0.05)
        Om2 = idzorek_omega(P, sigma_diagonal, c, tau=0.10)
        assert Om2[0, 0] == pytest.approx(Om1[0, 0] * 2, rel=1e-9)


# ─────────────────────────────────────────────────────────────
# construir_view_absoluta
# ─────────────────────────────────────────────────────────────

class TestConstruirViewAbsoluta:
    def test_p_shape_correto(self):
        P, Q = construir_view_absoluta("BTC", 5.0, 30, UNIVERSO_TEST)
        assert P.shape == (1, 4)
        assert Q.shape == (1,)

    def test_p_tem_1_na_posicao_correta(self):
        P, _ = construir_view_absoluta("ETH", 5.0, 30, UNIVERSO_TEST)
        assert P[0, UNIVERSO_TEST.index("ETH")] == 1.0
        assert P[0, UNIVERSO_TEST.index("BTC")] == 0.0

    def test_q_e_anual_positivo(self):
        _, Q = construir_view_absoluta("BTC", 5.0, 30, UNIVERSO_TEST)
        # +5% em 30 dias → retorno anual > +5%
        assert Q[0] > 0.05

    def test_q_negativo_para_view_baixista(self):
        _, Q = construir_view_absoluta("BTC", -5.0, 30, UNIVERSO_TEST)
        assert Q[0] < 0

    def test_conversao_1_ano_preserva_magnitude(self):
        """View de 5% em 365 dias → Q anual ≈ 5%."""
        _, Q = construir_view_absoluta("BTC", 5.0, 365, UNIVERSO_TEST)
        assert Q[0] == pytest.approx(0.05, rel=1e-6)

    def test_ativo_ausente_levanta_erro(self):
        with pytest.raises(ValueError, match="não encontrado"):
            construir_view_absoluta("DOGE", 5.0, 30, UNIVERSO_TEST)

    def test_horizonte_zero_levanta_erro(self):
        with pytest.raises(ValueError, match="horizonte_dias"):
            construir_view_absoluta("BTC", 5.0, 0, UNIVERSO_TEST)

    def test_horizonte_1_dia(self):
        _, Q = construir_view_absoluta("BTC", 2.0, 1, UNIVERSO_TEST)
        # 2%/dia composto por 365 dias → retorno anual muito alto
        esperado = (1.02) ** DIAS_ANO_CRIPTO - 1
        assert Q[0] == pytest.approx(esperado, rel=1e-9)


# ─────────────────────────────────────────────────────────────
# mini_backtest_view
# ─────────────────────────────────────────────────────────────

class TestMiniBacktestView:
    def test_retorna_dataframe_com_3_colunas(self, retornos_sinteticos, pesos_iguais):
        equity = mini_backtest_view(retornos_sinteticos, pesos_iguais, pesos_iguais)
        assert set(equity.columns) == {"BL_com_view", "Sem_view", "EW"}

    def test_equity_comeca_proximo_de_100(self, retornos_sinteticos, pesos_iguais):
        equity = mini_backtest_view(retornos_sinteticos, pesos_iguais, pesos_iguais, dias=90)
        # Primeiro valor = (1 + ret_dia1) * 100 ≠ exatamente 100
        # Mas deve estar entre 90 e 110 para dados sintéticos
        assert 80 < equity["BL_com_view"].iloc[0] < 120

    def test_n_linhas_igual_a_dias(self, retornos_sinteticos, pesos_iguais):
        dias = 60
        equity = mini_backtest_view(retornos_sinteticos, pesos_iguais, pesos_iguais, dias=dias)
        assert len(equity) == dias

    def test_pesos_iguais_bl_e_mkt_produzem_mesma_curva(self, retornos_sinteticos, pesos_iguais):
        equity = mini_backtest_view(retornos_sinteticos, pesos_iguais, pesos_iguais)
        pd.testing.assert_series_equal(
            equity["BL_com_view"], equity["Sem_view"], check_names=False
        )

    def test_dias_zero_levanta_erro(self, retornos_sinteticos, pesos_iguais):
        with pytest.raises(ValueError, match="dias"):
            mini_backtest_view(retornos_sinteticos, pesos_iguais, pesos_iguais, dias=0)

    def test_sem_ativos_comuns_levanta_erro(self, retornos_sinteticos):
        w_diferente = pd.Series(0.5, index=["DOGE", "XMR"])
        with pytest.raises(ValueError, match="Nenhum ativo"):
            mini_backtest_view(retornos_sinteticos, w_diferente, w_diferente)

    def test_pesos_fora_do_universo_sao_ignorados(self, retornos_sinteticos):
        """Pesos extras que não estão nos retornos devem ser descartados sem crash."""
        w_extra = pd.Series({"BTC": 0.5, "ETH": 0.3, "DOGE": 0.2})
        equity = mini_backtest_view(retornos_sinteticos, w_extra, w_extra)
        assert not equity.empty


# ─────────────────────────────────────────────────────────────
# tabela_metricas_comparativa
# ─────────────────────────────────────────────────────────────

class TestTabelaMetricas:
    @pytest.fixture
    def equity_simples(self):
        np.random.seed(1)
        idx = pd.date_range("2024-01-01", periods=90, freq="D")
        data = np.cumprod(1 + np.random.normal(0.001, 0.02, (90, 2)), axis=0) * 100
        return pd.DataFrame(data, index=idx, columns=["BL_com_view", "Sem_view"])

    def test_retorna_dataframe_com_linhas_corretas(self, equity_simples):
        df = tabela_metricas_comparativa(equity_simples)
        assert "BL com view" in df.index
        assert "Benchmark (sem view)" in df.index

    def test_colunas_obrigatorias(self, equity_simples):
        df = tabela_metricas_comparativa(equity_simples)
        for col in ("Retorno total (%)", "Sharpe", "Max DD (%)"):
            assert col in df.columns

    def test_max_dd_negativo_ou_zero(self, equity_simples):
        df = tabela_metricas_comparativa(equity_simples)
        for row in df.index:
            assert df.loc[row, "Max DD (%)"] <= 0


# ─────────────────────────────────────────────────────────────
# Gráficos Plotly (smoke tests)
# ─────────────────────────────────────────────────────────────

class TestGraficos:
    def test_plot_pesos_retorna_figura(self):
        import plotly.graph_objects as go

        w = pd.Series({"BTC": 0.5, "ETH": 0.3, "SOL": 0.2})
        fig = plot_pesos_comparacao(w, w)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # duas barras: benchmark e BL

    def test_plot_mini_backtest_retorna_figura(self, retornos_sinteticos, pesos_iguais):
        import plotly.graph_objects as go

        equity = mini_backtest_view(retornos_sinteticos, pesos_iguais, pesos_iguais)
        fig = plot_mini_backtest(equity)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3  # três linhas: BL, Bench, EW
