"""Testes unitários para data_quality.py."""
import numpy as np
import pandas as pd
import pytest

from src.data_quality import (
    verificar_gaps,
    verificar_nulos,
    detectar_outliers,
    estatisticas_basicas,
    relatorio_disponibilidade,
)


@pytest.fixture
def precos_completos():
    """Preços diários sem NaN para 3 ativos, 400 dias."""
    np.random.seed(0)
    idx = pd.date_range("2023-01-01", periods=400, freq="D")
    dados = np.abs(np.random.normal(100, 10, (400, 3)))
    return pd.DataFrame(dados, index=idx, columns=["BTC", "ETH", "SOL"])


@pytest.fixture
def precos_com_nans(precos_completos):
    """Preços onde ONDO só tem dados a partir do dia 200."""
    df = precos_completos.copy()
    df["ONDO"] = np.nan
    df.loc[df.index[200]:, "ONDO"] = np.abs(np.random.normal(1, 0.1, 200))
    return df


class TestRelatioDisponibilidade:
    def test_retorna_dataframe_com_colunas_corretas(self, precos_completos):
        df = relatorio_disponibilidade(precos_completos)
        colunas = {"data_inicio", "data_fim", "n_dias", "n_nan", "pct_nan", "dias_ate_hoje"}
        assert colunas.issubset(set(df.columns))

    def test_index_sao_os_ativos(self, precos_completos):
        df = relatorio_disponibilidade(precos_completos)
        assert set(df.index) == set(precos_completos.columns)

    def test_sem_nans_retorna_pct_zero(self, precos_completos):
        df = relatorio_disponibilidade(precos_completos)
        assert (df["pct_nan"] == 0.0).all()

    def test_ativo_curto_detectado(self, precos_com_nans):
        df = relatorio_disponibilidade(precos_com_nans)
        assert df.loc["ONDO", "n_dias"] < 365

    def test_ordenado_por_data_inicio(self, precos_com_nans):
        df = relatorio_disponibilidade(precos_com_nans)
        datas = df["data_inicio"].dropna()
        assert (datas.values[:-1] <= datas.values[1:]).all()

    def test_vazio_levanta_erro(self):
        with pytest.raises(ValueError, match="vazio"):
            relatorio_disponibilidade(pd.DataFrame())

    def test_n_dias_consistente_com_dropna(self, precos_com_nans):
        df = relatorio_disponibilidade(precos_com_nans)
        for ativo in precos_com_nans.columns:
            esperado = int(precos_com_nans[ativo].notna().sum())
            assert df.loc[ativo, "n_dias"] == esperado


class TestEstatisticasBasicas:
    def test_usa_365_para_anualizar(self):
        np.random.seed(1)
        retornos = pd.DataFrame(
            np.random.normal(0.001, 0.02, (365, 2)),
            columns=["A", "B"],
        )
        stats = estatisticas_basicas(retornos)
        media_A = retornos["A"].mean()
        # A função arredonda para 2 casas; comparamos com tolerância equivalente
        assert stats.loc["A", "retorno_anual_%"] == pytest.approx(
            round(media_A * 365 * 100, 2), abs=0.005
        )

    def test_colunas_obrigatorias(self):
        retornos = pd.DataFrame(
            {"X": [0.001, -0.002, 0.003]},
        )
        stats = estatisticas_basicas(retornos)
        for col in ("obs", "retorno_anual_%", "vol_anual_%", "sharpe_aprox"):
            assert col in stats.columns
