"""Testes unitários para backtest.py e estrategias.py."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import WalkForwardBacktest, ResultadoBacktest
from src.estrategias import (
    EqualWeight,
    MarketCapWeight,
    BlackLittermanNeutro,
)


# ────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────

@pytest.fixture
def dados_backtest():
    """500 dias sintéticos com 4 ativos — suficiente para lookback=365."""
    np.random.seed(7)
    n_dias, n_ativos = 500, 4
    ativos = ["BTC", "ETH", "SOL", "ADA"]

    ret = pd.DataFrame(
        np.random.normal(0.001, 0.02, (n_dias, n_ativos)),
        index=pd.date_range("2023-01-01", periods=n_dias, freq="D"),
        columns=ativos,
    )
    prec = (1 + ret).cumprod() * 100  # preços sintéticos
    mc   = pd.Series([500e9, 200e9, 50e9, 20e9], index=ativos)
    return ret, prec, mc


@pytest.fixture
def backtest(dados_backtest):
    ret, prec, mc = dados_backtest
    return WalkForwardBacktest(
        retornos=ret,
        precos=prec,
        market_caps_atuais=mc,
        lookback_days=365,
        holding_period_days=7,
    )


# ────────────────────────────────────────────────────────────
# 1. Sem lookahead bias
# ────────────────────────────────────────────────────────────

class TestSemLookahead:
    def test_janela_treino_nao_inclui_data_rebal(self, backtest):
        """Crítico: data_rebal NÃO deve aparecer na janela de treino."""
        datas = backtest.gerar_datas_rebalanceamento()
        for data_t in datas[:10]:  # verifica as primeiras 10
            treino = backtest.janela_treino(data_t)
            assert data_t not in treino.index, (
                f"Lookahead: {data_t.date()} encontrado na janela de treino."
            )

    def test_todas_datas_treino_anteriores_a_rebal(self, backtest):
        """Todos os timestamps do treino devem ser < data_rebal."""
        datas = backtest.gerar_datas_rebalanceamento()
        for data_t in datas[:5]:
            treino = backtest.janela_treino(data_t)
            assert (treino.index < data_t).all(), (
                f"Treino contém dados >= {data_t.date()}."
            )


# ────────────────────────────────────────────────────────────
# 2. Pesos somam 1
# ────────────────────────────────────────────────────────────

class TestPesosSomamUm:
    def test_equal_weight_soma_um(self, backtest):
        resultado = backtest.executar_estrategia(EqualWeight())
        for data, row in resultado.historico_pesos.iterrows():
            assert row.sum() == pytest.approx(1.0, abs=1e-6), (
                f"Pesos não somam 1 em {data.date()}: {row.sum()}"
            )

    def test_market_cap_soma_um(self, dados_backtest, backtest):
        _, _, mc = dados_backtest
        resultado = backtest.executar_estrategia(MarketCapWeight(mc))
        for data, row in resultado.historico_pesos.iterrows():
            assert row.sum() == pytest.approx(1.0, abs=1e-6)


# ────────────────────────────────────────────────────────────
# 3. Pesos não-negativos
# ────────────────────────────────────────────────────────────

class TestPesosNaoNegativos:
    def test_equal_weight_nao_negativo(self, backtest):
        resultado = backtest.executar_estrategia(EqualWeight())
        assert (resultado.historico_pesos >= -1e-9).all().all()

    def test_market_cap_nao_negativo(self, dados_backtest, backtest):
        _, _, mc = dados_backtest
        resultado = backtest.executar_estrategia(MarketCapWeight(mc))
        assert (resultado.historico_pesos >= -1e-9).all().all()


# ────────────────────────────────────────────────────────────
# 4. Equal-Weight correto
# ────────────────────────────────────────────────────────────

class TestEqualWeight:
    def test_pesos_exatamente_1_sobre_n(self, backtest, dados_backtest):
        ret, _, _ = dados_backtest
        n = ret.shape[1]
        resultado = backtest.executar_estrategia(EqualWeight())

        for data, row in resultado.historico_pesos.iterrows():
            np.testing.assert_array_almost_equal(
                row.values,
                np.full(len(row), 1.0 / n),
                decimal=10,
            )


# ────────────────────────────────────────────────────────────
# 5. BL neutro aproxima market cap
# ────────────────────────────────────────────────────────────

class TestBLNeutroAproximaMarketCap:
    def test_correlacao_alta_com_market_cap(self, dados_backtest, backtest):
        """BL neutro deve produzir pesos com correlação > 0.8 com market cap."""
        _, _, mc = dados_backtest
        bl = BlackLittermanNeutro(market_caps=mc, peso_maximo=1.0)
        resultado = backtest.executar_estrategia(bl)

        w_mkt = mc / mc.sum()
        media_pesos = resultado.historico_pesos.mean()

        # Correlação entre pesos médios e pesos de mercado
        corr = np.corrcoef(
            media_pesos.reindex(w_mkt.index).fillna(0).values,
            w_mkt.values,
        )[0, 1]

        assert corr > 0.80, (
            f"Correlação BL Neutro vs Market Cap: {corr:.3f} (esperado > 0.80)"
        )


# ────────────────────────────────────────────────────────────
# 6. Consistência temporal das datas
# ────────────────────────────────────────────────────────────

class TestConsistenciaTemporal:
    def test_datas_crescentes(self, backtest):
        datas = backtest.gerar_datas_rebalanceamento()
        assert (datas[1:] > datas[:-1]).all(), "Datas não estão em ordem crescente."

    def test_holding_period_respeitado(self, backtest, dados_backtest):
        """Datas consecutivas devem estar exatamente holding_period_days à frente no índice."""
        ret, _, _ = dados_backtest
        datas = backtest.gerar_datas_rebalanceamento()

        for i in range(len(datas) - 1):
            idx_atual  = ret.index.get_loc(datas[i])
            idx_proximo = ret.index.get_loc(datas[i + 1])
            assert idx_proximo - idx_atual == backtest.holding_period, (
                f"Intervalo entre rebalanceamentos não é {backtest.holding_period} dias."
            )

    def test_primeiro_rebal_tem_lookback_completo(self, backtest):
        datas  = backtest.gerar_datas_rebalanceamento()
        treino = backtest.janela_treino(datas[0])
        assert len(treino) == backtest.lookback_days


# ────────────────────────────────────────────────────────────
# 7. Métricas do ResultadoBacktest
# ────────────────────────────────────────────────────────────

class TestMetricas:
    def test_metricas_retorna_dict_completo(self, backtest):
        resultado = backtest.executar_estrategia(EqualWeight())
        met = resultado.metricas()
        chaves_esperadas = {
            "retorno_total_%", "retorno_anualizado_%", "volatilidade_%",
            "sharpe", "sortino", "max_drawdown_%", "calmar", "turnover_medio",
        }
        assert chaves_esperadas.issubset(set(met.keys()))

    def test_retornos_acumulados_comeca_em_1(self, backtest):
        resultado = backtest.executar_estrategia(EqualWeight())
        eq = resultado.retornos_acumulados
        assert eq.iloc[0] == pytest.approx(1 + resultado.retornos_diarios.iloc[0], abs=1e-9)

    def test_turnover_nao_negativo(self, backtest, dados_backtest):
        _, _, mc = dados_backtest
        resultado = backtest.executar_estrategia(MarketCapWeight(mc))
        assert (resultado.turnover() >= 0).all()
