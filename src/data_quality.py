"""
data_quality.py — Validação dos dados coletados.

Checagens essenciais antes de usar os dados no modelo:
1. Gaps temporais (dias faltando)
2. Valores nulos
3. Outliers extremos (possíveis erros de dados)
4. Estatísticas descritivas anualizadas
5. Resumo de market cap
6. Relatório de disponibilidade por ativo (data inicial, n_dias, NaNs)
"""
import logging
import pandas as pd
import numpy as np

from src.config import (
    ARQUIVO_PRECOS, ARQUIVO_RETORNOS, ARQUIVO_RELATORIO_QUALIDADE,
    ARQUIVO_MARKET_CAP, DIAS_ANO_CRIPTO,
)

log = logging.getLogger(__name__)


def verificar_gaps(df: pd.DataFrame) -> dict:
    """Verifica dias faltando na série. Cripto opera 24/7, esperado sem gaps."""
    data_inicio = df.index.min()
    data_fim    = df.index.max()
    dias_esperados = pd.date_range(start=data_inicio, end=data_fim, freq="D")

    gaps_por_ativo = {}
    for col in df.columns:
        serie = df[col].dropna()
        dias_presentes = pd.DatetimeIndex(serie.index.normalize().unique())
        dias_faltando  = dias_esperados.difference(dias_presentes)
        gaps_por_ativo[col] = len(dias_faltando)

    return gaps_por_ativo


def verificar_nulos(df: pd.DataFrame) -> pd.Series:
    """Conta valores nulos por ativo."""
    return df.isna().sum()


def detectar_outliers(retornos: pd.DataFrame, n_desvios: float = 5.0) -> dict:
    """Identifica retornos fora de N desvios padrão (5sigma para cripto).

    Outliers podem ser erros de dados (preço quebrado) ou eventos extremos
    (crashes, hacks de exchange, etc.).
    """
    outliers_por_ativo = {}
    for col in retornos.columns:
        serie = retornos[col].dropna()
        if len(serie) == 0:
            outliers_por_ativo[col] = 0
            continue
        media  = serie.mean()
        desvio = serie.std()
        outliers = serie[(serie - media).abs() > n_desvios * desvio]
        outliers_por_ativo[col] = len(outliers)
    return outliers_por_ativo


def estatisticas_basicas(retornos: pd.DataFrame) -> pd.DataFrame:
    """Estatísticas descritivas anualizadas (base DIAS_ANO_CRIPTO = 365, cripto 24/7)."""
    stats = pd.DataFrame({
        "obs":             retornos.count(),
        "retorno_anual_%": retornos.mean() * DIAS_ANO_CRIPTO * 100,
        "vol_anual_%":     retornos.std() * np.sqrt(DIAS_ANO_CRIPTO) * 100,
        "sharpe_aprox":    (retornos.mean() * DIAS_ANO_CRIPTO) / (retornos.std() * np.sqrt(DIAS_ANO_CRIPTO)),
        "min_diario_%":    retornos.min() * 100,
        "max_diario_%":    retornos.max() * 100,
    })
    return stats.round(2)


def relatorio_disponibilidade(precos: pd.DataFrame) -> pd.DataFrame:
    """Relatório de disponibilidade de dados por ativo.

    Para cada ativo: data inicial, data final, n_dias com dados, n_NaN.
    Útil para identificar ativos com histórico curto (ex: PENDLE, ONDO)
    que podem causar exclusão dinâmica em backtests com lookback longo.

    Args:
        precos: DataFrame de preços diários (linhas=datas, colunas=tickers).

    Returns:
        DataFrame ordenado por data_inicio com colunas:
        data_inicio, data_fim, n_dias, n_nan, pct_nan, dias_ate_hoje.

    Raises:
        ValueError: Se precos estiver vazio.
    """
    if precos.empty:
        raise ValueError("precos está vazio.")

    hoje = pd.Timestamp.now().normalize()
    registros = []

    for ativo in precos.columns:
        serie = precos[ativo]
        nao_nulos = serie.dropna()

        if nao_nulos.empty:
            data_inicio = pd.NaT
            data_fim = pd.NaT
            n_dias = 0
        else:
            data_inicio = nao_nulos.index.min()
            data_fim    = nao_nulos.index.max()
            n_dias      = len(nao_nulos)

        n_nan     = int(serie.isna().sum())
        n_total   = len(serie)
        pct_nan   = round(100 * n_nan / n_total, 2) if n_total > 0 else 0.0
        dias_ate_hoje = (hoje - data_inicio).days if pd.notna(data_inicio) else np.nan

        registros.append({
            "ativo":         ativo,
            "data_inicio":   data_inicio,
            "data_fim":      data_fim,
            "n_dias":        n_dias,
            "n_nan":         n_nan,
            "pct_nan":       pct_nan,
            "dias_ate_hoje": int(dias_ate_hoje) if not np.isnan(dias_ate_hoje) else np.nan,
        })

    df = (
        pd.DataFrame(registros)
        .set_index("ativo")
        .sort_values("data_inicio")
    )

    log.info("Relatório de disponibilidade gerado para %d ativos.", len(df))
    ativos_curtos = df[df["n_dias"] < 365]
    if not ativos_curtos.empty:
        log.warning(
            "Ativos com menos de 365 dias de histórico: %s",
            ativos_curtos.index.tolist(),
        )

    return df


def gerar_relatorio() -> str:
    """Executa todas as checagens e retorna relatório textual."""
    log.info("Carregando dados para validacao...")
    precos   = pd.read_parquet(ARQUIVO_PRECOS)
    retornos = pd.read_parquet(ARQUIVO_RETORNOS)

    if precos.empty:
        raise RuntimeError("Arquivo de precos vazio. Rode a coleta primeiro.")

    linhas = []
    linhas.append("=" * 70)
    linhas.append("RELATORIO DE QUALIDADE DOS DADOS")
    linhas.append("=" * 70)

    linhas.append("\nCOBERTURA TEMPORAL")
    linhas.append(f"   Inicio: {precos.index.min().date()}")
    linhas.append(f"   Fim:    {precos.index.max().date()}")
    linhas.append(f"   Total de dias:   {len(precos)}")
    linhas.append(f"   Total de ativos: {precos.shape[1]}")

    linhas.append("\nGAPS TEMPORAIS (dias faltando por ativo)")
    gaps = verificar_gaps(precos)
    for ativo, n_gaps in gaps.items():
        status = "OK " if n_gaps == 0 else "WARN"
        linhas.append(f"   [{status}] {ativo}: {n_gaps} dias faltando")

    linhas.append("\nVALORES NULOS")
    nulos = verificar_nulos(precos)
    for ativo, n_nulos in nulos.items():
        status = "OK " if n_nulos == 0 else "WARN"
        linhas.append(f"   [{status}] {ativo}: {n_nulos} nulos")

    linhas.append("\nOUTLIERS (retornos > 5 sigma)")
    outliers = detectar_outliers(retornos)
    for ativo, n_out in outliers.items():
        linhas.append(f"   {ativo}: {n_out} dias")

    linhas.append("\nESTATISTICAS BASICAS (anualizadas)")
    stats = estatisticas_basicas(retornos)
    linhas.append(stats.to_string())

    linhas.append("\nMARKET CAP E PESOS DE MERCADO")
    if not ARQUIVO_MARKET_CAP.exists():
        linhas.append("   [WARN] market_cap.parquet nao encontrado. Rode fetch_market_cap primeiro.")
    else:
        mc = pd.read_parquet(ARQUIVO_MARKET_CAP)
        market_cap_total = mc["market_cap_usd"].sum()
        linhas.append(f"   Market cap total do universo: ${market_cap_total/1e9:.2f}B")
        linhas.append(f"   Atualizado em: {mc['atualizado_em'].iloc[0][:10]}\n")
        preview = mc[["ticker", "nome", "market_cap_usd", "peso_mercado"]].copy()
        preview["market_cap_usd"] = preview["market_cap_usd"].apply(lambda x: f"${x/1e9:.2f}B")
        preview["peso_mercado"]   = preview["peso_mercado"].apply(lambda x: f"{x*100:.2f}%")
        linhas.append(preview.to_string(index=False))

    linhas.append("\n" + "=" * 70)

    return "\n".join(linhas)


def main():
    relatorio = gerar_relatorio()
    log.info("\n%s", relatorio)

    with open(ARQUIVO_RELATORIO_QUALIDADE, "w", encoding="utf-8") as f:
        f.write(relatorio)

    log.info("Relatorio salvo em: %s", ARQUIVO_RELATORIO_QUALIDADE)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
