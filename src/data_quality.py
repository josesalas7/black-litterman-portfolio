"""
data_quality.py — Validação dos dados coletados.

Por que isso importa? Dados ruins → modelo ruim → portfólio ruim.
Esse script faz checagens essenciais antes de você confiar nos dados:

1. Verificar gaps temporais (dias faltando)
2. Detectar valores nulos ou zero
3. Procurar outliers extremos (possíveis erros)
4. Validar consistência de tamanho entre ativos
5. Gerar relatório resumo
"""
import logging
import pandas as pd
import numpy as np

from src.config import ARQUIVO_PRECOS, ARQUIVO_RETORNOS, ARQUIVO_RELATORIO_QUALIDADE, ARQUIVO_MARKET_CAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def verificar_gaps(df: pd.DataFrame) -> dict:
    """
    Verifica se há dias faltando na série.

    Cripto opera 24/7 (sem fim de semana), então deveria ter
    dados todos os dias do período.
    """
    data_inicio = df.index.min()
    data_fim = df.index.max()
    dias_esperados = pd.date_range(start=data_inicio, end=data_fim, freq="D")

    gaps_por_ativo = {}
    for col in df.columns:
        serie = df[col].dropna()
        dias_presentes = pd.DatetimeIndex(serie.index.normalize().unique())
        dias_faltando = dias_esperados.difference(dias_presentes)
        gaps_por_ativo[col] = len(dias_faltando)

    return gaps_por_ativo


def verificar_nulos(df: pd.DataFrame) -> pd.Series:
    """Conta valores nulos por ativo."""
    return df.isna().sum()


def detectar_outliers(retornos: pd.DataFrame, n_desvios: float = 5.0) -> dict:
    """
    Identifica retornos fora de N desvios padrão.

    Cripto é volátil, então 5σ é razoável (vs ações onde 3σ é alarmante).
    Outliers além disso podem ser:
    - Erros de dados (preço quebrado)
    - Eventos extremos (LUNA crash, FTX, etc.)
    """
    outliers_por_ativo = {}
    for col in retornos.columns:
        serie = retornos[col].dropna()
        if len(serie) == 0:
            outliers_por_ativo[col] = 0
            continue
        media = serie.mean()
        desvio = serie.std()
        outliers = serie[(serie - media).abs() > n_desvios * desvio]
        outliers_por_ativo[col] = len(outliers)
    return outliers_por_ativo


def estatisticas_basicas(retornos: pd.DataFrame) -> pd.DataFrame:
    """Calcula estatísticas descritivas anualizadas."""
    # Anualiza assumindo 365 dias de trading (cripto)
    fator_anual = 365

    stats = pd.DataFrame({
        "obs": retornos.count(),
        "retorno_anual_%": retornos.mean() * fator_anual * 100,
        "vol_anual_%": retornos.std() * np.sqrt(fator_anual) * 100,
        "sharpe_aprox": (retornos.mean() * fator_anual) / (retornos.std() * np.sqrt(fator_anual)),
        "min_diario_%": retornos.min() * 100,
        "max_diario_%": retornos.max() * 100,
    })
    return stats.round(2)


def gerar_relatorio() -> str:
    """Roda todas as checagens e monta um relatório textual."""
    log.info("Carregando dados...")
    precos = pd.read_parquet(ARQUIVO_PRECOS)
    retornos = pd.read_parquet(ARQUIVO_RETORNOS)

    if precos.empty:
        raise RuntimeError("Arquivo de preços está vazio. Rode a coleta primeiro.")

    relatorio = []
    relatorio.append("=" * 70)
    relatorio.append("RELATÓRIO DE QUALIDADE DOS DADOS")
    relatorio.append("=" * 70)

    # --- 1. Cobertura temporal ---
    relatorio.append("\n📅 COBERTURA TEMPORAL")
    relatorio.append(f"   Início: {precos.index.min().date()}")
    relatorio.append(f"   Fim:    {precos.index.max().date()}")
    relatorio.append(f"   Total de dias: {len(precos)}")
    relatorio.append(f"   Total de ativos: {precos.shape[1]}")

    # --- 2. Gaps ---
    relatorio.append("\n🕳️  GAPS TEMPORAIS (dias faltando por ativo)")
    gaps = verificar_gaps(precos)
    for ativo, n_gaps in gaps.items():
        status = "✅" if n_gaps == 0 else "⚠️ "
        relatorio.append(f"   {status} {ativo}: {n_gaps} dias faltando")

    # --- 3. Nulos ---
    relatorio.append("\n❓ VALORES NULOS")
    nulos = verificar_nulos(precos)
    for ativo, n_nulos in nulos.items():
        status = "✅" if n_nulos == 0 else "⚠️ "
        relatorio.append(f"   {status} {ativo}: {n_nulos} nulos")

    # --- 4. Outliers ---
    relatorio.append("\n🎯 OUTLIERS (retornos > 5σ)")
    outliers = detectar_outliers(retornos)
    for ativo, n_out in outliers.items():
        relatorio.append(f"   • {ativo}: {n_out} dias")

    # --- 5. Estatísticas ---
    relatorio.append("\n📊 ESTATÍSTICAS BÁSICAS (anualizadas)")
    stats = estatisticas_basicas(retornos)
    relatorio.append(stats.to_string())

    # --- 6. Market Cap ---
    relatorio.append("\n💰 MARKET CAP E PESOS DE MERCADO")
    if not ARQUIVO_MARKET_CAP.exists():
        relatorio.append("   ⚠️  Arquivo market_cap.parquet não encontrado. Rode fetch_market_cap primeiro.")
    else:
        mc = pd.read_parquet(ARQUIVO_MARKET_CAP)
        market_cap_total = mc["market_cap_usd"].sum()
        relatorio.append(f"   Market cap total do universo: ${market_cap_total/1e9:.2f}B")
        relatorio.append(f"   Atualizado em: {mc['atualizado_em'].iloc[0][:10]}\n")
        preview = mc[["ticker", "nome", "market_cap_usd", "peso_mercado"]].copy()
        preview["market_cap_usd"] = preview["market_cap_usd"].apply(lambda x: f"${x/1e9:.2f}B")
        preview["peso_mercado"] = preview["peso_mercado"].apply(lambda x: f"{x*100:.2f}%")
        relatorio.append(preview.to_string(index=False))

    relatorio.append("\n" + "=" * 70)

    texto = "\n".join(relatorio)
    return texto


def main():
    relatorio = gerar_relatorio()
    print(relatorio)

    # Salva em arquivo
    with open(ARQUIVO_RELATORIO_QUALIDADE, "w", encoding="utf-8") as f:
        f.write(relatorio)

    log.info(f"\n💾 Relatório salvo em: {ARQUIVO_RELATORIO_QUALIDADE}")


if __name__ == "__main__":
    main()
