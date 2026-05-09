"""Pipeline único: Economatica -> Kozak -> correlações Pearson -> análise de fatores."""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_EXCEL = ROOT / "AvantGarde" / "Economatica Ibra + Parâmetros Completos.xlsx"
DEFAULT_OUT = ROOT / "data" / "panel_output"; DEFAULT_CORR_OUT = DEFAULT_OUT / "correlation"
HEADER_ROW = 3; MAX_GAP_TRADING_DAYS = 60; TRADING_DAYS_PER_MONTH = 21
ROLLING_VOL_WINDOW = 60; MIN_OBS_FOR_VOL = 40
REPORTING_LAG_BUSINESS_DAYS = 60  # ~90 dias corridos: cobre prazo CVM (ITR 45d, DFP 90d)

CANONICAL = ("date close volume_brl mcap shares book_equity net_income revenue assets pas_cir pas_nocir cpv capex "
             "ebitda pe_ratio ev_ebit ev_ebitda div_yield roe_rep roa_rep gross_margin_rep net_margin_rep ebit_margin_rep").split()
FUNDAMENTAL_COLUMNS = CANONICAL[5:]
INDICATORS = ("val_book_to_market val_earnings_to_price val_sales_to_price val_ebitda_to_ev val_ebit_to_ev "
              "val_dividend_yield val_price_to_book val_pe_ratio val_price_to_sales val_ev_to_sales val_ebit_to_mcap "
              "mom_ret_1m mom_ret_3m mom_ret_6m mom_ret_12m mom_ret_12_1 mom_ret_6_1 mom_ret_3_1 "
              "mom_short_term_reversal mom_vol_adj_12_1 size_log_mcap size_mcap liq_avg_vol_21d liq_turnover "
              "liq_amihud_21d liq_trading_days_21d liq_vol_per_share liq_dollar_vol liq_relative_vol prof_roe prof_roa "
              "prof_gross_profitability prof_gross_margin prof_ebit_margin prof_net_margin prof_ebitda_margin "
              "prof_asset_turnover prof_ebit_to_assets prof_ebitda_to_assets inv_asset_growth_yoy inv_sales_growth_yoy "
              "inv_earnings_growth_yoy inv_equity_growth_yoy inv_ebitda_growth_yoy inv_capex_to_assets inv_capex_to_revenue "
              "lev_liab_to_equity lev_liab_to_assets lev_assets_to_equity risk_vol_60d risk_downside_vol_60d oth_skew_126d oth_kurt_126d").split()

def para_numero(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.mask(s.astype(str).str.strip().isin(["-", "", "nan", "NaN", "N/A", "—"])), errors="coerce")

def divisao_segura(n: pd.Series, d: pd.Series) -> pd.Series:
    n, d = n.astype(float), d.astype(float)
    r = n / d
    return r.where(np.isfinite(r) & (d != 0))

def transformar_grupo(s: pd.Series, chave: pd.Series, fn): return s.groupby(chave, sort=False).transform(fn)

def carregar_dados(path: Path, start: str, end: str, tickers: list[str] | None = None) -> pd.DataFrame:
    xls, filtro = pd.ExcelFile(path, engine="openpyxl"), set(tickers) if tickers else None
    blocos = []
    for sh in xls.sheet_names:
        tk = sh.strip()
        if filtro and tk not in filtro: continue
        bruto = pd.read_excel(path, sheet_name=sh, header=HEADER_ROW, engine="openpyxl", usecols=range(len(CANONICAL)))
        if bruto.shape[1] < len(CANONICAL): continue
        bruto = bruto.iloc[:, :len(CANONICAL)].copy(); bruto.columns = CANONICAL; bruto["ticker"] = tk; blocos.append(bruto)
    df = pd.concat(blocos, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    for c in CANONICAL[1:]: df[c] = para_numero(df[c])
    df = df.dropna(subset=["date"]).drop_duplicates(["ticker", "date"], keep="last")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]

def preparar_painel_ffill(df: pd.DataFrame) -> pd.DataFrame:
    all_dates, partes = pd.DatetimeIndex(sorted(df["date"].dropna().unique())), []
    for tk, sub in df.groupby("ticker", sort=False):
        r = sub.drop_duplicates("date", keep="last").set_index("date").sort_index().reindex(all_dates)
        r = r.rename_axis("date").reset_index(); r["ticker"] = tk
        close = r["close"].to_numpy(dtype=float); active = np.ones(len(close), dtype=bool)
        gap = 0; run_start = None; inact = None
        for i, v in enumerate(close):
            if np.isfinite(v): gap = 0; run_start = None
            else:
                if gap == 0: run_start = i
                gap += 1
                if gap >= MAX_GAP_TRADING_DAYS and inact is None and run_start is not None:
                    inact = run_start
        if inact is not None: active[inact:] = False
        r["is_active"], r["has_price"] = active, np.isfinite(close)
        for c in FUNDAMENTAL_COLUMNS:
            if c in r.columns: r[c] = r[c].shift(REPORTING_LAG_BUSINESS_DAYS).where(active).ffill().where(active)
        partes.append(r)
    painel = pd.concat(partes, ignore_index=True)
    painel["liabilities"] = painel["pas_cir"].astype(float) + painel["pas_nocir"].astype(float)
    ev = painel["ev_ebitda"].astype(float) * painel["ebitda"].astype(float); painel["ev"] = ev; ev_ebit = painel["ev_ebit"].astype(float)
    painel["ebit"] = np.where(np.isfinite(ev) & np.isfinite(ev_ebit) & (ev_ebit > 0), ev / ev_ebit, np.nan)
    painel["r_daily"] = transformar_grupo(painel["close"].astype(float), painel["ticker"], lambda s: s.pct_change(fill_method=None))
    return painel

def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["ticker", "date"]).copy()
    use = out["is_active"] & out["has_price"]
    tk, dt = out["ticker"], out["date"]
    pl,ll,rev,a,mc = (out[c].astype(float) for c in ("book_equity","net_income","revenue","assets","mcap"))
    ev,ebit,ebitda,cpv,liab = (out[c].astype(float) for c in ("ev","ebit","ebitda","cpv","liabilities"))
    v,sh,r = out["volume_brl"].astype(float), out["shares"].astype(float), out["r_daily"].astype(float)
    out["val_book_to_market"]=divisao_segura(pl,mc); out["val_earnings_to_price"]=divisao_segura(ll,mc); out["val_sales_to_price"]=divisao_segura(rev,mc)
    out["val_ebitda_to_ev"]=divisao_segura(ebitda,ev); out["val_ebit_to_ev"]=divisao_segura(ebit,ev); out["val_dividend_yield"]=out["div_yield"].astype(float)
    out["val_price_to_book"]=divisao_segura(mc,pl); out["val_pe_ratio"]=out["pe_ratio"].astype(float); out["val_price_to_sales"]=divisao_segura(mc,rev)
    out["val_ev_to_sales"]=divisao_segura(ev,rev); out["val_ebit_to_mcap"]=divisao_segura(ebit,mc)
    log1p = np.log1p(r.where(r > -1.0))
    def comp(w):
        mp = max(int(0.8 * w), 5)
        return np.expm1(transformar_grupo(log1p, tk, lambda s: s.rolling(w, min_periods=mp).sum()))
    w1,w3,w6,w12 = TRADING_DAYS_PER_MONTH, 3*TRADING_DAYS_PER_MONTH, 6*TRADING_DAYS_PER_MONTH, 12*TRADING_DAYS_PER_MONTH
    out["mom_ret_1m"], out["mom_ret_3m"], out["mom_ret_6m"], out["mom_ret_12m"] = comp(w1), comp(w3), comp(w6), comp(w12)
    out["mom_ret_12_1"]=(1+out["mom_ret_12m"])/(1+out["mom_ret_1m"])-1; out["mom_ret_6_1"]=(1+out["mom_ret_6m"])/(1+out["mom_ret_1m"])-1
    out["mom_ret_3_1"]=(1+out["mom_ret_3m"])/(1+out["mom_ret_1m"])-1; out["mom_short_term_reversal"]=np.where(out["mom_ret_1m"]<0,-out["mom_ret_1m"],0.0)
    sig = transformar_grupo(r, tk, lambda s: s.rolling(w12, min_periods=int(0.8*w12)).std(ddof=1)); out["mom_vol_adj_12_1"] = out["mom_ret_12_1"] / sig
    out["size_log_mcap"] = np.where(use, np.log(np.maximum(mc, 1e-12)), np.nan); out["size_mcap"] = np.where(use, mc, np.nan)
    out["liq_avg_vol_21d"] = transformar_grupo(v, tk, lambda s: s.rolling(21, min_periods=10).mean()); out["liq_turnover"] = divisao_segura(v, mc)
    amihud = r.abs() / v.replace(0, np.nan)
    out["liq_amihud_21d"] = transformar_grupo(amihud, tk, lambda s: s.rolling(21, min_periods=10).mean())
    out["liq_trading_days_21d"] = transformar_grupo(v, tk, lambda s: s.gt(0).rolling(21, min_periods=10).sum()); out["liq_vol_per_share"] = divisao_segura(v, sh)
    out["liq_dollar_vol"] = v; out["liq_relative_vol"] = divisao_segura(v, transformar_grupo(v, dt, lambda s: s.sum(min_count=1)))
    gp = rev - cpv
    out["prof_roe"]=divisao_segura(ll,pl); out["prof_roa"]=divisao_segura(ll,a); out["prof_gross_profitability"]=divisao_segura(gp,a); out["prof_gross_margin"]=divisao_segura(gp,rev)
    out["prof_ebit_margin"]=divisao_segura(ebit,rev); out["prof_net_margin"]=divisao_segura(ll,rev); out["prof_ebitda_margin"]=divisao_segura(ebitda,rev)
    out["prof_asset_turnover"]=divisao_segura(rev,a); out["prof_ebit_to_assets"]=divisao_segura(ebit,a); out["prof_ebitda_to_assets"]=divisao_segura(ebitda,a)
    yoy = lambda col: transformar_grupo(out[col].astype(float), tk, lambda x: x/x.shift(252)-1.0)
    out["inv_asset_growth_yoy"], out["inv_sales_growth_yoy"], out["inv_earnings_growth_yoy"] = yoy("assets"), yoy("revenue"), yoy("net_income")
    out["inv_equity_growth_yoy"], out["inv_ebitda_growth_yoy"] = yoy("book_equity"), yoy("ebitda")
    out["inv_capex_to_assets"] = divisao_segura(out["capex"].astype(float), a); out["inv_capex_to_revenue"] = divisao_segura(out["capex"].astype(float), rev)
    out["lev_liab_to_equity"]=divisao_segura(liab,pl); out["lev_liab_to_assets"]=divisao_segura(liab,a); out["lev_assets_to_equity"]=divisao_segura(a,pl)
    out["risk_vol_60d"] = transformar_grupo(r, tk, lambda s: s.rolling(ROLLING_VOL_WINDOW, min_periods=MIN_OBS_FOR_VOL).std(ddof=1))
    out["risk_downside_vol_60d"] = transformar_grupo(r.where(r<0, 0.0), tk, lambda s: s.rolling(ROLLING_VOL_WINDOW, min_periods=MIN_OBS_FOR_VOL).std(ddof=1))
    out["oth_skew_126d"] = transformar_grupo(r, tk, lambda s: s.rolling(126, min_periods=80).skew())
    out["oth_kurt_126d"] = transformar_grupo(r, tk, lambda s: s.rolling(126, min_periods=80).kurt())
    for c in INDICATORS: out.loc[~use, c] = np.nan
    return out

def aplicar_kozak(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    base, d, out = df[["ticker", "date"]].copy(), df["date"], {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        s = s.where(np.isfinite(s))
        gb = s.groupby(d, sort=False)
        rc = gb.rank(method="average", ascending=True) / (gb.transform("count").astype(float) + 1.0)
        ctr = rc - rc.groupby(d, sort=False).transform("mean"); den = ctr.abs().groupby(d, sort=False).transform("sum")
        out[f"z_{c}"] = (ctr/den).where(den > 0).fillna(0.0).astype(float)
    return pd.concat([base, pd.DataFrame(out, index=df.index)], axis=1)

def painel_frequencia(panel: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == "daily": return aplicar_kozak(panel[["ticker", "date"] + INDICATORS], INDICATORS)
    t = panel[["ticker", "date", "is_active", "has_price"] + INDICATORS].copy(); t["date"] = pd.to_datetime(t["date"]).dt.normalize(); t["month"] = t["date"].dt.to_period("M")
    m = t.loc[t.groupby(["ticker", "month"], sort=False)["date"].idxmax()].copy()
    base = m[["ticker", "date", "is_active", "has_price"] + INDICATORS].sort_values(["date", "ticker"]).reset_index(drop=True)
    return aplicar_kozak(base[["ticker", "date"] + INDICATORS], INDICATORS)

def selecionar_fatores(paineis: dict[str, pd.DataFrame], min_cs: int, min_cov: float) -> list[str]:
    z = sorted(set.intersection(*[set(c for c in df.columns if c.startswith("z_")) for df in paineis.values()]))
    def cobertura(df: pd.DataFrame) -> pd.Series:
        return (df.groupby("date")[z].apply(lambda x: x.notna().sum() >= min_cs)).mean()
    cov = pd.concat([cobertura(df) for df in paineis.values()], axis=1).min(axis=1); keep = sorted(cov[cov >= min_cov].index.tolist())
    print(f"  fatores comuns={len(z)} | apos cobertura>={min_cov:.0%}: {len(keep)}")
    return keep

def correlacao_ponderada_por_par(df: pd.DataFrame, z: list[str], min_cs: int, label: str) -> pd.DataFrame:
    p = len(z); soma = np.zeros((p, p), float); peso = np.zeros((p, p), float)
    idx_map, grupos, t0 = {c: i for i, c in enumerate(z)}, df.groupby("date", sort=True), time.time()
    for i, (_, g) in enumerate(grupos, 1):
        x = g[z].replace([np.inf, -np.inf], np.nan)
        cols = x.columns[(x.notna().sum() >= min_cs)].tolist()
        if len(cols) < 2: continue
        x = x[cols]
        corr = x.corr(method="pearson", min_periods=min_cs).to_numpy(); valid = x.notna().astype(np.int16).to_numpy()
        nij = valid.T @ valid; ok = ~np.isnan(corr)
        idx = [idx_map[c] for c in cols]; bloco = np.where(ok, corr * nij, 0.0); pesos = np.where(ok, nij, 0.0)
        for a, ia in enumerate(idx):
            soma[ia, idx] += bloco[a]; peso[ia, idx] += pesos[a]
        if i % 500 == 0 or i == len(grupos): print(f"    [{label}] {i}/{len(grupos)} | {time.time()-t0:.0f}s")
    out = np.divide(soma, peso, out=np.full_like(soma, np.nan), where=peso > 0)
    return pd.DataFrame(out, index=z, columns=z)

def salvar_correlacao_excel(corr: pd.DataFrame, out_dir: Path, filename: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True); path = out_dir / filename
    with pd.ExcelWriter(path, engine="openpyxl") as writer: corr.to_excel(writer, sheet_name="correlation", index=True)
    return path

def analisar_fatores(corr: pd.DataFrame, out_dir: Path, label: str = "monthly", variabilidade_alvo: float = 0.85) -> dict[str, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    out_dir.mkdir(parents=True, exist_ok=True)
    corr = corr.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    np.fill_diagonal(corr.values, 1.0)
    fatores = corr.index.astype(str).tolist()

    heatmap_path = out_dir / f"heatmap_correlacao_{label}.png"
    plt.figure(figsize=(18, 14))
    sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, center=0, square=True, xticklabels=fatores, yticklabels=fatores)
    plt.title(f"Heatmap de Correlação entre Fatores ({label})", fontsize=14, pad=15)
    plt.xticks(fontsize=6, rotation=90); plt.yticks(fontsize=6, rotation=0); plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight"); plt.close()

    cluster = linkage(corr, method="ward", metric="euclidean")
    dendrogram_path = out_dir / f"dendrograma_fatores_{label}.png"
    plt.figure(figsize=(16, 12))
    dendrogram(cluster, labels=fatores, leaf_rotation=90, leaf_font_size=10)
    plt.title(f"Dendrograma - Clusterização Hierárquica dos Fatores ({label})")
    plt.tight_layout(); plt.savefig(dendrogram_path, dpi=150, bbox_inches="tight"); plt.close()

    corr_std = StandardScaler().fit_transform(corr)
    pca_full = PCA().fit(corr_std)
    var = pca_full.explained_variance_ratio_
    var_ac = np.cumsum(var)
    n_comp = int(np.searchsorted(var_ac, variabilidade_alvo, side="left") + 1)
    n_comp = min(n_comp, len(var))

    scree_path = out_dir / f"scree_plot_{label}.png"
    plt.figure(figsize=(13, 8))
    plt.plot(range(1, len(var) + 1), var, marker="o", linestyle="--", label="Variabilidade por componente")
    plt.axvline(x=n_comp, color="red", linestyle=":", linewidth=1.5, label=f"{n_comp} componentes -> {var_ac[n_comp-1]*100:.1f}% acumulado")
    plt.title(f"Scree Plot / Método do Cotovelo em PCA ({label})", fontsize=13)
    plt.xlabel("Número de Componentes Principais"); plt.ylabel("Variabilidade Explicada")
    plt.xticks(range(1, len(var) + 1)); plt.legend(fontsize=9); plt.grid(alpha=0.4); plt.tight_layout()
    plt.savefig(scree_path, dpi=150, bbox_inches="tight"); plt.close()

    pca = PCA(n_components=n_comp).fit(corr_std)
    pc_cols = [f"PC{i+1}" for i in range(n_comp)]
    loadings = pd.DataFrame(pca.components_.T, index=fatores, columns=pc_cols)
    variance = pd.DataFrame({"component": pc_cols, "explained_variance": pca.explained_variance_ratio_, "cumulative_variance": var_ac[:n_comp]})
    pca_path = out_dir / f"pca_fatores_{label}.xlsx"
    with pd.ExcelWriter(pca_path, engine="openpyxl") as writer:
        variance.to_excel(writer, sheet_name="variance", index=False)
        loadings.to_excel(writer, sheet_name="loadings", index=True)

    print(f"  PCA {label}: {n_comp} PCs explicam {var_ac[n_comp-1]*100:.2f}%")
    return {"heatmap": heatmap_path, "dendrogram": dendrogram_path, "scree": scree_path, "pca": pca_path}

def main() -> None:
    ap = argparse.ArgumentParser(description="Economatica -> Kozak daily/monthly")
    ap.add_argument("--excel", type=Path, default=DEFAULT_EXCEL); ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--corr-out", type=Path, default=DEFAULT_CORR_OUT); ap.add_argument("--start", default="2010-01-01"); ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--tickers", default="", help="Ex.: VALE3,PETR4"); ap.add_argument("--frequency", choices=["daily", "monthly", "both"], default="both", help="quais outputs gerar")
    ap.add_argument("--min-cs", type=int, default=30, help="minimo de acoes validas por fator/periodo"); ap.add_argument("--min-factor-coverage", type=float, default=0.80, help="cobertura minima por fator (0-1)")
    a = ap.parse_args(); tickers = [t.strip() for t in a.tickers.split(",") if t.strip()] or None
    freqs = {"daily", "monthly"} if a.frequency == "both" else {a.frequency}
    a.out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    print("[1] Lendo Excel..."); panel = carregar_dados(a.excel, a.start, a.end, tickers)
    print("[2] Reindex + ffill..."); panel = preparar_painel_ffill(panel)
    print("[3] Indicadores..."); panel = calcular_indicadores(panel)
    paineis: dict[str, pd.DataFrame] = {}
    for label in ("daily", "monthly"):
        if label not in freqs: continue
        print(f"[4] Kozak {label}...")
        paineis[label] = painel_frequencia(panel, label)
        paineis[label].to_parquet(a.out / f"panel_kozak_{label}.parquet", index=False)
        if label == "monthly":
            paineis[label].to_excel(a.out / "panel_kozak_monthly.xlsx", sheet_name="monthly_z", index=False, engine="openpyxl")
    print("[6] Correlação Pearson...")
    z = selecionar_fatores(paineis, a.min_cs, a.min_factor_coverage)
    if len(z) < 2:
        raise RuntimeError("fatores insuficientes apos filtros")
    paths, corrs = {}, {}
    for label, df in paineis.items():
        corrs[label] = correlacao_ponderada_por_par(df, z, a.min_cs, label)
        paths[label] = salvar_correlacao_excel(corrs[label], a.corr_out, f"corr_{label}_pearson.xlsx")
    analise_label = "monthly" if "monthly" in corrs else next(iter(corrs))
    print(f"[7] Análise de fatores ({analise_label})...")
    analise_paths = analisar_fatores(corrs[analise_label], a.corr_out, analise_label)
    sizes = " | ".join(f"{label}={len(df):,}" for label, df in paineis.items()); print(f"OK em {time.time()-t0:.1f}s | {sizes} | fatores={len(z)}")
    if "monthly" in paineis: print(f"  panel monthly z: {a.out / 'panel_kozak_monthly.xlsx'}")
    for label, path in paths.items():
        print(f"  corr {label}: {path}")
    for name, path in analise_paths.items():
        print(f"  {name}: {path}")

if __name__ == "__main__":
    main()
