# -*- coding: utf-8 -*-
"""
Gera tabela cross-sectional mensal a partir do panel_indicators diário.

Regra: para cada (ticker, mês), mantém a última observação disponível no mês.
Saída principal: panel_cross_sectional_monthly.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_OUTPUT_DIR, PROJECT_ROOT
from src.indicators.compute_indicators import INDICATOR_COLUMNS

DEFAULT_IND = PROJECT_ROOT / "data" / "panel_output" / "panel_indicators.parquet"
DEFAULT_OUT = PROJECT_ROOT / "data" / "panel_output" / "panel_cross_sectional_monthly.parquet"
def build_cross_sectional_monthly(df_ind: pd.DataFrame) -> pd.DataFrame:
    df = df_ind.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["ticker", "date"])
    df["month"] = df["date"].dt.to_period("M")

    # Último pregão disponível por ticker em cada mês.
    idx = df.groupby(["ticker", "month"], sort=False)["date"].idxmax()
    monthly = df.loc[idx].copy()
    monthly = monthly.sort_values(["date", "ticker"]).reset_index(drop=True)
    monthly["date_month"] = monthly["date"].dt.to_period("M").dt.to_timestamp("M")
    monthly["year_month"] = monthly["date_month"].dt.strftime("%Y-%m")

    keep = ["date_month", "year_month", "date", "ticker"] + INDICATOR_COLUMNS
    return monthly[keep]


def export_cross_sectional_monthly(df_monthly: pd.DataFrame, out_path: Path, out_csv: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_monthly.to_parquet(out_path, index=False)
    if out_csv:
        csv_path = out_path.with_suffix(".csv")
        df_monthly.to_csv(csv_path, index=False)

    meta = {
        "rows": int(len(df_monthly)),
        "months": int(df_monthly["date_month"].nunique()) if not df_monthly.empty else 0,
        "tickers": int(df_monthly["ticker"].nunique()) if not df_monthly.empty else 0,
        "indicator_columns": INDICATOR_COLUMNS,
        "n_indicators": len(INDICATOR_COLUMNS),
        "export_file": str(out_path.name),
    }
    meta_path = out_path.with_name(f"{out_path.stem}_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Build cross-sectional monthly panel from panel_indicators")
    p.add_argument("--indicators", type=Path, default=DEFAULT_IND, help="Input panel_indicators.parquet")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output Parquet path")
    p.add_argument("--tickers", type=str, default="", help="Opcional: filtrar tickers (VALE3,PETR4)")
    p.add_argument("--csv", action="store_true", help="Também exporta CSV")
    p.add_argument("--min-features", type=int, default=1, help="Mínimo de fatores não nulos por linha")
    args = p.parse_args()

    if not args.indicators.is_file():
        raise FileNotFoundError(f"Parquet de indicadores não encontrado: {args.indicators}")

    df_ind = pd.read_parquet(args.indicators)
    required = {"ticker", "date", *INDICATOR_COLUMNS}
    missing = required.difference(df_ind.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes no painel: {sorted(missing)}")

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if tickers:
        df_ind = df_ind[df_ind["ticker"].isin(tickers)].copy()

    df_monthly = build_cross_sectional_monthly(df_ind)

    feature_notna = df_monthly[INDICATOR_COLUMNS].notna().sum(axis=1)
    df_monthly = df_monthly.loc[feature_notna >= max(args.min_features, 1)].copy()
    df_monthly = df_monthly.reset_index(drop=True)

    export_cross_sectional_monthly(df_monthly, args.out, args.csv)
    print(f"OK: {args.out}")
    print(
        f"Rows: {len(df_monthly)}, months: {df_monthly['date_month'].nunique()}, "
        f"tickers: {df_monthly['ticker'].nunique()}"
    )


if __name__ == "__main__":
    main()
