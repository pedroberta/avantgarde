# -*- coding: utf-8 -*-
"""
Gera painéis cross-section com rank transform + normalização L1:
- diário
- mensal (último pregão de cada mês por ticker)

Permite rodar um ou ambos os modos para comparação de cobertura e rigor.
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

from src.config import PROJECT_ROOT
from src.cross_section_rank_l1 import apply_rank_and_l1_by_date
from src.indicators.compute_indicators import INDICATOR_COLUMNS

DEFAULT_IND = PROJECT_ROOT / "data" / "panel_output" / "panel_indicators.parquet"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "panel_output"


def _build_monthly_base(df_ind: pd.DataFrame) -> pd.DataFrame:
    df = df_ind.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["ticker", "date"])
    df["month"] = df["date"].dt.to_period("M")
    idx = df.groupby(["ticker", "month"], sort=False)["date"].idxmax()
    monthly = df.loc[idx].copy()
    monthly["source_date"] = monthly["date"]
    monthly["date"] = monthly["month"].dt.to_timestamp("M")
    keep = ["ticker", "date", "source_date"] + INDICATOR_COLUMNS
    return monthly[keep].sort_values(["date", "ticker"]).reset_index(drop=True)


def _compute_diagnostics(df_out: pd.DataFrame, features: list[str]) -> dict:
    per_feature: dict[str, dict] = {}
    for c in features:
        zc = f"z_{c}"
        if zc not in df_out.columns:
            continue
        s = pd.to_numeric(df_out[zc], errors="coerce")
        by_date_sum = s.groupby(df_out["date"], sort=False).sum(min_count=1)
        by_date_l1 = s.abs().groupby(df_out["date"], sort=False).sum(min_count=1)

        sum_abs_valid = by_date_sum.abs().dropna()
        l1_diff_valid = (by_date_l1 - 1.0).abs().dropna()
        per_feature[c] = {
            "non_null_share": float(s.notna().mean()),
            "dates_with_signal": int(by_date_sum.notna().sum()),
            "max_abs_cross_sectional_sum_z": (
                float(sum_abs_valid.max()) if not sum_abs_valid.empty else float("nan")
            ),
            "median_abs_cross_sectional_sum_z": (
                float(sum_abs_valid.median()) if not sum_abs_valid.empty else float("nan")
            ),
            "max_abs_l1_minus_1": float(l1_diff_valid.max()) if not l1_diff_valid.empty else float("nan"),
            "median_abs_l1_minus_1": (
                float(l1_diff_valid.median()) if not l1_diff_valid.empty else float("nan")
            ),
        }

    return {
        "rows": int(len(df_out)),
        "dates": int(pd.to_datetime(df_out["date"]).nunique()) if not df_out.empty else 0,
        "tickers": int(df_out["ticker"].nunique()) if not df_out.empty else 0,
        "n_features": len(features),
        "n_z_columns": int(len([c for c in df_out.columns if c.startswith("z_")])),
        "n_rc_columns": int(len([c for c in df_out.columns if c.startswith("rc_")])),
        "per_feature": per_feature,
    }


def _save_outputs(df_out: pd.DataFrame, out_path: Path, include_rc: bool, source: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, index=False)

    diag = _compute_diagnostics(df_out, INDICATOR_COLUMNS)
    meta = {
        "source_panel": source,
        "pipeline": "rank_average_rc_over_n_plus_1_then_l1_normalization",
        "include_rc": include_rc,
        "rows": diag["rows"],
        "dates": diag["dates"],
        "tickers": diag["tickers"],
        "n_features": diag["n_features"],
        "n_z_columns": diag["n_z_columns"],
        "n_rc_columns": diag["n_rc_columns"],
        "feature_columns": INDICATOR_COLUMNS,
    }

    meta_path = out_path.with_name(f"{out_path.stem}_metadata.json")
    qa_path = out_path.with_name(f"{out_path.stem}_qa.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2, ensure_ascii=False)

    # Também salva um resumo tabular por característica para leitura rápida.
    rows = []
    for feat, m in diag["per_feature"].items():
        rows.append({"feature": feat, **m})
    pd.DataFrame(rows).to_csv(out_path.with_name(f"{out_path.stem}_qa_by_feature.csv"), index=False)


def _transform_and_export(
    base_df: pd.DataFrame,
    out_path: Path,
    include_rc: bool,
    source_name: str,
) -> None:
    z_df = apply_rank_and_l1_by_date(base_df, INDICATOR_COLUMNS, date_col="date")
    if "source_date" in base_df.columns:
        z_df = z_df.merge(
            base_df[["ticker", "date", "source_date"]],
            on=["ticker", "date"],
            how="left",
        )
    if not include_rc:
        keep = [c for c in z_df.columns if c in {"ticker", "date", "source_date"} or c.startswith("z_")]
        z_df = z_df[keep].copy()
    _save_outputs(z_df, out_path=out_path, include_rc=include_rc, source=source_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build daily/monthly rank+L1 cross-sectional panels")
    parser.add_argument("--indicators", type=Path, default=DEFAULT_IND, help="Input panel_indicators.parquet")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["daily", "monthly", "both"],
        default="both",
        help="Frequência do experimento",
    )
    parser.add_argument("--include-rc", action="store_true", help="Exporta rc_* além de z_*")
    parser.add_argument("--tickers", type=str, default="", help="Opcional: filtro de tickers (VALE3,PETR4)")
    args = parser.parse_args()

    if not args.indicators.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {args.indicators}")

    df_ind = pd.read_parquet(args.indicators)
    need = {"ticker", "date", *INDICATOR_COLUMNS}
    missing = sorted(need.difference(df_ind.columns))
    if missing:
        raise ValueError(f"Colunas ausentes no input: {missing}")

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if tickers:
        df_ind = df_ind[df_ind["ticker"].isin(tickers)].copy()

    df_ind = df_ind.sort_values(["ticker", "date"]).reset_index(drop=True)

    if args.mode in {"daily", "both"}:
        out_daily = args.out_dir / "panel_indicators_z_daily.parquet"
        _transform_and_export(df_ind[["ticker", "date"] + INDICATOR_COLUMNS], out_daily, args.include_rc, "daily")
        print(f"OK daily: {out_daily}")

    if args.mode in {"monthly", "both"}:
        base_monthly = _build_monthly_base(df_ind[["ticker", "date"] + INDICATOR_COLUMNS])
        out_monthly = args.out_dir / "panel_indicators_z_monthly.parquet"
        _transform_and_export(base_monthly, out_monthly, args.include_rc, "monthly")
        print(f"OK monthly: {out_monthly}")


if __name__ == "__main__":
    main()
