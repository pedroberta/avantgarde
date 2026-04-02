# -*- coding: utf-8 -*-
"""
Gera painel diário com transformações cross-section (rank e normalização L1).
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
DEFAULT_OUT = PROJECT_ROOT / "data" / "panel_output" / "panel_indicators_z_daily.parquet"


def _write_metadata(df_out: pd.DataFrame, out_path: Path) -> None:
    z_cols = [c for c in df_out.columns if c.startswith("z_")]
    rc_cols = [c for c in df_out.columns if c.startswith("rc_")]
    meta = {
        "rows": int(len(df_out)),
        "dates": int(pd.to_datetime(df_out["date"]).nunique()) if not df_out.empty else 0,
        "tickers": int(df_out["ticker"].nunique()) if not df_out.empty else 0,
        "n_features": len(INDICATOR_COLUMNS),
        "n_rc_columns": len(rc_cols),
        "n_z_columns": len(z_cols),
        "feature_columns": INDICATOR_COLUMNS,
        "rc_columns": rc_cols,
        "z_columns": z_cols,
        "pipeline": "rank_average_rc_over_n_plus_1_then_l1_normalization",
    }
    meta_path = out_path.with_name(f"{out_path.stem}_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build daily rank+L1 transformed panel from panel_indicators")
    parser.add_argument("--indicators", type=Path, default=DEFAULT_IND, help="Input panel_indicators.parquet")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output Parquet path")
    parser.add_argument("--only-z", action="store_true", help="Exporta apenas z_* (sem rc_*)")
    parser.add_argument("--tickers", type=str, default="", help="Opcional: filtra tickers (VALE3,PETR4)")
    args = parser.parse_args()

    if not args.indicators.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {args.indicators}")

    df = pd.read_parquet(args.indicators)
    required = {"ticker", "date", *INDICATOR_COLUMNS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Colunas ausentes no input: {missing}")

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if tickers:
        df = df[df["ticker"].isin(tickers)].copy()

    try:
        out_df = apply_rank_and_l1_by_date(df, INDICATOR_COLUMNS)
    except MemoryError as exc:
        if args.only_z:
            raise
        raise MemoryError(
            "Sem memória para exportar rc_* + z_* neste volume. "
            "Tente novamente com --only-z."
        ) from exc
    if args.only_z:
        z_cols = [f"z_{c}" for c in INDICATOR_COLUMNS]
        out_df = out_df[["ticker", "date"] + z_cols].copy()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    _write_metadata(out_df, args.out)

    print(f"OK: {args.out}")
    print(f"Rows: {len(out_df)}, dates: {out_df['date'].nunique()}, tickers: {out_df['ticker'].nunique()}")


if __name__ == "__main__":
    main()
