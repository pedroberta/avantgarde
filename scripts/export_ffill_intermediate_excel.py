# -*- coding: utf-8 -*-
"""
Gera Excel **intermediário**: após reindex + forward-fill + inatividade + derivados (EV, EBIT, r_daily),
**antes** dos indicadores agregados — só para inspeção (ex.: Vale e Petrobras).

Uso (pasta Quant):
  python scripts/export_ffill_intermediate_excel.py
  python scripts/export_ffill_intermediate_excel.py --tickers VALE3,PETR3 --out data/panel_output/meu_ffill.xlsx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    DATE_END_DEFAULT,
    DATE_START_DEFAULT,
    DEFAULT_EXCEL_PATH,
    DEFAULT_OUTPUT_DIR,
    MAX_GAP_TRADING_DAYS_DEFAULT,
)
from src.derived import add_derived_columns
from src.fundamentals_ffill import apply_ffill_and_inactive
from src.io_excel import CANONICAL_COLS, load_excel_long


def _panel_pre_indicators_columns(df: pd.DataFrame) -> list[str]:
    """Mesmas colunas que build_panel grava em panel_raw_scale (sem colunas de indicadores)."""
    keep = [
        "ticker",
        "date",
        "close",
        "volume_thousands_brl",
        "mcap_thousands",
        "shares_thousands",
        "has_price",
        "is_active",
        "inactive_from",
        "r_daily",
    ]
    for c in CANONICAL_COLS[1:]:
        if c in df.columns:
            keep.append(c)
    for c in df.columns:
        if str(c).endswith("_raw"):
            keep.append(c)
    for c in ("liabilities_thousands", "ev_thousands", "ebit_thousands"):
        if c in df.columns:
            keep.append(c)
    return list(dict.fromkeys([c for c in keep if c in df.columns]))


def main() -> None:
    p = argparse.ArgumentParser(description="Excel intermediário pós-ffill (2 tickers típico)")
    p.add_argument("--excel", type=Path, default=DEFAULT_EXCEL_PATH)
    p.add_argument("--tickers", type=str, default="VALE3,PETR4")
    p.add_argument("--start", type=str, default=DATE_START_DEFAULT)
    p.add_argument("--end", type=str, default=DATE_END_DEFAULT)
    p.add_argument("--max-gap", type=int, default=MAX_GAP_TRADING_DAYS_DEFAULT)
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "preview_VALE3_PETR4_ffill_intermediate.xlsx",
    )
    args = p.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        raise SystemExit("Use --tickers VALE3,PETR4 (ou outros códigos das abas).")

    df = load_excel_long(
        args.excel,
        date_start=args.start,
        date_end=args.end,
        tickers=tickers,
    )
    df = apply_ffill_and_inactive(df, max_gap_trading_days=args.max_gap)
    df = add_derived_columns(df)

    cols = _panel_pre_indicators_columns(df)
    out = df[cols].sort_values(["ticker", "date"])
    args.out = Path(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="after_ffill_pre_indicators", index=False)

    print(f"OK: {args.out} ({len(out)} linhas, {out['ticker'].nunique()} tickers, {len(cols)} colunas)")


if __name__ == "__main__":
    main()
