# -*- coding: utf-8 -*-
"""
CLI: monta painel 2010–2024, ffill + inatividade, 55 indicadores (só dados da planilha + r do ativo).
Uso (na pasta Quant):
  python scripts/build_panel.py
  python scripts/build_panel.py --excel "C:/caminho/planilha.xlsx" --out data/out --max-gap 60
  python scripts/build_panel.py --tickers VALE3,PETR4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    DATE_END_DEFAULT,
    DATE_START_DEFAULT,
    DEFAULT_EXCEL_PATH,
    DEFAULT_OUTPUT_DIR,
    MAX_GAP_TRADING_DAYS_DEFAULT,
    RANDOM_SEED,
    UNITS_NOTE,
)
from src.derived import add_derived_columns
from src.export_parquet import export_panel, run_qa
from src.fundamentals_ffill import apply_ffill_and_inactive
from src.indicators.compute_indicators import INDICATOR_COLUMNS, compute_all_indicators
from src.io_excel import CANONICAL_COLS, FUNDAMENTAL_COLS, load_excel_long


RAW_BASE_COLS = [
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
    "liabilities_thousands",
    "ev_thousands",
    "ebit_thousands",
]


def _build_raw_output_columns(df_columns: list[str]) -> list[str]:
    """Schema explícito do panel_raw_scale: base + canônicas + versão *_raw dos fundamentais."""
    cols: list[str] = []
    for c in RAW_BASE_COLS + CANONICAL_COLS[1:] + [f"{c}_raw" for c in FUNDAMENTAL_COLS]:
        if c in df_columns:
            cols.append(c)
    return list(dict.fromkeys(cols))


def main() -> None:
    p = argparse.ArgumentParser(description="Build IBRA panel + 55 indicators")
    p.add_argument("--excel", type=Path, default=DEFAULT_EXCEL_PATH)
    p.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--start", type=str, default=DATE_START_DEFAULT)
    p.add_argument("--end", type=str, default=DATE_END_DEFAULT)
    p.add_argument("--max-gap", type=int, default=MAX_GAP_TRADING_DAYS_DEFAULT)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Lista separada por vírgula; lê só essas abas (ex.: VALE3,PETR4). Vazio = todas.",
    )
    args = p.parse_args()

    tickers_list = [t.strip() for t in args.tickers.split(",") if t.strip()] or None
    df = load_excel_long(
        args.excel,
        date_start=args.start,
        date_end=args.end,
        tickers=tickers_list,
    )
    df = apply_ffill_and_inactive(df, max_gap_trading_days=args.max_gap)
    df = add_derived_columns(df)

    df_ind = compute_all_indicators(df)

    keep_raw = _build_raw_output_columns(list(df_ind.columns))
    df_raw_out = df_ind[keep_raw].copy()

    ind_only = df_ind[["ticker", "date"] + INDICATOR_COLUMNS].copy()

    qa = run_qa(ind_only, INDICATOR_COLUMNS)
    qa_path = Path(args.out) / "qa_summary.json"
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa, f, indent=2, ensure_ascii=False)

    export_panel(
        df_raw_out,
        ind_only,
        Path(args.out),
        Path(args.excel),
        extra_meta={
            "date_start": args.start,
            "date_end": args.end,
            "max_gap_trading_days": args.max_gap,
            "seed": args.seed,
            "units": UNITS_NOTE,
            "qa_summary_path": str(qa_path.name),
            "n_indicator_columns": len(INDICATOR_COLUMNS),
            "tickers_filter": tickers_list,
        },
    )
    print(f"OK: {args.out}")
    print(f"Rows indicators: {len(ind_only)}, tickers: {ind_only['ticker'].nunique()}")


if __name__ == "__main__":
    main()
