# -*- coding: utf-8 -*-
"""
Exporta para Excel um recorte de tickers (p.ex. VALE3, PETR4) a partir dos Parquet gerados.
Útil para inspecionar visualmente se o pipeline está coerente.

Uso (pasta Quant):
  python scripts/export_tickers_excel.py
  python scripts/export_tickers_excel.py --tickers VALE3,PETR3 --out data/panel_output/preview.xlsx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_OUTPUT_DIR, PROJECT_ROOT

DEFAULT_IND = PROJECT_ROOT / "data" / "panel_output" / "panel_indicators.parquet"
DEFAULT_RAW = PROJECT_ROOT / "data" / "panel_output" / "panel_raw_scale.parquet"


def main() -> None:
    p = argparse.ArgumentParser(description="Export subset of tickers to Excel for inspection")
    p.add_argument(
        "--tickers",
        type=str,
        default="VALE3,PETR4",
        help="Separados por vírgula (nomes das abas / ticker no painel).",
    )
    p.add_argument("--indicators", type=Path, default=DEFAULT_IND, help="panel_indicators.parquet")
    p.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="panel_raw_scale.parquet (2ª aba)")
    p.add_argument("--no-raw", action="store_true", help="Só exporta aba de indicadores")
    p.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR / "preview_tickers.xlsx")
    args = p.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        raise SystemExit("Informe pelo menos um ticker em --tickers")

    if not args.indicators.is_file():
        raise SystemExit(f"Arquivo não encontrado: {args.indicators}")

    df_ind = pd.read_parquet(args.indicators)
    miss = [t for t in tickers if t not in set(df_ind["ticker"].unique())]
    if miss:
        print("Aviso: tickers sem linhas no parquet:", miss)
    sub_ind = df_ind[df_ind["ticker"].isin(tickers)].sort_values(["ticker", "date"])
    if sub_ind.empty:
        raise SystemExit("Nenhuma linha após filtro; confira códigos (ex.: VALE3, PETR4).")

    args.out = Path(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        sub_ind.to_excel(writer, sheet_name="indicators", index=False)
        if not args.no_raw and args.raw.is_file():
            df_raw = pd.read_parquet(args.raw)
            sub_raw = df_raw[df_raw["ticker"].isin(tickers)].sort_values(["ticker", "date"])
            sub_raw.to_excel(writer, sheet_name="raw_scale", index=False)

    print(f"OK: {args.out} ({len(sub_ind)} linhas, {sub_ind['ticker'].nunique()} tickers)")


if __name__ == "__main__":
    main()
