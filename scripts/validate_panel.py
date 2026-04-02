# -*- coding: utf-8 -*-
"""
Valida consistência financeira entre panel_raw_scale e panel_indicators.
Saída: relatório no stdout; exit code 1 se algum check crítico falhar.

Uso:
  python scripts/validate_panel.py
  python scripts/validate_panel.py --sample 50000 --tickers VALE3,PETR4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PROJECT_ROOT
from src.finance_validation import check_daily_return_vs_close, merge_raw_and_indicators, run_all_checks

DEFAULT_RAW = PROJECT_ROOT / "data" / "panel_output" / "panel_raw_scale.parquet"
DEFAULT_IND = PROJECT_ROOT / "data" / "panel_output" / "panel_indicators.parquet"


def main() -> int:
    p = argparse.ArgumentParser(description="Validação de contas do painel IBRA")
    p.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    p.add_argument("--indicators", type=Path, default=DEFAULT_IND)
    p.add_argument("--sample", type=int, default=0, help="Amostra aleatória de linhas após merge (0=todas)")
    p.add_argument("--tickers", type=str, default="", help="Filtrar tickers, vírgula")
    args = p.parse_args()

    if not args.raw.is_file() or not args.indicators.is_file():
        print("ERRO: Parquet não encontrado. Rode build_panel.py antes.")
        return 2

    import pandas as pd

    raw = pd.read_parquet(args.raw)
    ind = pd.read_parquet(args.indicators)
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] or None
    merged_full = merge_raw_and_indicators(raw, ind, max_rows=None, tickers=tickers)
    merged_full = merged_full.sort_values(["ticker", "date"])
    if merged_full.empty:
        print("ERRO: merge vazio (tickers/datas?).")
        return 2

    sampled = args.sample > 0 and len(merged_full) > args.sample
    if sampled:
        merged_base = merged_full.sample(n=args.sample, random_state=42)
        merged_base = merged_base.sort_values(["ticker", "date"])
    else:
        merged_base = merged_full

    results = run_all_checks(merged_base, include_return_check=not sampled)
    if sampled:
        results.append(check_daily_return_vs_close(merged_full))
    ok_all = True
    print("=== Validação painel (finanças / mercado) ===\n")
    if sampled:
        print(
            f"Nota: checks de identidade usam amostra de {args.sample} linhas; "
            "r_daily_close usa série completa para preservar shift temporal.\n"
        )
    for r in results:
        status = "OK" if r.ok else "FALHOU"
        if not r.ok:
            ok_all = False
        print(f"[{status}] {r.name}: {r.message} (n_checked={r.n_rows_checked})")
    print()
    if ok_all:
        print("Resumo: todos os checks passaram.")
        return 0
    print("Resumo: existe inconsistência — revisar fórmulas ou dados fonte.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
