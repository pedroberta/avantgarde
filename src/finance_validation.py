# -*- coding: utf-8 -*-
"""
Sanity checks de finanças / mercado entre painel bruto e indicadores.
Retorna lista de dicts: name, ok, message, n_violations (ou similar).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str
    n_rows_checked: int = 0
    max_abs_error: float = float("nan")


def _finite_mask(*arrays) -> pd.Series:
    m = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        x = np.asarray(a, dtype=float)
        m &= np.isfinite(x)
    return pd.Series(m, index=arrays[0].index if hasattr(arrays[0], "index") else None)


def _relative_check(
    *,
    name: str,
    lhs: pd.Series,
    rhs: pd.Series,
    mask: pd.Series,
    rtol: float,
    atol: float = 0.0,
) -> CheckResult:
    if not mask.any():
        return CheckResult(name, True, "sem linhas comparáveis", 0)
    err = (lhs[mask] - rhs[mask]).abs()
    rel = err / np.maximum(rhs[mask].abs(), 1e-12)
    bad = ((rel > rtol) & (err > atol)).sum()
    mx_rel = float(rel.max())
    mx_abs = float(err.max())
    return CheckResult(
        name,
        bad == 0,
        f"violations={bad}, max_rel_err={mx_rel:.2e}, max_abs_err={mx_abs:.2e}",
        int(mask.sum()),
        mx_rel,
    )


def _absolute_check(
    *,
    name: str,
    lhs: pd.Series,
    rhs: pd.Series,
    mask: pd.Series,
    atol: float,
) -> CheckResult:
    if not mask.any():
        return CheckResult(name, True, "sem linhas comparáveis", 0)
    err = (lhs[mask] - rhs[mask]).abs()
    bad = (err > atol).sum()
    mx_abs = float(err.max())
    return CheckResult(
        name,
        bad == 0,
        f"violations={bad}, max_abs_err={mx_abs:.2e}",
        int(mask.sum()),
        mx_abs,
    )


def check_book_to_market_vs_levels(merged: pd.DataFrame, rtol: float = 1e-5) -> CheckResult:
    """val_book_to_market ≈ book_equity_thousands / mcap_thousands."""
    need = {"val_book_to_market", "book_equity_thousands", "mcap_thousands"}
    if not need.issubset(merged.columns):
        return CheckResult("book_to_market", False, "colunas faltando", 0)
    a = merged["book_equity_thousands"].astype(float) / merged["mcap_thousands"].astype(float)
    b = merged["val_book_to_market"].astype(float)
    m = _finite_mask(a, b) & (merged["mcap_thousands"].astype(float) != 0)
    return _relative_check(
        name="book_to_market",
        lhs=a,
        rhs=b,
        mask=m,
        rtol=rtol,
    )


def check_earnings_to_price_vs_levels(merged: pd.DataFrame, rtol: float = 1e-5) -> CheckResult:
    """val_earnings_to_price ≈ net_income / mcap."""
    need = {"val_earnings_to_price", "net_income_thousands", "mcap_thousands"}
    if not need.issubset(merged.columns):
        return CheckResult("earnings_to_price", False, "colunas faltando", 0)
    a = merged["net_income_thousands"].astype(float) / merged["mcap_thousands"].astype(float)
    b = merged["val_earnings_to_price"].astype(float)
    m = _finite_mask(a, b) & (merged["mcap_thousands"].astype(float) != 0)
    return _relative_check(
        name="earnings_to_price",
        lhs=a,
        rhs=b,
        mask=m,
        rtol=rtol,
    )


def check_pe_ratio_column(merged: pd.DataFrame, rtol: float = 1e-4) -> CheckResult:
    """val_pe_ratio deve coincidir com pe_ratio (coluna fonte) quando ambos finitos."""
    if "val_pe_ratio" not in merged.columns or "pe_ratio" not in merged.columns:
        return CheckResult("pe_ratio_source", False, "colunas faltando", 0)
    a = merged["pe_ratio"].astype(float)
    b = merged["val_pe_ratio"].astype(float)
    m = _finite_mask(a, b)
    return _relative_check(
        name="pe_ratio_source",
        lhs=a,
        rhs=b,
        mask=m,
        rtol=rtol,
    )


def check_liabilities_sum(merged: pd.DataFrame, atol: float = 1e-3) -> CheckResult:
    """liabilities_thousands = pas_cir + pas_nocir."""
    need = {"liabilities_thousands", "pas_cir_thousands", "pas_nocir_thousands"}
    if not need.issubset(merged.columns):
        return CheckResult("liabilities_sum", False, "colunas faltando", 0)
    s = merged["pas_cir_thousands"].astype(float) + merged["pas_nocir_thousands"].astype(float)
    L = merged["liabilities_thousands"].astype(float)
    m = _finite_mask(s, L)
    return _absolute_check(
        name="liabilities_sum",
        lhs=s,
        rhs=L,
        mask=m,
        atol=atol,
    )


def check_ev_from_ev_ebitda(merged: pd.DataFrame, rtol: float = 1e-4) -> CheckResult:
    """ev_thousands ≈ ev_ebitda * ebitda_thousands."""
    need = {"ev_thousands", "ev_ebitda", "ebitda_thousands"}
    if not need.issubset(merged.columns):
        return CheckResult("ev_identity", False, "colunas faltando", 0)
    a = merged["ev_ebitda"].astype(float) * merged["ebitda_thousands"].astype(float)
    b = merged["ev_thousands"].astype(float)
    m = _finite_mask(a, b)
    return _relative_check(
        name="ev_identity",
        lhs=a,
        rhs=b,
        mask=m,
        rtol=rtol,
    )


def check_ebit_from_ev(merged: pd.DataFrame, rtol: float = 1e-4) -> CheckResult:
    """ebit_thousands ≈ ev_thousands / ev_ebit quando ev_ebit > 0."""
    need = {"ebit_thousands", "ev_thousands", "ev_ebit"}
    if not need.issubset(merged.columns):
        return CheckResult("ebit_identity", False, "colunas faltando", 0)
    ev_ebit = merged["ev_ebit"].astype(float)
    a = merged["ev_thousands"].astype(float) / ev_ebit
    b = merged["ebit_thousands"].astype(float)
    m = _finite_mask(a, b) & (ev_ebit > 0)
    return _relative_check(
        name="ebit_identity",
        lhs=a,
        rhs=b,
        mask=m,
        rtol=rtol,
    )


def check_daily_return_vs_close(
    merged: pd.DataFrame,
    rtol: float = 1e-5,
    atol: float = 1e-10,
) -> CheckResult:
    """r_daily = P_t/P_{t-1}-1 por ticker."""
    if "r_daily" not in merged.columns or "close" not in merged.columns or "ticker" not in merged.columns:
        return CheckResult("r_daily_close", False, "colunas faltando", 0)
    bad_total = 0
    nchk = 0
    mx = 0.0
    for _, g in merged.groupby("ticker", sort=False):
        c = g["close"].astype(float)
        r = g["r_daily"].astype(float)
        impl = c / c.shift(1) - 1.0
        m = np.isfinite(r.to_numpy()) & np.isfinite(impl.to_numpy())
        if not m.any():
            continue
        diff = (r.to_numpy()[m] - impl.to_numpy()[m])
        rel = np.abs(diff) / np.maximum(np.abs(r.to_numpy()[m]), 1e-12)
        bad = (rel > rtol) & (np.abs(diff) > atol)
        bad_total += int(bad.sum())
        nchk += int(m.sum())
        if rel.size:
            mx = max(mx, float(rel.max()))
    return CheckResult(
        "r_daily_close",
        bad_total == 0,
        f"violations={bad_total} em {nchk} comparações, max_rel_err={mx:.2e}",
        nchk,
        mx,
    )


def merge_raw_and_indicators(
    raw: pd.DataFrame,
    ind: pd.DataFrame,
    max_rows: Optional[int] = None,
    tickers: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Inner join em (ticker, date)."""
    r = raw.copy()
    i = ind.copy()
    if tickers:
        r = r[r["ticker"].isin(tickers)]
        i = i[i["ticker"].isin(tickers)]
    m = r.merge(i, on=["ticker", "date"], how="inner", suffixes=("_rawdup", ""))
    # evitar duplicata ticker/date
    if max_rows and len(m) > max_rows:
        m = m.sample(n=max_rows, random_state=42)
    return m


def run_non_temporal_checks(merged: pd.DataFrame) -> list[CheckResult]:
    return [
        check_book_to_market_vs_levels(merged),
        check_earnings_to_price_vs_levels(merged),
        check_pe_ratio_column(merged),
        check_liabilities_sum(merged),
        check_ev_from_ev_ebitda(merged),
        check_ebit_from_ev(merged),
    ]


def run_all_checks(merged: pd.DataFrame, include_return_check: bool = True) -> list[CheckResult]:
    out = run_non_temporal_checks(merged)
    if include_return_check:
        out.append(check_daily_return_vs_close(merged))
    return out
