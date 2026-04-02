# -*- coding: utf-8 -*-
"""
Cálculo dos 55 indicadores no painel (ativo × data), só a partir da tabela Economatica
e retornos do próprio ativo (sem benchmark de mercado).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.config import MIN_OBS_VOL, ROLL_VOL_WINDOW, TRADING_DAYS_PER_MONTH

INDICATOR_COLUMNS = [
    "val_book_to_market",
    "val_earnings_to_price",
    "val_sales_to_price",
    "val_ebitda_to_ev",
    "val_ebit_to_ev",
    "val_dividend_yield",
    "val_price_to_book",
    "val_pe_ratio",
    "val_price_to_sales",
    "val_ev_to_sales",
    "val_ebit_to_mcap",
    "mom_ret_1m",
    "mom_ret_3m",
    "mom_ret_6m",
    "mom_ret_12m",
    "mom_ret_12_1",
    "mom_ret_6_1",
    "mom_ret_3_1",
    "mom_short_term_reversal",
    "mom_vol_adj_12_1",
    "size_log_mcap_thousands",
    "size_mcap_thousands",
    "liq_avg_volume_thousands_21d",
    "liq_turnover_volume_to_mcap",
    "liq_amihud_21d",
    "liq_trading_days_21d",
    "liq_volume_per_share_thousands",
    "liq_dollar_volume_thousands",
    "liq_relative_volume",
    "prof_roe",
    "prof_roa",
    "prof_gross_profitability",
    "prof_gross_margin",
    "prof_ebit_margin",
    "prof_net_margin",
    "prof_ebitda_margin",
    "prof_asset_turnover",
    "prof_ebit_to_assets",
    "prof_ebitda_to_assets",
    "inv_asset_growth_yoy",
    "inv_sales_growth_yoy",
    "inv_earnings_growth_yoy",
    "inv_equity_growth_yoy",
    "inv_ebitda_growth_yoy",
    "inv_capex_to_assets",
    "inv_capex_to_revenue",
    "lev_liabilities_to_equity",
    "lev_liabilities_to_assets",
    "lev_assets_to_equity",
    "risk_realized_vol_60d",
    "risk_downside_vol_60d",
    "oth_return_skew_126d",
    "oth_return_kurt_126d",
    "oth_turnover_change_21d",
    "oth_volume_shock_21d",
]

VAL_COLS = INDICATOR_COLUMNS[0:11]
MOM_COLS = INDICATOR_COLUMNS[11:20]
LIQ_COLS = INDICATOR_COLUMNS[20:29]
PROF_COLS = INDICATOR_COLUMNS[29:39]
INV_COLS = INDICATOR_COLUMNS[39:46]
LEV_COLS = INDICATOR_COLUMNS[46:49]
RISK_OTH_COLS = INDICATOR_COLUMNS[49:55]


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    x = num.astype(float)
    d = den.astype(float)
    out = x / d
    return out.where(np.isfinite(out) & (d != 0))


def _rolling_compound_return(r: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    min_p = min_periods or max(int(0.8 * window), 5)
    return (
        (1.0 + r)
        .rolling(window, min_periods=min_p)
        .apply(lambda x: float(np.nanprod(1.0 + x) - 1.0), raw=True)
    )


def _apply_per_ticker(df: pd.DataFrame, fn: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
    """Aplica fn por ticker sem passar a coluna ticker a fn (evita FutureWarning / perda de coluna)."""
    parts: list[pd.DataFrame] = []
    for _, g in df.groupby("ticker", sort=False):
        idx = g.index
        t = g["ticker"].iloc[0]
        chunk = fn(g.drop(columns=["ticker"]))
        chunk["ticker"] = t
        chunk.index = idx
        parts.append(chunk)
    return pd.concat(parts)


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["ticker", "date"])

    use = out["is_active"] & out["has_price"]

    pl = out["book_equity_thousands"]
    ll = out["net_income_thousands"]
    rev = out["revenue_thousands"]
    a = out["assets_thousands"]
    mc = out["mcap_thousands"]
    ev = out["ev_thousands"]
    ebit = out["ebit_thousands"]
    ebitda = out["ebitda_thousands"]
    cpv = out["cpv_thousands"]
    liab = out["liabilities_thousands"]

    out["val_book_to_market"] = _safe_div(pl, mc)
    out["val_earnings_to_price"] = _safe_div(ll, mc)
    out["val_sales_to_price"] = _safe_div(rev, mc)
    out["val_ebitda_to_ev"] = _safe_div(ebitda, ev)
    out["val_ebit_to_ev"] = _safe_div(ebit, ev)
    out["val_dividend_yield"] = out["div_yield"].astype(float)
    out["val_price_to_book"] = _safe_div(mc, pl)
    out["val_pe_ratio"] = out["pe_ratio"].astype(float)
    out["val_price_to_sales"] = _safe_div(mc, rev)
    out["val_ev_to_sales"] = _safe_div(ev, rev)
    out["val_ebit_to_mcap"] = _safe_div(ebit, mc)

    for c in VAL_COLS:
        out.loc[~use, c] = np.nan

    w1 = TRADING_DAYS_PER_MONTH
    w3 = 3 * TRADING_DAYS_PER_MONTH
    w6 = 6 * TRADING_DAYS_PER_MONTH
    w12 = 12 * TRADING_DAYS_PER_MONTH

    def _mom_block(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.copy()
        r = sub["r_daily"].astype(float)
        sub["mom_ret_1m"] = _rolling_compound_return(r, w1)
        sub["mom_ret_3m"] = _rolling_compound_return(r, w3)
        sub["mom_ret_6m"] = _rolling_compound_return(r, w6)
        sub["mom_ret_12m"] = _rolling_compound_return(r, w12)
        r1 = sub["mom_ret_1m"]
        r3 = sub["mom_ret_3m"]
        r6 = sub["mom_ret_6m"]
        r12 = sub["mom_ret_12m"]
        sub["mom_ret_12_1"] = (1.0 + r12) / (1.0 + r1) - 1.0
        sub["mom_ret_6_1"] = (1.0 + r6) / (1.0 + r1) - 1.0
        sub["mom_ret_3_1"] = (1.0 + r3) / (1.0 + r1) - 1.0
        sub["mom_short_term_reversal"] = np.where(r1 < 0, -r1, 0.0)
        sig = r.rolling(w12, min_periods=int(0.8 * w12)).std(ddof=1)
        sub["mom_vol_adj_12_1"] = sub["mom_ret_12_1"] / sig
        return sub

    out = _apply_per_ticker(out, _mom_block)

    for c in MOM_COLS:
        out.loc[~use, c] = np.nan

    out["size_log_mcap_thousands"] = np.where(use, np.log(np.maximum(mc.astype(float), 1e-12)), np.nan)
    out["size_mcap_thousands"] = np.where(use, mc.astype(float), np.nan)

    def _liq_block(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.copy()
        v = sub["volume_thousands_brl"].astype(float)
        m = sub["mcap_thousands"].astype(float)
        sub["liq_avg_volume_thousands_21d"] = v.rolling(21, min_periods=10).mean()
        sub["liq_turnover_volume_to_mcap"] = _safe_div(v, m)
        absr = sub["r_daily"].astype(float).abs()
        sub["liq_amihud_21d"] = (absr / v.replace(0, np.nan)).rolling(21, min_periods=10).mean()
        sub["liq_trading_days_21d"] = v.gt(0).rolling(21, min_periods=10).sum()
        sub["liq_volume_per_share_thousands"] = _safe_div(v, sub["shares_thousands"].astype(float))
        sub["liq_dollar_volume_thousands"] = v
        return sub

    out = _apply_per_ticker(out, _liq_block)

    tot_vol = out.groupby("date")["volume_thousands_brl"].transform(
        lambda s: s.astype(float).sum(min_count=1)
    )
    out["liq_relative_volume"] = _safe_div(out["volume_thousands_brl"].astype(float), tot_vol)

    for c in LIQ_COLS:
        out.loc[~use, c] = np.nan

    gp = rev.astype(float) - cpv.astype(float)
    out["prof_roe"] = _safe_div(ll, pl)
    out["prof_roa"] = _safe_div(ll, a)
    out["prof_gross_profitability"] = _safe_div(gp, a)
    out["prof_gross_margin"] = _safe_div(gp, rev)
    out["prof_ebit_margin"] = _safe_div(ebit, rev)
    out["prof_net_margin"] = _safe_div(ll, rev)
    out["prof_ebitda_margin"] = _safe_div(ebitda, rev)
    out["prof_asset_turnover"] = _safe_div(rev, a)
    out["prof_ebit_to_assets"] = _safe_div(ebit, a)
    out["prof_ebitda_to_assets"] = _safe_div(ebitda, a)
    for c in PROF_COLS:
        out.loc[~use, c] = np.nan

    def _yoy(s: pd.Series) -> pd.Series:
        return s.astype(float) / s.astype(float).shift(252) - 1.0

    def _inv_block(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.copy()
        sub["inv_asset_growth_yoy"] = _yoy(sub["assets_thousands"])
        sub["inv_sales_growth_yoy"] = _yoy(sub["revenue_thousands"])
        sub["inv_earnings_growth_yoy"] = _yoy(sub["net_income_thousands"])
        sub["inv_equity_growth_yoy"] = _yoy(sub["book_equity_thousands"])
        sub["inv_ebitda_growth_yoy"] = _yoy(sub["ebitda_thousands"])
        sub["inv_capex_to_assets"] = _safe_div(sub["capex_thousands"], sub["assets_thousands"])
        sub["inv_capex_to_revenue"] = _safe_div(sub["capex_thousands"], sub["revenue_thousands"])
        return sub

    out = _apply_per_ticker(out, _inv_block)
    for c in INV_COLS:
        out.loc[~use, c] = np.nan

    out["lev_liabilities_to_equity"] = _safe_div(liab, pl)
    out["lev_liabilities_to_assets"] = _safe_div(liab, a)
    out["lev_assets_to_equity"] = _safe_div(a, pl)
    for c in LEV_COLS:
        out.loc[~use, c] = np.nan

    def _risk_oth_block(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.copy()
        r = sub["r_daily"].astype(float)
        sub["risk_realized_vol_60d"] = r.rolling(ROLL_VOL_WINDOW, min_periods=MIN_OBS_VOL).std(ddof=1)
        neg = r.where(r < 0)
        sub["risk_downside_vol_60d"] = neg.rolling(ROLL_VOL_WINDOW, min_periods=MIN_OBS_VOL).std(ddof=1)
        sub["oth_return_skew_126d"] = r.rolling(126, min_periods=80).skew()
        sub["oth_return_kurt_126d"] = r.rolling(126, min_periods=80).kurt()
        to = sub["liq_turnover_volume_to_mcap"].astype(float)
        sub["oth_turnover_change_21d"] = to / to.shift(21) - 1.0
        vloc = sub["volume_thousands_brl"].astype(float)
        lv = np.log(vloc.replace(0, np.nan))
        sub["oth_volume_shock_21d"] = lv - lv.rolling(21, min_periods=10).mean()
        return sub

    out = _apply_per_ticker(out, _risk_oth_block)

    for c in RISK_OTH_COLS:
        out.loc[~use, c] = np.nan

    for c in INDICATOR_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan

    return out
