# -*- coding: utf-8 -*-
"""Export Parquet particionado + metadata + definições de indicadores."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import UNITS_NOTE
from src.indicators.compute_indicators import INDICATOR_COLUMNS


def _file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


DEFINITIONS_ROWS: list[dict[str, str]] = [
    {"name": "val_book_to_market", "formula": "book_equity_thousands / mcap_thousands", "unit": "dimensionless"},
    {"name": "val_earnings_to_price", "formula": "net_income_thousands / mcap_thousands", "unit": "dimensionless"},
    {"name": "val_sales_to_price", "formula": "revenue_thousands / mcap_thousands", "unit": "dimensionless"},
    {"name": "val_ebitda_to_ev", "formula": "ebitda_thousands / ev_thousands", "unit": "dimensionless"},
    {"name": "val_ebit_to_ev", "formula": "ebit_thousands / ev_thousands", "unit": "dimensionless"},
    {"name": "val_dividend_yield", "formula": "div_yield (coluna)", "unit": "dimensionless"},
    {"name": "val_price_to_book", "formula": "mcap_thousands / book_equity_thousands", "unit": "dimensionless"},
    {"name": "val_pe_ratio", "formula": "pe_ratio (coluna)", "unit": "dimensionless"},
    {"name": "val_price_to_sales", "formula": "mcap_thousands / revenue_thousands", "unit": "dimensionless"},
    {"name": "val_ev_to_sales", "formula": "ev_thousands / revenue_thousands", "unit": "dimensionless"},
    {"name": "val_ebit_to_mcap", "formula": "ebit_thousands / mcap_thousands", "unit": "dimensionless"},
    {"name": "mom_ret_1m", "formula": "prod(1+r_daily)-1 over ~21d", "unit": "dimensionless"},
    {"name": "mom_ret_3m", "formula": "~63d compound", "unit": "dimensionless"},
    {"name": "mom_ret_6m", "formula": "~126d compound", "unit": "dimensionless"},
    {"name": "mom_ret_12m", "formula": "~252d compound", "unit": "dimensionless"},
    {"name": "mom_ret_12_1", "formula": "(1+R12)/(1+R1)-1", "unit": "dimensionless"},
    {"name": "mom_ret_6_1", "formula": "(1+R6)/(1+R1)-1", "unit": "dimensionless"},
    {"name": "mom_ret_3_1", "formula": "(1+R3)/(1+R1)-1", "unit": "dimensionless"},
    {"name": "mom_short_term_reversal", "formula": "-R1 if R1<0 else 0", "unit": "dimensionless"},
    {"name": "mom_vol_adj_12_1", "formula": "mom_ret_12_1 / std(r_daily 12m)", "unit": "dimensionless"},
    {"name": "size_log_mcap_thousands", "formula": "log(mcap_thousands)", "unit": "log(thousands BRL)"},
    {"name": "size_mcap_thousands", "formula": "mcap_thousands", "unit": "thousands BRL"},
    {"name": "liq_avg_volume_thousands_21d", "formula": "mean(volume_thousands_brl,21d)", "unit": "thousands BRL"},
    {"name": "liq_turnover_volume_to_mcap", "formula": "volume_thousands_brl / mcap_thousands", "unit": "dimensionless"},
    {"name": "liq_amihud_21d", "formula": "mean(|r|/volume_thousands_brl,21d)", "unit": "1/(thousands BRL)"},
    {"name": "liq_trading_days_21d", "formula": "count(volume>0,21d)", "unit": "days"},
    {"name": "liq_volume_per_share_thousands", "formula": "volume_thousands_brl / shares_thousands", "unit": "thousands BRL / thousands shares"},
    {"name": "liq_dollar_volume_thousands", "formula": "volume_thousands_brl", "unit": "thousands BRL"},
    {"name": "liq_relative_volume", "formula": "vol_i / sum_j vol_j por data", "unit": "dimensionless"},
    {"name": "prof_roe", "formula": "net_income / book_equity", "unit": "dimensionless"},
    {"name": "prof_roa", "formula": "net_income / assets", "unit": "dimensionless"},
    {"name": "prof_gross_profitability", "formula": "(revenue-cpv)/assets", "unit": "dimensionless"},
    {"name": "prof_gross_margin", "formula": "(revenue-cpv)/revenue", "unit": "dimensionless"},
    {"name": "prof_ebit_margin", "formula": "ebit/revenue", "unit": "dimensionless"},
    {"name": "prof_net_margin", "formula": "net_income/revenue", "unit": "dimensionless"},
    {"name": "prof_ebitda_margin", "formula": "ebitda/revenue", "unit": "dimensionless"},
    {"name": "prof_asset_turnover", "formula": "revenue/assets", "unit": "dimensionless"},
    {"name": "prof_ebit_to_assets", "formula": "ebit/assets", "unit": "dimensionless"},
    {"name": "prof_ebitda_to_assets", "formula": "ebitda/assets", "unit": "dimensionless"},
    {"name": "inv_asset_growth_yoy", "formula": "assets/assets.shift(252)-1", "unit": "dimensionless"},
    {"name": "inv_sales_growth_yoy", "formula": "revenue shift 252", "unit": "dimensionless"},
    {"name": "inv_earnings_growth_yoy", "formula": "NI shift 252", "unit": "dimensionless"},
    {"name": "inv_equity_growth_yoy", "formula": "book_equity shift 252", "unit": "dimensionless"},
    {"name": "inv_ebitda_growth_yoy", "formula": "ebitda shift 252", "unit": "dimensionless"},
    {"name": "inv_capex_to_assets", "formula": "capex/assets", "unit": "dimensionless"},
    {"name": "inv_capex_to_revenue", "formula": "capex/revenue", "unit": "dimensionless"},
    {"name": "lev_liabilities_to_equity", "formula": "liabilities_thousands/book_equity", "unit": "dimensionless"},
    {"name": "lev_liabilities_to_assets", "formula": "liabilities/assets", "unit": "dimensionless"},
    {"name": "lev_assets_to_equity", "formula": "assets/book_equity", "unit": "dimensionless"},
    {"name": "risk_realized_vol_60d", "formula": "std(r_daily,60)", "unit": "dimensionless"},
    {"name": "risk_downside_vol_60d", "formula": "std(negative r,60)", "unit": "dimensionless"},
    {"name": "oth_return_skew_126d", "formula": "skew(r,126)", "unit": "dimensionless"},
    {"name": "oth_return_kurt_126d", "formula": "kurtosis(r,126)", "unit": "dimensionless"},
    {"name": "oth_turnover_change_21d", "formula": "turnover/turnover.shift(21)-1", "unit": "dimensionless"},
    {"name": "oth_volume_shock_21d", "formula": "log(vol)-rolling mean log(vol)", "unit": "dimensionless"},
]


def _validate_indicator_contract() -> None:
    names = [row["name"] for row in DEFINITIONS_ROWS]
    if names != INDICATOR_COLUMNS:
        raise ValueError(
            "Inconsistência entre DEFINITIONS_ROWS e INDICATOR_COLUMNS. "
            "Verifique nomes/ordem/contagem de indicadores."
        )


def export_panel(
    df_raw_panel: pd.DataFrame,
    df_indicators: pd.DataFrame,
    output_dir: Path,
    excel_path: Path,
    extra_meta: dict[str, Any],
) -> None:
    _validate_indicator_contract()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "panel_raw_scale.parquet"
    ind_path = output_dir / "panel_indicators.parquet"

    df_raw_panel.to_parquet(raw_path, index=False)
    df_indicators.to_parquet(ind_path, index=False)

    by_year = output_dir / "panel_indicators_by_year"
    by_year.mkdir(exist_ok=True)
    di = df_indicators.copy()
    di["year"] = pd.to_datetime(di["date"]).dt.year
    for y, g in di.groupby("year"):
        g.drop(columns=["year"]).to_parquet(by_year / f"year={y}.parquet", index=False)

    defs = pd.DataFrame(DEFINITIONS_ROWS)
    defs.to_csv(output_dir / "indicator_definitions.csv", index=False)

    meta = {
        "units_note": UNITS_NOTE,
        "indicator_columns": INDICATOR_COLUMNS,
        "source_excel_md5": _file_md5(excel_path) if excel_path.is_file() else None,
        "n_indicators": len(INDICATOR_COLUMNS),
        "n_indicator_definitions": len(DEFINITIONS_ROWS),
        **extra_meta,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def run_qa(df: pd.DataFrame, indicator_cols: list[str]) -> dict[str, Any]:
    """Cobertura simples: % não-NaN por indicador e por ano."""
    out: dict[str, Any] = {}
    d = pd.to_datetime(df["date"])
    years = sorted(d.dt.year.unique().tolist())
    out["rows"] = int(len(df))
    out["tickers"] = int(df["ticker"].nunique())
    cov = {}
    for c in indicator_cols:
        if c in df.columns:
            cov[c] = float(df[c].notna().mean())
    out["indicator_non_null_share"] = cov
    by_y = {}
    for y in years:
        m = d.dt.year == y
        sub = df.loc[m, indicator_cols]
        by_y[str(y)] = {c: float(sub[c].notna().mean()) for c in indicator_cols if c in sub.columns}
    out["non_null_by_year"] = by_y
    return out
