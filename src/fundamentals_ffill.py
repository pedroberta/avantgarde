# -*- coding: utf-8 -*-
"""
Forward-fill de fundamentos com parada por gap longo em preço (delisting).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.calendar import reindex_ticker_to_calendar, union_trading_dates
from src.io_excel import FUNDAMENTAL_COLS


def _inactive_from_close(
    close: np.ndarray,
    max_gap_trading_days: int,
) -> tuple[int | None, np.ndarray]:
    """
    close alinhado ao calendário (NaN = sem preço).
    Primeira sequência com NaN consecutivo >= max_gap → inactive_from no 1º NaN do run.
    Após inactive_from, active=False até o fim da amostra (sem reativação).
    """
    n = len(close)
    inactive_from_idx: int | None = None
    gap = 0
    run_start: int | None = None
    for i in range(n):
        if np.isfinite(close[i]):
            gap = 0
            run_start = None
        else:
            if gap == 0:
                run_start = i
            gap += 1
            if gap >= max_gap_trading_days and inactive_from_idx is None and run_start is not None:
                inactive_from_idx = run_start

    active = np.ones(n, dtype=bool)
    if inactive_from_idx is not None:
        active[inactive_from_idx:] = False
    return inactive_from_idx, active


def apply_ffill_and_inactive(
    df_long: pd.DataFrame,
    max_gap_trading_days: int,
    fundamental_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Para cada ticker: reindex ao calendário global, marca inactive_from,
    ffill fundamentos apenas em linhas active (e com preço válido opcional para reset gap — já embutido no inactive).
    """
    fundamental_cols = fundamental_cols or FUNDAMENTAL_COLS
    trading_dates = union_trading_dates(df_long)
    tickers = df_long["ticker"].unique()
    parts: list[pd.DataFrame] = []

    for ticker in tickers:
        sub = df_long[df_long["ticker"] == ticker].copy()
        sub = sub.drop_duplicates("date", keep="last")
        r = reindex_ticker_to_calendar(sub, trading_dates)
        r["ticker"] = ticker

        close = r["close"].to_numpy(dtype=float)
        inactive_idx, active = _inactive_from_close(close, max_gap_trading_days)

        dates = trading_dates.to_numpy()
        inactive_from = pd.NaT
        if inactive_idx is not None:
            inactive_from = pd.Timestamp(dates[inactive_idx])

        r["has_price"] = np.isfinite(close)
        r["is_active"] = active
        r["inactive_from"] = inactive_from

        for col in fundamental_cols:
            if col not in r.columns:
                continue
            s = r[col].copy()
            r[col + "_raw"] = s
            # ffill só na região active; observações fora de active não propagam
            s_active = s.where(active)
            r[col] = s_active.ffill().where(active)

        parts.append(r)

    out = pd.concat(parts, ignore_index=True)
    return out
