# -*- coding: utf-8 -*-
"""Calendário de negociação observado no dataset."""
from __future__ import annotations

import pandas as pd


def union_trading_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Datas únicas presentes em qualquer ticker (ordenadas)."""
    return pd.DatetimeIndex(sorted(df["date"].dropna().unique()))


def reindex_ticker_to_calendar(
    sub: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    sub: colunas incluem date + features.
    Retorna sub reindexado a trading_dates.
    """
    t = sub.set_index("date").sort_index()
    out = t.reindex(trading_dates)
    out.index.name = "date"
    return out.reset_index()
