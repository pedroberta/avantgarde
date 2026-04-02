# -*- coding: utf-8 -*-
"""
Transformações cross-section por data:
- Passo A: rank com empates (average) e rc = rank / (n + 1)
- Passo B: z = (rc - média(rc)) / soma(|rc - média(rc)|)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rank_and_l1(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Recebe uma coluna de característica no corte transversal de uma data.
    Retorna (rc, z) alinhados ao índice original do grupo.
    """
    values = pd.to_numeric(series, errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=float))
    out_rc = pd.Series(np.nan, index=series.index, dtype=float)
    out_z = pd.Series(np.nan, index=series.index, dtype=float)

    if not finite.any():
        return out_rc, out_z

    valid = values.loc[finite]
    ranks = valid.rank(method="average", ascending=True)
    n_t = float(len(valid))
    rc = ranks / (n_t + 1.0)

    rc_bar = float(rc.mean())
    centered = rc - rc_bar
    denom = float(centered.abs().sum())

    out_rc.loc[valid.index] = rc
    if denom > 0.0:
        out_z.loc[valid.index] = centered / denom
    return out_rc, out_z


def apply_rank_and_l1_by_date(
    df: pd.DataFrame,
    feature_cols: list[str],
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Aplica rank+L1 para cada data e cada característica.
    Saída: df com colunas base (ticker/date) + rc_* + z_*.
    """
    required = {"ticker", date_col}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame precisa conter colunas ticker e {date_col}.")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Características ausentes no DataFrame: {missing[:5]}")

    out_base = df[["ticker", date_col]].copy()
    derived: dict[str, pd.Series] = {}
    g_date = df.groupby(date_col, sort=False)

    for col in feature_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        s_valid = s.where(np.isfinite(s.to_numpy(dtype=float)))
        n_t = s_valid.groupby(df[date_col], sort=False).transform("count").astype(float)
        ranks = s_valid.groupby(df[date_col], sort=False).rank(method="average", ascending=True)
        rc = ranks / (n_t + 1.0)

        rc_bar = rc.groupby(df[date_col], sort=False).transform("mean")
        centered = rc - rc_bar
        denom = centered.abs().groupby(df[date_col], sort=False).transform("sum")
        z = centered / denom
        z = z.where(denom > 0)

        derived[f"rc_{col}"] = rc.astype(float)
        derived[f"z_{col}"] = z.astype(float)

    out = pd.concat([out_base, pd.DataFrame(derived, index=df.index)], axis=1)
    return out
