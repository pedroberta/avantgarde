# -*- coding: utf-8 -*-
"""Variáveis derivadas por linha: passivo total, EV, EBIT."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Espera colunas canônicas pós-ffill. Mantém escala em milhares."""
    out = df.copy()
    pas_c = out["pas_cir_thousands"].astype(float)
    pas_nc = out["pas_nocir_thousands"].astype(float)
    out["liabilities_thousands"] = pas_c + pas_nc

    ev_ebitda = out["ev_ebitda"].astype(float)
    ebitda = out["ebitda_thousands"].astype(float)
    ev_ebit = out["ev_ebit"].astype(float)

    ev = ev_ebitda * ebitda
    out["ev_thousands"] = ev

    with np.errstate(divide="ignore", invalid="ignore"):
        ebit = np.where(
            (np.isfinite(ev)) & (np.isfinite(ev_ebit)) & (ev_ebit > 0),
            ev / ev_ebit,
            np.nan,
        )
    out["ebit_thousands"] = ebit

    out["r_daily"] = (
        out.groupby("ticker")["close"]
        .transform(lambda s: s.astype(float) / s.astype(float).shift(1) - 1.0)
    )
    return out
