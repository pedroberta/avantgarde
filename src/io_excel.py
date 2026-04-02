# -*- coding: utf-8 -*-
"""Leitura do Excel Economatica: multi-aba → painel longo."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# Linha de cabeçalho (0-based) na exportação verificada: linha 3 = 4ª linha
HEADER_ROW = 3

# Colunas canônicas após renomear (ordem igual ao Excel)
CANONICAL_COLS = [
    "date",
    "close",
    "volume_thousands_brl",
    "mcap_thousands",
    "shares_thousands",
    "book_equity_thousands",
    "net_income_thousands",
    "revenue_thousands",
    "assets_thousands",
    "pas_cir_thousands",
    "pas_nocir_thousands",
    "cpv_thousands",
    "capex_thousands",
    "ebitda_thousands",
    "pe_ratio",
    "ev_ebit",
    "ev_ebitda",
    "div_yield",
    "roe_reported",
    "roa_reported",
    "gross_margin_reported",
    "net_margin_reported",
    "ebit_margin_reported",
]

FUNDAMENTAL_COLS = [
    "book_equity_thousands",
    "net_income_thousands",
    "revenue_thousands",
    "assets_thousands",
    "pas_cir_thousands",
    "pas_nocir_thousands",
    "cpv_thousands",
    "capex_thousands",
    "ebitda_thousands",
    "pe_ratio",
    "ev_ebit",
    "ev_ebitda",
    "div_yield",
    "roe_reported",
    "roa_reported",
    "gross_margin_reported",
    "net_margin_reported",
    "ebit_margin_reported",
]

PRICE_VOLUME_COLS = ["close", "volume_thousands_brl", "mcap_thousands", "shares_thousands"]


def _coerce_numeric(series: pd.Series) -> pd.Series:
    s = series.copy()
    bad = s.astype(str).str.strip().isin(["-", "", "nan", "NaN"])
    s = s.mask(bad, np.nan)
    return pd.to_numeric(s, errors="coerce")


def load_excel_long(
    excel_path: Path,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    tickers: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Empilha todas as abas em (ticker, date, ...).
    Mantém escala original (milhares onde aplicável).
    Se tickers for informado, lê apenas abas cujo nome coincide (ex.: VALE3, PETR4).
    """
    excel_path = Path(excel_path)
    if not excel_path.is_file():
        raise FileNotFoundError(f"Excel não encontrado: {excel_path}")

    ticker_filter: Optional[set[str]] = None
    if tickers is not None:
        ticker_filter = {str(t).strip() for t in tickers if str(t).strip()}

    xl = pd.ExcelFile(excel_path, engine="openpyxl")
    frames: list[pd.DataFrame] = []

    for sheet in xl.sheet_names:
        sheet_name = sheet.strip()
        if ticker_filter is not None and sheet_name not in ticker_filter:
            continue
        raw = pd.read_excel(
            excel_path,
            sheet_name=sheet,
            header=HEADER_ROW,
            engine="openpyxl",
        )
        if raw.shape[1] < len(CANONICAL_COLS):
            continue
        raw = raw.iloc[:, : len(CANONICAL_COLS)].copy()
        raw.columns = CANONICAL_COLS
        raw["ticker"] = sheet_name
        frames.append(raw)

    if not frames:
        msg = "Nenhuma aba válida lida do Excel."
        if ticker_filter is not None:
            msg += f" Filtro tickers={sorted(ticker_filter)} — confira nomes das abas."
        raise ValueError(msg)

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    for c in CANONICAL_COLS[1:]:
        df[c] = _coerce_numeric(df[c])

    df = df.dropna(subset=["date"])
    df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
    df = df.sort_values(["ticker", "date"])

    if date_start:
        df = df[df["date"] >= pd.Timestamp(date_start)]
    if date_end:
        df = df[df["date"] <= pd.Timestamp(date_end)]

    df = df.reset_index(drop=True)
    return df


def classify_columns() -> tuple[list[str], list[str]]:
    """Retorna (cols_preço_volume, cols_fundamental)."""
    return list(PRICE_VOLUME_COLS), list(FUNDAMENTAL_COLS)
