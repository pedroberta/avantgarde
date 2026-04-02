# -*- coding: utf-8 -*-
"""Parâmetros globais e caminhos padrão."""
from __future__ import annotations

from pathlib import Path

# Raiz do projeto (pasta Quant)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EXCEL_PATH = PROJECT_ROOT / "AvantGarde" / "Economatica Ibra + Parâmetros Completos.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "panel_output"

# Janelas (dias úteis aproximados)
TRADING_DAYS_PER_MONTH = 21
MAX_GAP_TRADING_DAYS_DEFAULT = 60
MIN_OBS_VOL = 40
ROLL_VOL_WINDOW = 60

DATE_START_DEFAULT = "2010-01-01"
DATE_END_DEFAULT = "2024-12-31"

# Unidades: moeda e volume em MILHARES (export Economatica); preço por ação
UNITS_NOTE = (
    "Monetary fields (mcap, volume_thousands_brl, book_equity, net_income, revenue, "
    "assets, pas_cir, pas_nocir, cpv, capex, ebitda) are in thousands of original currency. "
    "shares_thousands is thousands of shares. close is price per share."
)

RANDOM_SEED = 42
