"""
formatters.py
Display helpers: number formatting, colour coding, tag rendering.
"""

import pandas as pd
import numpy as np


# ── Number formatters ──────────────────────────────────────────────────────────
def fmt_units(n) -> str:
    if pd.isna(n):
        return "—"
    return f"{int(round(n)):,}"


def fmt_lbs(n) -> str:
    if pd.isna(n):
        return "—"
    return f"{round(float(n), 1):,.1f} lbs"


def fmt_pct(n, decimals=1) -> str:
    if pd.isna(n):
        return "—"
    return f"{round(float(n) * 100, decimals):+.{decimals}f}%"


def fmt_currency(n) -> str:
    if pd.isna(n):
        return "—"
    return f"${float(n):,.2f}"


def fmt_weeks(n) -> str:
    if pd.isna(n) or n == 99:
        return "∞"
    if n >= 100:
        return "∞"
    return f"{round(float(n), 1):.1f} wks"


# ── Colour helpers ─────────────────────────────────────────────────────────────
def coverage_colour(weeks: float) -> str:
    """Return hex colour for inventory coverage weeks."""
    if weeks >= 8:
        return "#b8f542"   # green
    elif weeks >= 4:
        return "#f5a842"   # amber
    else:
        return "#f54242"   # red


def trend_colour(yoy: float) -> str:
    if yoy > 0.05:
        return "#b8f542"
    elif yoy < -0.05:
        return "#f54242"
    return "#6b7a6b"


def confidence_colour(conf: str) -> str:
    return {"High": "#b8f542", "Medium": "#f5a842", "Low": "#f54242"}.get(conf, "#6b7a6b")


def cv_label(cv: float) -> tuple[str, str]:
    """Return (label, colour) for coefficient of variation."""
    if cv < 0.3:
        return "Stable",   "#b8f542"
    elif cv < 0.6:
        return "Moderate", "#f5a842"
    else:
        return "Volatile", "#f54242"


# ── Streamlit metric delta helpers ────────────────────────────────────────────
def delta_str(yoy: float) -> str:
    pct = round(yoy * 100, 1)
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct}% YoY"


# ── Dataframe styling ─────────────────────────────────────────────────────────
def style_coverage(val):
    try:
        v = float(str(val).replace(" wks", "").replace("∞", "999"))
    except Exception:
        return ""
    if v >= 8:
        return "color: #b8f542"
    elif v >= 4:
        return "color: #f5a842"
    else:
        return "color: #f54242; font-weight: 600"


def style_trend(val):
    try:
        v = float(str(val).replace("%", "").replace("+", ""))
    except Exception:
        return ""
    if v > 5:
        return "color: #b8f542"
    elif v < -5:
        return "color: #f54242"
    return "color: #6b7a6b"


# ── Week label builder ─────────────────────────────────────────────────────────
def week_labels(current_week: int, current_year: int, n: int = 8) -> list[str]:
    """Generate list of 'YYYY-WW' strings starting from current_week."""
    labels = []
    w, y = current_week, current_year
    for _ in range(n):
        labels.append(f"{y}-{w:02d}")
        w += 1
        if w > 52:
            w = 1
            y += 1
    return labels


def week_display_labels(current_week: int, current_year: int, n: int = 8) -> list[str]:
    """Human-readable 'W15' style labels."""
    return [f"W{((current_week - 1 + i) % 52) + 1}" for i in range(n)]


# ── Allergen badge HTML ───────────────────────────────────────────────────────
ALLERGEN_COLOURS = {
    "Peanut":    "#f54242",
    "Tree Nut":  "#f5a842",
    "Wheat":     "#f5d442",
    "Milk":      "#42b8f5",
    "Soy":       "#a842f5",
    "Sesame":    "#f542b8",
    "Sulfites":  "#6b7a6b",
    "Shellfish": "#42f5e8",
    "Fish":      "#42f599",
    "Egg":       "#f5f542",
}


def allergen_badges(allergens: list[str]) -> str:
    if not allergens:
        return "<span style='color:#6b7a6b;font-size:11px'>None</span>"
    badges = []
    for a in allergens:
        col = ALLERGEN_COLOURS.get(a, "#6b7a6b")
        badges.append(
            f"<span style='background:rgba(0,0,0,0.3);border:1px solid {col};"
            f"color:{col};border-radius:3px;padding:1px 6px;font-size:10px;"
            f"margin-right:3px'>{a}</span>"
        )
    return "".join(badges)
