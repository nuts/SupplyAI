"""
formatters.py
Display helpers, colour coding, and HTML table generation for dense data views.
"""

import pandas as pd
import numpy as np


# ── Number formatters ──────────────────────────────────────────────────────────
def fmt_units(n):
    if pd.isna(n) or n is None: return "—"
    return f"{int(round(n)):,}"


def fmt_lbs(n):
    if pd.isna(n) or n is None: return "—"
    return f"{round(float(n), 0):,.0f} lbs"


def fmt_pct(n, decimals=1):
    if pd.isna(n) or n is None: return "—"
    return f"{round(float(n) * 100, decimals):+.{decimals}f}%"


def fmt_currency(n):
    if pd.isna(n) or n is None: return "—"
    return f"${float(n):,.0f}"


def fmt_weeks(n):
    if pd.isna(n) or n is None or n >= 99: return "∞"
    return f"{round(float(n), 1):.1f}wk"


# ── Colour helpers ─────────────────────────────────────────────────────────────
def coverage_colour(weeks):
    if pd.isna(weeks): return "#6b7a6b"
    if weeks >= 8:  return "#b8f542"
    if weeks >= 4:  return "#f5a842"
    return "#f54242"


def trend_colour(yoy):
    if pd.isna(yoy):       return "#6b7a6b"
    if yoy > 0.05:         return "#b8f542"
    if yoy < -0.05:        return "#f54242"
    return "#6b7a6b"


# ── Tag pill HTML ──────────────────────────────────────────────────────────────
def tag_pill(text, colour="#6b7a6b", bg=None):
    if bg is None:
        bg = colour + "22"   # 13% alpha
    return (f"<span style='background:{bg};color:{colour};"
            f"border-radius:4px;padding:2px 7px;font-size:10px;"
            f"font-weight:500;display:inline-block'>{text}</span>")


def confidence_tag(conf):
    colours = {"High": "#b8f542", "Medium": "#f5a842", "Low": "#f54242", "New": "#42b8f5"}
    return tag_pill(conf, colours.get(conf, "#6b7a6b"))


def status_tag(status):
    colours = {
        "Out of Stock":    "#f54242",
        "Critical":        "#f54242",
        "Below Reorder":   "#f5a842",
        "Low":             "#f5a842",
        "Adequate":        "#b8f542",
        "Healthy":         "#42f5a8",
    }
    return tag_pill(status, colours.get(status, "#6b7a6b"))


def trend_tag(yoy):
    if yoy is None or (isinstance(yoy, str) and yoy.lower() == "new"):
        return tag_pill("NEW", "#42b8f5")
    if pd.isna(yoy): return tag_pill("—", "#6b7a6b")

    pct = round(yoy * 100, 1)
    if pct > 5:  return f"<span style='color:#b8f542;font-weight:500'>▲ {pct}%</span>"
    if pct < -5: return f"<span style='color:#f54242;font-weight:500'>▼ {abs(pct)}%</span>"
    return f"<span style='color:#6b7a6b'>— {pct:+.1f}%</span>"


# ── Mini bar HTML ──────────────────────────────────────────────────────────────
def mini_bar(value, max_value, colour="#b8f542", width=60):
    if max_value <= 0 or pd.isna(value): return ""
    pct = min(100, round(float(value) / max_value * 100))
    return (f"<div style='width:{width}px;background:#1a1d1a;border-radius:2px;"
            f"height:3px;margin-top:3px'>"
            f"<div style='width:{pct}%;background:{colour};height:3px;border-radius:2px;opacity:.6'></div>"
            f"</div>")


# ── Allergen badges ────────────────────────────────────────────────────────────
ALLERGEN_COLOURS = {
    "Peanut":    "#f54242",  "Tree Nut":  "#f5a842",  "Wheat":     "#f5d442",
    "Milk":      "#42b8f5",  "Soy":       "#a842f5",  "Sesame":    "#f542b8",
    "Sulfites":  "#6b7a6b",  "Shellfish": "#42f5e8",  "Fish":      "#42f599",
    "Egg":       "#f5f542",
}


def allergen_badges(allergens):
    if not allergens:
        return "<span style='color:#3a3e3a;font-size:10px'>—</span>"
    return "".join(
        f"<span style='border:1px solid {ALLERGEN_COLOURS.get(a, '#6b7a6b')};"
        f"color:{ALLERGEN_COLOURS.get(a, '#6b7a6b')};border-radius:3px;"
        f"padding:1px 5px;font-size:9px;margin-right:2px'>{a[:3].upper()}</span>"
        for a in allergens
    )


# ── Week label builders ────────────────────────────────────────────────────────
def week_keys(current_week, current_year, n=8):
    """List of YYYY-WW strings."""
    out = []
    w, y = current_week, current_year
    for _ in range(n):
        out.append(f"{y}-{w:02d}")
        w += 1
        if w > 52: w, y = 1, y + 1
    return out


def week_short_labels(current_week, n=8):
    return [f"W{((current_week - 1 + i) % 52) + 1}" for i in range(n)]
