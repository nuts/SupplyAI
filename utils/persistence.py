"""
persistence.py
Disk-based caching for uploaded data files.
Keeps data in memory across Streamlit sessions.
"""

import pickle
import os
import json
from pathlib import Path
from datetime import datetime

CACHE_DIR     = Path("cache")
SNAPSHOT_DIR  = Path("snapshots")
SCENARIO_DIR  = Path("scenarios")

for d in [CACHE_DIR, SNAPSHOT_DIR, SCENARIO_DIR]:
    d.mkdir(exist_ok=True)


# ── Generic cache helpers ──────────────────────────────────────────────────────
def save_to_cache(key: str, data, metadata: dict = None) -> None:
    """Save any data object to disk cache."""
    path = CACHE_DIR / f"{key}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)

    meta = {
        "saved_at": datetime.now().isoformat(),
        **(metadata or {})
    }
    meta_path = CACHE_DIR / f"{key}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def load_from_cache(key: str):
    """Load data from disk cache. Returns None if missing."""
    path = CACHE_DIR / f"{key}.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def get_cache_metadata(key: str) -> dict:
    """Get metadata for a cached item (timestamp, filename, etc.)."""
    meta_path = CACHE_DIR / f"{key}.meta.json"
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception:
        return {}


def clear_cache(key: str = None) -> None:
    """Clear specific key or all cache."""
    if key:
        for ext in [".pkl", ".meta.json"]:
            p = CACHE_DIR / f"{key}{ext}"
            if p.exists():
                p.unlink()
    else:
        for f in CACHE_DIR.glob("*"):
            if f.is_file():
                f.unlink()


def list_cached_files() -> list:
    """Return list of cached file keys with metadata."""
    files = []
    for pkl in CACHE_DIR.glob("*.pkl"):
        key  = pkl.stem
        meta = get_cache_metadata(key)
        files.append({"key": key, **meta})
    return files


# ── Forecast snapshot management ───────────────────────────────────────────────
def save_forecast_snapshot(forecast_df, label: str = None) -> str:
    """Save a forecast snapshot for accuracy tracking. Returns snapshot ID."""
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    snap_id   = label.replace(" ", "_") if label else ts
    snap_id   = f"{ts}_{snap_id}" if label else ts

    path      = SNAPSHOT_DIR / f"{snap_id}.pkl"
    meta_path = SNAPSHOT_DIR / f"{snap_id}.meta.json"

    with open(path, "wb") as f:
        pickle.dump(forecast_df, f)

    with open(meta_path, "w") as f:
        json.dump({
            "snapshot_id": snap_id,
            "label":       label or "",
            "saved_at":    datetime.now().isoformat(),
            "n_variants":  len(forecast_df),
        }, f, indent=2)

    return snap_id


def list_snapshots() -> list:
    """List all forecast snapshots."""
    snaps = []
    for meta_path in SNAPSHOT_DIR.glob("*.meta.json"):
        try:
            with open(meta_path) as f:
                snaps.append(json.load(f))
        except Exception:
            continue
    return sorted(snaps, key=lambda x: x.get("saved_at", ""), reverse=True)


def load_snapshot(snap_id: str):
    """Load a specific forecast snapshot."""
    path = SNAPSHOT_DIR / f"{snap_id}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def delete_snapshot(snap_id: str) -> None:
    for ext in [".pkl", ".meta.json"]:
        p = SNAPSHOT_DIR / f"{snap_id}{ext}"
        if p.exists():
            p.unlink()


# ── Scenario management ────────────────────────────────────────────────────────
def save_scenario(name: str, params: dict, kpi_summary: dict = None) -> None:
    """Save a scenario with parameter overrides."""
    safe = name.replace(" ", "_").replace("/", "_")
    path = SCENARIO_DIR / f"{safe}.json"
    data = {
        "name":        name,
        "saved_at":    datetime.now().isoformat(),
        "params":      params,
        "kpi_summary": kpi_summary or {},
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def list_scenarios() -> list:
    scenarios = []
    for f in SCENARIO_DIR.glob("*.json"):
        try:
            with open(f) as fh:
                scenarios.append(json.load(fh))
        except Exception:
            continue
    return sorted(scenarios, key=lambda x: x.get("saved_at", ""), reverse=True)


def load_scenario(name: str) -> dict:
    safe = name.replace(" ", "_").replace("/", "_")
    path = SCENARIO_DIR / f"{safe}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def delete_scenario(name: str) -> None:
    safe = name.replace(" ", "_").replace("/", "_")
    p = SCENARIO_DIR / f"{safe}.json"
    if p.exists():
        p.unlink()
