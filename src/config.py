from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping at top level.")
    return cfg


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp_run_dir(runs_dir: str | Path) -> Path:
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(Path(runs_dir) / ts)

