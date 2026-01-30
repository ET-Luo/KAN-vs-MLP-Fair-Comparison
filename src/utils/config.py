from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_config_to_namespace(args, config: dict[str, Any]) -> None:
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
