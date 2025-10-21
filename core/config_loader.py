from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG_PATH = Path("config.yaml")


@dataclass
class AppConfig:
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def yolo(self) -> Dict[str, Any]:
        return self.raw.get("models", {}).get("yolo", {})

    @property
    def depth(self) -> Dict[str, Any]:
        return self.raw.get("models", {}).get("depth", {})

    @property
    def reference_classes(self) -> Dict[str, Any]:
        return self.raw.get("detection", {}).get("reference_classes", {})

    @property
    def portion(self) -> Dict[str, Any]:
        return self.raw.get("portion", {})

    @property
    def runtime(self) -> Dict[str, Any]:
        return self.raw.get("runtime", {})


def load_config(path: Path | str | None = None) -> AppConfig:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig(raw=raw)
