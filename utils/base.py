from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

from utils.config import PipelineConfig


class LegacyScriptRunner:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    @property
    def legacy_root(self) -> Path:
        return self.config.legacy_root

    def run_legacy_script(self, script_name: str, overrides: dict[str, Any]) -> dict[str, Any]:
        script_path = self.legacy_root / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Legacy script not found: {script_path}")

        prepared_overrides: dict[str, Any] = {}
        for key, value in overrides.items():
            if isinstance(value, Path):
                prepared_overrides[key] = str(value)
            else:
                prepared_overrides[key] = value

        return runpy.run_path(str(script_path), init_globals=prepared_overrides, run_name="__main__")
