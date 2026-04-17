from __future__ import annotations

from utils.base import LegacyScriptRunner
from utils.config import PipelineConfig


class SceneConfigBuilder(LegacyScriptRunner):
    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)

    def run(self) -> None:
        self.config.ensure_output_directories()
        self.run_legacy_script(
            "cvat_scene_to_config.py",
            {
                "ANNOTATIONS_XML": self.config.annotations_xml,
                "OUTPUT_JSON": self.config.scene_regions_json,
                "IMAGE_NAME": self.config.scene_image_name,
                "ROUND_DIGITS": self.config.scene_round_digits,
            },
        )
