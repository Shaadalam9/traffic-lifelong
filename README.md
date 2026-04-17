# Traffic lifelong pipeline

This bundle removes the old `legacy/` dependency.

## How configuration works

This pipeline reads values from your project `config` file through `common.get_configs(...)`.

The expected keys are:

- `logger_level`
- `input_path`
- `annotations_xml`
- `output_root`
- `run_video_preparation`
- `run_scene_config_export`
- `run_tracking`
- `run_event_extraction`
- `run_event_merge`
- `model_weights`
- `device`
- `confidence_threshold`
- `tracker_config`

## Main behaviour

- only `input_path` and `annotations_xml` are required as inputs
- all outputs are written inside `_output/`
- scene config is saved to `_output/scene/scene_regions.json`
- tracking reads prepared clips from `_output/standardized_videos/`
- event extraction scans all `tracks.csv` files under `_output/tracking_outputs/`
- merge writes `_output/event_tables/master_events.csv`

## Notes

- time OCR is not implemented here
- video preparation currently copies videos into standardized output rather than transcoding or splitting them
- tracking requires `ultralytics`


This version does not use utils/config.py. Runtime settings are loaded from the repository config file through common.get_configs(...), and the in-memory PipelineContext lives in utils/base.py.
