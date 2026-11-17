"""Explicitly reuse the pinned YOLOX preparer; no model loading during graph open."""

import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "tracking_yolox_preparation", HERE.parent / "detection_tile_fusion/prepare_tile_model.py"
)
preparer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparer)
PARAMETERS = preparer.PARAMETERS


def prepare(source=None):
    return preparer.prepare(source, directory=HERE / ".local/yolox_raw")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Reuse the identical pinned source weights")
    prepare(parser.parse_args().model)
