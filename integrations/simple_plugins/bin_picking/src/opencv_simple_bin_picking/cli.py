"""Offline suction-planning recipe creation and first-return replay."""

import argparse
import json
from pathlib import Path

from .persistence import MAX_BYTES, load_recipe, read_array, save_recipe, strict_json
from .planning import Picker
from .specs import Camera, Recipe, SuctionSettings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("reference", type=Path, help="Depth NPY, 0/nonfinite denotes missing")
    train.add_argument("configuration", type=Path)
    train.add_argument("recipe", type=Path)
    plan = commands.add_parser("plan")
    plan.add_argument("recipe", type=Path)
    plan.add_argument("depth", type=Path)
    plan.add_argument("--output", type=Path)
    plan.add_argument("--bin-pose", type=Path, help="Explicit bin-to-camera 4x4 JSON matrix")
    args = parser.parse_args(argv)
    try:
        path = args.reference if args.command == "train" else args.depth
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("Depth file exceeds 64 MiB")
        depth = read_array(path.read_bytes())
        if args.command == "train":
            if args.configuration.stat().st_size > 1024 * 1024:
                raise ValueError("Configuration exceeds 1 MiB")
            config = strict_json(args.configuration.read_bytes())
            recipe = Recipe(reference_depth=depth, **{**config,
                             "camera": Camera(**config["camera"]),
                             "settings": SuctionSettings(**config.get("settings", {}))})
            digest = save_recipe(args.recipe, recipe)
            result = {"sha256": digest, "reference": Picker(recipe).plan(depth).report()}
        else:
            transform = None
            if args.bin_pose is not None:
                if args.bin_pose.stat().st_size > 65536:
                    raise ValueError("Bin pose exceeds 64 KiB")
                transform = strict_json(args.bin_pose.read_bytes())
            result = Picker(load_recipe(args.recipe)).plan(depth, bin_to_camera=transform).report()
        value = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if getattr(args, "output", None):
            args.output.write_bytes(value.encode("utf-8"))
        else:
            print(value, end="")
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.exit(2, str(error) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
