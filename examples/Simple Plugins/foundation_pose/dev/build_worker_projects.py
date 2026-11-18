"""Build new global-registration and explicitly assisted two-proposal replay projects."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from opencv_dataflow.assistant_api import ProjectSession, make_registry


def build(destination):
    source = Path(__file__).resolve().parent
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    assets = destination / "assets"
    assets.mkdir()
    shutil.copyfile(source / "trained.sfp", assets / "trained.sfp")
    digest = hashlib.sha256((assets / "trained.sfp").read_bytes()).hexdigest()
    registry = make_registry([source / "plugin.py"])
    for case in ("held_out_frame13", "different_object_negative"):
        shutil.copyfile(source / case / "capture.npz", assets / (case + ".npz"))
        report = json.loads((source / case / "report.json").read_bytes())
        # A short replay starts from previous NETWORK proposals, never annotated truth.
        # This is additional initialization assistance and is not global registration.
        seeds = [
            dict(
                model="pose_3d_v1",
                units="m",
                source_frame=report["source_frame"],
                target_frame=report["target_frame"],
                source_to_target=c["source_to_target"],
            )
            for c in report["candidates"][:2]
        ]
        (assets / (case + "_prior_proposals.json")).write_text(
            json.dumps(seeds, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        for assisted in (False, True):
            name = case + ("_assisted_replay" if assisted else "_global")
            description = (
                "ASSISTED replay from two prior network proposals; no annotated pose input. "
                "Additional refinement and changed score cohort; not independent global inference."
                if assisted
                else "Global 252-rotation registration with a supplied visible instance mask. "
                "No initial/annotated pose input. CPU execution takes several minutes."
            )
            operations = [
                dict(
                    op="add",
                    type="input.foundation_pose_example_capture",
                    id="capture",
                    parameters=dict(
                        capture_path=f"assets/{case}.npz",
                        capture_id=f"tless:{case}",
                        coarse_path=f"assets/{case}_prior_proposals.json" if assisted else "",
                    ),
                ),
                dict(
                    op="add",
                    type="3d.simple_foundation_pose",
                    id="pose",
                    parameters=dict(recipe_path="assets/trained.sfp", sha256=digest),
                ),
                dict(
                    op="update",
                    id="capture",
                    fields=dict(position=[0, 0], node_size="large", preview_output="image"),
                ),
                dict(
                    op="update",
                    id="pose",
                    fields=dict(
                        position=[460, 0],
                        node_size="large",
                        preview_output="cloud",
                        description=description,
                    ),
                ),
            ]
            for port in ("image", "depth", "mask", "camera", "capture_id") + (
                ("coarse_poses",) if assisted else ()
            ):
                operations.append(
                    dict(op="connect", source="capture", output=port, target="pose", input=port)
                )
            with ProjectSession(registry, directory=destination) as session:
                session.apply(operations)
                assert session.inspect()["validation"]["valid"]
                session.save(destination / (name + ".json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="New portable project folder")
    build(parser.parse_args().destination)
