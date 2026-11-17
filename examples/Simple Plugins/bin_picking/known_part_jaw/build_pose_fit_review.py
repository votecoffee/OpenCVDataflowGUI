"""Export exact runtime pose-fit clouds and computed closeups; no re-training or motion."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.plugins import load_plugin
from opencv_dataflow.pointclouds import PointCloud, render_cloud


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def closeup(cloud, camera_from_bin, title):
    """Display-only crop around tested model rows; saved cloud keeps metric bin coordinates."""
    components = cloud.metadata["components"]
    rows = np.concatenate(
        [
            np.arange(*item["rows"])
            for item in components
            if item["role"].startswith("derived_pose_")
        ]
    )
    model = cloud.points[rows]
    if not len(model):
        return np.zeros((680, 840, 3), np.uint8)
    lo, hi = model.min(0), model.max(0)
    margin = max(float(np.ptp(model, axis=0).max()) * 0.25, 0.01)
    keep = np.all(cloud.points >= lo - margin, axis=1) & np.all(cloud.points <= hi + margin, axis=1)
    points = cloud.points[keep] @ camera_from_bin[:3, :3].T + camera_from_bin[:3, 3]
    points[:, 1] *= -1  # Display camera Y down; do not change saved metric geometry.
    view = PointCloud(points, "m", "display_view", colors=cloud.colors[keep])
    picture = render_cloud(view, width=840, height=500, yaw=0, pitch=0, limit=len(points))
    image = cv2.copyMakeBorder(picture, 45, 135, 0, 0, cv2.BORDER_CONSTANT, value=(32, 25, 18))
    cv2.putText(image, title, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 1)
    counts = {
        key: sum(item["counts"][key] for item in cloud.metadata["pose_fit"])
        for key in cloud.metadata["pose_fit_legend"]
    }
    total = sum(counts.values())
    for i, (name, rgb) in enumerate(cloud.metadata["pose_fit_legend"].items()):
        x, y = 15 + (i % 2) * 415, 570 + (i // 2) * 25
        cv2.circle(image, (x + 6, y - 5), 5, tuple(reversed(rgb)), -1)
        text = f"{name.replace('_', ' ')}: {counts[name]}/{total}"
        cv2.putText(image, text, (x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)
    cv2.putText(
        image,
        "Display crop; sampled derived CAD queries. Use native cloud to orbit.",
        (15, 662),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (180, 185, 195),
        1,
    )
    return image


def build(root, output):
    output.mkdir(parents=True, exist_ok=False)
    registry = builtin_registry()
    load_plugin(root / "plugin.py", registry)
    rows, panels = [], {}
    for path in sorted(root.rglob("project.json")):
        relative = path.parent.relative_to(root).as_posix()
        expected = json.loads(path.with_name("expected.json").read_bytes())
        destination = output / relative
        destination.mkdir(parents=True)
        with ProjectSession.open(path, registry) as session:
            session.run()
            block = session.graph.blocks["jaw"]
            result = block.last_result
            assert result is not None, block.error
            assert result.outputs["report"] == expected
            cloud = result.outputs["pose_fit"]
            if cloud is NO_VALUE:
                assert expected["pose_evidence"] == []
                rows.append(dict(case=relative, evaluated_instances=0, report_exact=True))
                continue
            metadata = cloud.metadata
            for fit, evidence in zip(metadata["pose_fit"], expected["pose_evidence"]):
                assert fit["counts"]["supported"] == evidence["supported_samples"]
                assert sum(fit["counts"].values()) == evidence["samples"]
            write_cloud(destination / "pose_fit.ply", cloud, binary=True)
            write_json(destination / "graphics.json", metadata)
            camera = block._planner.recipe.bin.bin_to_camera
            title = relative + (" / annotation pose" if relative.startswith("tless/") else "")
            picture = closeup(cloud, camera, title)
            assert cv2.imwrite(str(destination / "closeup.png"), picture)
            panels[relative] = picture
            rows.append(
                dict(
                    case=relative,
                    report_exact=True,
                    evaluated_instances=len(metadata["pose_fit"]),
                    summaries=metadata["pose_fit"],
                    measured_and_derived_rows_labeled=True,
                    cloud_sha256=hashlib.sha256(
                        (destination / "pose_fit.ply").read_bytes()
                    ).hexdigest(),
                    display_crop_only=True,
                )
            )
    overview = np.concatenate(
        [
            np.concatenate([panels["clear"], panels["unknown"]], axis=1),
            np.concatenate([panels["tless/frame000003"], panels["tless/frame000013"]], axis=1),
        ],
        axis=0,
    )
    assert cv2.imwrite(str(output / "overview.png"), overview)
    write_json(output / "checks.json", rows)
    print(
        json.dumps([dict(case=row["case"], instances=row["evaluated_instances"]) for row in rows])
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.root.resolve(), args.output.resolve())
