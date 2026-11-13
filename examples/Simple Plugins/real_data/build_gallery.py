"""Replay every real-data showcase project and export actual inputs/results."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE


def write_png(path, image):
    ok, raw = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode a gallery image.")
    path.write_bytes(raw.tobytes())


def displayed(image):
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image


def place(page, pixels, rectangle):
    x, y, width, height = rectangle
    pixels = displayed(pixels)
    scale = min(width / pixels.shape[1], height / pixels.shape[0])
    size = max(1, round(pixels.shape[1] * scale)), max(1, round(pixels.shape[0] * scale))
    pixels = cv2.resize(pixels, size, interpolation=cv2.INTER_AREA)
    left, top = x + (width - size[0]) // 2, y + (height - size[1]) // 2
    page[top : top + size[1], left : left + size[0]] = pixels


def text(page, value, x, y, scale=0.55):
    value = str(value).encode("ascii", "replace").decode()[:180]
    cv2.putText(
        page, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 238, 242), 1, cv2.LINE_AA
    )


def numeric(example, output, expected=None):
    tool = example["tool"]
    if tool == "SimplePatternMatch":
        return dict(count=output["count"], score=output["score"], matches=output["matches"])
    if tool == "SimpleSegmentation":
        return dict(
            foreground_pixels=int(np.count_nonzero(output["mask"])),
            inspected_pixels=int(np.count_nonzero(output["inspected"])),
            report=output["report"],
        )
    if tool == "SimpleClassify":
        return dict(
            label=output["label"],
            score=output["score"],
            confusion=output["confusion"],
            probabilities=output["probabilities"],
            expected=expected,
            correct=output["label"] == expected,
        )
    if tool == "SimplePathWidth":
        return output["measurements"]
    if tool == "SimpleReferenceInspect":
        return output["inspection"]
    return dict(count=output["count"], poses=output["poses"], report=output["report"])


def build(target):
    root = Path(__file__).resolve().parents[1]
    catalog = json.loads(Path(__file__).with_name("catalog.json").read_bytes())
    target = Path(target)
    target.mkdir(parents=True, exist_ok=False)
    receipt, chosen = {}, []
    for example in catalog["examples"]:
        tool = example["tool"]
        project = root / example["project"]
        print("Replaying", tool, flush=True)
        graph = Graph.load(project, builtin_registry())
        truth = {}
        if tool == "SimpleClassify":
            truth = {
                row["image"]: row["label"]
                for row in json.loads((project.parent / "ground_truth.json").read_bytes())
            }
        records = []
        try:
            source = graph.blocks[example["source_block"]]
            block = graph.blocks[example["block"]]
            for frame in range(example["frames"]):
                graph.run()
                if block.state not in ("pass", "fail"):
                    raise ValueError(f"{tool}: {block.error}")
                output = block.last_result.outputs
                preview_block = graph.blocks[example["preview_block"]]
                if preview_block.last_result is None:
                    raise ValueError(f"{tool}: preview did not complete")
                result = preview_block.last_result.outputs[example["preview_port"]]
                if result is NO_VALUE:
                    raise ValueError(f"{tool}: saved recipe does not render its preview")
                source_path = (source.parameters.get("paths") or [source.parameters["path"]])[frame]
                input_file = project.parent / source_path
                if tool == "Simple3DLocate":
                    photo_file = root / example["input_photo"]
                    image = cv2.imdecode(np.frombuffer(photo_file.read_bytes(), np.uint8), 1)
                else:
                    image = source.last_result.outputs["image"]
                summary = numeric(example, output, truth.get(source_path))
                folder = target / tool / f"{frame:02}"
                folder.mkdir(parents=True)
                write_png(folder / "input.png", displayed(image))
                write_png(folder / "result.png", result)
                page = np.full((760, 1280, 3), (30, 25, 20), np.uint8)
                text(page, f"{tool} | {example['dataset']} | frame {frame + 1}", 15, 32, 0.72)
                text(page, "Real input", 15, 69)
                text(page, "Actual saved-project result", 500, 69)
                place(page, image, (10, 90, 470, 540))
                place(page, result, (500, 90, 770, 540))
                text(page, f"Tool state: {block.state} | {source_path}", 15, 660)
                text(page, example["scope"], 15, 690, 0.43)
                text(
                    page,
                    f"Data license: {example['license']} | full attribution linked in catalog",
                    15,
                    720,
                    0.43,
                )
                write_png(folder / "comparison.png", page)
                record = dict(
                    frame=frame,
                    state=block.state,
                    input_file=input_file.relative_to(root).as_posix(),
                    input_file_sha256=hashlib.sha256(input_file.read_bytes()).hexdigest(),
                    result_pixels_sha256=hashlib.sha256(result.tobytes()).hexdigest(),
                    result_size=list(result.shape[1::-1]),
                    measurements=summary,
                )
                if tool == "Simple3DLocate":
                    cloud = block.last_inputs["scene"]
                    record["source_points"] = len(cloud.points)
                    for name in ("fit", "aligned_model"):
                        if output[name] is not NO_VALUE:
                            write_cloud(folder / f"{name}.ply", output[name], binary=True)
                (folder / "result.json").write_bytes(
                    (json.dumps(record, indent=2, allow_nan=False) + "\n").encode()
                )
                records.append(record)
                if frame == example["overview_frame"]:
                    chosen.append((tool, result.copy()))
            receipt[tool] = dict(
                project=example["project"],
                dataset=example["dataset"],
                license=example["license"],
                attribution=example["attribution"],
                frames=records,
            )
            print(tool, "completed", len(records), "real frames", flush=True)
        finally:
            graph.close()
    overview = np.full((1500, 1280, 3), (30, 25, 20), np.uint8)
    for index, (tool, pixels) in enumerate(chosen):
        x, y = (index % 2) * 640, (index // 2) * 500
        text(overview, tool, x + 12, y + 28, 0.68)
        place(overview, pixels, (x + 10, y + 48, 620, 430))
    write_png(target / "overview.png", overview)
    versions = {}
    for name in (
        "opencv-geometric-pattern-match",
        "opencv-defect-segmentation",
        "opencv-simple-classifier",
        "opencv-simple-path-width",
        "opencv-simple-reference-inspect",
        "opencv-simple-3d-locate",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        revision = (
            subprocess.run(
                ["git", "log", "-1", "--format=%H", "--", "src/opencv_dataflow", "pyproject.toml"],
                cwd=root.parents[1],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            or None
        )
    except OSError:
        revision = None
    runtime = dict(
        python=sys.version.split()[0],
        opencv=cv2.__version__,
        numpy=np.__version__,
        host_runtime_revision=revision,
        tool_packages=versions,
    )
    (target / "runtime.json").write_bytes((json.dumps(runtime, indent=2) + "\n").encode())
    (target / "receipt.json").write_bytes(
        (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode()
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory; existing directories are refused")
    build(parser.parse_args().output)
