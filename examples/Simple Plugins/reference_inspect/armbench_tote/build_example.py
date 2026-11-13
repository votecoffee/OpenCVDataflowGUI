"""Build a real clear-tote reference example from the pinned ARMBench tiny archive."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np
from opencv_simple_reference_inspect import Settings, save_model, train
from opencv_simple_reference_inspect.result_review import ResultReview

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

ARCHIVE_SHA256 = "30f2539a2c8ae32ae25000ee9025657e2c49529cf9e513196597f9a94e7ca0d7"
REFERENCE = "LO95CK8OLH"
CASES = (
    ("EZ5895NGB8", "empty"),
    ("P2S0835EOA", "loaded"),
    ("9TARTKRF1X", "loaded"),
    ("OAR1KXNIME", "loaded"),
)
MATRIX = [[0.7, -0.35, -0.35, 128.0]]


def write_image(path, image):
    options = [cv2.IMWRITE_JPEG_QUALITY, 93] if path.suffix == ".jpg" else []
    ok, encoded = cv2.imencode(path.suffix, image, options)
    if not ok:
        raise ValueError("Could not encode a sample image.")
    path.write_bytes(encoded.tobytes())


def build(archive, target):
    archive, target = Path(archive), Path(target)
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != ARCHIVE_SHA256:
            raise ValueError("ARMBench source archive differs from the pinned sample.")
    target.mkdir(parents=True, exist_ok=False)
    assets = target / "assets"
    assets.mkdir()
    registry = builtin_registry()
    transform = registry.create("cv.channel_transform", parameters={"matrix": MATRIX})
    sources, signals = {}, {}
    with tarfile.open(archive, "r:gz") as data:
        for image_id in (REFERENCE, *(image_id for image_id, _ in CASES)):
            member = f"./armbench-segmentation-tiny/{image_id}.jpg"
            raw = data.extractfile(member).read()
            color = cv2.imdecode(np.frombuffer(raw, np.uint8), 1)
            color = cv2.resize(color, (612, 512), interpolation=cv2.INTER_AREA)
            filename = assets / f"{image_id}.jpg"
            write_image(filename, color)
            color = cv2.imdecode(np.frombuffer(filename.read_bytes(), np.uint8), 1)
            transform.execute({"image": color})
            signals[image_id] = transform.last_result.outputs["image"].copy()
            write_image(assets / f"{image_id}-blue.png", signals[image_id])
            sources[image_id] = dict(
                member=member,
                original_sha256=hashlib.sha256(raw).hexdigest(),
                prepared_sha256=hashlib.sha256(filename.read_bytes()).hexdigest(),
            )
    region = np.zeros((512, 612), np.uint8)
    region[70:424, 40:571] = 255
    write_image(assets / "inspection-mask.png", region)
    # A declared occupancy allowance, not a threshold fit to replay labels.
    settings = Settings(
        normalization="none",
        min_correlation=None,
        min_defect_area=500,
        max_total_defect_area=int(np.count_nonzero(region) * 0.05),
    )
    model = train([signals[REFERENCE]], settings=settings, inspection_mask=region)
    digest = save_model(assets / "reference.sri", model)
    rows = []
    for image_id, expected in CASES:
        result = model.inspect(signals[image_id])
        ResultReview(signals[image_id], model, result).save(target / "results" / image_id)
        rows.append(
            dict(
                image=f"assets/{image_id}.jpg",
                expected_occupancy=expected,
                predicted_occupancy="empty" if result.passed else "loaded",
                inspection=result.report,
            )
        )
    (target / "evaluation.json").write_bytes((json.dumps(rows, indent=2) + "\n").encode())
    (target / "provenance.json").write_bytes(
        (
            json.dumps(
                dict(
                    archive_sha256=ARCHIVE_SHA256,
                    images=sources,
                    reference=REFERENCE,
                    occupancy_labels="Visual image-level labels for this example; "
                    "not publisher annotations.",
                ),
                indent=2,
            )
            + "\n"
        ).encode()
    )
    graph = Graph()
    try:
        source = graph.add(
            registry.create(
                "input.file",
                block_id="source",
                parameters={
                    "source_mode": "files",
                    "paths": [row["image"] for row in rows],
                    "path_base": "project",
                },
            )
        )
        source.name = "Independent empty and loaded ARMBench captures"
        blue = graph.add(
            registry.create("cv.channel_transform", block_id="blue", parameters={"matrix": MATRIX})
        )
        blue.position = [360.0, 0.0]
        blue.name = "Measure blue dominance"
        block = graph.add(
            registry.create(
                "inspect.simple_reference",
                block_id="inspection",
                parameters={
                    "recipe_path": "assets/reference.sri",
                    "sha256": digest,
                    "render_assessment": True,
                },
            )
        )
        block.position = [720.0, 0.0]
        block.name = "SimpleReferenceInspect · Large occupancy changes"
        block.preview_output = "assessment"
        if "assessment" not in block.exposed_outputs:
            block.exposed_outputs.append("assessment")
        note = graph.add(registry.create("layout.comment", block_id="credits"))
        note.name = "Real ARMBench images · CC BY 4.0"
        note.description = (
            "Amazon.com, Inc., ARMBench Sparrow tiny archive (CC BY 4.0). "
            "One empty tote is the retained good reference; a separate empty and three loaded "
            "captures are replayed. Cavity changes up to 5% are allowed; regions below 500 pixels "
            "are ignored. Exposure fitting/reference-correlation gating are disabled for the "
            "flat blue signal. Fixed camera and lighting are assumed. See README.md and SOURCES.md."
        )
        note.position = [0.0, -260.0]
        note.node_size = "large"
        graph.connect(source.id, "image", blue.id, "image")
        graph.connect(blue.id, "image", block.id, "image")
        graph.save(target / "project.json")
    finally:
        graph.close()
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path, help="New directory; existing directories are refused")
    args = parser.parse_args()
    for row in build(args.archive, args.output):
        print(row["image"], row["expected_occupancy"], row["predicted_occupancy"])
