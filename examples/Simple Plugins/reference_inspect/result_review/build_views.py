"""Export completed inspections and real-photo gate failures without retuning."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import cv2
import opencv_simple_reference_inspect.model as core
from opencv_simple_reference_inspect import save_model
from opencv_simple_reference_inspect.cli import read_image
from opencv_simple_reference_inspect.result_review import ResultReview

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.processing import gray_image


def build(target):
    root = Path(__file__).resolve().parents[1]
    target = Path(target)
    target.mkdir(parents=True, exist_ok=False)
    receipt = {}
    original_inspect = core.inspect

    def save(name, image, model, result, **context):
        before = json.dumps(result.report, sort_keys=True, allow_nan=False)
        ResultReview(image, model, result).save(target / name)
        assert before == json.dumps(result.report, sort_keys=True, allow_nan=False)
        receipt[name] = {
            **context,
            "state": "pass"
            if result.passed
            else "fail"
            if result.report["evaluated"]
            else "not_evaluated",
            "input_pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "reference_mean_sha256": hashlib.sha256(model.mean.tobytes()).hexdigest(),
            "report_sha256": hashlib.sha256(before.encode()).hexdigest(),
        }

    for relative, count in (("", 2), ("fixture_capture", 3)):
        project = root / relative / "project.json"
        graph = Graph.load(project, builtin_registry())
        completed = []

        def capture(*args, **kwargs):
            result = original_inspect(*args, **kwargs)
            completed.append(result)
            return result

        try:
            block = graph.blocks["inspection"]
            with patch.object(core, "inspect", side_effect=capture):
                for index in range(count):
                    graph.run()
                    if block.state not in ("pass", "fail"):
                        raise ValueError(block.error)
                    assert len(completed) == index + 1
                    model = block._verified[1]
                    image = gray_image(block.last_inputs["image"])
                    save(
                        f"{relative or 'aligned'}_{index}",
                        image,
                        model,
                        completed[-1],
                        project=project.relative_to(root).as_posix(),
                        frame_index=index,
                        recipe_sha256=block.parameters["sha256"],
                    )
            if not relative:
                result = model.inspect(image, reference_to_image=[[1, 0, 1000], [0, 1, 0]])
                save(
                    "unsupported_reference",
                    image,
                    model,
                    result,
                    description="Explicit off-image pose; unchanged saved recipe settings.",
                )
        finally:
            graph.close()

    # Reuse the already attributed prepared photographs. One reference and no
    # alignment are deliberately insufficient for the independent normal photo.
    # Retain the default gates and expose that failure instead of tuning them.
    assets = root.parent / "pattern_matching/visa_pcb2/assets"
    images = [
        cv2.cvtColor(read_image(assets / name), cv2.COLOR_BGR2GRAY)
        for name in ("pcb2-0000.jpg", "pcb2-0621.jpg")
    ]
    model = core.train(images[:1])
    project_folder = target / "visa_pcb2_project"
    (project_folder / "assets").mkdir(parents=True)
    digest = save_model(project_folder / "assets/reference.sri", model)
    for name in ("pcb2-0000.jpg", "pcb2-0621.jpg"):
        shutil.copyfile(assets / name, project_folder / "assets" / name)
    project = json.loads((root / "project.json").read_bytes())
    for block in project["blocks"]:
        if block["id"] == "source":
            block["parameters"]["paths"] = ["assets/pcb2-0000.jpg", "assets/pcb2-0621.jpg"]
        elif block["id"] == "inspection":
            block["parameters"].update(sha256=digest, render_assessment=True)
            block["preview_output"] = "assessment"
            block["exposed_outputs"].append("assessment")
    (project_folder / "project.json").write_bytes(
        (json.dumps(project, indent=2, allow_nan=False) + "\n").encode()
    )
    for index, image in enumerate(images):
        result = model.inspect(image)
        save(
            f"visa_pcb2_{index}",
            image,
            model,
            result,
            source=f"pattern_matching/visa_pcb2/assets/pcb2-{'0000' if not index else '0621'}.jpg",
            source_file_sha256=hashlib.sha256(
                (assets / f"pcb2-{'0000' if not index else '0621'}.jpg").read_bytes()
            ).hexdigest(),
            training_sample_count=1,
            source_ground_truth="normal",
            description="Default settings; one good reference; no pose alignment.",
        )
    (target / "receipt.json").write_bytes(
        (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode()
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory; existing paths are refused")
    print(f"Exported {len(build(parser.parse_args().output))} inspection reviews.")
