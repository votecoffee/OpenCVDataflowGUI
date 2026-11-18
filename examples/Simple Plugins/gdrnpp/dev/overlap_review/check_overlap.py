"""Actual external worker: repeated real proposal, final selection and cached review."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2

from opencv_simple_gdrnpp.bridge import WorkerClient
from opencv_simple_gdrnpp.review import write_cloud


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )


def save(folder, name, metadata, arrays):
    write_json(folder / f"{name}.json", metadata)
    if "overlay" in arrays:
        assert cv2.imwrite(str(folder / f"{name}.png"), arrays["overlay"][..., ::-1])
    if "points" in arrays:
        write_cloud(
            folder / f"{name}.ply",
            arrays["points"],
            arrays["colors"],
            metadata["report"]["target_frame"],
        )
    if "gallery" in arrays:
        assert cv2.imwrite(str(folder / "gallery.png"), arrays["gallery"][..., ::-1])


def request(recipe, models, **options):
    return dict(
        base=dict(
            kind="recipe",
            path=str(recipe.resolve()),
            sha256=hashlib.sha256(recipe.read_bytes()).hexdigest(),
        ),
        models_dir=str(models.resolve()),
        **options,
    )


def run(enabled, disabled, models, output, *, worker_python=sys.executable):
    if output.exists():
        raise FileExistsError("Choose a fresh overlap check folder")
    output.mkdir(parents=True)
    with WorkerClient(python=worker_python) as client:
        result, arrays = client.call(
            "preview", request(enabled, models, graphics=True, want_gallery=True)
        )
        report = result["report"]
        assert len(report["candidates"]) == 6
        assert report["depth_accepted_candidates"] == 2 and report["accepted_candidates"] == 1
        assert result["candidate_index"] == 0
        duplicate = report["candidates"][5]
        assert duplicate["selection"]["status"] == "overlap_suppressed"
        assert duplicate["selection"]["overlap"] == 1
        assert duplicate["selection"]["retained_by"] == report["candidates"][0]["id"]
        save(output, "best", result, arrays)
        cached, graphics = client.call("render", dict(token=result["token"], index=5))
        assert cached["review"]["status"] == "overlap_suppressed"
        assert cached["review"]["depth_status"] == "accepted"
        assert {k: v for k, v in cached["report"].items() if k != "request_timing"} == {
            k: v for k, v in report.items() if k != "request_timing"
        }
        assert cached["report"]["worker"]["inference_request_count"] == 1
        save(output, "suppressed", cached, graphics)
        baseline, no_arrays = client.call("preview", request(disabled, models, graphics=False))
        assert no_arrays == {} and baseline["report"]["accepted_candidates"] == 2
        assert "overlap_selection" not in baseline["report"]
        assert baseline["report"]["best_pose"] == report["best_pose"]
        for selected, original in zip(
            report["candidates"], baseline["report"]["candidates"], strict=True
        ):
            assert selected["pose"] == original["pose"]
            assert selected["depth_fit"] == original["fit"]
        save(output, "disabled", baseline, no_arrays)
        numeric, no_arrays = client.call(
            "preview", request(enabled, models, graphics=False, want_gallery=True)
        )
        assert no_arrays == {} and numeric["report"]["accepted_candidates"] == 1
        assert numeric["report"]["candidates"] == report["candidates"]
        assert numeric["report"]["best_pose"] == report["best_pose"]
        assert numeric["report"]["worker"]["model_load_count"] == 1
    write_json(
        output / "checks.json",
        dict(
            proposals=6,
            depth_accepted=2,
            unique_accepted=1,
            suppressed=1,
            maximum_overlap=0.6,
            repeated_proposal_overlap=1.0,
            all_poses_and_depth_metrics_unchanged=True,
            best_pose_unchanged=True,
            suppressed_review_retains_depth_accept=True,
            cached_review_inference_requests=1,
            three_inspections_one_model_load=True,
            graphics_off_numeric_parity=True,
            weights_bundled=False,
            scope=(
                "Real T-LESS scene with one intentionally repeated external box; "
                "duplicate-proposal contract, not broader pose accuracy"
            ),
            attribution="T-LESS, Hodan et al., WACV 2017, CC BY 4.0",
        ),
    )
    print(
        "Six real proposals: two depth accepts -> one unique pose; "
        "cached suppression review and numeric parity pass"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("enabled", "disabled", "models", "new_output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.enabled, args.disabled, args.models, args.new_output)
