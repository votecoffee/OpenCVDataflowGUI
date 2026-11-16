"""Actual persistent worker: best-pose view and cached rendering of every proposal."""

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


def stable(report):
    return dict(
        best_pose=report["best_pose"],
        accepted_candidates=report["accepted_candidates"],
        candidates=[dict(id=c["id"], pose=c["pose"], fit=c["fit"]) for c in report["candidates"]],
    )


def save(folder, name, metadata, arrays):
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
    write_json(folder / f"{name}.json", metadata)


def run(recipe, models, output, *, worker_python=sys.executable):
    if output.exists():
        raise FileExistsError("Choose a fresh live-review check folder")
    output.mkdir(parents=True)
    data = dict(
        base=dict(
            kind="recipe",
            path=str(recipe.resolve()),
            sha256=hashlib.sha256(recipe.read_bytes()).hexdigest(),
        ),
        models_dir=str(models.resolve()),
        graphics=True,
        want_gallery=True,
    )
    with WorkerClient(python=worker_python) as client:
        first, arrays = client.call("preview", data)
        assert first["candidate_index"] == 1 and first["review"]["candidate_number"] == 2
        assert first["review"]["is_best_accepted"] and first["report"]["accepted_candidates"] == 1
        assert first["candidate_table"][0]["status"] == "rejected"
        save(output, "best", first, arrays)
        baseline = stable(first["report"])
        views = []
        for index in range(len(first["report"]["candidates"])):
            view, graphics = client.call("render", dict(token=first["token"], index=index))
            assert view["candidate_index"] == index and stable(view["report"]) == baseline
            assert view["report"]["worker"]["inference_request_count"] == 1
            assert view["report"]["worker"]["model_load_count"] == 1
            assert "gallery" not in graphics
            save(output, f"candidate_{index + 1}", view, graphics)
            views.append(view["review"])
        without, none = client.call("preview", {**data, "graphics": False})
        assert none == {} and stable(without["report"]) == baseline
        assert without["review"]["is_best_accepted"] and without["candidate_index"] == 1
    write_json(
        output / "checks.json",
        dict(
            default_selected_number=2,
            default_selected_is_best_accepted=True,
            candidate_count=len(views),
            cached_candidate_views=views,
            inference_requests_through_cached_views=1,
            model_loads_through_cached_views=1,
            accepted_pose_and_all_fits_unchanged=True,
            graphics_off_has_no_arrays=True,
            gallery_opt_in=True,
            weights_bundled=False,
            scope=(
                "Same real T-LESS scene with proposal order changed; selected view "
                "does not change acceptance, not new accuracy qualification"
            ),
            attribution="T-LESS, Hodan et al., WACV 2017, CC BY 4.0",
        ),
    )
    print(
        "Best view = proposal 2; all five cached reviews preserve acceptance; graphics-off parity"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("recipe", "models", "new_output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.recipe, args.models, args.new_output)
