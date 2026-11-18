"""Retain every fixed-setting result and make honest image/native-cloud review artifacts."""

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from validate_tless import load_frame, write_json

from opencv_simple_gdrnpp.review import result_cloud, write_cloud
from opencv_simple_gdrnpp.setup import load_recipe


def pose_label(row, mode):
    value = row[mode]
    if not row["selection"]["target_present"]:
        return "TARGET ABSENT"
    error = value["pose_error"]
    return (
        f"GT error {error['translation_error_mm']:.2f} mm / "
        f"{error['rotation_error_degrees']:.2f} deg"
    )


def panel(results, row):
    selected = row["selection"]
    x, y, w, h = selected["bbox_xywh"]
    canvas = np.full((360, 920, 3), 246, np.uint8)
    title = f"{row['case']} | {'target present' if selected['target_present'] else 'TARGET ABSENT'}"
    cv2.putText(
        canvas, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (25, 25, 25), 1, cv2.LINE_AA
    )
    for col, mode in enumerate(("raw", "icp")):
        value = row[mode]
        decision = "depth ACCEPTED" if value["accepted"] else "depth REJECTED"
        color = (50, 125, 20) if value["accepted"] else (25, 75, 180)
        if value["accepted"] and (
            not selected["target_present"] or not value["pose_error"]["correct_5mm_5deg"]
        ):
            decision += " / WRONG by GT"
            color = (20, 20, 200)
        for i, text in enumerate(
            (
                f"{mode.upper()}: {decision}",
                pose_label(row, mode),
                f"Support {value['fit']['coverage']:.1%}; "
                f"contradiction {value['fit']['contradiction']:.1%}",
            )
        ):
            cv2.putText(
                canvas,
                text,
                (12 + col * 460, 51 + i * 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )
        image = cv2.imread(str(results / row["case"] / f"{mode}_overlay.png"))
        height, width = image.shape[:2]
        image = image[
            max(62, y - 40) : min(height - 31, y + h + 40), max(0, x - 40) : min(width, x + w + 40)
        ]
        scale = min(440 / image.shape[1], 245 / image.shape[0])
        image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
        left = col * 460 + (460 - image.shape[1]) // 2
        canvas[106 : 106 + image.shape[0], left : left + image.shape[1]] = image
    return canvas


def select_reviews(rows):
    positives = [r for r in rows if r["selection"]["target_present"]]
    negatives = [r for r in rows if not r["selection"]["target_present"]]
    chosen = []

    def add(row, reason):
        if row and row["case"] not in {item[0]["case"] for item in chosen}:
            chosen.append((row, reason))

    add(
        next((r for r in positives if r["icp"]["accepted"] and not r["raw"]["accepted"]), None),
        "First positive converted to depth acceptance, in frozen case order",
    )
    rejected = [r for r in positives if not r["icp"]["accepted"]]
    if rejected:
        add(
            max(
                rejected,
                key=lambda r: max(
                    r["icp"]["pose_error"]["translation_error_mm"] / 5,
                    r["icp"]["pose_error"]["rotation_error_degrees"] / 5,
                ),
            ),
            "Worst remaining positive annotation error among depth rejections",
        )
    accepted = [r for r in positives if r["icp"]["accepted"]]
    if accepted:
        add(
            max(
                accepted,
                key=lambda r: max(
                    r["icp"]["pose_error"]["translation_error_mm"] / 5,
                    r["icp"]["pose_error"]["rotation_error_degrees"] / 5,
                ),
            ),
            "Worst accepted-positive annotation error; depth acceptance is separate",
        )
    if negatives:
        add(
            max(negatives, key=lambda r: (r["icp"]["accepted"], r["icp"]["fit"]["coverage"])),
            "Target-absent control with highest depth support; wrong accepts take priority",
        )
    return chosen


def curate(source, results, recipe_path, output):
    source, results, output = Path(source), Path(results), Path(output)
    if output.exists():
        raise FileExistsError("Choose a fresh curated evidence folder")
    output.mkdir(parents=True)
    summary = json.loads((results / "summary.json").read_bytes())
    rows = summary["cases"]
    recipe = load_recipe(recipe_path)
    summary["recipe_archive_sha256"] = hashlib.sha256(Path(recipe_path).read_bytes()).hexdigest()
    summary["recipe_content_sha256"] = recipe.content_sha256
    write_json(output / "summary.json", summary)
    for old, new in (
        ("selection.json", "selection.json"),
        ("provenance.json", "source_manifest.json"),
    ):
        shutil.copyfile(source / old, output / new)
    gallery = []
    for row in rows:
        target = output / "cases" / row["case"]
        target.mkdir(parents=True)
        for name in ("raw_report.json", "icp_report.json", "evaluation.json"):
            shutil.copyfile(results / row["case"] / name, target / name)
        image = panel(results, row)
        cv2.imwrite(str(target / "comparison.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 91])
        gallery.append((row, image))
    for present, filename in ((True, "positive_overview.jpg"), (False, "negative_overview.jpg")):
        tiles = [
            cv2.resize(image, (690, 270))
            for row, image in gallery
            if row["selection"]["target_present"] == present
        ]
        while len(tiles) % 2:
            tiles.append(np.full_like(tiles[0], 246))
        sheet = np.vstack([np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)])
        cv2.imwrite(str(output / filename), sheet, [cv2.IMWRITE_JPEG_QUALITY, 91])
    selected_reviews = []
    for row, reason in select_reviews(rows):
        name, selected = row["case"], row["selection"]
        target = output / "reviews" / name
        target.mkdir(parents=True)
        frame, detection = load_frame(source, selected)
        for mode in ("raw", "icp"):
            report = json.loads((results / name / f"{mode}_report.json").read_bytes())
            points, colors = result_cloud(recipe.mesh, frame, report)
            write_cloud(target / f"{mode}_fit.ply", points, colors, frame.frame)
            shutil.copyfile(results / name / f"{mode}_overlay.png", target / f"{mode}_overlay.png")
        folder = source / f"test_primesense/{selected['scene']:06d}"
        shutil.copyfile(folder / f"rgb/{selected['frame']:06d}.png", target / "source.png")
        shutil.copyfile(folder / f"depth/{selected['frame']:06d}.png", target / "depth.png")
        camera = json.loads((folder / "scene_camera.json").read_bytes())[str(selected["frame"])]
        request = dict(
            schema="simple_gdrnpp_request/v1",
            capture_id=name,
            rgb_path="source.png",
            depth_path="depth.png",
            depth_scale_to_m=camera["depth_scale"] / 1000,
            camera_matrix=frame.camera.tolist(),
            model_frame=recipe.mesh.frame,
            camera_frame=frame.frame,
            extents_m=np.ptp(recipe.mesh.vertices, axis=0).tolist(),
            detections=[
                dict(identifier=detection.identifier, box=detection.box, source=detection.source)
            ],
            distortion=[],
        )
        write_json(target / "capture.json", request)
        selected_reviews.append(dict(case=name, reason=reason))
    write_json(
        output / "review_selection.json",
        dict(
            scope=(
                "Post-run review selection only; "
                "aggregate evaluation includes every preselected case"
            ),
            reviews=selected_reviews,
        ),
    )
    sections = []
    for row in rows:
        name = row["case"]
        flags = "positive" if row["selection"]["target_present"] else "negative"
        if row["icp"]["accepted"]:
            flags += " accepted"
        sections.append(
            f'<section data-kind="{flags}"><h2>{html.escape(name)}</h2>'
            f'<a href="cases/{name}/comparison.jpg"><img alt="Raw versus local alignment" '
            f'src="cases/{name}/comparison.jpg"></a>'
            f'<p><a href="cases/{name}/evaluation.json">Pose and fit evaluation</a> · '
            f'<a href="cases/{name}/raw_report.json">Raw report</a> · '
            f'<a href="cases/{name}/icp_report.json">Alignment report</a></p></section>'
        )
    document = (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<title>SimpleGDRNPP scene extension</title>
<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:0 16px;
background:#f6f6f6;color:#222}img{width:100%;max-width:920px}
section{background:white;border:1px solid #bbb;padding:12px;margin:18px 0}
h2{font-size:18px}button{padding:8px;margin:5px}a{color:#145bb0}</style>
<h1>Real scene extension: raw pose and optional local depth alignment</h1>
<p>All preselected cases are retained. Depth acceptance and annotation error are separate.
Supplied rectangles; unchanged 5 mm / 70% support / 10% contradiction gates.
Fitting and verification use the same depth.
This is not detection, heavy-occlusion or grasp qualification.</p>
<p>T-LESS, Hodan et al., WACV 2017,
<a href="https://cmp.felk.cvut.cz/t-less/">CC BY 4.0 and attribution</a>.
Derived crops/overlays/clouds are modifications; no endorsement implied.</p>
<button onclick="filter('')">All cases</button>
<button onclick="filter('positive')">Target present</button>
<button onclick="filter('negative')">Target absent</button>
<button onclick="filter('accepted')">Depth accepted</button>
<p><a href="summary.json">Complete summary and timing scopes</a> ·
<a href="README.md">Method and limitations</a></p>
"""
        + "\n".join(sections)
        + """<script>function filter(kind){document.querySelectorAll('section').forEach(
e=>e.hidden=kind&&!e.dataset.kind.split(' ').includes(kind));}</script></html>"""
    )
    (output / "gallery.html").write_text(document, encoding="utf-8", newline="\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("source", "results", "recipe", "new_output"):
        parser.add_argument(argument, type=Path)
    args = parser.parse_args()
    curate(args.source, args.results, args.recipe, args.new_output)
