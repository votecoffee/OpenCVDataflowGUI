"""Rebuild photo-backed demos from pinned, already bundled open-data captures.

No downloads, controller access or new training are performed. Original synthetic
graphs are retained as independent regression fixtures on the first migration.
"""

import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "examples/assets/showcase"
FIXTURES = ROOT / "examples/assets/synthetic_graphs"
DEMOS = ROOT / "examples/demos"
REGISTRY = builtin_registry()
NAMES = (
    "demo",
    "starter_inspection",
    "regions",
    "object_collections",
    "overlay_result_graphics",
    "image_set_workspace",
    "validation_images",
    "sequence_memory",
    "rgbd_cloud",
)
VISA_CREDIT = (
    "VisA, CC BY 4.0. Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang, "
    "Onkar Dabeer; SPot-the-Difference, ECCV 2022. registry.opendata.aws/visa/. "
    "Resized/letterboxed for this example; no defects were added."
)
ARM_CREDIT = (
    "ARMBench, CC BY 4.0. Amazon.com, Inc.; Chaitanya Mitash, Fan Wang, Shiyang Lu, "
    "Vikedo Terhuja, Tyler Garaas, Felipe Polido, Manikantan Nambi; ICRA 2023. "
    "armbench.com. Resized/letterboxed; occupancy labels are example-authored."
)
TLESS_CREDIT = (
    "T-LESS, CC BY 4.0. Tomas Hodan, Pavel Haluza, Stepan Obdrzalek, Jiri Matas, "
    "Manolis Lourakis, Xenophon Zabulis; WACV 2017. cmp.felk.cvut.cz/t-less/. "
    "Real Primesense scene 1, frame 220; original depth and intrinsics retained."
)
TUM_CREDIT = (
    "TUM RGB-D, CC BY 4.0. J. Sturm, N. Engelhard, F. Endres, W. Burgard, D. Cremers; "
    "IROS 2012. cvg.cit.tum.de/data/datasets/rgbd-dataset. freiburg1_rpy RGB frames "
    "384-443, resized to 320x240 grayscale FFV1; replayed here without injected motion."
)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def write(path, data):
    old = path.read_bytes() if path.exists() else b""
    eol = "\r\n" if b"\r\n" in old and old.count(b"\r\n") == old.count(b"\n") else "\n"
    path.write_bytes(
        (json.dumps(data, indent=2, allow_nan=False) + "\n").replace("\n", eol).encode("utf8")
    )


def relative(target, owner):
    return os.path.relpath(target, owner.parent).replace("\\", "/")


def original(name):
    path = FIXTURES / (name + ".json")
    if not path.exists():
        shutil.copyfile(DEMOS / name / (name + ".json"), path)
    return read(path)


def block(kind, ident, name, position, parameters=None):
    value = REGISTRY.create(kind, block_id=ident, parameters=parameters or {})
    value.name, value.position = name, list(position)
    return value.to_project_item()


def photo_source(old, path, owner, name):
    value = block(
        "input.file",
        old["id"],
        name,
        old["position"],
        {
            "path": relative(path, owner),
            "path_base": "project",
            "source_mode": "single",
            "repeat": "stop",
            "filename_ticket": True,
            "color_format": "bgr",
            "pixel_depth": "uint8",
        },
    )
    return value


def wire(data, source, output, target, input):
    data["connections"].append(
        {"source": source, "output": output, "target": target, "input": input}
    )


def note(data, text):
    if any(b["id"] == "showcase_source" for b in data["blocks"]):
        return
    value = block("layout.comment", "showcase_source", "Real data: source and scope", [0, -680])
    value["description"] = text
    value["node_size"] = "large"
    data["blocks"].append(value)


def fit_photo(source, target):
    image = cv2.imread(str(source))
    scale = min(480 / image.shape[1], 320 / image.shape[0])
    resized = cv2.resize(
        image,
        (round(image.shape[1] * scale), round(image.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )
    left, top = (480 - resized.shape[1]) // 2, (320 - resized.shape[0]) // 2
    output = cv2.copyMakeBorder(
        resized,
        top,
        320 - resized.shape[0] - top,
        left,
        480 - resized.shape[1] - left,
        cv2.BORDER_CONSTANT,
    )
    assert cv2.imwrite(str(target), output)
    return {
        "source": str(source.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": sha(source),
        "file": target.name,
        "sha256": sha(target),
        "source_size": list(image.shape[1::-1]),
        "resized_size": list(resized.shape[1::-1]),
        "padding_left_top": [left, top],
        "changes": "aspect-preserving area resize and black letterbox to 480x320",
        "license": "CC-BY-4.0",
    }


def tote_graph(owner, validation=False):
    data = original("validation_images" if validation else "image_set_workspace")
    source_id, final_id = "validation_source", "validation_final"
    # Both old fixtures use these IDs; the graph's saved groups/view stay intact.
    old = next(b for b in data["blocks"] if b["id"] == source_id)
    paths = [
        ASSETS / (n + ".png")
        for n in ("armbench-EZ5895NGB8", "armbench-P2S0835EOA", "armbench-9TARTKRF1X")
    ]
    source = photo_source(old, paths[0], owner, "Real tote photographs")
    if not validation:
        source["parameters"].update(source_mode="files", paths=[relative(p, owner) for p in paths])
    count = block(
        "inspect.contours",
        "validation_inspect",
        "Inspect clear tote cavity",
        [850, 200],
        {"min_area": 400, "expected_count": 0, "region_mode": "clip"},
    )
    final = next(b for b in data["blocks"] if b["id"] == final_id)
    final["position"] = [1150, 200]
    final["parameters"].update(
        fail_reason="tote_occupied",
        reason_message="Non-blue foreground in the inspected tote cavity",
    )
    data["blocks"] = [
        source,
        block(
            "cv.channel_transform",
            "blue",
            "Blue-plastic signal",
            [300, 80],
            {"matrix": [[2.0, -1.0, -1.0]]},
        ),
        block(
            "region.shape",
            "cavity",
            "Inspected tote cavity",
            [300, 400],
            {"rectangle": [105, 40, 290, 245]},
        ),
        block(
            "cv.compare_pixels",
            "foreground",
            "Non-blue foreground",
            [580, 80],
            {"operation": "less", "scalar": 25},
        ),
        count,
        final,
        block(
            "cv.overlay_result_graphics",
            "photo_overlay",
            "Inspection on the real photograph",
            [1450, 200],
        ),
        block("output.filmstrip_image", "filmstrip", "Ticketed photo result", [1780, 200]),
    ]
    data["connections"] = []
    for args in [
        (source_id, "image", "blue", "image"),
        (source_id, "image", "cavity", "image"),
        ("blue", "image", "foreground", "image_a"),
        ("cavity", "region", "foreground", "region"),
        ("foreground", "mask", count["id"], "image"),
        (count["id"], "count", final_id, "value"),
        (source_id, "image", "photo_overlay", "image"),
        (count["id"], "result_graphics", "photo_overlay", "graphics_1"),
        ("photo_overlay", "image", "filmstrip", "image"),
    ]:
        wire(data, *args)
    count["exposed_outputs"].append("result_graphics")
    if data.get("validation_sets"):
        cases = [
            ("empty_reference", "LO95CK8OLH", "must_pass"),
            ("empty_capture", "EZ5895NGB8", "should_pass"),
            ("mixed_inventory", "P2S0835EOA", "must_fail"),
            ("packages", "9TARTKRF1X", "should_fail"),
            ("bagged_items", "OAR1KXNIME", "unknown"),
        ]
        data["validation_sets"][0]["cases"] = [
            {
                "id": i,
                "path": relative(ASSETS / ("armbench-" + n + ".png"), owner),
                "expectation": e,
                **({"preferred_fail_reason": "tote_occupied"} if e.endswith("fail") else {}),
            }
            for i, n, e in cases
        ]
        data["validation_sets"][0]["name"] = "Real tote occupancy regression"
    note(
        data,
        ARM_CREDIT
        + " Native fixed-color cavity check, not item counting or trained instance segmentation. "
        "Touching items can merge; colors, pose and lighting affect the rule. "
        "Labels are scoped to these photographs.",
    )
    return data


def main():
    ASSETS.mkdir(exist_ok=True)
    FIXTURES.mkdir(exist_ok=True)
    for name in NAMES:
        original(name)
    records = []
    for stem in ("0000", "0621"):
        records.append(
            {
                "dataset": "VisA",
                **fit_photo(
                    ROOT / f"examples/assets/visa-pcb2-normal-{stem}.jpg",
                    ASSETS / f"visa-pcb2-{stem}.png",
                ),
            }
        )
    for name in ("LO95CK8OLH", "EZ5895NGB8", "P2S0835EOA", "9TARTKRF1X", "OAR1KXNIME"):
        records.append(
            {
                "dataset": "ARMBench",
                **fit_photo(
                    ROOT
                    / f"examples/Simple Plugins/reference_inspect/armbench_tote/assets/{name}.jpg",
                    ASSETS / f"armbench-{name}.png",
                ),
            }
        )
    pcb = ASSETS / "visa-pcb2-0000.png"
    for name in ("demo", "starter_inspection", "regions", "overlay_result_graphics"):
        owner = DEMOS / name / (name + ".json")
        data = original(name)
        old = data["blocks"][0]
        data["blocks"][0] = photo_source(old, pcb, owner, "Real VisA PCB photograph")
        by = {b["id"]: b for b in data["blocks"]}
        if name == "demo":
            by["demo_contours"]["parameters"].update(min_area=5000, expected_count=1)
        elif name == "starter_inspection":
            by["starter_crop"]["parameters"].update(x=60, y=45, width=325, height=195)
            by["starter_morphology"]["parameters"].update(operation="close", kernel=5)
            by["starter_components"]["parameters"].update(min_area=5000, expected_count=1)
            by["starter_components"]["name"] = "Measure the PCB foreground"
        elif name == "regions":
            by["roi_circle"]["parameters"].update(center=[130, 155], radius=65)
            by["roi_rectangle"]["parameters"].update(rectangle=[285, 120, 80, 90])
            by["roi_polygon"]["parameters"].update(
                vertices=[[75, 85], [370, 85], [370, 230], [75, 230]]
            )
            by["roi_left"]["parameters"].update(min_area=500, region_mode="clip", expected_count=2)
            by["roi_right"]["parameters"].update(min_area=500, region_mode="clip")
            by["roi_left"]["name"] = "Measure left PCB area"
            by["roi_right"]["name"] = "Inspect right PCB area"
            by["roi_circle"]["name"] = "Circle selects left PCB area"
            by["roi_rectangle"]["name"] = "Rectangle selects right PCB area"
            by["roi_polygon"]["name"] = "Polygon selects the board"
        else:
            by["region"]["parameters"].update(rectangle=[75, 85, 295, 145])
            by["contours"]["parameters"].update(min_area=5000, expected_count=1)
        note(
            data,
            VISA_CREDIT + " Thresholded regions illustrate graph editing and overlays. "
            "No calibrated defect labels are supplied.",
        )
        write(owner, data)

    owner = DEMOS / "object_collections/object_collections.json"
    data = original("object_collections")
    old = data["blocks"][0]
    data["blocks"][0] = photo_source(
        old, ASSETS / "armbench-P2S0835EOA.png", owner, "Real warehouse inventory"
    )
    by = {b["id"]: b for b in data["blocks"]}
    by["objects_roi"]["parameters"].update(x=0, y=0, width=480, height=320)
    by["objects_roi"]["name"] = "Full photograph"
    by["objects_gray"].update(
        type="cv.channel_transform",
        parameters={"matrix": [[2.0, -1.0, -1.0]]},
        name="Blue-plastic signal",
    )
    by["objects_threshold"].update(
        type="cv.compare_pixels",
        parameters={"operation": "less", "scalar": 25},
        name="Non-blue foreground",
    )
    by["objects_threshold"]["exposed_inputs"] = ["image_a", "region"]
    by["objects_threshold"]["exposed_outputs"] = ["mask", "count"]
    for edge in data["connections"]:
        if edge["target"] == "objects_threshold":
            edge["input"] = "image_a"
        if edge["source"] == "objects_threshold":
            edge["output"] = "mask"
    data["blocks"].append(
        block(
            "region.shape", "cavity", "Tote cavity", [300, -350], {"rectangle": [105, 40, 290, 245]}
        )
    )
    wire(data, old["id"], "image", "cavity", "image")
    wire(data, "cavity", "region", "objects_threshold", "region")
    by["objects_blobs"]["parameters"].update(min_area=400)
    by["objects_blob_filter"]["parameters"].update(metric="area", minimum=1000)
    by["objects_blob_filter"]["name"] = "Keep larger foreground regions"
    by["objects_contours"]["parameters"].update(min_area=400, expected_count=4)
    by["objects_contour_filter"]["parameters"].update(minimum=0.4)
    by["objects_line_length"]["parameters"].update(minimum=50)
    note(
        data,
        ARM_CREDIT + " Connected foreground regions can merge touching objects. "
        "This example demonstrates collections, sorting and overlays.",
    )
    write(owner, data)

    for name in ("image_set_workspace", "validation_images"):
        owner = DEMOS / name / (name + ".json")
        write(owner, tote_graph(owner, name == "validation_images"))

    owner = DEMOS / "sequence_memory/sequence_memory.json"
    data = original("sequence_memory")
    old = data["blocks"][0]
    clip = ROOT / "examples/opencv tools/recorded_camera_stabilization/assets/fr1-rpy-0384-0443.mkv"
    data["blocks"][0] = block(
        "input.video_file",
        old["id"],
        "Real TUM camera frames",
        old["position"],
        {
            "path": relative(clip, owner),
            "path_base": "project",
            "end_frame": 12,
            "repeat": "stop",
            "color_format": "gray",
        },
    )
    note(
        data,
        TUM_CREDIT + " Camera motion contributes to optical flow and background change. "
        "No encoder or robot pose is supplied.",
    )
    write(owner, data)

    owner = DEMOS / "rgbd_cloud/rgbd_cloud.json"
    data = original("rgbd_cloud")
    source = ROOT / "examples/Simple Plugins/3d_locate/real_tless/source/test_primesense/000001"
    camera = read(source / "scene_camera.json")["220"]
    k = camera["cam_K"]
    data["blocks"] = [b for b in data["blocks"] if b["id"] != "pair"]
    data["blocks"][:0] = [
        block(
            "input.file",
            "rgb",
            "Real T-LESS RGB capture",
            [0, 0],
            {"path": relative(source / "rgb/000220.png", owner), "path_base": "project"},
        ),
        block(
            "input.depth_file",
            "depth",
            "Measured Primesense depth",
            [0, 300],
            {
                "path": relative(source / "depth/000220.png", owner),
                "path_base": "project",
                "units": "m",
                "units_per_sample": camera["depth_scale"] * 0.001,
            },
        ),
        block(
            "3d.camera",
            "camera",
            "Publisher camera intrinsics",
            [0, 600],
            {
                "fx": k[0],
                "fy": k[4],
                "cx": k[2],
                "cy": k[5],
                "width": 720,
                "height": 540,
                "units": "m",
            },
        ),
    ]
    for b in data["blocks"]:
        if b["id"] == "filter":
            b["parameters"].update(minimum=[-0.3, -0.3, 0.3], maximum=[0.3, 0.3, 1.5])
        if b["id"] == "sample":
            b["parameters"].update(voxel_size=0.004)
        if b["id"] in ("project", "filter", "sample", "normals"):
            b["preview_output"] = "cloud"
    for edge in data["connections"]:
        if edge["source"] == "pair":
            edge["source"] = {"depth": "depth", "left": "rgb", "camera": "camera"}[edge["output"]]
            edge["output"] = {"depth": "depth", "rgb": "image", "camera": "camera"}[edge["source"]]
    note(
        data,
        TLESS_CREDIT
        + " Publisher depth_scale=0.1 mm/sample (0.0001 m/sample). No object mask is used.",
    )
    write(owner, data)

    # Two material/color applications use real PCB regions; known synthetic color
    # values remain in independent regression fixtures.
    for name in ("lab_color_match", "lab_color_classification"):
        owner = ROOT / f"examples/applications/{name}/{name}.json"
        saved = FIXTURES / (name + ".json")
        if not saved.exists():
            shutil.copyfile(owner, saved)
        data = read(saved)
        data["blocks"][0] = photo_source(data["blocks"][0], pcb, owner, "Real blue PCB substrate")
        by = {b["id"]: b for b in data["blocks"]}
        if name == "lab_color_match":
            by["reference_region"]["parameters"].update(rectangle=[180, 204, 14, 8])
            by["sample_region"]["parameters"].update(rectangle=[285, 204, 14, 8])
            by["compare"]["parameters"].update(maximum_delta_e=15)
        else:
            by["region"]["parameters"].update(rectangle=[285, 204, 14, 8])
            photo = cv2.imread(str(pcb)).astype(np.float32) / 255
            lab = cv2.cvtColor(photo, cv2.COLOR_BGR2Lab)
            colors = {
                "blue PCB substrate": lab[204:212, 180:194].mean(axis=(0, 1)).tolist(),
                "gray IC package": lab[150:162, 310:335].mean(axis=(0, 1)).tolist(),
            }
            by["classify"]["parameters"].update(
                references=colors, maximum_delta_e=15, minimum_margin=1
            )
        note(
            data,
            VISA_CREDIT + " Uncalibrated photographic Lab comparison. Regions and palette are "
            "example-authored. No colorimeter ground truth is supplied.",
        )
        write(owner, data)

    owner = (
        ROOT
        / "examples/industrial integration/controllogix_trigger_and_writeback"
        / "controllogix_trigger_and_writeback.json"
    )
    saved = FIXTURES / "controllogix_trigger_and_writeback.json"
    if not saved.exists():
        shutil.copyfile(owner, saved)
    data = read(saved)
    by = {b["id"]: b for b in data["blocks"]}
    capture = photo_source(by["capture"], pcb, owner, "Real PCB image on PLC ticket")
    capture["parameters"]["filename_ticket"] = False
    capture["ticket_settings"] = {"mode": "external", "value": 1}
    data["blocks"][data["blocks"].index(by["capture"])] = capture
    by["count"]["parameters"].update(min_area=5000, expected_count=1)
    note(
        data,
        VISA_CREDIT + " Saved photo replay. PLC reader/writer remain disabled with blank routes. "
        "No live-controller operation is performed.",
    )
    write(owner, data)
    offline = copy.deepcopy(data)
    offline["blocks"] = [b for b in offline["blocks"] if b["id"] not in ("read", "write")]
    offline["connections"] = [
        e for e in offline["connections"] if e["source"] != "read" and e["target"] != "write"
    ]
    source = next(b for b in offline["blocks"] if b["id"] == "capture")
    source.pop("ticket_settings", None)
    source["parameters"]["filename_ticket"] = True
    write(owner.with_name("offline_photo_replay.json"), offline)

    write(
        ASSETS / "sources.json",
        {
            "license_checked": "2026-11-13",
            "images": records,
            "datasets": {
                "VisA": {
                    "publisher": "https://registry.opendata.aws/visa/",
                    "credit": VISA_CREDIT,
                    "release": "VisA_20220922",
                    "license": "CC-BY-4.0",
                },
                "ARMBench": {
                    "publisher": "https://www.armbench.com/",
                    "credit": ARM_CREDIT,
                    "license": "CC-BY-4.0",
                    "archive_sha256": read(
                        ROOT
                        / "examples/Simple Plugins/reference_inspect/armbench_tote/provenance.json"
                    )["archive_sha256"],
                },
                "TUM RGB-D": {
                    "publisher": "https://cvg.cit.tum.de/data/datasets/rgbd-dataset",
                    "credit": TUM_CREDIT,
                    "clip_sha256": sha(clip),
                },
                "T-LESS": {
                    "publisher": "https://cmp.felk.cvut.cz/t-less/",
                    "credit": TLESS_CREDIT,
                    "license": "CC-BY-4.0",
                    "camera_sha256": sha(
                        ROOT
                        / "examples/Simple Plugins/3d_locate/real_tless/source"
                        / "test_primesense/000001/scene_camera.json"
                    ),
                    "revision": read(
                        ROOT / "examples/Simple Plugins/3d_locate/real_tless/source/provenance.json"
                    )["revision"],
                    "rgb_sha256": sha(
                        source_path := ROOT
                        / "examples/Simple Plugins/3d_locate/real_tless/source"
                        / "test_primesense/000001/rgb/000220.png"
                    ),
                    "depth_sha256": sha(source_path.parent.parent / "depth/000220.png"),
                },
            },
        },
    )
    print(
        "Built nine real-data demos, two real-photo color applications "
        "and a disabled PLC template/offline replay"
    )


if __name__ == "__main__":
    main()
