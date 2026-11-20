"""Prepare exclusive private launch copies; never edit the native runtime."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from .scene import bridges, world_xml


_SHARED_SIM_ROOT = re.compile(r"\b[a-z]:[/\\]+dev[/\\]+ros2-sim2?(?=[/\\'\"\s]|$)", re.IGNORECASE)


def _rebase_launch(raw, runtime, output):
    root = str(runtime)
    variants = sorted({runtime.as_posix(), root, root.replace("\\", "\\\\")}, key=len, reverse=True)
    pattern = "(?:" + "|".join(re.escape(value) for value in variants) + r")(?=[/\\'\"\s]|$)"
    flags = re.IGNORECASE if runtime.drive else 0
    text = re.sub(pattern, lambda _: output.as_posix(), raw.decode("utf-8"), flags=flags)
    if _SHARED_SIM_ROOT.search(text):
        raise ValueError("Launch copy still references a shared simulator root")
    return text


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(runtime, config, output):
    runtime = Path(runtime).resolve(strict=True)
    config = Path(config).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists() or output == runtime or runtime in output.parents:
        raise ValueError("Choose a new overlay outside the native runtime")
    scene_raw = config.read_bytes()
    scene = json.loads(scene_raw)
    if scene.get("schema") != "opencv.hand_eye_scene/v1" or scene.get("units") != "m":
        raise ValueError("Expected a metric hand-eye scene")
    copies = {
        f"launch/{name}": (runtime / "launch" / name).read_bytes()
        for name in (
            "ur_sim_control_windows.launch.py",
            "ur_moveit_windows.launch.py",
            "gripper_description.py",
        )
    }
    for item in (runtime / "config").iterdir():
        if item.is_file() and item.suffix in {".yaml", ".rviz", ".config"}:
            copies[f"config/{item.name}"] = item.read_bytes()
    rebased = {}
    for name, raw in copies.items():
        if name.startswith("launch/"):
            rebased[name] = _rebase_launch(raw, runtime, output)
    output.mkdir(parents=True)
    for name in ("generated", "logs", "state"):
        (output / name).mkdir()
    (output / "scene.json").write_bytes(scene_raw)
    (output / "world.sdf").write_text(world_xml(scene), encoding="utf-8")
    (output / "bridge.yaml").write_text(json.dumps(bridges(), indent=2), encoding="utf-8")
    for name, raw in copies.items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name.startswith("launch/"):
            text = rebased[name]
            if name.endswith("gripper_description.py"):
                text += "\nfrom opencv_hand_eye_demo.scene import add_camera as _add_camera\n"
                text += "import json as _json\n"
                text += "_original_gripper = add_gripper_urdf\n"
                text += "def add_gripper_urdf(description):\n"
                text += f"    with open({str(output / 'scene.json')!r}, encoding='utf-8') as f:\n"
                text += (
                    "        return _add_camera(_original_gripper(description), _json.load(f))\n"
                )
            compile(text, name, "exec")
            destination.write_text(text, encoding="utf-8", newline="\n")
        else:
            destination.write_bytes(raw)
    if any(digest((runtime / n).read_bytes()) != digest(raw) for n, raw in copies.items()):
        raise ValueError("Runtime changed during preparation")
    record = dict(
        schema="opencv.hand_eye_preparation/v1",
        runtime_root=str(runtime),
        overlay=str(output),
        scene_sha256=digest(scene_raw),
        copied_source_sha256={n: digest(b) for n, b in copies.items()},
        prepared_sha256={
            n: digest((output / n).read_bytes())
            for n in [*copies, "scene.json", "world.sdf", "bridge.yaml"]
        },
    )
    package = Path(__file__).resolve().parents[1]
    record["package_source_sha256"] = {
        p.relative_to(package).as_posix(): digest(p.read_bytes())
        for folder in ("opencv_hand_eye_demo", "launch")
        for p in (package / folder).glob("*.py")
    }
    (output / "preparation.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    print(json.dumps(prepare(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
