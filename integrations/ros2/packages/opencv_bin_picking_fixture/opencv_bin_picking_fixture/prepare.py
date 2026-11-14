"""Create a new private runtime overlay without changing the installed environment."""
import argparse
import hashlib
import json
from pathlib import Path

from .scene import bridge_config, default_scene, digest, load, world_xml


def prepare(runtime_root, output, scene=None):
    runtime_root = Path(runtime_root).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists() or output == runtime_root or runtime_root in output.parents:
        raise ValueError("Choose a new output directory outside the runtime root")
    scene = default_scene() if scene is None else scene
    world = world_xml(scene)
    copies = {}
    for name in ("ur_sim_control_windows.launch.py", "ur_moveit_windows.launch.py", "gripper_description.py"):
        path = runtime_root / "launch" / name
        copies[f"launch/{name}"] = path.read_bytes()
    for path in (runtime_root / "config").iterdir():
        if path.is_file() and path.suffix in {".yaml", ".rviz", ".config"}:
            copies[f"config/{path.name}"] = path.read_bytes()
    native = runtime_root / ".pixi/envs/default/Library/bin/gz-sim-detachable-joint-system.dll"
    if b"initial_attach" not in native.read_bytes():
        raise ValueError("Installed DetachableJoint lacks the explicit initially-unattached setting")
    original_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in copies.items()}
    output.mkdir(parents=True)
    (output / "generated").mkdir()
    (output / "logs").mkdir()
    (output / "scene.json").write_text(json.dumps(scene, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "world.sdf").write_text(world, encoding="utf-8")
    # JSON is also valid YAML and accepted by yaml-cpp's bridge-config parser.
    (output / "bridge.yaml").write_text(json.dumps(bridge_config(scene), indent=2) + "\n", encoding="utf-8")
    for name, data in copies.items():
        path = output / name; path.parent.mkdir(parents=True, exist_ok=True)
        if name.startswith("launch/"):
            text = data.decode("utf-8")
            text = text.replace(runtime_root.as_posix(), output.as_posix()).replace(str(runtime_root), output.as_posix())
            if name.endswith("gripper_description.py"):
                text += "\n# Additive fixture hook; this copy is private to one simulation run.\n"
                text += "from opencv_bin_picking_fixture.scene import add_assistance as _fixture_add, load as _fixture_load\n"
                text += "_fixture_original_gripper = add_gripper_urdf\n"
                text += "def add_gripper_urdf(description):\n"
                text += f"    return _fixture_add(_fixture_original_gripper(description), _fixture_load({str(output / 'scene.json')!r}))\n"
            if "C:/dev/ros2-sim/generated" in text or "C:/dev/ros2-sim2/generated" in text:
                raise ValueError("Native generated output path was not isolated")
            compile(text, name, "exec")
            path.write_text(text, encoding="utf-8", newline="\n")
        else:
            path.write_bytes(data)
    for name, digest_value in original_hashes.items():
        if hashlib.sha256((runtime_root / name).read_bytes()).hexdigest() != digest_value:
            raise ValueError("Runtime source changed while preparing the overlay; discard and retry with its owner")
    record = {"runtime_root": str(runtime_root), "overlay": str(output), "scene_sha256": digest(scene),
              "native_detachable_joint_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
              "copied_original_sha256": original_hashes, "scope": "private additive simulation overlay; no simulator launched"}
    (output / "preparation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scene", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.runtime_root, args.output, load(args.scene) if args.scene else None), indent=2))


if __name__ == "__main__":
    main()
