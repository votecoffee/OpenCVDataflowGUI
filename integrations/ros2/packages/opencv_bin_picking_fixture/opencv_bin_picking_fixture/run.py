"""Apply isolation after Pixi activation, then enter the source ROS launch."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import uuid


def environment(overlay, partition, current=None):
    overlay = Path(overlay).resolve(strict=True)
    if not re.fullmatch(r"opencv_bin_pick_[A-Za-z0-9_-]{1,80}", partition):
        raise ValueError("Choose a unique opencv_bin_pick_ partition")
    preparation = json.loads((overlay / "preparation.json").read_text(encoding="utf-8"))
    if Path(preparation["overlay"]).resolve() != overlay:
        raise ValueError("Preparation receipt belongs to another overlay")
    values = dict(os.environ if current is None else current)
    values.update(ROS_DOMAIN_ID="130", ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST", GZ_PARTITION=partition,
                  ROS_HOME=str(overlay / "state/ros"), ROS_LOG_DIR=str(overlay / "logs"))
    if values.get("ROS_STATIC_PEERS", "").strip():
        raise ValueError("Remove explicit static ROS peers before this localhost-only simulation")
    package = Path(__file__).resolve().parents[1]
    values["PYTHONPATH"] = str(package) + os.pathsep + values.get("PYTHONPATH", "")
    return preparation, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    preparation, values = environment(args.overlay, args.partition)
    package = Path(__file__).resolve().parents[1]
    launch = package / "launch/fixture.launch.py"
    if not launch.is_file():
        from ament_index_python.packages import get_package_share_directory
        launch = Path(get_package_share_directory("opencv_bin_picking_fixture")) / "launch/fixture.launch.py"
    plan = {"runtime_root": preparation["runtime_root"], "overlay": str(args.overlay.resolve()),
            "ros_domain_id": values["ROS_DOMAIN_ID"], "gz_partition": values["GZ_PARTITION"],
            "simulation_only": True, "headless": not args.gui, "launch_file": str(launch),
            "note": "Isolation is applied inside the activated child; saved native defaults remain unchanged"}
    if args.plan_only:
        print(json.dumps(plan, indent=2)); return 0
    native_python = Path(preparation["runtime_root"]) / ".pixi/envs/default/python.exe"
    if Path(sys.executable).resolve() != native_python.resolve(strict=True):
        raise RuntimeError("Run this module with the preparation receipt's activated native runtime Python")
    # One explicit launch per overlay; a crash is not permission to replace its owner.
    lease = args.overlay / "session-launch.json"
    with lease.open("x", encoding="utf-8") as stream:
        json.dump({**plan, "launch_id": str(uuid.uuid4()), "owner_pid": os.getpid()}, stream, indent=2)
        stream.write("\n")
    os.environ.update(values)
    from ros2cli.cli import main as ros_main
    return ros_main(args=["launch", str(launch), f"overlay:={args.overlay.resolve()}",
                          "gui:=true" if args.gui else "gui:=false"])


if __name__ == "__main__":
    raise SystemExit(main())
