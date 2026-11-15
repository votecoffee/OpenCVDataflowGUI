"""Launch one privately prepared simulation with generation-bound shutdown."""

import _thread
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import uuid
from pathlib import Path


def close_owned_world(overlay):
    """The Windows Ruby wrapper can outlive launch's child list; match only our world."""
    if os.name != "nt":
        return {"native_cleanup": "not_required"}
    script = """
$worldPath = $env:OPENCV_HAND_EYE_OWNED_WORLD.Replace('\\','/').ToLowerInvariant()
$owned = @(Get-CimInstance Win32_Process -Filter "Name='gz-sim-main.exe'" | Where-Object {
    $_.CommandLine -and $_.CommandLine.Replace('\\','/').ToLowerInvariant().Contains($worldPath)
})
$closed = @()
foreach ($item in $owned) {
    $current = Get-CimInstance Win32_Process -Filter ("ProcessId=" + $item.ProcessId)
    if ($current -and $current.CreationDate -eq $item.CreationDate -and
        $current.CommandLine -eq $item.CommandLine) {
        Stop-Process -Id $current.ProcessId -ErrorAction Stop
        $closed += $current.ProcessId
    }
}
@{closed_world_server_pids=$closed} | ConvertTo-Json -Compress
"""
    env = {**os.environ, "OPENCV_HAND_EYE_OWNED_WORLD": str(Path(overlay).resolve() / "world.sdf")}
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    return json.loads(result.stdout)


def environment(overlay, partition):
    overlay = Path(overlay).resolve(strict=True)
    if not re.fullmatch(r"opencv_hand_eye_[A-Za-z0-9_-]{1,80}", partition):
        raise ValueError("Use a unique opencv_hand_eye_ partition")
    record = json.loads((overlay / "preparation.json").read_text())
    if Path(record["overlay"]).resolve() != overlay:
        raise ValueError("Preparation belongs to another overlay")
    for name, expected in record["prepared_sha256"].items():
        if hashlib.sha256((overlay / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Prepared input changed: " + name)
    package = Path(__file__).resolve().parents[1]
    for name, expected in record.get("package_source_sha256", {}).items():
        if hashlib.sha256((package / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Example source changed after preparation: " + name)
    values = dict(os.environ)
    if values.get("ROS_STATIC_PEERS", "").strip():
        raise ValueError("Explicit discovery peers must be empty")
    values.update(
        ROS_DOMAIN_ID="44",
        ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST",
        GZ_PARTITION=partition,
        ROS_HOME=str(overlay / "state/ros"),
        ROS_LOG_DIR=str(overlay / "logs"),
    )
    values["PYTHONPATH"] = (
        str(Path(__file__).resolve().parents[1]) + os.pathsep + values.get("PYTHONPATH", "")
    )
    return record, values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--overlay", type=Path, required=True)
    p.add_argument("--partition", required=True)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--gui", action="store_true")
    args = p.parse_args()
    record, values = environment(args.overlay, args.partition)
    if args.plan_only:
        print(
            json.dumps(
                dict(
                    overlay=str(args.overlay),
                    domain=44,
                    partition=args.partition,
                    headless=not args.gui,
                )
            )
        )
        return
    if Path(sys.executable).resolve() != (
        Path(record["runtime_root"]) / ".pixi/envs/default/python.exe"
    ).resolve(strict=True):
        raise RuntimeError("Use the allocated activated native interpreter")
    os.environ.update(values)
    launch_id = str(uuid.uuid4())
    with (args.overlay / "session-launch.json").open("x") as f:
        json.dump(
            dict(launch_id=launch_id, owner_pid=os.getpid(), domain=44, partition=args.partition), f
        )
    finished = threading.Event()

    def stop():
        while not finished.wait(0.2):
            try:
                request = json.loads((args.overlay / "stop-request.json").read_text())
            except OSError, ValueError:
                continue
            if request == dict(action="stop", launch_id=launch_id):
                _thread.interrupt_main()
                return

    threading.Thread(target=stop, daemon=True).start()
    from ros2cli.cli import main as ros_main

    result = None
    try:
        launch = Path(__file__).resolve().parents[1] / "launch/hand_eye.launch.py"
        if not launch.is_file():
            from ament_index_python.packages import get_package_share_directory

            launch = (
                Path(get_package_share_directory("opencv_hand_eye_demo"))
                / "launch/hand_eye.launch.py"
            )
        result = ros_main(
            argv=[
                "launch",
                str(launch),
                f"overlay:={args.overlay.resolve()}",
                f"gui:={str(args.gui).lower()}",
            ]
        )
    finally:
        finished.set()
        cleanup = close_owned_world(args.overlay)
        (args.overlay / "session-exit.json").write_text(
            json.dumps(
                dict(
                    launch_id=launch_id,
                    return_code=result,
                    cleanup=cleanup,
                    note="Owned world server cleanup attempted; verify remaining child completion",
                )
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
