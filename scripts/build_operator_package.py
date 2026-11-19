"""Assemble an offline Windows x64 operator installer from a verified app wheel."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import tempfile
import zipfile
from email import message_from_bytes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    "numpy",
    "opencv-contrib-python",
    "scipy",
    "PySide6",
    "PySide6_Addons",
    "PySide6_Essentials",
    "shiboken6",
    "Pillow",
)
INSTALLER = r"""@echo off
setlocal
set "PYTHONPATH="
set "PYTHONHOME="
pushd "%~dp0" || exit /b 2
if not exist ".venv\Scripts\python.exe" (
    py -3.14 -m venv .venv
    if errorlevel 1 (
        echo Install Windows x64 CPython 3.14 with the Python launcher and try again.
        exit /b 2
    )
)
".venv\Scripts\python.exe" -c ^
    "import sys; assert sys.version_info[:2] == (3,14)"
if errorlevel 1 exit /b 2
".venv\Scripts\python.exe" -c "import sysconfig; assert sysconfig.get_platform()=='win-amd64'"
if errorlevel 1 exit /b 2
".venv\Scripts\python.exe" -m pip install --no-index --only-binary=:all: ^
    --require-hashes --force-reinstall --find-links wheelhouse -r requirements.txt
if errorlevel 1 exit /b 2
".venv\Scripts\python.exe" -m pip check
exit /b %ERRORLEVEL%
"""


def launcher(module, arguments):
    return (
        '@echo off\nsetlocal\nset "PYTHONPATH="\nset "PYTHONHOME="\n'
        'if not exist "%~dp0.venv\\Scripts\\python.exe" (\n'
        "    echo Run Install.cmd first.\n    exit /b 2\n)\n"
        f'"%~dp0.venv\\Scripts\\python.exe" -m {module} {arguments}\n'
        "exit /b %ERRORLEVEL%\n"
    )


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(app_wheel, output):
    app_wheel, output = app_wheel.resolve(strict=True), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if sys.platform != "win32" or sys.version_info[:2] != (3, 14):
        raise ValueError("Build with the application's Windows x64 Python 3.14")
    with tempfile.TemporaryDirectory(prefix="dataflow-operator-package-") as scratch:
        root = Path(scratch)
        wheels = root / "wheelhouse"
        wheels.mkdir()
        requirements = [f"{name}=={importlib.metadata.version(name)}" for name in PACKAGES]
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "--only-binary=:all:",
                "--dest",
                str(wheels),
                str(app_wheel),
                *requirements,
            ],
            check=True,
        )
        hashes = {}
        locks = []
        for path in sorted(wheels.glob("*.whl")):
            with zipfile.ZipFile(path) as wheel:
                metadata = next(n for n in wheel.namelist() if n.endswith(".dist-info/METADATA"))
                package = message_from_bytes(wheel.read(metadata))
            sha = digest(path)
            hashes[path.name] = sha
            locks.append(f"{package['Name']}=={package['Version']} --hash=sha256:{sha}")
        (root / "requirements.txt").write_text("\n".join(locks) + "\n", encoding="utf-8")
        (root / "WHEEL-SHA256.json").write_text(
            json.dumps(hashes, indent=2) + "\n", encoding="utf-8"
        )
        (root / "Install.cmd").write_text(INSTALLER, encoding="ascii", newline="\r\n")
        operator = launcher("opencv_dataflow.operator_window", "%*")
        command = '"%~dp0.venv\\Scripts\\python.exe" -m opencv_dataflow.operator_window'
        operator = operator.replace(
            f"{command} %*",
            'if "%~1"=="" (\n'
            f'    {command} "%~dp0examples\\smoke-project.json"\n'
            "    exit /b\n)\n"
            f"{command} %*",
        )
        (root / "Start.cmd").write_text(operator, encoding="ascii", newline="\r\n")
        (root / "Run.cmd").write_text(
            launcher("opencv_dataflow.operator_runtime", "%*"), encoding="ascii", newline="\r\n"
        )
        (root / "examples").mkdir()
        (root / "examples/smoke-project.json").write_bytes(
            (ROOT / "examples/user_tools/assistant_workflow/assistant_workflow.json").read_bytes()
        )
        (root / "README.md").write_text(
            "# OpenCV Dataflow Operator for Windows x64\n\n"
            "Requires 64-bit CPython 3.14 with the Windows Python launcher. "
            "Extract the ZIP to a writable folder and run Install.cmd. "
            "All application and dependency wheels are included and verified by hash.\n\n"
            "Open Start.cmd for the operator window. It opens a synthetic smoke example "
            "by default. Pick a trusted saved project to run your inspection.\n\n"
            "For headless execution, run `Run.cmd run examples\\smoke-project.json`.\n"
            "Query recent tickets with `Run.cmd history`, or add `--sessions` for run history.\n"
            "The default database is %LOCALAPPDATA%/OpenCVDataflowGUI/runtime.sqlite3.\n\n"
            "Move your projects together with their relative assets. Install optional "
            "integrations or isolated model workers separately for jobs that require them.\n",
            encoding="utf-8",
        )
        archive = root / output.name
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
            for path in sorted(root.rglob("*")):
                if path.is_file() and path != archive:
                    bundle.write(path, path.relative_to(root).as_posix())
        output.parent.mkdir(parents=True, exist_ok=True)
        with archive.open("rb") as source, output.open("xb") as target:
            import shutil

            shutil.copyfileobj(source, target)
    return dict(file=str(output), sha256=digest(output), wheels=len(hashes))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--wheel", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    print(json.dumps(build(args.wheel, args.output), indent=2))


if __name__ == "__main__":
    main()
