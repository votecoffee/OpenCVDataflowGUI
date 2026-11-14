"""Install the maintained skills without overwriting different local content."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from tempfile import mkdtemp

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("opencv-dataflow-apps", "opencv-dataflow-plugins")


def install(destination, *, root=ROOT, interpreter=None):
    destination, root = Path(destination).resolve(), Path(root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    interpreter = Path(interpreter or sys.executable).resolve()
    records = []
    for name in NAMES:
        source = root / "skills" / name
        files = {
            p.relative_to(source): p.read_bytes()
            for p in source.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        }
        files[Path("references/installation.md")] = (
            "# Local application installation\n\n"
            f"Source checkout: `{root}`\n\nPython interpreter: `{interpreter}`\n\n"
            f"Documentation map: [{root / 'docs/README.md'}](<{root / 'docs/README.md'}>)\n\n"
            f"Plan template: [{root / 'docs/application-plan-template.md'}]"
            f"(<{root / 'docs/application-plan-template.md'}>)\n\n"
            "Use this interpreter with `-m opencv_dataflow.assistant_cli`. "
            "Verify it using `info` before application work.\n"
        ).encode("utf-8")
        target = destination / name
        if target.exists():
            differences = [
                str(p)
                for p, raw in files.items()
                if not (target / p).is_file() or (target / p).read_bytes() != raw
            ]
            if differences:
                raise FileExistsError(
                    f"Preserved existing skill {target}; different files: {differences}. "
                    "Choose another --destination to review the new version."
                )
            state = "unchanged"
        else:
            temporary = Path(mkdtemp(prefix=f".{name}-", dir=destination))
            try:
                for path, raw in files.items():
                    output = temporary / path
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(raw)
                temporary.rename(target)
            finally:
                if temporary.exists():
                    if temporary.resolve().parent != destination or temporary.is_symlink():
                        raise ValueError("Refusing cleanup outside the skill staging directory")
                    shutil.rmtree(temporary)  # Only this function's known staging directory.
            state = "installed"
        records.append(
            {
                "skill": name,
                "path": str(target),
                "state": state,
                "files": {str(p): hashlib.sha256(raw).hexdigest() for p, raw in files.items()},
            }
        )
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path.home() / ".codex" / "skills")
    args = parser.parse_args(argv)
    try:
        print(json.dumps({"skills": install(args.destination)}, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
