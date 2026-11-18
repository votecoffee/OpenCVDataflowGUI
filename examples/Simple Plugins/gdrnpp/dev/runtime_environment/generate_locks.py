"""Regenerate reviewed Windows CPU wheel locks; never install into any environment."""

import argparse
import subprocess
from pathlib import Path

SCOPE = (
    "sys_platform == 'win32' and platform_machine == 'AMD64' "
    "and implementation_name == 'cpython' and python_full_version == '3.13.9'"
)


def generate(output, uv="uv"):
    root = Path(__file__).resolve().parent
    if output.exists():
        raise FileExistsError("Choose a fresh directory for proposed locks")
    output.mkdir(parents=True)
    for mode in ("runtime", "validation"):
        path = output / f"pylock.{mode}.toml"
        subprocess.run(
            [
                uv,
                "pip",
                "compile",
                str(root / f"{mode}.in"),
                "--constraint",
                str(root / "constraints.txt"),
                "--python-version",
                "3.13.9",
                "--python-platform",
                "x86_64-pc-windows-msvc",
                "--torch-backend",
                "cpu",
                "--index-url",
                "https://pypi.org/simple",
                "--format",
                "pylock.toml",
                "--only-binary",
                ":all:",
                "--no-annotate",
                "--no-header",
                "--output-file",
                str(path),
                "--quiet",
            ],
            check=True,
        )
        raw = path.read_bytes()
        assert raw.count(b'lock-version = "1.0"\n') == 1
        path.write_bytes(
            raw.replace(
                b'lock-version = "1.0"\n',
                ('lock-version = "1.0"\nenvironments = ["' + SCOPE + '"]\n').encode(),
                1,
            )
        )
        print(f"Generated {path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new_output", type=Path)
    parser.add_argument("--uv", default="uv")
    args = parser.parse_args()
    generate(args.new_output, args.uv)
