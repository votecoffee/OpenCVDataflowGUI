"""Keep practical examples discoverable without exposing their development bundles."""

import json
from pathlib import Path

import pytest

from opencv_dataflow.assistant_cli import example_index


def test_simple_plugin_discovery_omits_dev_only(tmp_path):
    payload = '{"schema_version":1,"blocks":[],"connections":[]}'
    names = (
        "Simple Plugins/tool/pcb/project.json",
        "Simple Plugins/tool/dev/regression/project.json",
        "Simple Plugins/dev/gallery/project.json",
        "Simple Plugins/tool/DEV/hidden.json",
        "opencv tools/dev/project.json",
    )
    for name in names:
        path = tmp_path / "examples" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    paths = {Path(r["project"]).relative_to(tmp_path / "examples").as_posix()
             for r in example_index(tmp_path)}
    assert paths == {names[0], names[-1]}


@pytest.mark.local_examples
def test_every_tool_has_documented_direct_runnable_folders():
    root = Path(__file__).resolve().parents[1] / "examples/Simple Plugins"
    layout = json.loads((root / "dev/layout.json").read_bytes())
    assert len(layout["tools"]) == 14
    for tool, entries in layout["tools"].items():
        folder = root / tool
        assert (folder / "README.md").is_file()
        assert (folder / "dev/README.md").is_file()
        assert {p.name for p in folder.iterdir() if p.is_dir() and p.name != "dev"} == {
            entry["folder"] for entry in entries
        }
        for entry in entries:
            case = folder / entry["folder"]
            assert (case / "README.md").is_file()
            assert entry["projects"]
            for name in entry["projects"]:
                data = json.loads((case / name).read_bytes())
                assert data["schema_version"] == 1 and isinstance(data["blocks"], list)
