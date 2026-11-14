"""Exercise application authoring, conflict protection, and runtime receipts."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from opencv_dataflow.assistant_api import (
    ProjectSession,
    RevisionConflict,
    block_schema,
    catalog,
    file_revision,
    make_registry,
    validate_graph,
)
from opencv_dataflow.assistant_artifacts import export_previews
from opencv_dataflow.assistant_cli import example_index, main
from opencv_dataflow.core import Block, Port, Registry
from opencv_dataflow.toolblocks import encapsulate


def inspection_operations():
    return [
        {"op": "add", "type": "input.demo", "id": "source"},
        {"op": "add", "type": "cv.gray", "id": "gray"},
        {"op": "add", "type": "cv.gaussian_blur", "id": "blur"},
        {"op": "add", "type": "cv.threshold", "id": "threshold", "parameters": {"value": 100}},
        {"op": "add", "type": "inspect.contours", "id": "parts", "parameters": {"min_area": 1000}},
        *(
            {"op": "connect", "source": a, "output": "image", "target": b, "input": "image"}
            for a, b in zip(
                ["source", "gray", "blur", "threshold"], ["gray", "blur", "threshold", "parts"]
            )
        ),
        {"op": "arrange"},
    ]


@pytest.fixture
def session():
    with ProjectSession() as session:
        session.apply(inspection_operations())
        yield session


def test_catalog_discovers_optional_without_loading_and_describes_real_ports():
    registry = make_registry()
    before = set(registry.types)
    records = catalog(registry)
    assert "3d.simple_measure" in {r["type"] for r in records}
    assert set(registry.types) == before
    assert "cv.gaussian_blur" in {r["type"] for r in catalog(registry, "GaussianBlur")}
    schema = block_schema("cv.gaussian_blur", registry)
    assert schema["parameters"]["kernel"]["range"]["odd"]
    assert "reflect" in schema["parameters"]["border"]["choices"]
    assert {p["name"] for p in schema["inputs"]} >= {"image", "mask", "region"}
    source = block_schema("input.demo", registry)
    assert "ticket" in {p["name"] for p in source["inputs"]}
    assert source["capabilities"]["ticket_source"]
    assert any(p["name"] == "result_graphics" and not p["exposed"] for p in schema["outputs"])


def test_create_edit_reopen_run_and_review(session, tmp_path):
    path = tmp_path / "inspection.json"
    session.apply(
        [
            {
                "op": "update",
                "id": "parts",
                "fields": {
                    "description": "Count the two synthetic parts",
                    "documentation": "<p>Test fixture.</p>",
                    "node_size": "large",
                },
            }
        ]
    )
    receipt = session.save(path)
    assert receipt["revision"] == file_revision(path)
    with ProjectSession.open(path) as loaded:
        assert loaded.inspect()["validation"]["valid"]
        assert loaded.graph.blocks["parts"].node_size == "large"
        assert loaded.graph.blocks["parts"].description.startswith("Count")
        assert loaded.graph.blocks["gray"].position[0] > loaded.graph.blocks["source"].position[0]
        report = loaded.run(repeat=2)
        assert report["passed"] and len(report["runs"]) == 2
        parts = next(b for b in report["runs"][0]["blocks"] if b["id"] == "parts")
        assert parts["outputs"]["count"] == 2
        assert report["ticket_results"]["tickets"]
        assert any(e["kind"] == "start" for e in report["activity"]["events"])
        previews = export_previews(loaded.graph, tmp_path / "review", max_images=3)
        assert len(previews["images"]) == 3 and previews["omitted"] > 0
        for preview in previews["images"]:
            assert Image.open(preview["path"]).size[0] <= 1024
        with pytest.raises(FileExistsError):
            export_previews(loaded.graph, tmp_path / "review")


@pytest.mark.parametrize(
    "operation",
    [
        {"op": "update", "id": "blur", "fields": {"parameters": {"kernel": 4}}},
        {"op": "update", "id": "blur", "fields": {"parameters": {"typo": 1}}},
        {
            "op": "connect",
            "source": "parts",
            "output": "count",
            "target": "source",
            "input": "ticket",
        },  # Introduces a cycle.
        {
            "op": "connect",
            "source": "parts",
            "output": "count",
            "target": "blur",
            "input": "region",
        },  # Scalar to region data may be allowed; image is not.
        {"op": "connect", "source": "parts", "output": "count", "target": "blur", "input": "image"},
        {"op": "remove", "id": "gray"},
        {"op": "update", "id": "blur", "fields": {"node_size": "enormous"}},
        {"op": "add", "id": "source", "type": "input.demo"},
        {"op": "update", "id": "blur", "fields": {"parameterz": {}}},
        {"op": "settings", "fields": {"branch_workers": 0}},
    ],
)
def test_bad_batch_does_not_change_session_or_file(session, tmp_path, operation):
    path = tmp_path / "inspection.json"
    session.save(path)
    before = session.inspect()
    raw = path.read_bytes()
    # A successful first operation must also roll back when the second fails.
    with pytest.raises((ValueError, KeyError, TypeError)):
        session.apply([{"op": "update", "id": "parts", "fields": {"name": "changed"}}, operation])
    assert session.inspect() == before
    assert path.read_bytes() == raw


def test_static_validation_does_not_process_or_open_inputs(tmp_path):
    with ProjectSession(directory=tmp_path) as session:
        session.apply(
            [
                {
                    "op": "add",
                    "type": "input.file",
                    "id": "image",
                    "parameters": {"path": "does-not-exist.png", "path_base": "project"},
                },
                {"op": "add", "type": "cv.gray", "id": "gray"},
            ]
        )
        validation = validate_graph(session.graph)
        assert not validation["valid"] and not validation["executed"]
        assert validation["issues"][0]["fields"] == ["image"]
        assert all(b.last_result is None for b in session.graph.blocks.values())
        session.apply(
            [
                {
                    "op": "connect",
                    "source": "image",
                    "output": "image",
                    "target": "gray",
                    "input": "image",
                }
            ]
        )
        assert validate_graph(session.graph)["valid"]  # Existence is a runtime rule.
        report = session.run()
        assert not report["passed"]
        assert session.graph.blocks["image"].state == "error"


def test_stale_revision_and_existing_destination_preserve_bytes(session, tmp_path):
    path = tmp_path / "inspection.json"
    session.save(path)
    with ProjectSession.open(path) as second:
        second.apply([{"op": "update", "id": "parts", "fields": {"name": "User edit"}}])
        second.save()
    raw = path.read_bytes()
    with pytest.raises(RevisionConflict):
        session.save()
    assert path.read_bytes() == raw
    with pytest.raises(RevisionConflict):
        ProjectSession().save(path)
    assert not list(tmp_path.glob(".*.assistant.lock"))
    assert not list(tmp_path.glob(".assistant-*"))


def test_active_writer_lock_is_not_removed(session, tmp_path):
    path = tmp_path / "inspection.json"
    lock = tmp_path / ".inspection.json.assistant.lock"
    lock.write_text("another writer", encoding="utf-8")
    with pytest.raises(RevisionConflict):
        session.save(path)
    assert lock.read_text() == "another writer"
    assert not path.exists()


def test_nested_scopes_accept_bound_inputs_and_edit_children(session, tmp_path):
    tool = encapsulate(
        session.graph,
        ["gray", "blur", "threshold", "parts"],
        "Reusable inspection",
        session.registry,
    )
    assert validate_graph(session.graph)["valid"]
    session.apply(
        [
            {
                "op": "update",
                "scope": [tool.id],
                "id": "blur",
                "fields": {"parameters": {"kernel": 7}},
            }
        ]
    )
    assert session.graph.blocks[tool.id].graph.blocks["blur"].parameters["kernel"] == 7
    path = tmp_path / "nested.json"
    session.save(path)
    with ProjectSession.open(path) as loaded:
        assert loaded.run()["passed"]


def test_save_as_preserves_project_relative_inputs(session, tmp_path):
    original = tmp_path / "source"
    destination = tmp_path / "destination"
    original.mkdir()
    destination.mkdir()
    Image.fromarray(np.full((40, 60, 3), 150, np.uint8)).save(original / "input.png")
    with ProjectSession(directory=original) as project:
        project.apply(
            [
                {
                    "op": "add",
                    "type": "input.file",
                    "id": "image",
                    "parameters": {
                        "path": "input.png",
                        "path_base": "project",
                    },
                }
            ]
        )
        project.save(original / "project.json")
        project.save(destination / "copy.json")
        assert project.graph.blocks["image"].parameters["path"] == str(
            Path("..") / "source" / "input.png"
        )
    with ProjectSession.open(destination / "copy.json") as loaded:
        assert loaded.run()["passed"]


def test_remove_connected_block_is_explicit_and_conversion_rules_match_runtime(session):
    session.apply([{"op": "remove", "id": "blur", "disconnect": True}])
    assert "blur" not in session.graph.blocks
    assert not validate_graph(session.graph)["valid"]
    assert session.connection("number", "integer")["compatible"]
    assert not session.connection("image", "integer")["compatible"]


class Broken(Block):
    type_id = "test.broken"
    outputs = (Port("value", "number"),)

    def process(self, inputs):
        raise RuntimeError("Plugin failure fixture")


def test_plugin_exception_has_diagnostics_and_does_not_crash_client(tmp_path):
    registry = Registry()
    registry.register(Broken)
    with ProjectSession(registry) as session:
        session.apply([{"op": "add", "type": "test.broken", "id": "broken"}])
        report = session.run()
        assert not report["passed"]
        events = report["activity"]["events"]
        assert any(
            "Plugin failure fixture" in (e.get("exception") or {}).get("traceback", "")
            for e in events
        )


@pytest.mark.parametrize(
    "options", [{"repeat": 0}, {"repeat": 3, "max_runs": 2}, {"all_images": True}]
)
def test_execution_bounds(session, options):
    with pytest.raises(ValueError):
        session.run(**options)


def test_cancelled_run_is_incomplete(session):
    report = session.run(cancelled=lambda: True)
    assert not report["complete"] and not report["passed"]
    assert report["stop_reason"] == "cancelled" and not report["runs"]


def test_finite_replay_and_limit_are_distinguished(tmp_path):
    for index in range(3):
        Image.fromarray(np.full((20, 30), index * 50, np.uint8)).save(tmp_path / f"{index}.png")

    def replay():
        session = ProjectSession(directory=tmp_path)
        session.apply(
            [
                {
                    "op": "add",
                    "type": "input.file",
                    "id": "source",
                    "parameters": {
                        "source_mode": "directory",
                        "directory": ".",
                        "path_base": "project",
                    },
                }
            ]
        )
        return session

    with replay() as session:
        report = session.run(all_images=True)
        assert report["passed"] and len(report["runs"]) == 3
        assert report["stop_reason"] == "sources_exhausted"
    with replay() as session:
        report = session.run(all_images=True, max_runs=2)
        assert not report["complete"] and report["stop_reason"] == "max_runs"


def test_cli_json_protocol_errors_and_working_copy(session, tmp_path, capsys):
    path = tmp_path / "inspection.json"
    revision = session.save(path)["revision"]
    operations = tmp_path / "operations.json"
    operations.write_text(
        json.dumps([{"op": "update", "id": "parts", "fields": {"name": "CLI edit"}}])
    )
    assert main(["apply", str(path), "--operations", str(operations)]) == 3
    assert json.loads(capsys.readouterr().out)["error"]["type"] == "RevisionConflict"
    assert main(["apply", str(path), "--operations", str(operations), "--dry-run"]) == 0
    assert not json.loads(capsys.readouterr().out)["data"]["saved"]
    assert file_revision(path) == revision
    assert (
        main(["apply", str(path), "--operations", str(operations), "--expected-revision", revision])
        == 0
    )
    changed = json.loads(capsys.readouterr().out)["data"]["revision"]
    assert changed != revision
    assert (
        main(
            [
                "apply",
                str(path),
                "--operations",
                str(operations),
                "--dry-run",
                "--expected-revision",
                revision,
            ]
        )
        == 3
    )
    capsys.readouterr()
    assert main(["no-such-command"]) == 2
    assert json.loads(capsys.readouterr().out)["api_version"] == 1


def test_cli_new_run_reports_previews_exports_and_json_only(tmp_path, capsys):
    operations = tmp_path / "operations.json"
    operations.write_text(json.dumps(inspection_operations()))
    path = tmp_path / "new.json"
    assert main(["new", "--operations", str(operations), "--output", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["data"]["saved"]
    report = tmp_path / "receipt.json"
    assert (
        main(["run", str(path), "--report", str(report), "--artifacts", str(tmp_path / "review")])
        == 0
    )
    response = json.loads(capsys.readouterr().out)
    assert response["data"]["passed"]
    assert json.loads(report.read_text())["runs"][0]["blocks"]
    assert response["data"]["artifacts"]["images"]
    before = report.read_bytes()
    assert main(["run", str(path), "--report", str(report)]) == 2
    capsys.readouterr()
    assert report.read_bytes() == before
    for format in ("xml", "html"):
        output = tmp_path / f"program.{format}"
        assert main(["export", str(path), "--format", format, "--output", str(output)]) == 0
        assert json.loads(capsys.readouterr().out)["data"]["format"] == format
        assert output.stat().st_size > 100
        assert main(["export", str(path), "--format", format, "--output", str(output)]) == 2
        capsys.readouterr()


def test_plugin_stdout_does_not_corrupt_cli_json(tmp_path):
    plugin = tmp_path / "plugin.py"
    plugin.write_text("print('plugin loaded')\ndef register(registry):\n    pass\n")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "opencv_dataflow.assistant_cli",
            "--plugin",
            str(plugin),
            "catalog",
            "--query",
            "grayscale",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["ok"]
    assert "plugin loaded" in result.stderr


def test_example_index_omits_private_data_and_does_not_execute(tmp_path):
    for category in ("demos", "private", "custom tools"):
        directory = tmp_path / "examples" / category / "fixture"
        directory.mkdir(parents=True)
        (directory / "project.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "blocks": [{"type": "not.installed"}],
                    "connections": [],
                }
            )
        )
        (directory / "recipe.json").write_text('{"recipe": true}')
    index = example_index(tmp_path)
    assert len(index) == 2
    assert all("private" not in record["project"] for record in index)


def test_invalid_parameter_error_is_structured(session, tmp_path, capsys):
    path = tmp_path / "invalid.json"
    session.graph.blocks["blur"].parameters["kernel"] = 4
    session.graph.save(path)
    assert main(["validate", str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["data"]["issues"][0]["fields"] == ["kernel"]
    assert main(["run", str(path)]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["issues"][0]["block"] == "blur"


def test_operation_json_rejects_duplicate_fields(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text('[{"op":"add","op":"remove"}]')
    assert main(["new", "--operations", str(path), "--output", str(tmp_path / "p.json")]) == 2
    assert "Duplicate" in json.loads(capsys.readouterr().out)["error"]["message"]


def test_example_index_honors_local_git_ignores(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / ".gitignore").write_text("examples/applications/customer/\n")
    for folder in ("customer", "public_sample"):
        directory = tmp_path / "examples/applications" / folder
        directory.mkdir(parents=True)
        (directory / "project.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "blocks": [{"type": "input.file"}],
                    "connections": [],
                }
            )
        )
    index = example_index(tmp_path)
    assert len(index) == 1 and "public_sample" in index[0]["project"]


def test_compact_run_keeps_full_report_in_file(session, tmp_path, capsys):
    path = tmp_path / "project.json"
    session.save(path)
    report = tmp_path / "report.json"
    assert main(["run", str(path), "--report", str(report), "--summary"]) == 0
    data = json.loads(capsys.readouterr().out)["data"]
    assert data["run_count"] == 1 and not data["failures"]
    assert "activity" not in data
    assert json.loads(report.read_text())["activity"]["events"]
