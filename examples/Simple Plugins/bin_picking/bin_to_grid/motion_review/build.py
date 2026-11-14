"""Build offline native-cloud reviews from retained simulation evidence."""

from pathlib import Path

from opencv_simple_bin_picking.sim_dataflow import register

from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry


def build(root):
    registry = builtin_registry()
    register(registry)
    for name, cell in (("baseline", 6), ("shifted_stacks", 6), ("held_transfer", 3)):
        directory = Path(root) / name
        project = directory / "project.json"
        if project.exists():
            raise FileExistsError(project)
        with ProjectSession(registry, directory=directory) as session:
            operations = [
                dict(
                    op="add",
                    type="input.sim_pick_review",
                    id="review",
                    parameters={"archive_directory": "assets/run", "cell": cell},
                )
            ]
            for identifier, filename in (
                ("final_camera", "final-camera.png"),
                ("summary", "summary.png"),
            ):
                if (directory / filename).is_file():
                    operations.append(
                        dict(
                            op="add",
                            type="input.file",
                            id=identifier,
                            parameters={
                                "path": filename,
                                "path_base": "project",
                                "filename_ticket": False,
                            },
                        )
                    )
            operations.append(dict(op="arrange"))
            session.apply(operations)
            block = session.graph.blocks["review"]
            block.preview_output = "motion"
            block.description = "Phase-colored planned TCP lines; pink measured-joint FK dots."
            if not session.inspect()["validation"]["valid"]:
                raise RuntimeError("Offline motion review failed validation")
            session.save(project)


if __name__ == "__main__":
    build(Path(__file__).parent)
