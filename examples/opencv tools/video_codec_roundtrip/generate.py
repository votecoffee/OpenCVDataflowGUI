"""Build original video fixtures and portable recording/replay graphs."""

import json
from pathlib import Path

import cv2

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.video_qualification import fixture_frames


def main():
    folder = Path(__file__).resolve().parent
    assets = folder / "assets"
    assets.mkdir(exist_ok=True)
    paths = []
    for index, image in enumerate(fixture_frames()):
        path = assets / f"frame-{index:02d}.png"
        assert cv2.imwrite(str(path), image)
        paths.append(path.relative_to(folder).as_posix())
    truth = {
        "origin": "Original independently specified color bars and moving-part fixture; MIT",
        "frame_count": 8,
        "frame_size": [128, 96],
        "fps": 12.5,
        "media_times_ms": [80 * i for i in range(8)],
        "part_centers": [[16 + 9 * i, 60] for i in range(8)],
        "bgr_primary_samples": [[255, 0, 0], [0, 255, 0], [0, 0, 255], [125, 125, 125]],
        "gray_primary_samples": [29, 150, 76, 125],
        "ffv1_maximum_absolute_error_counts": 0,
    }
    (assets / "truth.json").write_text(
        json.dumps(truth, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    registry = builtin_registry()
    for filename, record in (("project.json", True), ("replay.json", False)):
        graph = Graph()
        try:
            source = graph.add(
                registry.create(
                    "input.file" if record else "input.video_file",
                    block_id="source",
                    parameters={
                        "path_base": "project",
                        **(
                            {"source_mode": "files", "paths": paths, "color_format": "rgb"}
                            if record
                            else {"path": "recorded.avi", "backend": "ffmpeg"}
                        ),
                    },
                )
            )
            source.name = (
                "Eight original color/moving targets"
                if record
                else "Replay finalized FFV1 recording"
            )
            source.position, source.node_size = [0, 60], "large"
            if record:
                writer = graph.add(
                    registry.create(
                        "output.video",
                        block_id="record",
                        parameters={
                            "path": "recorded.avi",
                            "path_base": "project",
                            "codec": "FFV1",
                            "backend": "ffmpeg",
                            "fps": 12.5,
                        },
                    )
                )
                writer.position = [440, 60]
                graph.connect(source.id, "image", writer.id, "image")
            note = graph.add(registry.create("layout.comment", block_id="about"))
            note.name, note.position, note.node_size = (
                "Original video fixtures - MIT",
                [0, -260],
                "large",
            )
            note.description = (
                "Original synthetic images, generator and graphs under the repository's "
                "MIT license. "
                "Run all in project.json to record eight RGB-tagged images as lossless BGR FFV1 "
                "at 12.5 FPS. Closing execution finalizes recorded.avi. "
                "Open replay.json and Run all "
                "to decode the eight frames at media times 0,80,...,560 ms. The part center moves "
                "from (16,60) to (79,60) in steps of 9 pixels. Existing recordings are preserved; "
                "choose a new destination to record again. The block info acknowledges frame "
                "submission; the headless qualification command verifies finalized streams. "
                "API: https://docs.opencv.org/5.0/main_modules/classcv_1_1VideoWriter.html"
            )
            graph.save(folder / filename)
        finally:
            graph.close()


if __name__ == "__main__":
    main()
