"""Headless, local encoder qualification with original moving/color fixtures.

Run with ``python -m opencv_dataflow.video_qualification --output-dir NEW_FOLDER``.
An opened writer is only a submission endpoint; qualification decodes the
finalized clip and checks pixels, frame order, count, rate and media time.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .payloads import NO_VALUE
from .video_files import VideoFile, WriteVideo

CASES = {
    "ffv1_bgr": ("FFV1", "ffmpeg", "bgr", 12.5, "avi", True),
    "ffv1_gray": ("FFV1", "ffmpeg", "gray", 12.5, "avi", True),
    "mjpg_bgr": ("MJPG", "ffmpeg", "bgr", 12.5, "avi", False),
    "mjpg_gray": ("MJPG", "ffmpeg", "gray", 12.5, "avi", False),
    "mp4v_bgr": ("mp4v", "ffmpeg", "bgr", 12.5, "mp4", False),
    "opencv_mjpeg_bgr": ("MJPG", "opencv_mjpeg", "bgr", 10.0, "avi", False),
}


def fixture_frames():
    """Eight independent BGR targets, with known primaries and moving-part center."""
    images = []
    for index in range(8):
        image = np.full((96, 128, 3), 40 + index, np.uint8)
        for column, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255), (125,) * 3)):
            image[:32, column * 32 : (column + 1) * 32] = color
        # Analytic center is (16 + 9*index, 60); no detection API defines truth.
        image[48:73, 8 + 9 * index : 25 + 9 * index] = 225
        images.append(image)
    return images


def expected_gray(image):
    """Independent BT.601 reference for these gray/primary-color fixtures."""
    return np.rint(np.asarray(image, float) @ np.array([0.114, 0.587, 0.299])).astype(np.uint8)


def qualify_video_codec(directory, case):
    """Write and close a fresh clip, then decode it through the real source block.

    Returns explicit passed/failed/unavailable status. A missing native codec is
    unqualified, rather than a passing skip. Existing files are never replaced.
    """
    codec, backend, color, fps, extension, lossless = CASES[case]
    path = Path(directory) / f"{case}.{extension}"
    report = {
        "case": case,
        "requested": {"codec": codec, "backend": backend, "color": color, "fps": fps},
        "file": path.name,
        "status": "failed",
        "lossless_required": lossless,
    }
    writer = WriteVideo(parameters={"path": str(path), **report["requested"]})
    reader = VideoFile(parameters={"path": str(path), "backend": "ffmpeg", "color_format": color})
    expected = fixture_frames()
    if color == "gray":
        # Feed BGR images to the gray writer to qualify the actual conversion.
        expected = [expected_gray(image) for image in expected]
    try:
        for index, image in enumerate(fixture_frames()):
            result = writer.execute({"image": image})
            if result is None:
                report["error"] = writer.error
                if index == 0 and "OpenCV cannot open video codec" in writer.error:
                    report["status"] = "unavailable"
                return report
        report["submission"] = result.outputs["info"]
        writer.close()
        capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        try:
            if not capture.isOpened():
                report.update(
                    status="unavailable", error="FFmpeg decoder cannot open finalized clip"
                )
                return report
            report["container"] = {
                "reported_frame_count": capture.get(cv2.CAP_PROP_FRAME_COUNT),
                "fps": capture.get(cv2.CAP_PROP_FPS),
                "decoder_backend": capture.getBackendName(),
            }
        finally:
            capture.release()
        decoded = []
        for index in range(len(expected) + 1):
            if reader.exhausted:
                break
            result = reader.execute({})
            if result is None:
                report["error"] = reader.error
                return report
            image = result.outputs["image"]
            if image is NO_VALUE:
                break
            if (
                index >= len(expected)
                or image.shape != expected[index].shape
                or image.dtype != np.uint8
            ):
                report["error"] = "Decoded an extra frame or incorrect frame shape/depth"
                return report
            error = np.abs(np.asarray(image, float) - expected[index])
            plane = expected_gray(image) if image.ndim == 3 else image
            yy, xx = np.nonzero(plane[44:80] > 170)
            center_error = (
                float(np.hypot(xx.mean() - (16 + 9 * index), yy.mean() + 44 - 60))
                if len(xx)
                else None
            )
            patches = [
                float(error[4:28, column * 32 + 4 : column * 32 + 28].mean(axis=(0, 1)).max())
                for column in range(4)
            ]
            info = result.outputs["info"]
            decoded.append(
                {
                    **info,
                    "ticket": result.outputs["ticket"]["value"],
                    "mae_per_channel": np.atleast_1d(error.mean(axis=(0, 1))).tolist(),
                    "maximum_absolute_error": float(error.max()),
                    "maximum_patch_mean_error": max(patches),
                    "part_center_error_pixels": center_error,
                }
            )
        report["frames"] = decoded
        checks = {
            "frame_count_and_eof": len(decoded) == len(expected) and reader.exhausted,
            "reported_count": report["container"]["reported_frame_count"] == len(expected),
            "rate": abs(report["container"]["fps"] - fps) <= 0.001,
            "order": all(
                frame["part_center_error_pixels"] is not None
                and frame["part_center_error_pixels"] <= 0.5
                and frame["frame_index"] == index
                for index, frame in enumerate(decoded)
            ),
            "media_time": all(
                frame["timestamp_ms"] is not None
                and abs(frame["timestamp_ms"] - 1000 * index / fps) <= 0.01
                for index, frame in enumerate(decoded)
            ),
            "pixels": all(
                frame["maximum_absolute_error"] == 0
                if lossless
                else max(frame["mae_per_channel"]) <= 8 and frame["maximum_patch_mean_error"] <= 12
                for frame in decoded
            ),
        }
        report["checks"] = checks
        report["status"] = "passed" if all(checks.values()) else "failed"
        return report
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        return report
    finally:
        writer.close()
        reader.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="New folder for clips and report.json"
    )
    parser.add_argument(
        "--case", choices=tuple(CASES), action="append", help="Default: all six cases"
    )
    args = parser.parse_args(argv)
    # Exclusive directory creation preserves every prior recording and receipt.
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.error(f"Cannot create a new output directory: {error}")
    reports = [
        qualify_video_codec(args.output_dir, case) for case in dict.fromkeys(args.case or CASES)
    ]
    report = {
        "opencv_version": cv2.__version__,
        "passed": all(case["status"] == "passed" for case in reports),
        "cases": reports,
    }
    with (args.output_dir / "report.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for case in reports:
        print(f"{case['case']}: {case['status']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
