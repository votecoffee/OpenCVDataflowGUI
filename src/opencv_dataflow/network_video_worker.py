"""Private fixed-protocol decoder. Wait for parent ownership before importing OpenCV."""

import json
import math
import os
import sys
import time
from multiprocessing.shared_memory import SharedMemory


def reply(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def main():
    settings = json.loads(sys.stdin.buffer.readline(16384))
    # Native diagnostics can contain passwords/URL query strings. The parent reports codes.
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 2)
    import cv2
    import numpy as np

    memory = SharedMemory(name=settings["memory"], track=False)
    capture = cv2.VideoCapture()
    previous, index, shape = None, 0, None
    try:
        reply({"status": "ready"})
        for line in sys.stdin.buffer:
            command = json.loads(line)
            if command["op"] == "open":
                ok = capture.open(
                    settings["url"],
                    cv2.CAP_FFMPEG,
                    [
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                        settings["open_timeout_ms"],
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                        settings["read_timeout_ms"],
                        cv2.CAP_PROP_N_THREADS,
                        settings["decoder_threads"],
                    ],
                )
                if not ok:
                    reply({"status": "error", "code": "open_failed", "retryable": True})
                    return
                if not capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0):
                    raise ValueError("orientation")
                width, height = (
                    capture.get(key)
                    for key in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)
                )
                if all(math.isfinite(n) and n > 0 for n in (width, height)) and (
                    width * height > settings["maximum_frame_pixels"]
                    or width * height * 3 > settings["maximum_frame_bytes"]
                ):
                    reply({"status": "error", "code": "frame_limit_or_format", "retryable": False})
                    return
                reply({"status": "opened"})
            elif command["op"] == "read":
                ok, image = capture.read()
                received = time.time_ns()
                if not ok:
                    reply({"status": "error", "code": "read_failed", "retryable": True})
                    return
                if (
                    image.dtype != np.uint8
                    or image.ndim != 3
                    or image.shape[2] != 3
                    or image.size > settings["maximum_frame_bytes"]
                    or image.shape[0] * image.shape[1] > settings["maximum_frame_pixels"]
                ):
                    reply({"status": "error", "code": "frame_limit_or_format", "retryable": False})
                    return
                if shape is not None and shape != image.shape:
                    reply({"status": "error", "code": "frame_size_changed", "retryable": False})
                    return
                shape = image.shape
                np.ndarray(shape, np.uint8, buffer=memory.buf)[:] = image
                stamp, fps = (capture.get(key) for key in (cv2.CAP_PROP_POS_MSEC, cv2.CAP_PROP_FPS))
                usable = math.isfinite(stamp) and 0 <= stamp <= 1e15
                increasing = usable and (previous is None or stamp > previous)
                if increasing:
                    previous = stamp
                reply(
                    {
                        "status": "frame",
                        "shape": list(shape),
                        "connection_frame_index": index,
                        "timestamp_ms": stamp if increasing else None,
                        "timestamp_source": "decoder" if increasing else "unavailable",
                        "fps": fps if math.isfinite(fps) and 0 < fps <= 100000 else None,
                        "backend": capture.getBackendName(),
                        "received_unix_ns": received,
                    }
                )
                index += 1
            else:
                raise ValueError("command")
    except Exception:
        reply({"status": "error", "code": "decoder_error", "retryable": False})
    finally:
        capture.release()
        memory.close()


if __name__ == "__main__":
    main()
