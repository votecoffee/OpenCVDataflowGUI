"""Original MIT loopback-only MJPEG fixture; /drop closes after four frames."""

import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def make_server(port=0, images=None):
    if images is None:
        images = [p.read_bytes() for p in sorted((Path(__file__).parent / "assets").glob("*.jpg"))]
    if not images:
        raise ValueError("No fixture images")
    stopped = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path not in ("/stream", "/drop", "/stall", "/open-stall"):
                self.send_error(404)
                return
            if self.path == "/open-stall":
                stopped.wait(30)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            index = 0
            try:
                while not stopped.is_set():
                    if self.path == "/drop" and index == 4:
                        break
                    if self.path == "/stall" and index == 6:
                        stopped.wait(30)
                        break
                    image = images[index % len(images)]
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(image)).encode()
                        + b"\r\n\r\n"
                        + image
                        + b"\r\n"
                    )
                    self.wfile.flush()
                    index += 1
                    stopped.wait(0.04)
            except ConnectionError, OSError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.stopped = stopped
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18764)
    args = parser.parse_args()
    server = make_server(args.port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"READY network-video {server.server_port}", flush=True)
    try:
        # Program lifecycle sends EOF on shutdown; no unrelated server is stopped.
        import sys

        for _ in sys.stdin:
            pass
    finally:
        server.stopped.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


if __name__ == "__main__":
    main()
