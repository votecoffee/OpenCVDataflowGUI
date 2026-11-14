"""Fixed stdio RPC entry point for a separately configured ROS Python interpreter."""

import os
import sys

if __package__:
    from .native import Endpoint, PublishUncertain
    from .wire import read_frame, write_frame
else:
    from native import Endpoint, PublishUncertain
    from wire import read_frame, write_frame


def main():
    incoming, outgoing = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # Python/ROS logging must not share protocol stdout.
    os.environ["RCUTILS_LOGGING_USE_STDOUT"] = "0"
    endpoint = None
    try:
        while True:
            try:
                request = read_frame(incoming)
            except EOFError:
                break
            ident = request.get("id")
            try:
                operation = request["op"]
                if operation == "init":
                    if endpoint is not None:
                        raise ValueError("ROS worker is already initialized")
                    endpoint = Endpoint(request["config"])
                    result = endpoint.info
                elif operation == "close":
                    write_frame(outgoing, {"id": ident, "ok": True, "result": {}})
                    break
                elif endpoint is None:
                    raise ValueError("Initialize the ROS worker first")
                elif operation == "poll":
                    result = endpoint.poll(request["timeout"])
                elif operation == "publish":
                    result = endpoint.publish(
                        request["value"],
                        request["timeout"],
                        request["require_subscriber"],
                        request["wait_for_ack"],
                        request["stamp_mode"],
                    )
                else:
                    raise ValueError("Unknown ROS worker operation")
                write_frame(outgoing, {"id": ident, "ok": True, "result": result})
            except Exception as exc:
                write_frame(
                    outgoing,
                    {
                        "id": ident,
                        "ok": False,
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "unknown_publish": isinstance(exc, PublishUncertain),
                        },
                    },
                )
    finally:
        if endpoint is not None:
            endpoint.close()


if __name__ == "__main__":
    main()
