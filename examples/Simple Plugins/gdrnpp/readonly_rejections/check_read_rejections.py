"""Check stale read-only rejection without inference or changing a runtime profile."""

import argparse
import hashlib
import json
from pathlib import Path


def check(recipe, python, *, host_adapter=False):
    if host_adapter:
        from opencv_dataflow.simple_gdrnpp.bridge import WorkerClient
    else:
        from opencv_simple_gdrnpp.bridge import WorkerClient
    request = dict(
        base=dict(
            kind="recipe",
            path=str(recipe.resolve()),
            sha256=hashlib.sha256(recipe.read_bytes()).hexdigest(),
        )
    )
    with WorkerClient(python=str(python.resolve())) as client:
        first, arrays = client.call("inspect", request)
        process = client.process
        for operation in (
            "capture_completed",
            "capture_completed_candidate",
            "capture_completed_graphics",
        ):
            try:
                client.call(operation, dict(identity={}))
            except ValueError as error:
                assert "expired" in str(error)
            else:
                raise AssertionError("Expected missing/expired completion rejection")
            assert client.process is process and process.poll() is None
            assert not list(client.root.iterdir())
        again, other = client.call("inspect", request)
        assert again["setup"] == first["setup"]
        for key in arrays:
            import numpy as np

            np.testing.assert_array_equal(arrays[key], other[key])
    assert process.poll() is not None
    return dict(
        read_rejections=3,
        process_preserved=True,
        private_request_files_retired=True,
        recipe_unchanged=True,
        model_loaded=False,
        inference_performed=False,
        final_close_retired_worker=True,
        scope="Protocol/cache-lifetime control only",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-adapter", action="store_true")
    args = parser.parse_args()
    args.output.write_bytes(
        (
            json.dumps(check(args.recipe, args.python, host_adapter=args.host_adapter), indent=2)
            + "\n"
        ).encode()
    )
