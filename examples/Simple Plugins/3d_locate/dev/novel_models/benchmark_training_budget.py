"""Validate unchanged default training across the pinned thirty-model CAD archive."""

import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from build_tless_demo import tless_mesh, write_json
from opencv_simple_3d_locate import train_cad
from opencv_simple_3d_locate.model import prepared

ARCHIVE_SHA256 = "6a29c59766b8d2af05e62e71739f0ca7243ad81bc3c7f9a24925504a8cb37928"


def benchmark(archive, output):
    archive = Path(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Expected the pinned original T-LESS model archive.")
    results = []
    with ZipFile(archive) as files, TemporaryDirectory() as temporary:
        for object_id in range(1, 31):
            path = Path(temporary) / "model.ply"
            path.write_bytes(files.read(f"models_cad/obj_{object_id:06}.ply"))
            try:
                recipe = train_cad(tless_mesh(path), units="mm")
                health = recipe.health
                old_error = None
                try:
                    diagonal = np.linalg.norm(np.ptp(recipe.reference[:, :3].astype(float), axis=0))
                    legacy = prepared(
                        recipe.reference,
                        recipe.settings.sampling_distance * diagonal,
                        recipe.settings.normal_neighbors,
                        recipe.settings.model_viewpoint,
                        2000,
                    )
                    np.testing.assert_array_equal(recipe.samples, legacy)
                except ValueError as exc:
                    old_error = str(exc)
                result = dict(
                    object_id=object_id,
                    trained=True,
                    health=health,
                    previous_training_error=old_error,
                )
            except ValueError as exc:
                result = dict(object_id=object_id, trained=False, error=str(exc))
            results.append(result)
            print(
                object_id,
                result["trained"],
                result.get("health", {}).get("sample_points"),
                flush=True,
            )
    report = dict(
        dataset="T-LESS",
        archive_sha256=ARCHIVE_SHA256,
        models=results,
        scope="Default CAD training only; no pose-detection qualification. "
        "No parameters vary between models. Legacy-accepted sample arrays are compared exactly.",
    )
    write_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    benchmark(args.archive, args.output)
