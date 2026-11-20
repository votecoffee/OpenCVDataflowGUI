"""Exclusive publication shared by point-cloud and triangle geometry writers."""

import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

MAX_GEOMETRY_BYTES = 256 * 1024 * 1024


@contextmanager
def geometry_output(path, prefix):
    """Stage complete geometry in the destination directory before publishing it."""
    temporary = None
    try:
        with NamedTemporaryFile(prefix=prefix, dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            yield stream
            if stream.tell() > MAX_GEOMETRY_BYTES:
                raise ValueError("Geometry files are limited to 256 MiB")
        if os.name == "nt":
            temporary.rename(path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
