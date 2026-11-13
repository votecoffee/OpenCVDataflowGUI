"""Byte-preserving publication of declared, project-contained plugin assets."""

import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

MAX_ASSET_BYTES = 256 * 1024 * 1024


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class AssetCopies:
    def __init__(self, destination, reserved=None):
        self.root = Path(destination).resolve()
        self.reserved = reserved
        self.created, self.directories, self.seen = [], [], {}
        self.remaining = MAX_ASSET_BYTES

    def copy(self, original, relative):
        if not isinstance(relative, str) or not relative:
            raise ValueError("Project assets must be nonempty relative file paths")
        path = Path(relative)
        if path.is_absolute() or path.anchor or ".." in path.parts or ":" in relative:
            raise ValueError("Project assets must stay inside the project folder")
        source_root = Path(original).resolve()
        source, target = (source_root / path).resolve(), (self.root / path).resolve()
        if not source.is_relative_to(source_root) or not target.is_relative_to(self.root):
            raise ValueError("Project asset resolves outside the project folder")
        if target == self.reserved:
            raise ValueError("The program filename conflicts with a project asset")
        size = source.stat().st_size
        if target not in self.seen and size > self.remaining:
            raise ValueError("Save As assets exceed the 256 MiB copy budget")
        expected = digest(source)
        if target in self.seen:
            if self.seen[target] != expected:
                raise FileExistsError(f"Conflicting project assets: {relative}")
            return
        self.seen[target] = expected
        self.remaining -= size
        if target.exists():
            self._check_existing(target, expected)
            return
        missing, parent = [], target.parent
        while not parent.exists():
            missing.append(parent)
            parent = parent.parent
        for directory in reversed(missing):
            try:
                directory.mkdir()
            except FileExistsError:
                pass
            else:
                self.directories.append(directory)
        temporary = None
        try:
            with NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                with source.open("rb") as incoming:
                    copied = 0
                    while chunk := incoming.read(1024 * 1024):
                        copied += len(chunk)
                        if copied > size:
                            raise ValueError("Project asset changed during Save As")
                        stream.write(chunk)
            if digest(temporary) != expected:
                raise ValueError("Project asset changed during Save As")
            try:
                # Exclusive publication: an existing destination is never overwritten.
                os.link(temporary, target)
            except FileExistsError:
                self._check_existing(target, expected)
            else:
                self.created.append((target, expected))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _check_existing(target, expected):
        if not target.is_file() or digest(target) != expected:
            raise FileExistsError(f"Save As would replace an existing asset: {target.name}")

    def rollback(self):
        for path, expected in reversed(self.created):
            try:
                if digest(path) == expected:
                    path.unlink()
            except OSError:
                pass  # Preserve files changed by another owner during the save.
        for directory in reversed(self.directories):
            try:
                directory.rmdir()
            except OSError:
                pass
