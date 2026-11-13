"""Preserve declared project-relative input references when a program is saved elsewhere."""

import copy
import os
from contextlib import contextmanager
from pathlib import Path

from .project_assets import AssetCopies

STANDARD_FIELDS = ("path", "paths", "image_paths", "directory", "icc_path")


def relocated(value, original, destination):
    if isinstance(value, list):
        return [relocated(item, original, destination) for item in value]
    if not isinstance(value, str) or not value or "://" in value or Path(value).is_absolute():
        return value
    target = os.path.abspath(original / value)
    try:
        return os.path.relpath(target, destination)
    except ValueError:  # Different Windows drives cannot share a relative path.
        return target


@contextmanager
def rebase_project_paths(graph, destination, project_path=None):
    """Commit relocated asset references only when saving succeeds; never move assets."""
    from .core import Graph

    destination = Path(destination)
    snapshots, visited = [], set()
    assets = AssetCopies(destination, project_path)

    def visit(current):
        if id(current) in visited:
            return
        visited.add(id(current))
        original = current.project_dir
        for block in current.blocks.values():
            if original is not None and Path(original) != destination:
                for relative in block.project_assets():
                    assets.copy(original, relative)
                fields = set(block.project_path_parameters) - set(block.project_asset_parameters)
                if block.parameters.get("path_base") == "project" and block.block_type != "Output":
                    fields.update(STANDARD_FIELDS)
                parameters = copy.deepcopy(block.parameters)
                for field in fields & parameters.keys():
                    parameters[field] = relocated(parameters[field], Path(original), destination)
                if parameters != block.parameters:
                    snapshots.append((block, "parameters", block.parameters))
                    block.parameters = parameters
            nested = getattr(block, "graph", None)
            if isinstance(nested, Graph):
                visit(nested)
        if original is not None and Path(original) != destination and current.validation_sets:
            manifests = copy.deepcopy(current.validation_sets)
            for manifest in manifests:
                for case in manifest["cases"]:
                    case["path"] = relocated(case["path"], Path(original), destination)
            if manifests != current.validation_sets:
                snapshots.append((current, "validation_sets", current.validation_sets))
                current.validation_sets = manifests

    try:
        visit(graph)
        yield
    except BaseException:
        for owner, field, value in reversed(snapshots):
            setattr(owner, field, value)
        assets.rollback()
        raise
