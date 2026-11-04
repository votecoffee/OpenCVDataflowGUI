"""Explicit, session-only loading of trusted local Python plugins."""

import importlib.util
import uuid

from .core import Registry


def load_plugin(path, registry):
    """Module exports register(registry). Stage registrations to avoid partial loads.

    This is arbitrary Python execution, not a sandbox. Project JSON never imports
    modules and never embeds executable code or automatically loads plugins.
    """
    spec = importlib.util.spec_from_file_location(f"dataflow_plugin_{uuid.uuid4().hex}", path)
    if spec is None or spec.loader is None:
        raise ValueError("Select a Python plugin file")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    staged = Registry()
    staged.types.update(registry.types)
    module.register(staged)
    registry.types = staged.types
