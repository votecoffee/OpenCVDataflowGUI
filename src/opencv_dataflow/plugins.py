"""Session loading of local plugins and known installed integrations."""

import importlib.util
import sys
import uuid

from .core import Registry
from .diagnostics import exception_message

GEOMETRIC_MATCHER_TYPES = (
    "pattern.ogpm_recipe",
    "pattern.ogpm_load_recipe",
    "pattern.ogpm_train_recipe",
    "pattern.ogpm_match_model",
)
PIXEL_SEGMENTATION_TYPES = (
    "inspection.ocv_segment_image",
    "inspection.ocv_segment_parts",
)
PATH_WIDTH_TYPES = ("inspect.simple_path_width",)
REFERENCE_INSPECT_TYPES = ("inspect.simple_reference",)


def _register_blocks(register, registry):
    staged = Registry()
    staged.types.update(registry.types)
    register(staged)
    registry.types = staged.types


def load_installed_geometric_matcher(registry):
    try:
        from opencv_geometric_pattern_match.dataflow import register
    except ModuleNotFoundError as exc:
        if exc.name not in (
            "opencv_geometric_pattern_match",
            "opencv_geometric_pattern_match.dataflow",
        ):
            raise
        raise ValueError(
            "Install OpenCVGeometricPatternMatch in the Python environment running DataflowGUI. "
            "See examples/Simple Plugins/README.md for setup."
        ) from exc
    _register_blocks(register, registry)


def load_installed_pixel_segmentation(registry):
    try:
        from opencv_defect_segmentation.dataflow import register
    except ModuleNotFoundError as exc:
        if exc.name not in ("opencv_defect_segmentation", "opencv_defect_segmentation.dataflow"):
            raise
        raise ValueError(
            "Install OpenCV Pixel Segmentation in the Python environment running DataflowGUI. "
            "See examples/Simple Plugins/pixel_segmentation/README.md for setup."
        ) from exc
    _register_blocks(register, registry)


def load_installed_path_width(registry):
    try:
        from opencv_simple_path_width.dataflow import register
    except ModuleNotFoundError as exc:
        if exc.name not in ("opencv_simple_path_width", "opencv_simple_path_width.dataflow"):
            raise
        raise ValueError(
            "Install SimplePathWidth in the Python environment running DataflowGUI. "
            "See examples/Simple Plugins/path_width/README.md for setup."
        ) from exc
    _register_blocks(register, registry)


def load_installed_reference_inspect(registry):
    try:
        from opencv_simple_reference_inspect.dataflow import register
    except ModuleNotFoundError as exc:
        if exc.name not in ("opencv_simple_reference_inspect", "opencv_simple_reference_inspect.dataflow"):
            raise
        raise ValueError(
            "Install SimpleReferenceInspect in the Python environment running DataflowGUI. "
            "See examples/Simple Plugins/reference_inspect/README.md for setup."
        ) from exc
    _register_blocks(register, registry)


def load_plugin(path, registry):
    """Module exports register(registry). Stage registrations to avoid partial loads.

    This executes arbitrary Python. Project JSON cannot select a plugin file.
    """
    spec = importlib.util.spec_from_file_location(f"dataflow_plugin_{uuid.uuid4().hex}", path)
    if spec is None or spec.loader is None:
        raise ValueError("Select a Python plugin file")
    module = importlib.util.module_from_spec(spec)
    # Standard import semantics matter to dataclasses and annotation introspection.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        _register_blocks(module.register, registry)
    except BaseException as exc:
        sys.modules.pop(spec.name, None)
        if isinstance(exc, SystemExit):
            raise RuntimeError(f"Plugin load failed: {exception_message(exc)}") from exc
        raise
