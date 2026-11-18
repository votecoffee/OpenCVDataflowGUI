"""Session loading of local plugins and known installed integrations."""

import importlib
import importlib.util
import sys
import uuid

from .core import Registry
from .diagnostics import exception_message
from .simple_plugin_bundles import prepare_bundle

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
LOCATE_3D_TYPES = ("3d.simple_locate",)
MEASURE_3D_TYPES = ("3d.simple_measure",)
BIN_PICKING_TYPES = ("3d.simple_bin_picking",)

# Application-selected providers, including the recorded sources used by examples.
SIMPLE_EXTRA_PROVIDERS = {
    "3d.simple_bin_picking_jaw": ("opencv_simple_bin_picking", "jaw_dataflow", "Bin picking"),
    "3d.simple_part_instances": ("opencv_simple_bin_picking", "pose_dataflow", "Bin picking"),
    **dict.fromkeys(
        ("input.sim_bin_grid", "3d.sim_bin_pick_one", "3d.sim_bin_grid_summary",
         "input.sim_pick_review"),
        ("opencv_simple_bin_picking", "sim_dataflow", "Bin picking"),
    ),
    "3d.simple_laser_profile": ("opencv_simple_laser_profile", "dataflow", "Laser profiles"),
    "image.simple_surface_enhance": (
        "opencv_simple_surface_enhance", "dataflow", "Photometric inspection"
    ),
    "3d.simple_fringe_scan": ("opencv_simple_fringe_scan", "dataflow", "Structured light"),
    "image.simple_deflectometry": ("opencv_simple_deflectometry", "dataflow", "Deflectometry"),
    "3d.simple_foundation_pose": ("opencv_simple_foundation_pose", "dataflow", "Learned 3D pose"),
    "3d.simple_gdrnpp": ("opencv_simple_gdrnpp", "dataflow", "Learned 3D pose"),
    "input.simple_jaw_example": ("opencv_simple_bin_picking", None, "Bin picking"),
    "input.simple_surface_capture": (
        "opencv_simple_surface_enhance", None, "Photometric inspection"
    ),
    "input.simple_fringe_capture": ("opencv_simple_fringe_scan", None, "Structured light"),
    "input.simple_deflect_capture": ("opencv_simple_deflectometry", None, "Deflectometry"),
    "input.foundation_pose_example_capture": (
        "opencv_simple_foundation_pose", None, "Learned 3D pose"
    ),
}


def register_simple_providers(registry):
    for type_id in SIMPLE_EXTRA_PROVIDERS:
        registry.add_provider(
            (type_id,), lambda target, requested=type_id: load_simple_integration(target, requested)
        )


def load_simple_integration(registry, type_id):
    package, module, _ = SIMPLE_EXTRA_PROVIDERS[type_id]
    prepare_bundle(package)
    if package == "opencv_simple_deflectometry":
        prepare_bundle("opencv_simple_fringe_scan")
    if module is None:
        from .simple_plugin_inputs import RECORDED_INPUTS

        _register_blocks(lambda target: target.register(RECORDED_INPUTS[type_id]), registry)
    else:
        try:
            integration = importlib.import_module(f"{package}.{module}")
        except ModuleNotFoundError as exc:
            if exc.name not in (package, f"{package}.{module}"):
                raise
            raise ValueError(
                f"The already loaded {package} cannot supply {type_id}. "
                "Restart DataflowGUI to use its bundled Simple Plugins."
            ) from exc
        _register_blocks(integration.register, registry)


def _register_blocks(register, registry):
    staged = Registry()
    staged.types.update(registry.types)
    register(staged)
    registry.types = staged.types


def load_installed_geometric_matcher(registry):
    prepare_bundle("opencv_geometric_pattern_match")
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
    prepare_bundle("opencv_defect_segmentation")
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
    prepare_bundle("opencv_simple_path_width")
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
    prepare_bundle("opencv_simple_reference_inspect")
    try:
        from opencv_simple_reference_inspect.dataflow import register
    except ModuleNotFoundError as exc:
        if exc.name not in (
            "opencv_simple_reference_inspect", "opencv_simple_reference_inspect.dataflow"
        ):
            raise
        raise ValueError(
            "Install SimpleReferenceInspect in the Python environment running DataflowGUI. "
            "See examples/Simple Plugins/reference_inspect/README.md for setup."
        ) from exc
    _register_blocks(register, registry)


def load_installed_3d_locate(registry):
    prepare_bundle("opencv_simple_3d_locate")
    try:
        from opencv_simple_3d_locate.dataflow import register
    except ModuleNotFoundError as exc:
        if exc.name not in ("opencv_simple_3d_locate", "opencv_simple_3d_locate.dataflow"):
            raise
        raise ValueError(
            "Install Simple3DLocate in the Python environment running DataflowGUI. "
            "See examples/Simple Plugins/3d_locate/README.md for setup."
        ) from exc
    _register_blocks(register, registry)


def load_installed_3d_measure(registry):
    prepare_bundle("opencv_simple_3d_measure")
    try:
        from opencv_simple_3d_measure.dataflow import register
    except ModuleNotFoundError as exc:
        if exc.name not in ("opencv_simple_3d_measure", "opencv_simple_3d_measure.dataflow"):
            raise
        raise ValueError(
            "Install Simple3DMeasure in the Python environment running DataflowGUI. "
            "See examples/Simple Plugins/3d_measure/README.md for setup."
        ) from exc
    _register_blocks(register, registry)


def load_installed_bin_picking(registry):
    prepare_bundle("opencv_simple_bin_picking")
    try:
        from opencv_simple_bin_picking.dataflow import register
    except ModuleNotFoundError as exc:
        if exc.name not in ("opencv_simple_bin_picking", "opencv_simple_bin_picking.dataflow"):
            raise
        raise ValueError(
            "Install SimpleBinPicking in the Python environment running DataflowGUI. "
            "See examples/Simple Plugins/bin_picking/README.md for setup."
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
