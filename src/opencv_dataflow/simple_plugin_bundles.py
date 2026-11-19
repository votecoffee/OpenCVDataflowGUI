"""Host-selected, pinned Simple Plugin packages; project files cannot select code."""

import hashlib
import sys
import threading
from pathlib import Path

# Unchanged wheels already shipped with public examples. Keep their embedded licenses.
# Names/hashes are application code, never read from a project's directory or settings.
BUNDLES = {
    "opencv_geometric_pattern_match": (
        "opencv_geometric_pattern_match-0.1.0.dev3-py3-none-any.whl",
        "a3e4025d4100d7d0b7fd3a1c469cd2468de685c10ffdec29ba8fff98f62eb2eb",
    ),
    "opencv_defect_segmentation": (
        "opencv_defect_segmentation-0.1.0.dev3-py3-none-any.whl",
        "f703100682e6e0c1cc6a190d82419825e1e39cbaeef18b80a26b3ff981aafcfc",
    ),
    "opencv_simple_path_width": (
        "opencv_simple_path_width-0.1.0.dev14-py3-none-any.whl",
        "4276490dbf0d291418088367be7fa83801008e19edd5e99ba144ba3e62242840",
    ),
    "opencv_simple_reference_inspect": (
        "opencv_simple_reference_inspect-0.1.0.dev5-py3-none-any.whl",
        "0dc6ccea2289c093cd8aff35b1eb97ead68d2d01e05e3abcdbec68a6a15a177c",
    ),
    "opencv_simple_3d_locate": (
        "opencv_simple_3d_locate-0.1.0.dev22-py3-none-any.whl",
        "c299e0d8433c10d245f5ec6849eed9843fb761eab53a12cdd3c99e1ee495275c",
    ),
    "opencv_simple_3d_measure": (
        "opencv_simple_3d_measure-0.1.0.dev7-py3-none-any.whl",
        "32fca513e9839ca7d78053bca447359289193950d8a3daec6eb53b00a05cfc6a",
    ),
    "opencv_simple_bin_picking": (
        "opencv_simple_bin_picking-0.1.0.dev14-py3-none-any.whl",
        "4eff5f15afb2b3827badf0c20871ca2a3c447e1d9d26c15f49655935974dedd3",
    ),
    "opencv_simple_laser_profile": (
        "opencv_simple_laser_profile-0.1.0.dev4-py3-none-any.whl",
        "1de8610194093aa8fb523032f90d369f7ed59e18f8b789663a73b603400f4771",
    ),
    "opencv_simple_surface_enhance": (
        "opencv_simple_surface_enhance-0.1.0.dev5-py3-none-any.whl",
        "392947769d3245021ab2a5c43a8eaba40908cea3e9278973f7a6b007134c0b01",
    ),
    "opencv_simple_fringe_scan": (
        "opencv_simple_fringe_scan-0.1.0.dev3-py3-none-any.whl",
        "0106d63746e36244358454432e9f707628989134e5af2288de3e436c18737f87",
    ),
    "opencv_simple_deflectometry": (
        "opencv_simple_deflectometry-0.1.0.dev3-py3-none-any.whl",
        "417c4699aed97edbba0f531038cbb8fb9a88d9d6d8b3e426fe70996340aec7f9",
    ),
    "opencv_simple_foundation_pose": (
        "opencv_simple_foundation_pose-0.1.0.dev4-py3-none-any.whl",
        "aa1618f0bb6d0932656a69e26d690c635169a982abc6946e74e597b7025ec4c0",
    ),
    "opencv_simple_gdrnpp": (
        "opencv_simple_gdrnpp-0.1.0.dev3-py3-none-any.whl",
        "d8fd2193f010f1888eb5c36a40a10ca0e5498f5227acbf669f1a473231c6886c",
    ),
}
BUNDLE_DIRECTORY = Path(__file__).resolve().parent / "simple_plugin_wheels"
_LOCK = threading.RLock()


def prepare_bundle(package):
    """Make a checked app-owned wheel importable, without installing dependencies.

    Preserve packages loaded explicitly earlier in this Python session. Hot-swapping
    their modules would invalidate existing blocks, recipes and worker state.
    """
    with _LOCK:
        filename, expected = BUNDLES[package]
        path = BUNDLE_DIRECTORY / filename
        if not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError(f"Bundled {package} is missing or oversized; reinstall DataflowGUI")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Bundled {package} checksum differs; reinstall DataflowGUI")
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
