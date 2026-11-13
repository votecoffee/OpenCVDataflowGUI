"""Toolbox organization for native wrappers, built-in workflows and user tools."""

import re
from dataclasses import dataclass

from PySide6.QtCore import QMimeData, Qt
from PySide6.QtWidgets import QAbstractItemView, QTreeWidget, QTreeWidgetItem

from .core import block_display_title
from .plugins import (
    GEOMETRIC_MATCHER_TYPES,
    LOCATE_3D_TYPES,
    PATH_WIDTH_TYPES,
    PIXEL_SEGMENTATION_TYPES,
    REFERENCE_INSPECT_TYPES,
)
from .visual_style import INDUSTRIAL

TOOL_MIME = "application/x-opencv-dataflow-tool"
SIMPLE_PLUGINS = "Simple Plugins"
TOOL_SOURCES = ("OpenCV", "scikit-image", INDUSTRIAL, "Custom", SIMPLE_PLUGINS, "User")
_OPTIONAL_SIMPLE_FAMILIES = {
    **dict.fromkeys(GEOMETRIC_MATCHER_TYPES, "Pattern matching"),
    **dict.fromkeys(PIXEL_SEGMENTATION_TYPES, "Segmentation"),
    **dict.fromkeys(PATH_WIDTH_TYPES, "Measurements"),
    **dict.fromkeys(REFERENCE_INSPECT_TYPES, "Reference inspection"),
    **dict.fromkeys(LOCATE_3D_TYPES, "3D localization"),
}
_SIMPLE_FAMILIES = {**_OPTIONAL_SIMPLE_FAMILIES, "inspect.simple_classify_image": "Classification"}

OPENCV_MODULES = {
    "imgproc": "Image Processing",
    "core": "Core",
    "imgcodecs": "Image Codecs",
    "features": "Features Framework",
    "calib": "Camera Calibration",
    "geometry": "Computational Geometry",
    "video": "Video Analysis",
    "bgsegm": "Background and Foreground Segmentation",
    "videoio": "Video I/O",
    "objdetect": "Object Detection",
    "dnn": "Deep Neural Networks",
    "flann": "Clustering and Search",
    "photo": "Computational Photography",
    "stitching": "Image Stitching",
    "stereo": "Stereo Correspondence",
    "ptcloud": "Point Cloud Processing",
    "ccm": "Color Correction",
    "mcc": "Color Chart Detection",
    "ml": "Machine Learning",
    "quality": "Image Quality",
    "img_hash": "Image Hashing",
    "rgbd": "RGB-D Processing",
    "structured_light": "Structured Light",
    "surface_matching": "Surface Matching",
    "tracking": "Tracking",
    "ximgproc": "Extended Image Processing",
    "xobjdetect": "Extended Object Detection",
}
_MODULE_ORDER = {
    f"{title} ({module})": index for index, (module, title) in enumerate(OPENCV_MODULES.items())
}

# Most implementation files cover one OpenCV module. Blocks in mixed files
# use the overrides below. Unmapped built-in workflows belong under Custom.
IMPLEMENTATION_MODULES = {
    "imgproc": (
        "analysis_tools annotation basic_processing blocks collection_tools features "
        "generalized_hough "
        "histogram histogram_comparison image_blend image_colorize path_strip "
        "point_set_lines polar_image "
        "laplacian_pyramids processing "
        "pyramids raster_hough_lines registration segmentation subpixel_crop template_instances "
        "tilted_statistics"
    ),
    "core": (
        "channel_transform clustering feature_covariance fourier_correlation gram_matrix "
        "image_layout image_noise image_norm image_statistics image_validity linear_program "
        "linear_system matrix_eigen "
        "matrix_inverse matrix_multiply matrix_structure matrix_svd nearest_rows polynomial_roots "
        "principal_components spectral"
    ),
    "imgcodecs": "image_codecs tiff_files",
    "calib": "calibration calibration_validation chessboard_quality hand_eye remap_maps workplane",
    "geometry": "contour_analysis convex_overlap geometry3d line_pair shape_fitting "
    "translation_registration triangulation voronoi",
    "video": (
        "ecc_registration frame_interpolation image_correlation "
        "native_background sequence_tools"
    ),
    "videoio": "video_files",
    "objdetect": "aruco_markers charuco code_readers",
    "xobjdetect": "hog",
    "dnn": "dnn_blocks ocr",
    "photo": "exposure hdr photo_filters restoration",
    "stitching": "image_stitching scan_stitching tiled_mosaic",
    "stereo": "stereo3d",
    "ccm": "color_calibration",
    "quality": "image_quality",
    "img_hash": "image_hash",
    "rgbd": "ptcloud3d",
    "tracking": "native_trackers",
    "ximgproc": "local_threshold region_proposals ridge_response superpixels",
}
BLOCK_MODULES = {
    "core": (
        "cv.intensity_transform cv.scale_absolute cv.image_arithmetic cv.compare_pixels "
        "cv.cart_to_polar cv.polar_to_cart cv.normalize cv.intensity_lut cv.split_channels "
        "cv.merge_channels cv.extract_channel cv.insert_channel cv.apply_mask "
        "cv.find_nonzero_pixels "
        "cv.absdiff cv.flip cv.rotate90 cv.transpose_image cv.bitwise cv.blend cv.copy_border "
        "cv.transform_points cv.alpha_over"
    ),
    "imgproc": (
        "cv.hann_window cv.decode_yuv calibration.color_balance cv.pack_remap_map "
        "inspect.intensity_moments "
        "inspect.contour_hierarchy inspect.focus inspect.slanted_edge cv.decode_raw_sensor "
        "inspect.structure_tensor"
    ),
    "imgcodecs": "input.file input.depth_file input.raw_yuv input.raw_sensor output.image",
    "calib": (
        "3d.camera 3d.project 3d.undistort_points 3d.stereo_rig 3d.stereo_calibrate "
        "3d.stereo_rectify 3d.rectify_camera 3d.check_epipolar"
    ),
    "geometry": "inspect.affine inspect.homography collection.clip_lines",
    "features": (
        "inspect.affine_features "
        "inspect.fast inspect.orb inspect.sift inspect.akaze inspect.kaze inspect.brisk "
        "inspect.mser inspect.simple_blob_detector "
        "inspect.match_orb inspect.match_sift inspect.match_akaze inspect.match_brisk "
        "inspect.match_descriptor_radius"
    ),
    "flann": "inspect.match_flann",
    "bgsegm": "sequence.gsoc_background sequence.lsbp_background",
    "videoio": "input.camera",
    "objdetect": "3d.aruco_pose",
    "ptcloud": "3d.depth_cloud",
    "rgbd": "3d.register_depth",
    "structured_light": "3d.graycode 3d.graycode_decode",
    "surface_matching": "3d.icp 3d.ppf_train 3d.ppf_match",
    "ximgproc": "cv.guided_filter cv.domain_transform_filter inspect.thin_mask",
    "mcc": "calibration.detect_color_chart",
    None: (
        "input.demo output.results input.calibration output.calibration "
        "calibration.pixel_to_physical "
        "calibration.lens_map_displacement collection.filter collection.sort collection.draw "
        "core.attach_payload core.read_payload sequence.background sequence.image_memory "
        "input.sequence_demo cv.paste_patch cv.filter_fourier_spectrum cv.mask_fourier_spectrum "
        "inspect.contour_profile"
    ),
}
_MODULE_BY_FILE = {
    file: module for module, files in IMPLEMENTATION_MODULES.items() for file in files.split()
}
_MODULE_BY_BLOCK = {
    type_id: module for module, types in BLOCK_MODULES.items() for type_id in types.split()
}
_NATIVE_NAMES = {
    "cv.gray": ("cvtColor",),
    "cv.alpha_over": ("alpha", "premultiplied", "source over", "opacity", "transparency"),
    "inspect.intensity_moments": ("moments", "centroid", "center of mass", "principal axis"),
    "cv.interpolate_frames": ("Farneback", "optical flow", "remap", "intermediate frame"),
    "inspect.structure_tensor": ("cornerEigenValsAndVecs", "tensor", "orientation", "coherence"),
    "inspect.affine_features": ("AffineFeature", "ASIFT", "tilt", "SIFT", "ORB"),
    "cv.decode_raw_sensor": ("MIPI", "RAW10", "RAW12", "unpack", "Bayer", "row stride"),
    "input.raw_sensor": ("MIPI", "RAW10", "RAW12", "sensor frame"),
    "cv.laplacian_pyramid": ("createLaplacePyr", "pyrDown", "pyrUp", "signed bands"),
    "cv.reconstruct_laplacian_pyramid": ("restoreImageFromLaplacePyr", "band gains"),
    "inspect.focus": ("Laplacian", "Sobel", "Tenengrad", "sharpness", "blur"),
    "inspect.image_correlation": ("computeECC", "Pearson", "NCC", "reference comparison"),
    "inspect.slanted_edge": ("ESF", "LSF", "SFR", "MTF50", "MTF10", "edge spread", "sharpness"),
    "inspect.local_binary_patterns": ("LBP", "local_binary_pattern"),
    "inspect.gray_level_cooccurrence": ("GLCM", "Haralick", "graycoprops"),
    "cv.graph_segments": ("createGraphSegmentation",),
    "inspect.selective_search": ("createSelectiveSearchSegmentation",),
    "cv.calibrate_debevec": ("createCalibrateDebevec",),
    "cv.merge_debevec": ("createMergeDebevec",),
    "cv.calibrate_robertson": ("createCalibrateRobertson",),
    "cv.merge_robertson": ("createMergeRobertson",),
    "cv.tonemap_hdr": ("createTonemap", "TonemapDrago", "TonemapReinhard", "TonemapMantiuk"),
    "cv.gaussian_blur": ("GaussianBlur",),
    "cv.morphology": ("morphologyEx", "erode", "dilate"),
    "cv.adaptive_threshold": ("adaptiveThreshold",),
    "cv.local_threshold": ("niBlackThreshold", "Sauvola", "Wolf", "NICK"),
    "inspect.multiscale_ridges": ("Sato", "tubeness", "Hessian", "crack"),
    "cv.ridge_response": ("RidgeDetectionFilter_create", "Hessian", "crack"),
    "cv.slic_superpixels": ("createSuperpixelSLIC", "SLICO", "MSLIC"),
    "cv.lsc_superpixels": ("createSuperpixelLSC", "Linear Spectral Clustering"),
    "cv.seeds_superpixels": ("createSuperpixelSEEDS",),
    "cv.distance_labels": ("distanceTransformWithLabels",),
    "cv.demosaic_bayer": ("demosaicing",),
    "input.file": ("imread", "imdecode"),
    "output.image": ("imwrite", "imencode"),
    "inspect.contours": ("findContours",),
    "inspect.corners": ("goodFeaturesToTrack",),
    "inspect.blobs": ("connectedComponentsWithStats",),
    "inspect.simple_blob_detector": ("SimpleBlobDetector", "SimpleBlobDetector_create"),
    "cv.seamless_clone": ("seamlessClone",),
    "inspect.components": ("connectedComponents", "connectedComponentsWithStats"),
    "inspect.line_segments": ("HoughLinesP", "createLineSegmentDetector"),
    "inspect.fit_shape": ("fitEllipse", "fitLine", "minEnclosingCircle"),
    "inspect.bounding_rotated_rectangle": ("minAreaRect",),
    "inspect.match_template": ("matchTemplate",),
    "inspect.histogram": ("calcHist",),
    "sequence.dis": ("DISOpticalFlow",),
    "sequence.gsoc_background": ("createBackgroundSubtractorGSOC",),
    "sequence.lsbp_background": ("createBackgroundSubtractorLSBP",),
    "sequence.farneback": ("calcOpticalFlowFarneback",),
    "sequence.lk_grid_flow": ("calcOpticalFlowPyrLK",),
    "sequence.track_points": ("calcOpticalFlowPyrLK",),
    "calibration.lens": ("calibrateCamera", "calib3d"),
    "inspect.homography": ("findHomography", "calib3d"),
    "3d.pnp": ("solvePnP", "calib3d"),
    "ml.dnn": ("dnn.readNetFromONNX", "readNetFromONNX"),
}


@dataclass(frozen=True)
class ToolEntry:
    path: tuple[str, ...]
    title: str
    type_id: str
    keywords: tuple[str, ...] = ()


def block_entry(block):
    kind, subtype = block.block_type or block.category, block.subtype
    package, _, file = block.__module__.rpartition(".")
    type_id = block.type_id
    if type_id in _SIMPLE_FAMILIES:
        path = (SIMPLE_PLUGINS, _SIMPLE_FAMILIES[type_id])
    elif package != "opencv_dataflow":
        path = ("User", kind, subtype)
    elif kind == INDUSTRIAL:
        path = (INDUSTRIAL, subtype)
    elif file == "multiscale_ridges":
        path = ("scikit-image", "Filtering")
    elif file == "skeleton":
        family = "Graph" if type_id == "inspect.prune_skeleton" else "Morphology"
        path = ("scikit-image", family)
    elif file in ("local_binary_patterns", "gray_level_cooccurrence"):
        path = ("scikit-image", "Texture")
    elif type_id.startswith("gapi."):
        path = ("OpenCV", "Advanced", "G-API")
    else:
        module = _MODULE_BY_BLOCK.get(type_id, _MODULE_BY_FILE.get(file))
        path = (
            ("OpenCV", f"{OPENCV_MODULES[module]} ({module})", subtype)
            if module
            else ("Custom", kind, subtype)
        )
    aliases = ("skimage",) if path[0] == "scikit-image" else ()
    title = block_display_title(block)
    former_title = (block.title,) if title != block.title else ()
    return ToolEntry(
        path, title, type_id,
        (kind, subtype, *former_title, *aliases, *_NATIVE_NAMES.get(type_id, ())),
    )


def optional_simple_entries(registry):
    """Expose optional tools without importing their packages during toolbox setup."""
    return [
        ToolEntry((SIMPLE_PLUGINS, family), block_display_title(type_id), type_id, ("Inspection",))
        for type_id, family in _OPTIONAL_SIMPLE_FAMILIES.items()
        if type_id not in registry.types
    ]


def _compact(text):
    return re.sub(r"[\W_]+", "", text)


def _matches(entry, terms):
    fields = tuple(
        field.casefold() for field in (*entry.path, entry.title, entry.type_id, *entry.keywords)
    )
    text = " ".join(fields)
    compact = " ".join(_compact(field) for field in fields)
    return all(term in text or (short and short in compact) for term, short in terms)


class ToolLibrary(QTreeWidget):
    def __init__(self):
        super().__init__()
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.setDefaultDropAction(Qt.DropAction.CopyAction)
        self._groups = {}
        self._expanded = set()
        self._filtered = False

    def mimeTypes(self):
        return [TOOL_MIME]

    def mimeData(self, items):
        type_id = items[0].data(0, Qt.ItemDataRole.UserRole) if len(items) == 1 else None
        if not type_id:
            return None
        data = QMimeData()
        data.setData(TOOL_MIME, type_id.encode("utf-8"))
        return data

    def supportedDropActions(self):
        return Qt.DropAction.CopyAction

    def _group(self, path):
        if path not in self._groups:
            parent = self._group(path[:-1]) if len(path) > 1 else self
            item = QTreeWidgetItem(parent, [path[-1]])
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsDragEnabled)
            item.setToolTip(0, " / ".join(path))
            self._groups[path] = item
        return self._groups[path]

    def populate(self, entries, query):
        if not self._filtered:
            self._expanded = {path for path, item in self._groups.items() if item.isExpanded()}
        terms = [word.removeprefix("cv2.").removeprefix("cv.") for word in query.casefold().split()]
        terms = [(word, _compact(word)) for word in terms if word]
        self._filtered = bool(terms)
        self._groups.clear()
        self.clear()
        if not self._filtered:
            for source in TOOL_SOURCES:
                self._group((source,))
            self._groups[("User",)].setToolTip(0, "Load a plugin or import a reusable ToolBlock.")
        entries = sorted(
            entries,
            key=lambda entry: (
                TOOL_SOURCES.index(entry.path[0]),
                _MODULE_ORDER.get(entry.path[1], len(_MODULE_ORDER))
                if entry.path[0] == "OpenCV"
                else 0,
                tuple(part.casefold() for part in entry.path[1:]),
                entry.title.casefold(),
                entry.type_id,
            ),
        )
        for entry in entries:
            if self._filtered and not _matches(entry, terms):
                continue
            item = QTreeWidgetItem(self._group(entry.path), [entry.title])
            item.setData(0, Qt.ItemDataRole.UserRole, entry.type_id)
            item.setToolTip(0, " / ".join((*entry.path, entry.title)))
        if self._filtered:
            self.expandAll()
        else:
            for path in self._expanded & self._groups.keys():
                self._groups[path].setExpanded(True)
