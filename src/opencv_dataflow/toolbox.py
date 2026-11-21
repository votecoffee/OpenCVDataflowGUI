"""Toolbox organization for native wrappers, built-in workflows and user tools."""

import re
from dataclasses import dataclass

from PySide6.QtCore import QMimeData, Qt
from PySide6.QtWidgets import QAbstractItemView, QTreeWidget, QTreeWidgetItem

from .core import block_display_title
from .plugins import (
    BIN_PICKING_TYPES,
    GEOMETRIC_MATCHER_TYPES,
    LOCATE_3D_TYPES,
    MEASURE_3D_TYPES,
    PATH_WIDTH_TYPES,
    PIXEL_SEGMENTATION_TYPES,
    REFERENCE_INSPECT_TYPES,
    SIMPLE_EXTRA_PROVIDERS,
)
from .visual_style import INDUSTRIAL

TOOL_MIME = "application/x-opencv-dataflow-tool"
SIMPLE_PLUGINS = "Simple Plugins"
TOOL_SOURCES = ("OpenCV", "scikit-image", "SimpleITK", INDUSTRIAL, "Custom", SIMPLE_PLUGINS, "User")
_OPTIONAL_SIMPLE_FAMILIES = {
    **dict.fromkeys(GEOMETRIC_MATCHER_TYPES, "Pattern matching"),
    **dict.fromkeys(PIXEL_SEGMENTATION_TYPES, "Segmentation"),
    **dict.fromkeys(PATH_WIDTH_TYPES, "Measurements"),
    **dict.fromkeys(REFERENCE_INSPECT_TYPES, "Reference inspection"),
    **dict.fromkeys(LOCATE_3D_TYPES, "3D localization"),
    **dict.fromkeys(MEASURE_3D_TYPES, "3D measurements"),
    **dict.fromkeys(BIN_PICKING_TYPES, "Bin picking"),
    **{type_id: family for type_id, (_, _, family) in SIMPLE_EXTRA_PROVIDERS.items()},
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
    "phase_unwrapping": "Phase Unwrapping",
    "rapid": "Silhouette Tracking",
    "linemod": "RGB-D Template Matching",
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
        "analysis_tools annotation basic_processing blocks collection_tools contour_raster "
        "features focus_stacking generalized_hough hit_miss "
        "histogram histogram_comparison image_blend image_colorize path_strip "
        "logpolar_registration mask_boundary mask_resampling point_set_lines polar_image "
        "laplacian_pyramids processing differential_exposure "
        "pixel_moments pyramids raster_hough_lines registration run_contours segmentation "
        "stack_blur selection_remap visible_support "
        "subpixel_crop pose_crop "
        "template_instances "
        "tilted_statistics"
    ),
    "core": (
        "channel_transform clustering feature_covariance fourier_components fourier_correlation "
        "frequency_response gram_matrix "
        "image_layout image_noise image_norm image_statistics image_validity linear_program "
        "linear_system matrix_eigen "
        "matrix_inverse matrix_multiply matrix_structure matrix_svd nearest_rows polynomial_roots "
        "principal_components spectral"
    ),
    "imgcodecs": (
        "animation_blocks codec_probe image_codecs metadata_blocks multipage_blocks tiff_files "
        "tiff_page_access"
    ),
    "calib": (
        "calibration calibration_capture calibration_validation chessboard_quality hand_eye multicamera_calibration "
        "random_pattern remap_maps "
        "robot_world workplane"
    ),
    "geometry": "circle_fitting contour_analysis convex_overlap geometry3d line_pair shape_fitting "
    "planar_pose scene_scale sparse_multiview bundler_reconstruction translation_registration triangulation voronoi",
    "video": (
        "background_reference color_tracking ecc_registration flow_files flow_refinement "
        "flow_warp frame_interpolation "
        "mil_tracking motion_trajectory motion_trajectory_io neighbor_restoration scanline_motion "
        "stabilization_fill "
        "image_correlation image_tracking tracking_appearance refresh_tracking "
        "native_background sequence_tools"
    ),
    "videoio": (
        "capture_settings network_video paced_video video_buffer video_files "
        "video_time video_timing"
    ),
    "objdetect": "aruco_board_recovery aruco_dictionary aruco_markers charuco code_readers fiducial_targets qr_encoding qr_append",
    "xobjdetect": "hog hog_detection cascade_detection",
    "dnn": "batch_inference detection_decode detection_fusion dnn_blocks dnn_profiling soft_nms "
    "tensor_comparison multiscale_super_resolution "
    "instance_segmentation instance_tracking landmarks multi_inference ocr rotated_detection "
    "semantic_segmentation prompt_segmentation prompt_mask_decode regression local_anomaly "
    "input_shifts",
    "photo": "chromatic_aberration exposure hdr hdr_motion photo_filters restoration",
    "ml": "svm_blocks image_mixtures visual_words",
    "stitching": "image_stitching scan_stitching tiled_mosaic mosaic_layout_io registered_replay "
    "rotating_panorama panorama_points",
    "stereo": (
        "stereo3d stereo_omnidir_reconstruction binary_stereo binary_stereo_costs "
        "stereo_color disparity_speckles disparity_wls "
        "disparity_evaluation quasi_dense_stereo collinear_rectification"
    ),
    "ccm": "color_calibration",
    "quality": "image_quality",
    "img_hash": "image_hash",
    "rgbd": "ptcloud3d rgbd_registration",
    "tracking": "native_trackers",
    "features": "native_lightglue native_index",
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
        "3d.camera 3d.project 3d.inspect_projection_matrix 3d.undistort_points 3d.planar_camera_motion "
        "3d.stereo_rig 3d.stereo_calibrate "
        "3d.validate_stereo_calibration 3d.calibrate_multicamera 3d.validate_multicamera "
        "3d.stereo_rectify 3d.stereo_rectified_camera 3d.rectify_camera "
        "3d.check_epipolar 3d.refine_epipolar "
        "3d.rectify_uncalibrated_stereo 3d.fisheye_calibrate "
        "3d.omnidir_calibrate 3d.wide_stereo_calibrate calibration.refine_target "
        "calibration.marked_checkerboard 3d.register_cameras"
    ),
    "geometry": "inspect.affine inspect.homography collection.clip_lines 3d.estimate_matched_transform",
    "features": (
        "inspect.affine_features inspect.evaluate_features "
        "inspect.fast inspect.orb inspect.sift inspect.akaze inspect.kaze inspect.brisk "
        "inspect.mser inspect.simple_blob_detector "
        "inspect.match_orb inspect.match_sift inspect.match_akaze inspect.match_brisk "
        "inspect.match_descriptor_radius"
    ),
    "flann": "inspect.match_flann",
    "bgsegm": "sequence.gsoc_background sequence.lsbp_background",
    "videoio": "input.camera",
    "objdetect": "3d.aruco_pose input.coded_rig_observations",
    "ptcloud": (
        "3d.depth_cloud 3d.backproject_depth_pixels 3d.separate_depth_plane 3d.cloud_project 3d.mesh_project 3d.evaluate_depth "
        "3d.reproject_depth_region 3d.fuse_registered_depth 3d.normalize_depth "
        "3d.warp_rgbd_pose 3d.tsdf_hash 3d.organized_mesh 3d.organized_planes 3d.mesh_texture "
        "3d.depth_components 3d.depth_silhouette 3d.track_depth_fusion 3d.track_deforming_surface"
    ),
    "rgbd": "3d.register_depth",
    "structured_light": (
        "3d.graycode 3d.graycode_decode 3d.phase_shift_patterns 3d.phase_shift_decode "
        "3d.camera_projector_calibrate 3d.camera_projector_reconstruct 3d.projector_inverse_rectify"
    ),
    "phase_unwrapping": "3d.unwrap_phase",
    "rapid": "3d.track_silhouette",
    "linemod": "3d.train_linemod_pose 3d.match_linemod_pose",
    "surface_matching": "3d.icp 3d.ppf_train 3d.ppf_match 3d.register_cloud_features",
    "ximgproc": "cv.guided_filter cv.domain_transform_filter inspect.thin_mask 3d.smooth_depth",
    "mcc": "calibration.detect_color_chart",
    None: (
        "input.demo output.results input.calibration output.calibration "
        "calibration.pixel_to_physical "
        "calibration.lens_map_displacement collection.filter collection.sort collection.draw "
        "core.attach_payload core.read_payload flow.detection_cadence input.cloud_tiled_radius "
        "sequence.background sequence.image_memory "
        "cv.distortion_fixture "
        "inspect.region_contrast "
        "sequence.foreground_debounce "
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
    "3d.tsdf_mesh": ("TSDF", "marching cubes", "surface reconstruction"),
    "3d.organized_mesh": ("depth grid", "triangulation", "discontinuity", "organized cloud"),
    "3d.organized_planes": ("findPlanes", "plane coefficients", "depth labels", "foreground"),
    "3d.tsdf_color": ("ColorTSDF", "RGB-D fusion", "colored point cloud"),
    "3d.tsdf_hash": ("HashTSDF", "sparse volume", "bounded growth"),
    "3d.cloud_compare": ("dimensional comparison", "cloud deviation", "point to plane", "overlap"),
    "3d.cloud_merge": ("scan merge", "registered views", "duplicate suppression", "provenance"),
    "3d.cloud_datum": ("datum", "flatness", "surface height", "peak to valley"),
    "3d.cloud_section": ("cross section", "metric slab", "width profile"),
    "3d.cloud_oriented_extents": ("oriented bounding box", "PCA", "dimensions", "pose"),
    "3d.estimate_matched_transform": ("estimateAffine3D", "estimateTranslation3D", "3D landmarks"),
    "3d.cloud_project": ("projectPoints", "z-buffer", "occlusion"),
    "3d.cloud_range_image": ("range image", "LiDAR", "spherical projection", "return intensity"),
    "3d.backproject_depth_pixels": ("depthTo3dSparse", "sparse depth", "metric XYZ"),
    "3d.cloud_octree_query": ("octree", "nearest neighbors", "radius search"),
    "3d.cloud_cluster": ("euclidean clustering", "connected components", "part segmentation"),
    "3d.cloud_region_grow": ("region growing", "surface segmentation", "normal curvature"),
    "3d.cloud_model_fit": ("ransac", "fit plane", "fit sphere", "sample consensus"),
    "3d.cloud_cylinder": ("bore", "fit cylinder", "cylinder axis", "diameter", "metrology"),
    "3d.cloud_surface_quality": ("estimate normals", "curvature", "local PCA"),
    "3d.scan_mesh_compare": ("scan to mesh", "closest triangle", "surface deviation"),
    "3d.mesh_inspect": ("mesh validity", "boundaries", "nonmanifold", "surface area", "volume"),
    "3d.mesh_simplify": ("mesh decimation", "edge collapse", "face budget", "reduction"),
    "3d.mesh_texture": ("triangleRasterizeDepth", "UV", "texture atlas"),
    "3d.mesh_project": ("triangleRasterize", "triangleRasterizeColor", "triangleRasterizeDepth"),
    "input.mesh": ("obj", "ply", "loadMesh"),
    "output.mesh_ply": ("obj", "ply", "saveMesh"),
    "input.cloud": ("obj", "ply", "pcd", "loadPointCloud"),
    "output.cloud": ("obj", "ply", "pcd", "savePointCloud"),
    "input.cloud_recording": ("scan replay", "recording", "PLY", "PCD", "OBJ"),
    "input.cloud_tiled_radius": ("large scan", "tiled PLY", "radius neighbors"),
    "3d.cloud_outlier": ("point density", "remove noise", "statistical outlier removal"),
    "3d.cloud_sample": ("randomSampling", "farthestPointSampling", "downsample", "FPS"),
    "3d.cloud_perturb": ("addNoisePC", "getRandomPose", "synthetic point cloud", "seeded noise"),
    "3d.cloud_voxel": ("voxelGridSampling", "nearest centroid", "representative point"),
    "3d.normalize_depth": ("rescaleDepth", "depth units", "decode depth"),
    "3d.warp_rgbd_pose": ("warpFrame", "RGB-D pose", "camera viewpoint"),
    "3d.quantize_depth": ("depth codes", "depth units", "encode depth"),
    "3d.separate_depth_plane": ("depthTo3d", "RANSAC", "foreground segmentation"),
    "3d.smooth_depth": ("jointBilateralFilter", "confidence-weighted depth denoising"),
    "3d.clean_tof_depth": ("ToF", "flying pixels", "depth artifact rejection"),
    "inspect.differential_exposure": ("boxFilter", "illumination", "reference comparison"),
    "cv.resample_mask": ("integral", "area coverage", "validity mask"),
    "cv.stack_blur": ("stackBlur", "large-radius smoothing"),
    "cv.gray": ("cvtColor",),
    "cv.alpha_over": ("alpha", "premultiplied", "source over", "opacity", "transparency"),
    "inspect.intensity_moments": ("moments", "centroid", "center of mass", "principal axis"),
    "inspect.region_contrast": ("CNR", "meanStdDev", "Weber", "Michelson", "visibility"),
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
    "cv.correct_chromatic_aberration": ("correctChromaticAberration", "color fringe", "lens"),
    "input.chromatic_calibration": ("loadChromaticAberrationParams", "lens calibration"),
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
    "3d.rectify_collinear": ("rectify3Collinear", "trinocular", "shared stereo grid"),
    "3d.register_cameras": ("registerCameras", "registerCamerasExtended", "non-overlapping cameras"),
    "calibration.marked_checkerboard": (
        "findChessboardCornersSBWithMeta", "oversized checkerboard", "marker origin"
    ),
    "calibration.refine_target": ("calibrateCameraRO", "calibrateCameraROExtended", "release object"),
    "inspect.homography": ("findHomography", "calib3d"),
    "3d.pnp": ("solvePnP", "calib3d"),
    "3d.recover_aruco_board": ("ArUco board", "GridBoard", "refineDetectedMarkers", "matchImagePoints"),
    "3d.planar_pose_hypotheses": ("solvePnPGeneric", "IPPE", "IPPE square"),
    "ml.dnn": ("dnn.readNetFromONNX", "readNetFromONNX"),
}


@dataclass(frozen=True)
class ToolEntry:
    path: tuple[str, ...]
    title: str
    type_id: str
    keywords: tuple[str, ...] = ()


_GEOMETRY_FILE_TITLES = {
    "input.cloud": "Load point cloud (PLY / PCD / OBJ)",
    "output.cloud": "Save point cloud (PLY / PCD / OBJ)",
    "output.mesh_ply": "Save triangle mesh (OBJ / PLY)",
}


def block_entry(block):
    kind, subtype = block.block_type or block.category, block.subtype
    package, _, file = block.__module__.rpartition(".")
    type_id = block.type_id
    if type_id in _SIMPLE_FAMILIES:
        path = (SIMPLE_PLUGINS, _SIMPLE_FAMILIES[type_id])
    elif package.split(".")[0] != "opencv_dataflow":
        path = ("User", kind, subtype)
    elif kind == INDUSTRIAL:
        path = (INDUSTRIAL, subtype)
    elif file == "simpleitk_blocks":
        path = ("SimpleITK", subtype)
    elif file == "multiscale_ridges":
        path = ("scikit-image", "Filtering")
    elif type_id == "3d.tsdf_mesh":
        path = ("scikit-image", subtype)
    elif file == "skimage_regions":
        path = ("scikit-image", "Measurements")
    elif file == "skimage_thresholds":
        path = ("scikit-image", "Thresholding")
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
    title = _GEOMETRY_FILE_TITLES.get(type_id) or block_display_title(block)
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
