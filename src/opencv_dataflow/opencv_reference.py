"""References for the OpenCV bindings that implement built-in tools."""

from html import escape

import cv2

# Keep this explicit so wrapper helpers and result drawing calls do not appear as
# the tool's algorithm. A block can use several bindings or select one by mode.
BINDINGS = {
    "3d.aruco_pose": "aruco.ArucoDetector.detectMarkers solvePnP",
    "3d.disparity": "StereoBM.create StereoSGBM.create",
    "3d.essential_pose": "findEssentialMat recoverPose",
    "3d.check_epipolar": "computeCorrespondEpilines",
    "3d.fundamental": "findFundamentalMat",
    "3d.pnp": "solvePnP solvePnPRansac solvePnPRefineLM solvePnPRefineVVS",
    "3d.project": "projectPoints fisheye.projectPoints omnidir.projectPoints",
    "3d.reproject": "reprojectImageTo3D",
    "3d.stereo_calibrate": "stereoCalibrate",
    "3d.stereo_rectify": "stereoRectify",
    "3d.triangulate": "triangulatePoints",
    "3d.undistort_points": "undistortPoints fisheye.undistortPoints omnidir.undistortPoints",
    "calibration.charuco": "aruco.CharucoDetector.detectBoard",
    "calibration.chessboard_quality": "findChessboardCornersSB estimateChessboardSharpness",
    "calibration.grid": "findChessboardCorners findChessboardCornersSB findHomography",
    "calibration.lens": "calibrateCameraExtended",
    "calibration.lens_maps": "initUndistortRectifyMap getOptimalNewCameraMatrix",
    "calibration.undistort": "undistort",
    "calibration.undistort_lens_points": "undistortPoints projectPoints",
    "cv.absdiff": "absdiff",
    "cv.adaptive_threshold": "adaptiveThreshold",
    "cv.align_exposures": "createAlignMTB AlignMTB",
    "cv.calibrate_debevec": "createCalibrateDebevec CalibrateDebevec",
    "cv.apply_mask": "copyTo",
    "cv.auto_threshold": "threshold thresholdWithMask",
    "cv.bilateral": "bilateralFilter",
    "cv.bitwise": "bitwise_and bitwise_or bitwise_xor bitwise_not",
    "cv.blend": "addWeighted",
    "cv.box_blur": "blur",
    "cv.box_filter": "boxFilter",
    "cv.canny": "Canny",
    "cv.canny_derivatives": "Canny",
    "cv.clahe": "createCLAHE CLAHE.apply",
    "cv.colorize_scalar": "applyColorMap",
    "cv.compare_pixels": "compare",
    "cv.control_point_transform": "getAffineTransform getPerspectiveTransform",
    "cv.copy_border": "copyMakeBorder",
    "cv.crop_subpixel": "getRectSubPix",
    "cv.dct": "dct",
    "cv.decode_image": "imdecode",
    "cv.demosaic_bayer": "demosaicing",
    "cv.detail_enhance": "detailEnhance",
    "cv.dft": "dft",
    "cv.distance_labels": "distanceTransformWithLabels",
    "cv.distance_transform": "distanceTransform",
    "cv.domain_transform_filter": "ximgproc.dtFilter",
    "cv.edge_preserving_photo": "edgePreservingFilter",
    "cv.encode_image": "imencode",
    "cv.equalize": "equalizeHist",
    "cv.fuse_exposures": "createMergeMertens MergeMertens",
    "cv.extract_channel": "extractChannel",
    "cv.filter2d": "filter2D",
    "cv.flip": "flip",
    "cv.gabor_bank": "getGaborKernel filter2D",
    "cv.gabor_response": "getGaborKernel filter2D",
    "cv.gaussian_blur": "GaussianBlur",
    "cv.graph_segments": "ximgproc.segmentation.createGraphSegmentation",
    "cv.gradient": "Sobel Scharr Laplacian",
    "cv.gray": "cvtColor",
    "cv.guided_filter": "ximgproc.guidedFilter",
    "cv.idct": "idct",
    "cv.idft": "idft",
    "cv.image_arithmetic": "add subtract multiply divide min max absdiff",
    "cv.image_pyramid": "pyrDown",
    "cv.in_range": "inRange",
    "cv.inpaint": "inpaint",
    "cv.insert_channel": "insertChannel",
    "cv.mean_shift_filter": "pyrMeanShiftFiltering",
    "cv.median_blur": "medianBlur",
    "cv.merge_channels": "merge",
    "cv.merge_debevec": "createMergeDebevec MergeDebevec",
    "cv.morphology": "morphologyEx getStructuringElement",
    "cv.non_local_means": "fastNlMeansDenoising fastNlMeansDenoisingColored",
    "cv.normalize": "normalize",
    "cv.pack_remap_map": "convertMaps",
    "cv.pencil_sketch": "pencilSketch",
    "cv.pyramid_down": "pyrDown",
    "cv.pyramid_up": "pyrUp",
    "cv.remap": "remap",
    "cv.remap_packed": "remap",
    "cv.resize": "resize copyMakeBorder",
    "cv.rotate": "getRotationMatrix2D warpAffine",
    "cv.rotate90": "rotate",
    "cv.route_channels": "mixChannels",
    "cv.seamless_clone": "seamlessClone",
    "cv.separable_filter": "sepFilter2D",
    "cv.spatial_gradient": "spatialGradient",
    "cv.split_channels": "split",
    "cv.stitch_image_pair": "warpPerspective",
    "cv.stylize_photo": "stylization",
    "cv.tonemap_hdr": "createTonemap createTonemapDrago createTonemapReinhard createTonemapMantiuk",
    "cv.threshold": "threshold",
    "cv.transform_points": "perspectiveTransform",
    "cv.transpose_image": "transpose",
    "cv.unsharp": "GaussianBlur addWeighted",
    "cv.warp_affine": "warpAffine",
    "cv.warp_perspective": "warpPerspective",
    "cv.warp_polar": "warpPolar",
    "gapi.blur": "gapi.gaussianBlur",
    "gapi.canny": "gapi.Canny",
    "gapi.gray": "gapi.BGR2Gray",
    "input.camera": "VideoCapture.open VideoCapture.read",
    "input.depth_file": "imdecode",
    "input.file": "imdecode",
    "input.tiff_file": "imdecodemulti",
    "input.video_file": "VideoCapture.open VideoCapture.read",
    "inspect.affine": "estimateAffine2D estimateAffinePartial2D",
    "inspect.akaze": "xfeatures2d.AKAZE_create",
    "inspect.align_ecc": "findTransformECCMultiScale findTransformECCWithMask",
    "inspect.approximate_contour": "approxPolyDP",
    "inspect.approximate_convex_contour": "approxPolyN",
    "inspect.aruco_markers": "aruco.ArucoDetector.detectMarkers",
    "inspect.blobs": "connectedComponentsWithStatsWithAlgorithm",
    "inspect.bounding_convex_polygon": "convexHull",
    "inspect.bounding_rotated_rectangle": "minAreaRect boxPoints",
    "inspect.brisk": "xfeatures2d.BRISK_create",
    "inspect.compare_contours": "matchShapes",
    "inspect.compare_edge_maps": "distanceTransform",
    "inspect.compare_histograms": "compareHist",
    "inspect.components": "connectedComponentsWithStatsWithAlgorithm",
    "inspect.contour_hierarchy": "findContours",
    "inspect.contour_hull": "convexHull convexityDefects",
    "inspect.contour_points": "pointPolygonTest",
    "inspect.contours": "findContours contourArea moments",
    "inspect.corner_response": "cornerHarris cornerMinEigenVal",
    "inspect.corners": "goodFeaturesToTrack",
    "inspect.fast": "FastFeatureDetector_create",
    "inspect.flood_fill": "floodFill",
    "inspect.grabcut": "grabCut",
    "inspect.histogram_backproject": "calcBackProject",
    "inspect.hog": "HOGDescriptor HOGDescriptor.compute",
    "inspect.homography": "findHomography",
    "inspect.hough_circles": "HoughCircles HoughCirclesWithAccumulator",
    "inspect.hough_lines": "HoughLinesP",
    "inspect.hough_point_lines": "HoughLinesPointSet",
    "inspect.hough_polar_lines": "HoughLinesWithAccumulator",
    "inspect.kaze": "xfeatures2d.KAZE_create",
    "inspect.kmeans_image": "kmeans",
    "inspect.line_segments": "createLineSegmentDetector LineSegmentDetector.detect",
    "inspect.match_descriptor_radius": "DescriptorMatcher.radiusMatch",
    "inspect.match_template": "matchTemplate",
    "inspect.mser": "MSER_create MSER.detectRegions",
    "inspect.orb": "ORB_create Feature2D.detectAndCompute",
    "inspect.phase_correlation": "phaseCorrelate",
    "inspect.read_barcode": "barcode.BarcodeDetector.detectAndDecodeWithType",
    "inspect.read_qr": "GraphicalCodeDetector.detectAndDecode",
    "inspect.refine_corners": "cornerSubPix",
    "inspect.sift": "SIFT_create Feature2D.detectAndCompute",
    "inspect.selective_search": "ximgproc.segmentation.createSelectiveSearchSegmentation",
    "inspect.simple_blob_detector": "SimpleBlobDetector.create Feature2D.detect",
    "inspect.thin_mask": "ximgproc.thinning",
    "inspect.watershed": "watershed",
    "ml.dnn": "dnn.readNetFromONNX dnn.Net.setInput dnn.Net.forward",
    "ml.kmeans": "kmeans",
    "output.image": "imencode",
    "output.tiff_file": "imencodemulti",
    "output.video": "VideoWriter.open VideoWriter.write",
    "sequence.dis": "DISOpticalFlow.create DenseOpticalFlow.calc",
    "sequence.farneback": "calcOpticalFlowFarneback",
    "sequence.gsoc_background": "bgsegm.createBackgroundSubtractorGSOC BackgroundSubtractor.apply",
    "sequence.lsbp_background": "bgsegm.createBackgroundSubtractorLSBP BackgroundSubtractor.apply",
    "sequence.knn_background": "createBackgroundSubtractorKNN BackgroundSubtractor.apply",
    "sequence.lk_grid_flow": "calcOpticalFlowPyrLK remap",
    "sequence.mog2": "createBackgroundSubtractorMOG2 BackgroundSubtractor.apply",
    "sequence.track_points": "calcOpticalFlowPyrLK",
}

_DOCS = "https://docs.opencv.org/5.0"


def _binding_reference(name):
    binding = cv2
    for part in name.split("."):
        binding = getattr(binding, part, None)
    if not callable(binding):
        return ""
    doc = getattr(binding, "__doc__", None)
    if not doc or not doc.strip():
        return ""
    # The Python generator prefixes the body with ".   " and preserves Doxygen
    # commands. Keep the content verbatim apart from display formatting.
    lines = []
    for line in doc.strip().splitlines():
        if line.startswith("."):
            line = line.removeprefix(".").strip()
            lines.append(escape(line.removeprefix("@brief ")))
        elif line:
            lines.append(f"<code>{escape(line)}</code>")
        else:
            lines.append("")
    return f"<h4>cv2.{escape(name)}</h4><p>{'<br>'.join(lines)}</p>"


def opencv_reference(block):
    """Return an offline HTML fragment from the installed binding, if available."""
    if type(block).__module__.rpartition(".")[0] != "opencv_dataflow":
        return ""
    sections = "".join(
        _binding_reference(name) for name in BINDINGS.get(block.type_id, "").split()
    )
    if not sections:
        return ""
    return (
        f"<h3>OpenCV {escape(cv2.__version__)} Python reference</h3>"
        "<p>From the installed OpenCV Python bindings. Parameters and overloads "
        "describe the underlying API. The block may expose a subset or add its own controls.</p>"
        f'<p><a href="{_DOCS}/main_modules/api_root.html">OpenCV main modules</a> · '
        f'<a href="{_DOCS}/extra_modules/api_root.html">OpenCV extra modules</a> · '
        '<a href="https://opencv.org/license/">OpenCV license (Apache 2.0)</a></p>'
        + sections
    )
