"""References for the OpenCV bindings that implement built-in tools."""

from html import escape

import cv2

# Keep this explicit so wrapper helpers and result drawing calls do not appear as
# the tool's algorithm. A block can use several bindings or select one by mode.
BINDINGS = {
    "ml.super_resolution_multi": "dnn.readNetFromTensorflow dnn.Net.forward dnn.blobFromImage",
    "cv.crop_pose": "warpAffine erode",
    "ml.cascade_detect": (
        "CascadeClassifier CascadeClassifier.read CascadeClassifier.detectMultiScale "
        "CascadeClassifier.detectMultiScale2 CascadeClassifier.detectMultiScale3"
    ),
    "ml.hog_detect": "HOGDescriptor HOGDescriptor.setSVMDetector HOGDescriptor.detectMultiScale",
    "ml.hog_detector_train": (
        "HOGDescriptor.compute HOGDescriptor.detect HOGDescriptor.setSVMDetector "
        "ml.SVM_create ml.SVM.train ml.SVM.getDecisionFunction"
    ),
    "ml.hog_detector_model": "HOGDescriptor.getDefaultPeopleDetector "
    "HOGDescriptor.getDaimlerPeopleDetector ml.SVM.getDecisionFunction",
    "ml.visual_features": "SIFT_create ORB_create",
    "ml.bow_train": "kmeans BFMatcher",
    "ml.bow_encode": "BFMatcher",
    "ml.bow_rank": "norm compareHist",
    "cv.feature_covariance": "calcCovarMatrix",
    "inspect.mahalanobis_distance": "Mahalanobis eigen invert",
    "ml.em_train_image": "ml.EM_create ml.EM.trainEM ml.EM.trainM",
    "ml.em_classify_image": "ml.EM.predict ml.EM.read",
    "cv.fit_pca": "PCACompute2",
    "cv.transform_pca": "PCAProject PCABackProject",
    "sequence.track_vit": (
        "TrackerVit.create TrackerVit.init TrackerVit.update TrackerVit.getTrackingScore"
    ),
    "ml.superpoint_onnx_features": (
        "dnn.readNetFromONNX dnn.Net.setInput dnn.Net.forward dilate resize cvtColor"
    ),
    "3d.reconstruct_sparse_views": "undistortPoints projectPoints",
    "input.bundler_reconstruction": "projectPoints",
    "calibration.generate_random_pattern": "resize circle fillConvexPoly",
    "calibration.detect_random_pattern": (
        "SIFT_create Feature2D.detectAndCompute BFMatcher DescriptorMatcher.knnMatch "
        "DescriptorMatcher.match findHomography convexHull contourArea"
    ),
    "3d.recover_scene_scale": "estimateAffine3D",
    "3d.mesh_texture": "triangleRasterizeDepth",
    "3d.estimate_matched_transform": "estimateAffine3D estimateTranslation3D",
    "ml.optical_flow": "dnn.readNetFromONNX dnn.Net.setInput dnn.Net.forward remap",
    "ml.stereo_disparity": "dnn.readNetFromONNX dnn.Net.setInput dnn.Net.forward remap",
    "ml.monocular_depth": "dnn.readNetFromONNX dnn.Net.setInput dnn.Net.forward remap",
    "ml.restore_image": "dnn.readNetFromONNX dnn.Net.setInput dnn.Net.forward",
    "ml.super_resolution": (
        "dnn_superres.DnnSuperResImpl_create dnn_superres.DnnSuperResImpl.readModel "
        "dnn_superres.DnnSuperResImpl.setModel dnn_superres.DnnSuperResImpl.upsample "
        "dnn_superres.DnnSuperResImpl.setPreferableBackend "
        "dnn_superres.DnnSuperResImpl.setPreferableTarget"
    ),
    "inspect.estimate_motion_blur": "dft",
    "sequence.fill_stabilization_borders": "remap connectedComponents distanceTransform dilate",
    "cv.panorama_points": "perspectiveTransform",
    "cv.stitch_rotating_panorama": (
        "Stitcher_create Stitcher.estimateTransform PyRotationWarper detail.waveCorrect remap "
        "detail.computeImageFeatures2 detail.BestOf2NearestMatcher detail.HomographyBasedEstimator "
        "detail.BundleAdjusterRay detail.BundleAdjusterReproj"
    ),
    "cv.merge_hdr_motion": "LUT",
    "inspect.visible_support": "remap perspectiveTransform",
    "cv.remap_selection": "remap",
    "cv.remap_selection_packed": "remap convertMaps",
    "input.paced_video_file": "VideoCapture",
    "input.network_video": "VideoCapture.open VideoCapture.read",
    "input.video_at_time": "VideoCapture.set VideoCapture.read",
    "sequence.video_timing": "VideoCapture.get",
    "sequence.track_refreshed": (
        "KalmanFilter calcOpticalFlowPyrLK goodFeaturesToTrack estimateAffinePartial2D"
    ),
    "sequence.track_image": "KalmanFilter KalmanFilter.predict KalmanFilter.correct",
    "inspect.scanline_motion": "goodFeaturesToTrack calcOpticalFlowPyrLK",
    "sequence.restore_neighbors": "Sobel GaussianBlur blur boxFilter erode dilate",
    "sequence.replay_trajectory": "warpAffine",
    "inspect.frequency_response": "dft divSpectrums createHanningWindow",
    "cv.replay_registered_views": (
        "warpPerspective detail.Timelapser.createDefault "
        "detail.Timelapser.initialize detail.Timelapser.process detail.Timelapser.getDst"
    ),
    "inspect.evaluate_features": "perspectiveTransform BFMatcher.knnMatch",
    "inspect.evaluate_keypoints": "perspectiveTransform",
    "sequence.multi_frame_super_resolution": "phaseCorrelate findTransformECC resize",
    "inspect.read_qr_append": (
        "QRCodeDetector QRCodeDetectorAruco GraphicalCodeDetector.detectMulti "
        "GraphicalCodeDetector.decodeBytes"
    ),
    "inspect.differential_exposure": "boxFilter",
    "cv.resample_mask": "integral",
    "cv.stack_blur": "stackBlur",
    "input.animation_file": "Animation imreadanimation",
    "cv.decode_animation": "Animation imdecodeanimation",
    "cv.encode_animation": "Animation imencodeanimation imdecodeanimation",
    "cv.encode_pages": "imencodemulti imdecodemulti",
    "cv.decode_pages": "imdecodemulti",
    "cv.probe_image_file": "haveImageReader haveImageWriter imcount",
    "cv.hit_or_miss": "morphologyEx getStructuringElement",
    "output.animation_file": "Animation imwriteanimation imdecodeanimation",
    "cv.encode_qr": (
        "QRCodeEncoder_create QRCodeEncoder.encode QRCodeEncoder.encodeStructuredAppend"
    ),
    "cv.refine_flow": "VariationalRefinement_create VariationalRefinement.calc",
    "sequence.color_tracker": "meanShift CamShift calcHist calcBackProject",
    "cv.correct_chromatic_aberration": "correctChromaticAberration",
    "input.chromatic_calibration": "loadChromaticAberrationParams",
    "ml.svm_train": "ml.SVM_create ml.SVM.train ml.SVM.write",
    "ml.svm_predict": "ml.SVM.predict",
    "ml.svm_load": "ml.SVM.read",
    "ml.one_class_svm_train": "ml.SVM_create ml.SVM.train ml.SVM.getDecisionFunction",
    "ml.one_class_svm_predict": "ml.SVM.predict ml.SVM.read",
    "ml.one_class_svm_load": "ml.SVM.read",
    "cv.reference_shadows": "connectedComponentsWithStats erode",
    "inspect.logpolar_registration": (
        "dft warpPolar phaseCorrelate findTransformECCWithMask warpAffine"
    ),
    "cv.focus_stack": "Sobel Laplacian GaussianBlur boxFilter erode",
    "inspect.mask_boundary": "distanceTransform erode findContours",
    "3d.aruco_pose": "aruco.ArucoDetector.detectMarkers solvePnP",
    "3d.disparity": "StereoBM.create StereoSGBM.create getValidDisparityROI",
    "3d.binary_disparity": "boxFilter erode",
    "3d.binary_stereo_costs": "boxFilter",
    "3d.quasi_dense_disparity": (
        "stereo.QuasiDenseStereo.create stereo.QuasiDenseStereo.process "
        "stereo.QuasiDenseStereo.getDenseMatches stereo.QuasiDenseStereo.getSparseMatches"
    ),
    "3d.filter_disparity_speckles": "filterSpeckles",
    "3d.filter_disparity_wls": (
        "ximgproc.createDisparityWLSFilterGeneric ximgproc.DisparityFilter.filter "
        "ximgproc.DisparityWLSFilter.getConfidenceMap ximgproc.createFastGlobalSmootherFilter"
    ),
    "3d.evaluate_disparity": "applyColorMap",
    "3d.stereo_color_match": "applyColorMap",
    "3d.essential_pose": "findEssentialMat recoverPose",
    "3d.independent_relative_pose": (
        "undistortPoints fisheye.undistortPoints omnidir.undistortPoints findEssentialMat "
        "recoverPose findFundamentalMat triangulatePoints projectPoints "
        "fisheye.projectPoints omnidir.projectPoints"
    ),
    "3d.check_epipolar": "computeCorrespondEpilines",
    "3d.refine_epipolar": "sampsonDistance correctMatches",
    "3d.rectify_uncalibrated_stereo": "stereoRectifyUncalibrated warpPerspective",
    "3d.fundamental": "findFundamentalMat",
    "3d.recover_aruco_board": (
        "aruco.ArucoDetector.refineDetectedMarkers aruco.Board.matchImagePoints solvePnP"
    ),
    "3d.planar_pose_hypotheses": "solvePnPGeneric projectPoints",
    "3d.planar_camera_motion": "findHomography decomposeHomographyMat filterHomographyDecompByVisibleRefpoints",
    "3d.pnp": "solvePnP solvePnPRansac solvePnPRefineLM solvePnPRefineVVS",
    "3d.wide_pnp": (
        "fisheye.solvePnP fisheye.solvePnPRansac fisheye.undistortPoints fisheye.projectPoints "
        "omnidir.undistortPoints omnidir.projectPoints solvePnP solvePnPRansac"
    ),
    "3d.project": "projectPoints fisheye.projectPoints omnidir.projectPoints",
    "3d.unwrap_phase": (
        "phase_unwrapping.HistogramPhaseUnwrapping.create "
        "phase_unwrapping.PhaseUnwrapping.unwrapPhaseMap "
        "phase_unwrapping.HistogramPhaseUnwrapping.getInverseReliabilityMap"
    ),
    "3d.train_linemod_pose": (
        "linemod.ColorGradient.create linemod.DepthNormal.create "
        "linemod.Detector.addTemplate linemod.Detector.getTemplates"
    ),
    "3d.match_linemod_pose": "linemod.Detector.match linemod.Detector.readClasses matchTemplate",
    "3d.track_depth_fusion": "Odometry.compute Volume.integrate Volume.raycast Rodrigues",
    "3d.track_deforming_surface": "calcOpticalFlowPyrLK",
    "3d.tsdf_color": (
        "VolumeSettings Volume.integrateColor Volume.fetchPointsNormalsColors Volume.raycastExColor"
    ),
    "3d.tsdf_fuse": "VolumeSettings Volume.integrate Volume.fetchPointsNormals Volume.raycastEx",
    "3d.tsdf_hash": (
        "VolumeSettings Volume.integrate Volume.setEnableGrowth Volume.getVisibleBlocks "
        "Volume.getTotalVolumeUnits Volume.getBoundingBox Volume.fetchPointsNormals"
    ),
    "3d.track_silhouette": (
        "rapid.extractControlPoints rapid.extractLineBundle rapid.findCorrespondencies "
        "projectPoints Rodrigues"
    ),
    "3d.phase_shift_patterns": (
        "structured_light.SinusoidalPattern.create structured_light.SinusoidalPattern.generate"
    ),
    "3d.cloud_project": "projectPoints",
    "3d.backproject_depth_pixels": "depthTo3dSparse",
    "3d.organized_mesh": "depthTo3d",
    "3d.organized_planes": "depthTo3d RgbdNormals findPlanes",
    "3d.cloud_oriented_extents": "PCACompute2 Rodrigues",
    "3d.normalize_depth": "rescaleDepth",
    "3d.warp_rgbd_pose": "warpFrame",
    "3d.evaluate_depth": "applyColorMap",
    "3d.fuse_registered_depth": "merge",
    "3d.reproject_depth_region": (
        "undistortPoints fisheye.undistortPoints omnidir.undistortPoints "
        "projectPoints fisheye.projectPoints omnidir.projectPoints"
    ),
    "input.coded_rig_observations": "aruco.ArucoDetector.detectMarkers aruco.drawDetectedMarkers",
    "3d.calibrate_multicamera": (
        "calibrateMultiviewExtended calibrateCamera fisheye.calibrate solvePnP projectPoints"
    ),
    "3d.validate_multicamera": "solvePnP solvePnPRefineLM projectPoints fisheye.projectPoints",
    "3d.camera_projector_calibrate": "calibrateCamera stereoCalibrateExtended",
    "3d.camera_projector_reconstruct": "undistortPoints triangulatePoints projectPoints",
    "3d.projector_inverse_rectify": "initInverseRectificationMap remap projectPoints",
    "3d.inspect_projection_matrix": "decomposeProjectionMatrix calibrationMatrixValues RQDecomp3x3",
    "3d.reproject": "reprojectImageTo3D",
    "3d.smooth_depth": "ximgproc.jointBilateralFilter",
    "3d.separate_depth_plane": "depthTo3d findContours approxPolyDP",
    "3d.depth_components": "connectedComponentsWithAlgorithm",
    "3d.depth_silhouette": "erode convexHull minAreaRect boxPoints",
    "3d.omnidir_calibrate": "omnidir.calibrate omnidir.projectPoints convexHull contourArea",
    "3d.fisheye_calibrate": "fisheye.calibrate fisheye.projectPoints convexHull contourArea",
    "3d.stereo_calibrate": "stereoCalibrateExtended",
    "3d.validate_stereo_calibration": "solvePnP solvePnPRefineLM projectPoints",
    "3d.wide_stereo_calibrate": "fisheye.stereoCalibrate omnidir.stereoCalibrate",
    "3d.omnidir_stereo_reconstruct": (
        "omnidir.stereoReconstruct omnidir.initUndistortRectifyMap omnidir.stereoRectify"
    ),
    "3d.stereo_rectify": (
        "stereoRectify fisheye.stereoRectify omnidir.stereoRectify "
        "initUndistortRectifyMap fisheye.initUndistortRectifyMap "
        "omnidir.initUndistortRectifyMap remap"
    ),
    "3d.triangulate": "triangulatePoints",
    "3d.undistort_points": "undistortPoints fisheye.undistortPoints omnidir.undistortPoints",
    "calibration.generate_fiducial_target": (
        "aruco.generateImageMarker aruco.Board.generateImage"
    ),
    "calibration.charuco": "aruco.CharucoDetector.detectBoard",
    "calibration.chessboard_quality": "findChessboardCornersSB estimateChessboardSharpness",
    "calibration.grid": "findChessboardCorners findChessboardCornersSB findHomography",
    "calibration.lens": "calibrateCameraExtended Rodrigues",
    "calibration.audit_capture": "projectPoints",
    "calibration.measure_uncertainty": "undistortPoints Rodrigues projectPoints",
    "3d.cloud_fit_line": "fitLine",
    "calibration.validate_lens": (
        "solvePnP projectPoints fisheye.solvePnP fisheye.projectPoints "
        "omnidir.undistortPoints omnidir.projectPoints"
    ),
    "calibration.lens_maps": "initUndistortRectifyMap getOptimalNewCameraMatrix",
    "calibration.undistort": "undistort",
    "calibration.undistort_lens_points": "undistortPoints projectPoints",
    "collection.rasterize_contours": "drawContours fillPoly polylines",
    "cv.absdiff": "absdiff",
    "cv.interpolate_frames": "calcOpticalFlowFarneback remap",
    "cv.adaptive_threshold": "adaptiveThreshold",
    "cv.local_threshold": "ximgproc.niBlackThreshold",
    "cv.ridge_response": (
        "ximgproc.RidgeDetectionFilter_create ximgproc.RidgeDetectionFilter.getRidgeFilteredImage"
    ),
    "cv.lsc_superpixels": (
        "ximgproc.createSuperpixelLSC ximgproc.SuperpixelLSC.iterate "
        "ximgproc.SuperpixelLSC.enforceLabelConnectivity ximgproc.SuperpixelLSC.getLabels "
        "ximgproc.SuperpixelLSC.getLabelContourMask"
    ),
    "cv.seeds_superpixels": (
        "ximgproc.createSuperpixelSEEDS ximgproc.SuperpixelSEEDS.iterate "
        "ximgproc.SuperpixelSEEDS.getLabels ximgproc.SuperpixelSEEDS.getLabelContourMask"
    ),
    "cv.slic_superpixels": (
        "ximgproc.createSuperpixelSLIC ximgproc.SuperpixelSLIC.iterate "
        "ximgproc.SuperpixelSLIC.enforceLabelConnectivity ximgproc.SuperpixelSLIC.getLabels "
        "ximgproc.SuperpixelSLIC.getLabelContourMask"
    ),
    "cv.align_exposures": "createAlignMTB AlignMTB",
    "cv.annotation_layer": "polylines fillPoly putText",
    "cv.calibrate_debevec": "createCalibrateDebevec CalibrateDebevec",
    "cv.calibrate_robertson": "createCalibrateRobertson CalibrateRobertson",
    "cv.apply_mask": "copyTo",
    "cv.auto_threshold": "threshold thresholdWithMask",
    "cv.bilateral": "bilateralFilter",
    "cv.bitwise": "bitwise_and bitwise_or bitwise_xor bitwise_not",
    "cv.blend": "addWeighted",
    "cv.box_blur": "blur",
    "cv.box_filter": "boxFilter",
    "cv.canny": "Canny",
    "cv.canny_derivatives": "Canny",
    "cv.local_contrast_edges": "GaussianBlur blur Sobel magnitude Canny",
    "cv.clahe": "createCLAHE CLAHE.apply",
    "cv.colorize_scalar": "applyColorMap",
    "cv.color_change": "colorChange",
    "cv.compare_pixels": "compare",
    "cv.control_point_transform": "getAffineTransform getPerspectiveTransform",
    "cv.copy_border": "copyMakeBorder",
    "cv.crop_subpixel": "getRectSubPix",
    "cv.dct": "dct",
    "cv.decode_image": "imdecode imdecodeWithMetadata",
    "cv.decolor": "decolor",
    "cv.warp_flow": "remap",
    "input.flow_file": "readOpticalFlow",
    "output.flow_file": "writeOpticalFlow",
    "cv.demosaic_bayer": "demosaicing",
    "cv.detail_enhance": "detailEnhance",
    "cv.illumination_change": "illuminationChange",
    "cv.texture_flattening": "textureFlattening",
    "cv.dft": "dft",
    "cv.reconstruct_fourier_image": "dft idft getOptimalDFTSize",
    "cv.distance_labels": "distanceTransformWithLabels",
    "cv.distance_transform": "distanceTransform",
    "cv.domain_transform_filter": "ximgproc.dtFilter",
    "cv.edge_preserving_photo": "edgePreservingFilter",
    "cv.encode_image": "imencode imencodeWithMetadata",
    "cv.equalize": "equalizeHist",
    "cv.fuse_exposures": "createMergeMertens MergeMertens",
    "cv.extract_channel": "extractChannel",
    "cv.filter2d": "filter2D",
    "cv.flip": "flip",
    "cv.gabor_bank": "getGaborKernel filter2D",
    "cv.gabor_response": "getGaborKernel filter2D",
    "cv.gaussian_blur": "GaussianBlur",
    "cv.graph_segments": "ximgproc.segmentation.createGraphSegmentation",
    "cv.hfs_segments": "hfs.HfsSegment.create hfs.HfsSegment.performSegmentCpu",
    "cv.bimef": "intensity_transform.BIMEF intensity_transform.BIMEF2",
    "cv.align_mapper": (
        "reg.MapperGradShift reg.MapperGradEuclid reg.MapperGradSimilar reg.MapperGradAffine "
        "reg.MapperGradProj reg.MapperPyramid.calculate reg.MapTypeCaster.toShift "
        "reg.MapTypeCaster.toAffine reg.MapTypeCaster.toProjec warpPerspective"
    ),
    "cv.gradient": "Sobel Scharr Laplacian",
    "cv.gray": "cvtColor",
    "cv.guided_filter": "ximgproc.guidedFilter",
    "cv.adaptive_manifold_filter": (
        "ximgproc.createAMFilter ximgproc.AdaptiveManifoldFilter.filter ximgproc.dtFilter"
    ),
    "cv.joint_bilateral_filter": "ximgproc.jointBilateralFilter",
    "cv.deriche_gradient": "ximgproc.GradientDericheX ximgproc.GradientDericheY",
    "cv.structured_edges": (
        "ximgproc.createStructuredEdgeDetection ximgproc.StructuredEdgeDetection.detectEdges "
        "ximgproc.StructuredEdgeDetection.computeOrientation ximgproc.StructuredEdgeDetection.edgesNms"
    ),
    "cv.anisotropic_diffusion": "ximgproc.anisotropicDiffusion",
    "cv.weighted_median_filter": "ximgproc.weightedMedianFilter",
    "cv.variance_adaptive_smoothing": "ximgproc.edgePreservingFilter",
    "cv.rolling_guidance_filter": "ximgproc.rollingGuidanceFilter ximgproc.jointBilateralFilter",
    "cv.fast_bilateral_solver": (
        "ximgproc.createFastBilateralSolverFilter ximgproc.FastBilateralSolverFilter.filter"
    ),
    "cv.fast_global_smoother": (
        "ximgproc.createFastGlobalSmootherFilter ximgproc.FastGlobalSmootherFilter.filter"
    ),
    "cv.idct": "idct",
    "cv.idft": "idft",
    "cv.image_hash": (
        "img_hash.AverageHash_create img_hash.PHash_create "
        "img_hash.BlockMeanHash_create img_hash.MarrHildrethHash_create "
        "img_hash.ColorMomentHash_create img_hash.RadialVarianceHash_create "
        "img_hash.ImgHashBase.compute"
    ),
    "inspect.compare_image_hashes": "norm",
    "inspect.compare_perceptual_descriptors": (
        "img_hash.ColorMomentHash_create img_hash.RadialVarianceHash_create "
        "img_hash.ImgHashBase.compare"
    ),
    "inspect.focus": "Laplacian Sobel meanStdDev mean",
    "cv.image_arithmetic": "add subtract multiply divide min max absdiff",
    "cv.image_pyramid": "pyrDown",
    "cv.in_range": "inRange",
    "cv.inpaint": "inpaint",
    "cv.extended_inpaint": "xphoto.inpaint",
    "cv.fuzzy_filter": "ft.filter ft.createKernel ft.createKernel1",
    "cv.fuzzy_inpaint": "ft.inpaint ft.FT02D_iteration ft.createKernel",
    "cv.bm3d_denoise": "xphoto.bm3dDenoising",
    "cv.oil_painting": "xphoto.oilPainting",
    "cv.tonemap_durand": "xphoto.createTonemapDurand xphoto.TonemapDurand",
    "cv.dct_denoise": "xphoto.dctDenoising",
    "cv.insert_channel": "insertChannel",
    "cv.mean_shift_filter": "pyrMeanShiftFiltering",
    "cv.median_blur": "medianBlur",
    "cv.merge_channels": "merge",
    "cv.merge_debevec": "createMergeDebevec MergeDebevec",
    "cv.merge_robertson": "createMergeRobertson MergeRobertson.process",
    "cv.morphology": "morphologyEx getStructuringElement",
    "cv.non_local_means": "fastNlMeansDenoising fastNlMeansDenoisingColored",
    "cv.normalize": "normalize",
    "cv.pack_remap_map": "convertMaps",
    "cv.pencil_sketch": "pencilSketch",
    "cv.pyramid_down": "pyrDown",
    "cv.pyramid_up": "pyrUp",
    "cv.distortion_fixture": "remap erode",
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
    "cv.stitch_image_pair": "warpPerspective pyrDown pyrUp detail.MultiBandBlender",
    "cv.stitch_scan_sequence": (
        "SIFT_create estimateTranslation2D estimateAffinePartial2D warpPerspective "
        "pyrDown pyrUp detail.GraphCutSeamFinder detail.MultiBandBlender"
    ),
    "cv.render_mosaic_tile": "warpPerspective pyrDown pyrUp detail.MultiBandBlender",
    "cv.stylize_photo": "stylization",
    "cv.tonemap_hdr": "createTonemap createTonemapDrago createTonemapReinhard createTonemapMantiuk",
    "cv.threshold": "threshold",
    "cv.transform_points": "perspectiveTransform",
    "cv.transpose_image": "transpose",
    "cv.unsharp": "GaussianBlur addWeighted",
    "cv.warp_affine": "warpAffine",
    "cv.peilin_normalization": "ximgproc.PeiLinNormalization moments warpAffine",
    "cv.static_saliency": (
        "saliency.StaticSaliencySpectralResidual_create saliency.StaticSaliencyFineGrained_create "
        "saliency.StaticSaliencySpectralResidual.computeSaliency "
        "saliency.StaticSaliencyFineGrained.computeSaliency"
    ),
    "cv.thin_plate_spline": (
        "createThinPlateSplineShapeTransformer ShapeTransformer.estimateTransformation "
        "ShapeTransformer.applyTransformation remap"
    ),
    "cv.rectify_quadrilateral": "getPerspectiveTransform warpPerspective",
    "cv.warp_perspective": "warpPerspective",
    "cv.warp_polar": "warpPolar",
    "calibration.rectify_workplane": "warpPerspective",
    "calibration.fit_planar": "findHomography",
    "cv.unwrap_path_strip": "remap",
    "gapi.blur": "gapi.gaussianBlur",
    "gapi.canny": "gapi.Canny",
    "gapi.gray": "gapi.BGR2Gray",
    "input.camera": (
        "VideoCapture.open VideoCapture.set VideoCapture.get "
        "VideoCapture.getBackendName VideoCapture.read"
    ),
    "input.capture_file": (
        "VideoCapture.open VideoCapture.set VideoCapture.get "
        "VideoCapture.getBackendName VideoCapture.retrieve"
    ),
    "input.depth_file": "imdecode",
    "input.file": "imdecode imdecodeWithMetadata",
    "input.tiff_file": "imdecodemulti",
    "input.tiff_page": "imdecodemulti",
    "input.video_file": "VideoCapture.open VideoCapture.read",
    "input.rgbd_recording": "imdecode",
    "3d.register_rgbd": "projectPoints",
    "cv.decode_video": "VideoCapture.open VideoCapture.read VideoCapture.release",
    "inspect.affine": "estimateAffine2D estimateAffinePartial2D",
    "inspect.akaze": "xfeatures2d.AKAZE_create",
    "inspect.align_ecc": "findTransformECCMultiScale findTransformECCWithMask",
    "inspect.approximate_contour": "approxPolyDP",
    "inspect.find_quadrilateral": "findContours approxPolyDP isContourConvex",
    "inspect.approximate_convex_contour": "approxPolyN",
    "inspect.aruco_markers": "aruco.ArucoDetector.detectMarkers",
    "calibration.create_aruco_dictionary": "aruco.extendDictionary aruco.Dictionary.getMarkerBits",
    "inspect.gmsd_quality": "quality.QualityGMSD_compute",
    "cv.learned_white_balance": (
        "xphoto.createLearningBasedWB xphoto.LearningBasedWB.setHistBinNum "
        "xphoto.LearningBasedWB.setRangeMaxVal xphoto.LearningBasedWB.setSaturationThreshold "
        "xphoto.LearningBasedWB.extractSimpleFeatures xphoto.WhiteBalancer.balanceWhite"
    ),
    "cv.percentile_white_balance": (
        "xphoto.createSimpleWB xphoto.SimpleWB.setP xphoto.SimpleWB.setInputMin "
        "xphoto.SimpleWB.setInputMax xphoto.SimpleWB.setOutputMin xphoto.SimpleWB.setOutputMax "
        "xphoto.WhiteBalancer.balanceWhite"
    ),
    "cv.gray_world_white_balance": (
        "xphoto.createGrayworldWB xphoto.GrayworldWB.setSaturationThreshold "
        "xphoto.WhiteBalancer.balanceWhite xphoto.applyChannelGains"
    ),
    "inspect.brisque_quality": "quality.QualityBRISQUE_create quality.QualityBRISQUE.compute",
    "inspect.blobs": "connectedComponentsWithStatsWithAlgorithm",
    "inspect.bounding_convex_polygon": "convexHull",
    "inspect.bounding_rotated_rectangle": "minAreaRect boxPoints",
    "inspect.brisk": "xfeatures2d.BRISK_create",
    "inspect.brief_descriptor": "xfeatures2d.BriefDescriptorExtractor_create",
    "inspect.freak_descriptor": "xfeatures2d.FREAK_create",
    "inspect.latch_descriptor": "xfeatures2d.LATCH_create",
    "inspect.beblid_descriptor": "xfeatures2d.BEBLID_create",
    "inspect.boost_descriptor": "xfeatures2d.BoostDesc.create Feature2D.compute",
    "inspect.vgg_descriptor": "xfeatures2d.VGG_create Feature2D.compute",
    "inspect.lucid_descriptor": "xfeatures2d.LUCID_create Feature2D.compute",
    "inspect.teblid_descriptor": "xfeatures2d.TEBLID_create",
    "inspect.daisy_descriptor": "xfeatures2d.DAISY_create",
    "inspect.compare_contours": "matchShapes",
    "inspect.shape_context_distance": "createShapeContextDistanceExtractor",
    "inspect.fourier_contour": "ximgproc.contourSampling ximgproc.fourierDescriptor",
    "inspect.fit_fourier_contours": (
        "ximgproc.contourSampling ximgproc.fourierDescriptor ximgproc.createContourFitting "
        "ximgproc.ContourFitting.estimateTransformation ximgproc.transformFD"
    ),
    "inspect.compare_edge_maps": "distanceTransform",
    "inspect.mask_clearance": "distanceTransform",
    "inspect.compare_histograms": "compareHist",
    "inspect.components": "connectedComponentsWithStatsWithAlgorithm",
    "inspect.contour_hierarchy": "findContours",
    "inspect.contour_hull": "convexHull convexityDefects",
    "inspect.contour_points": "pointPolygonTest",
    "inspect.contours": "findContours contourArea moments",
    "inspect.run_contours": "findContoursLinkRuns contourArea moments",
    "inspect.corner_response": "cornerHarris cornerMinEigenVal",
    "inspect.corners": "goodFeaturesToTrack",
    "inspect.affine_features": "AffineFeature_create SIFT_create ORB_create",
    "inspect.structure_tensor": "cornerEigenValsAndVecs",
    "inspect.intensity_moments": "moments",
    "inspect.region_contrast": "meanStdDev mean findContours",
    "inspect.fast": "FastFeatureDetector_create",
    "inspect.flood_fill": "floodFill",
    "inspect.grabcut": "grabCut",
    "inspect.histogram_backproject": "calcBackProject",
    "inspect.hog": "HOGDescriptor HOGDescriptor.compute",
    "inspect.homography": "findHomography",
    "inspect.hough_circles": "HoughCircles HoughCirclesWithAccumulator",
    "inspect.find_ellipses": "ximgproc.findEllipses",
    "inspect.tbmr": "xfeatures2d.TBMR_create xfeatures2d.TBMR.detect",
    "inspect.hough_lines": "HoughLinesP",
    "inspect.hough_point_lines": "HoughLinesPointSet",
    "inspect.hough_polar_lines": "HoughLinesWithAccumulator",
    "inspect.fast_hough_lines": "ximgproc.FastHoughTransform ximgproc.HoughPoint2Line",
    "inspect.edge_boxes": "ximgproc.createEdgeBoxes ximgproc.EdgeBoxes.getBoundingBoxes",
    "inspect.filter_matches_gms": "xfeatures2d.matchGMS",
    "inspect.harris_laplace": (
        "xfeatures2d.HarrisLaplaceFeatureDetector_create xfeatures2d.HarrisLaplaceFeatureDetector.detect"
    ),
    "inspect.line_descriptors": (
        "line_descriptor.BinaryDescriptor.createBinaryDescriptor "
        "line_descriptor.BinaryDescriptor.detect line_descriptor.BinaryDescriptor.compute"
    ),
    "inspect.match_line_descriptors": (
        "line_descriptor.BinaryDescriptorMatcher.match line_descriptor.BinaryDescriptorMatcher.knnMatch"
    ),
    "inspect.kaze": "xfeatures2d.KAZE_create",
    "inspect.kmeans_image": "kmeans",
    "inspect.line_segments": "createLineSegmentDetector LineSegmentDetector.detect",
    "inspect.fast_line_segments": "ximgproc.createFastLineDetector ximgproc.FastLineDetector.detect",
    "inspect.edge_drawing": (
        "ximgproc.createEdgeDrawing ximgproc.EdgeDrawing.detectEdges "
        "ximgproc.EdgeDrawing.getSegments ximgproc.EdgeDrawing.getSegmentIndicesOfLines "
        "ximgproc.EdgeDrawing.getEdgeImage ximgproc.EdgeDrawing.getGradientImage "
        "ximgproc.EdgeDrawing.detectLines ximgproc.EdgeDrawing.detectEllipses"
    ),
    "inspect.match_descriptor_radius": "DescriptorMatcher.radiusMatch",
    "inspect.match_template": "matchTemplate",
    "inspect.match_color_template": "ximgproc.colorMatchTemplate",
    "inspect.radon_projection": "ximgproc.RadonTransform",
    "inspect.mser": "MSER_create MSER.detectRegions",
    "cv.text_extremal_regions": (
        "text.createERFilterNM1 text.createERFilterNM2 text.detectRegions text.erGrouping "
        "text.computeNMChannels"
    ),
    "inspect.orb": "ORB_create Feature2D.detectAndCompute",
    "inspect.phase_correlation": "phaseCorrelate",
    "inspect.read_barcode": "barcode.BarcodeDetector.detectAndDecodeWithType",
    "inspect.read_qr": (
        "QRCodeDetector QRCodeDetectorAruco GraphicalCodeDetector.detectAndDecode "
        "GraphicalCodeDetector.detectAndDecodeMulti QRCodeDetector.detectAndDecodeCurved"
    ),
    "inspect.read_wechat_qr": (
        "wechat_qrcode.WeChatQRCode wechat_qrcode.WeChatQRCode.detectAndDecode "
        "wechat_qrcode.WeChatQRCode.setScaleFactor"
    ),
    "inspect.refine_corners": "cornerSubPix",
    "inspect.sift": "SIFT_create Feature2D.detectAndCompute",
    "inspect.selective_search": "ximgproc.segmentation.createSelectiveSearchSegmentation",
    "inspect.simple_blob_detector": "SimpleBlobDetector.create Feature2D.detect",
    "inspect.thin_mask": "ximgproc.thinning",
    "inspect.watershed": "watershed",
    "ml.decode_detections": "dnn.NMSBoxes",
    "ml.group_detection_votes": "groupRectangles",
    "ml.soft_nms": "dnn.softNMSBoxes dnn.NMSBoxes",
    "ml.fuse_tile_detections": "dnn.NMSBoxes",
    "ml.box_appearance": "calcHist cvtColor",
    "ml.efficientsam": "dnn.readNetFromONNX resize cvtColor",
    "ml.decode_prompt_masks": "resize findContours",
    "ml.decode_rotated_detections": "dnn.NMSBoxesRotated rotatedRectangleIntersection boxPoints",
    "ml.decode_segmentation": "remap findContours",
    "ml.decode_instances": "dnn.NMSBoxes remap findContours",
    "ml.decode_landmarks": "minMaxLoc",
    "ml.dnn_batch": (
        "dnn.readNetFromONNX dnn.blobFromImages dnn.Net.setInput dnn.Net.forward "
        "dnn.getAvailableTargets"
    ),
    "ml.dnn_multi": "dnn.readNetFromONNX dnn.Net.setInput dnn.Net.forward dnn.getAvailableTargets",
    "ml.image_tensor": "resize",
    "ml.dnn": "dnn.readNetFromONNX dnn.Net.setInput dnn.Net.forward dnn.getAvailableTargets",
    "ml.dnn_profile": (
        "dnn.readNetFromONNX dnn.Net.forward dnn.Net.getPerfProfile "
        "dnn.Net.getLayerNames dnn.Net.getLayerId getTickFrequency"
    ),
    "ml.dnn_multi_profile": (
        "dnn.Net.setInput dnn.Net.forward dnn.Net.getPerfProfile dnn.Net.getLayerNames"
    ),
    "ml.dnn_batch_profile": (
        "dnn.blobFromImages dnn.Net.forward dnn.Net.getPerfProfile dnn.Net.getLayerNames"
    ),
    "ml.kmeans": "kmeans",
    "output.image": "imencode imwriteWithMetadata",
    "output.tiff_file": "imencodemulti",
    "output.video": "VideoWriter.open VideoWriter.write",
    "sequence.deepflow": "optflow.createOptFlow_DeepFlow DenseOpticalFlow.calc",
    "sequence.pcaflow": "optflow.createOptFlow_PCAFlow DenseOpticalFlow.calc",
    "sequence.simpleflow": "optflow.createOptFlow_SimpleFlow optflow.calcOpticalFlowSF",
    "cv.edge_aware_flow": "ximgproc.createEdgeAwareInterpolator EdgeAwareInterpolator.interpolate",
    "cv.ric_flow": "ximgproc.createRICInterpolator ximgproc.RICInterpolator.interpolate",
    "sequence.sparse_to_dense": "optflow.calcOpticalFlowSparseToDense calcOpticalFlowPyrLK",
    "sequence.dense_rlof": "optflow.calcOpticalFlowDenseRLOF optflow.RLOFOpticalFlowParameter",
    "cv.sparse_rlof": "optflow.calcOpticalFlowSparseRLOF optflow.RLOFOpticalFlowParameter",
    "sequence.dis": "DISOpticalFlow.create DenseOpticalFlow.calc",
    "sequence.dual_tvl1": "optflow.createOptFlow_DualTVL1 optflow.DualTVL1OpticalFlow_create DenseOpticalFlow.calc",
    "sequence.foreground_debounce": "connectedComponentsWithStats findContours moments",
    "sequence.farneback": "calcOpticalFlowFarneback",
    "sequence.mog_background": "bgsegm.createBackgroundSubtractorMOG BackgroundSubtractor.apply",
    "input.synthetic_background_sequence": "bgsegm.createSyntheticSequenceGenerator bgsegm.SyntheticSequenceGenerator.getNextFrame",
    "sequence.gmg_background": "bgsegm.createBackgroundSubtractorGMG BackgroundSubtractor.apply",
    "sequence.cnt_background": "bgsegm.createBackgroundSubtractorCNT BackgroundSubtractor.apply",
    "sequence.gsoc_background": "bgsegm.createBackgroundSubtractorGSOC BackgroundSubtractor.apply",
    "sequence.lsbp_background": "bgsegm.createBackgroundSubtractorLSBP BackgroundSubtractor.apply",
    "sequence.knn_background": "createBackgroundSubtractorKNN BackgroundSubtractor.apply",
    "sequence.lk_grid_flow": "calcOpticalFlowPyrLK remap",
    "sequence.mog2": "createBackgroundSubtractorMOG2 BackgroundSubtractor.apply",
    "sequence.motion_history": (
        "motempl.updateMotionHistory motempl.calcMotionGradient "
        "motempl.calcGlobalOrientation motempl.segmentMotion"
    ),
    "sequence.retina": (
        "bioinspired.Retina_create Retina.setupOPLandIPLParvoChannel "
        "Retina.setupIPLMagnoChannel Retina.getParvoRAW Retina.getMagnoRAW"
    ),
    "sequence.motion_saliency": (
        "saliency.MotionSaliencyBinWangApr2014_create "
        "saliency.MotionSaliencyBinWangApr2014.computeSaliency "
        "bioinspired.Retina.getMagnoRAW bioinspired.TransientAreasSegmentationModule.run"
    ),
    "sequence.track_points": "calcOpticalFlowPyrLK",
    "sequence.track_mil": "TrackerMIL.create TrackerMIL.init TrackerMIL.update",
    "sequence.pixel_moments": "accumulate accumulateSquare accumulateProduct accumulateWeighted",
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
