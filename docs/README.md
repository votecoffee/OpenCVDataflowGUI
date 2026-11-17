# Documentation map

## Plan, build and review an application

| Task | Start here |
| --- | --- |
| Install and make a first graph | [README](../README.md), [workflow guide](workflow-guide.md) |
| Work with a coding chat | [Assistant development](assistant-development.md), [application skill](../skills/opencv-dataflow-apps/SKILL.md) |
| Discover blocks and automate saved projects | [JSON/Python API](../skills/opencv-dataflow-apps/references/api.md) |
| Define requirements, inputs and acceptance checks | [Application plan template](application-plan-template.md) |
| Find runnable projects and image rights | [Example catalog](../examples/README.md) |
| Check implemented tools and remaining work | [Coverage](opencv-coverage.md), [checklist](vision-coverage-checklist.md), [roadmap](roadmap.md) |
| Look up controls and native bindings | [Block parameters](block-parameters.md), [binding inventory](opencv-bindings.tsv) |

## Runtime and extension contracts

| Topic | Reference |
| --- | --- |
| Blocks, persistence, errors, conversions and scheduling | [Architecture](architecture.md) |
| Program startup/run/shutdown scripts and owned ROS/external services | [Program lifecycle](program-lifecycle.md) |
| Reusable subgraphs and interfaces | [ToolBlocks](toolblocks.md) |
| Custom Python blocks | [Plugin skill](../skills/opencv-dataflow-plugins/SKILL.md), [plugin contract](../skills/opencv-dataflow-plugins/references/plugin-contract.md) |
| Dialogs, captures and worker training | [Plugin dialogs](plugin-dialogs.md) |
| HDR brackets, rejected motion and measured-versus-review support | [HDR motion](hdr-motion.md) |
| Unordered rotating views, explicit camera refinement, reserved validation and source-point mapping | [Rotating panoramas](rotating-panorama.md) |
| Neighboring-frame border fills, review images and unchanged measured support | [Stabilization border fills](stabilization-fill.md) |
| Conditional single-image motion PSF estimates and qualified Wiener restoration | [Motion blur estimation](blur-estimation.md) |
| Pixel formats and high-depth images | [Image formats](image-formats.md) |
| Ordered multipage buffers, typed page metadata and decode limits | [Multipage TIFF in memory](multipage-memory.md) |
| Random TIFF pages, bounded LRU caching and snapshot refresh | [On-demand TIFF pages](tiff-page-access.md) |
| Reader/encoder availability and native image page inventory | [Codec preflight](codec-preflight.md) |
| Stored optical-flow fields, native I/O and verified image-pair association | [Flow files](flow-files.md) |
| Detection association, appearance, prediction, expiry and identity metrics | [Image tracking](image-tracking.md) |
| Periodic full-frame detector scheduling, local motion and freshness | [Detector refresh tracking](refresh-tracking.md) |
| Common-pitch measured profiles, physical scale and antialiasing | [Profile resampling](profile-resampling.md) |
| Seeded MIL tracking, loss/reseed and native confidence limits | [MIL tracking](mil-tracking.md) |
| Precomputed optical-flow pyramids and Python consumer qualification | [Pyramid binding audit](optical-flow-pyramid-audit.md) |
| Encoded video bytes, bounded FFmpeg callbacks and per-frame groups | [Video buffers](video-buffer.md) |
| Video clock reliability, valid variable rate and independent timing references | [Video timing monitor](video-timing.md) |
| Requested-time frame sampling, bracket selection and native seek diagnostics | [Video at time](video-time-seeking.md) |
| Live network video, decoder ownership, deadlines and reconnect | [Network video](network-video.md) |
| Timestamp-paced files, playback policy, loop/reset and Stop/Resume | [Paced video replay](paced-video.md) |
| Requested camera settings, backend readback, decoded geometry and recorded qualification | [Capture negotiation](capture-negotiation.md) |
| Binary corners, endpoints and junctions with signed kernels | [Hit-or-miss morphology](hit-or-miss.md) |
| Sparse-mask contours with bounded run input and flat native polygons | [Run-linked contours](run-contours.md) |
| Template appearance support before pick clearance/ranking | [Visible support](visible-support.md) |
| Whole selections through nonlinear lens correction | [Selection remap](selection-remap.md) |
| Regions, poses, pyramids and spatial provenance | [Workflow guide](workflow-guide.md), [architecture](architecture.md) |
| Tickets, branching, batching and joins | [Control flow](control-flow.md), [input coordination](input-coordination.md) |
| Acquisition queues and memory | [Input queues](input-queues.md), [sequence memory](sequence-memory.md) |
| Per-pixel temporal noise, weighted support and paired covariance | [Temporal pixel statistics](temporal-pixel-statistics.md) |
| QR backend selection, finder controls and module/camera result grids | [QR readers](qr-backends.md), [barcode reading](barcode-reading.md) |
| Cylindrical QR labels, approximate source outlines and independent sampled modules | [Curved QR](curved-qr.md) |
| Multipart QR recovery with header/order/count/parity checks | [Structured-append QR](qr-structured-append.md) |
| Detector repeatability and thresholded descriptor precision/recall | [Image feature evaluation](feature-evaluation.md) |
| Reference-based frequency gain/phase, excited bands and radial profiles | [Relative frequency response](frequency-response.md) |
| Recipes and sealed trained assets | [Recipes](recipes.md) |
| Fixed-camera lighting, position and focus monitoring | [Camera health and reference training](camera-health.md) |
| Multiview composition, native/floating blending and bounded tiles | [Multiband mosaics](multiband-mosaics.md) |
| Stable source IDs, missing views and sealed calibrated layouts | [Mosaic layout reuse](mosaic-layout-reuse.md) |
| Independent registered frames, union/common crops and source timestamps | [Registered-view replay](registered-view-replay.md) |
| Content-bound motion capture, JSON assets and no-fit clip replay | [Motion trajectories](motion-trajectory.md) |
| Blur-model checked detail reuse, change gates and target coverage | [Neighbor restoration](neighbor-restoration.md) |
| Validated row motion, rolling-shutter correction and nonlinear maps | [Rolling-shutter baseline](rolling-shutter.md) |
| Sealed clean backgrounds, first-image restoration and protected learning | [Background references](background-reference.md) |
| ROS, MQTT, PLC and robot connection status | [Connection health monitors](connection-health.md) |
| Judgments, reasons and ticket histories | [Ticket results](ticket-results.md) |
| 3D/depth/point clouds | [3D vision](vision-3d.md) |
| ML/DNN and optional dependencies | [ML](ml.md) |
| Explicit external SuperPoint tensors, learned matching and extraction limits | [Local features](local-features.md) |
| Learned DISK extraction, native controls and matching local weights | [DISK features](disk-features.md) |
| ALIKED presets, subpixel features, fallback/score policies and matching | [ALIKED features](aliked-features.md) |
| Saved image color palettes, descriptor vocabularies and unknown-sample inspection | [Reusable clusters](reusable-clusters.md) |
| Native neighbor classes, explicit votes/distances and unknown-image rejection | [K-nearest classification](knn-classification.md) |
| Native forest training, class votes, importance and portable checked trees | [Random forest classification](rtrees-classification.md) |
| Labeled image sets, grouped splits, confusion matrices and classifier reports | [Classifier evaluation](classifier-evaluation.md) |
| Labeled boxes, class-aware IoU matches, precision/recall/AP and detector reports | [Detector evaluation](detector-evaluation.md) |
| Explicit ONNX engine dispatch, immutable snapshots and import evidence | [ONNX engine loading](onnx-engine-loading.md) |
| Object detection heads, class-aware NMS and source box geometry | [DNN detection](object-detection.md) |
| Shared-ticket tile boxes, exact source mapping and duplicate suppression | [Tile detection fusion](tile-detection-fusion.md) |
| Oriented heads, rotated overlap and exact affine source poses | [Rotated detection](rotated-detection.md) |
| Shared-ticket segmentation tiles, overlap weights, valid coverage and source-grid decisions | [Tiled segmentation](tiled-segmentation.md) |
| Pinned PP-OCRv3, observable maps, DB-style rectangles and source-grid text regions | [Text detection](text-detection.md) |
| Per-pixel class scores/IDs, source-grid masks and confidence/area decisions | [Semantic segmentation](semantic-segmentation.md) |
| Per-object masks, prototype/ROI heads, overlap and individual inspection | [Instance segmentation](instance-segmentation.md) |
| Image/point/box prompts, native mask candidates and source-grid inspection | [Prompted segmentation](prompted-segmentation.md) |
| Ordered heatmap/direct landmarks, subpixel fitting and local inspection poses | [Landmarks](landmarks.md) |
| Independent image/tensor preparation, named model inputs and ticket roles | [Multi-input inference](multi-input-inference.md) |
| Image batches, dynamic sizes, tails and ordered per-image inference results | [Batch inference](batch-inference.md) |
| Pinned ONNX model/settings bundles and execution/fallback evidence | [Model contracts](model-contracts.md) |
| PLC, robot and messaging interfaces | [Industrial integration](industrial-integration.md), [ROS 2 Lyrical / Fast DDS](ros2.md) |
| scikit-image tools | [scikit-image](scikit-image.md) |
| Medical images, physical geometry and SimpleITK | [SimpleITK](simpleitk.md), [medical roadmap](simpleitk-roadmap.md), [labeled MRI examples](<../examples/SimpleITK Tools/README.md>) |

For narrower tools, search file names under `docs/` and the machine-readable block
catalog. The catalog reflects the running environment. Optional providers are
listed without loading them; resolving a particular type can require its package.
