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
| Pixel formats and high-depth images | [Image formats](image-formats.md) |
| Ordered multipage buffers, typed page metadata and decode limits | [Multipage TIFF in memory](multipage-memory.md) |
| Random TIFF pages, bounded LRU caching and snapshot refresh | [On-demand TIFF pages](tiff-page-access.md) |
| Reader/encoder availability and native image page inventory | [Codec preflight](codec-preflight.md) |
| Stored optical-flow fields, native I/O and verified image-pair association | [Flow files](flow-files.md) |
| Detection association, oriented prediction, uncertainty and expiry | [Image tracking](image-tracking.md) |
| Periodic full-frame detector scheduling, local motion and freshness | [Detector refresh tracking](refresh-tracking.md) |
| Common-pitch measured profiles, physical scale and antialiasing | [Profile resampling](profile-resampling.md) |
| Seeded MIL tracking, loss/reseed and native confidence limits | [MIL tracking](mil-tracking.md) |
| Precomputed optical-flow pyramids and Python consumer qualification | [Pyramid binding audit](optical-flow-pyramid-audit.md) |
| Encoded video bytes, bounded FFmpeg callbacks and per-frame groups | [Video buffers](video-buffer.md) |
| Video clock reliability, valid variable rate and independent timing references | [Video timing monitor](video-timing.md) |
| Requested-time frame sampling, bracket selection and native seek diagnostics | [Video at time](video-time-seeking.md) |
| Live network video, decoder ownership, deadlines and reconnect | [Network video](network-video.md) |
| Timestamp-paced files, playback policy, loop/reset and Stop/Resume | [Paced video replay](paced-video.md) |
| Binary corners, endpoints and junctions with signed kernels | [Hit-or-miss morphology](hit-or-miss.md) |
| Sparse-mask contours with bounded run input and flat native polygons | [Run-linked contours](run-contours.md) |
| Template appearance support before pick clearance/ranking | [Visible support](visible-support.md) |
| Whole selections through nonlinear lens correction | [Selection remap](selection-remap.md) |
| Regions, poses, pyramids and spatial provenance | [Workflow guide](workflow-guide.md), [architecture](architecture.md) |
| Tickets, branching, batching and joins | [Control flow](control-flow.md), [input coordination](input-coordination.md) |
| Acquisition queues and memory | [Input queues](input-queues.md), [sequence memory](sequence-memory.md) |
| Per-pixel temporal noise, weighted support and paired covariance | [Temporal pixel statistics](temporal-pixel-statistics.md) |
| QR backend selection, finder controls and module/camera result grids | [QR readers](qr-backends.md), [barcode reading](barcode-reading.md) |
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
| PLC, robot and messaging interfaces | [Industrial integration](industrial-integration.md), [ROS 2 Lyrical / Fast DDS](ros2.md) |
| scikit-image tools | [scikit-image](scikit-image.md) |
| Medical images, physical geometry and SimpleITK | [SimpleITK](simpleitk.md), [medical roadmap](simpleitk-roadmap.md), [labeled MRI examples](<../examples/SimpleITK Tools/README.md>) |

For narrower tools, search file names under `docs/` and the machine-readable block
catalog. The catalog reflects the running environment. Optional providers are
listed without loading them; resolving a particular type can require its package.
