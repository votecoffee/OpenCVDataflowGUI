# Simple Plugins examples

Open a tool folder below. Its runnable example folders are directly inside it,
with a short README linking the project JSON files. No manual plugin-loading step
is needed in the current app.

| Tool | Example folders |
| --- | --- |
| [Simple3DLocate](3d_locate/README.md) | `cad_surface`, `generated_surface`, `tless_depth_verified`, `tless_match`, `tless_no_match` |
| [Simple3DMeasure](3d_measure/README.md) | `generated_good`, `inspection`, `missing_line_target`, `tless_flatness`, `tless_line`, `tless_plane`, `training` |
| [SimpleBinPicking](bin_picking/README.md) | `bin_to_grid`, `inspection`, `jaw_clear`, `jaw_obstacle`, `suction_generated`, `suction_tless`, `tless_mesh_grid`, `training` |
| [SimpleDeflectometry](deflectometry/README.md) | `inspection` |
| [SimpleFoundationPose](foundation_pose/README.md) | `held_out_frame13`, `wrong_object` |
| [SimpleFringeScan](fringe_scan/README.md) | `inspection` |
| [SimpleGDRNPP](gdrnpp/README.md) | `accepted_parts`, `live_review`, `rejected_part`, `setup` |
| [SimpleClassify](image_classification/README.md) | `vegqual` |
| [SimpleLaserProfile](laser_profile/README.md) | `inspection`, `training` |
| [SimplePathWidth](path_width/README.md) | `calibrated_bead`, `calibrated_curve`, `fixture_capture`, `generated_bead`, `weld_bead` |
| [SimplePatternMatch](pattern_matching/README.md) | `asymmetric_brackets`, `visa_pcb2` |
| [SimpleSegmentation](pixel_segmentation/README.md) | `armbench_tote`, `brick_mortar`, `cup_rim`, `pcb_laminate` |
| [SimpleReferenceInspect](reference_inspect/README.md) | `armbench_tote`, `fixture_capture`, `generated_inspection`, `visa_pcb2` |
| [SimpleSurfaceEnhance](surface_enhance/README.md) | `inspection`, `training` |

Each tool's **dev** folder contains builders, old-version plugins/releases, benchmark
studies, frozen reviews and verification bundles. Shared development material is
in [dev](dev/README.md). The coding API's example search omits these development
projects by default, while they remain loadable through File → Open.

Neural inference and live simulation require their configured external workers,
weights and services. Saved file-only reviews are identified in the tool READMEs.
Attribution, source data, recipes and qualification limits remain with their examples.
