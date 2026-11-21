# Import a Bundler sparse reconstruction

Open the [example project](bundler_reconstruction.json) and run once. The
importer reads five camera slots and 42 colored points from
[bundle.out](assets/bundle.out). It preserves the unregistered slot and
uses a measured **0.4 m** baseline between cameras 0 and 1 to restore
metric scale. The projection block uses camera 4, which has no tracks in
the imported observations. All **42 points** land in the expected
held-out pixels with depths and colors that match the
[synthetic reference](assets/truth.json).

For an existing scene, select a Bundler **v0.3** file and choose the
camera index to expose directly. Enter a common image size as
[width, height] or supply one size or null per original camera slot.
Bundler v0.3 carries centered observations. An image size lets the block
emit the corresponding top-left pixel coordinates and a camera model
for downstream use. Supply a measured camera baseline when metric
geometry is available. Unitless mode preserves the bundle's original scale.

The importer reports fit error for the original observations and retains
the point and feature indices for the selected camera. A positive
reprojection limit marks over-limit observations as a failed import
judgement and preserves the diagnostic report. The format check accepts
the Bundler v0.3 header and rejects unsupported versions and formats.

The scene and held-out reference are original synthetic fixtures by
OpenCVDataflowGUI contributors under the project's MIT license.
Run the example builder with an empty output directory to regenerate them.
[Bundler file format and camera conventions](https://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html#S6)
