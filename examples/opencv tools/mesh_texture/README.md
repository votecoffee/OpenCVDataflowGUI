# Bake a registered camera image onto a mesh

Open [mesh_texture.json](mesh_texture.json) and run the graph. The camera
image shows a slanted foreground panel over a larger rear panel. The two
surfaces have separate islands in a 128×64 UV atlas. The rear panel is
on the left and the slanted panel is on the right.

Inspect **Texture** and **Texture mask** in each bake. The rear island has
a gap where the nearer panel blocks the camera and a strip where the
surface extends past the image. The foreground island has distinct colors.
The solid parts of the two islands cover 6,720 texels. Nearest sampling
observes 5,835, and bilinear sampling observes 5,490 because every
contributing pixel must be visible and selected.

The bake uses the existing UV coordinates and preserves every mesh
attribute. UV pixel centers are `u = (x + 0.5) / width` and
`v = 1 - (y + 0.5) / height`. The texture retains the source encoding
and pixel depth. Use the mask to distinguish unsupported texels from valid
black. Save the texture and mask as images alongside the mesh when exporting.

The source needs a registered rectified pinhole image and a mesh with
non-overlapping normalized UVs. Visibility comes from the mesh at
camera pixel centers. Use a source mask for camera occluders omitted
from the mesh. The bake checks UV overlap at the selected atlas resolution.
[OpenCV triangle rasterization reference](https://docs.opencv.org/5.0/main_modules/ptcloud.html#trianglerasterizedepth)

The images, geometry, and [builder](build_example.py) are original
OpenCVDataflowGUI contributions under the repository's MIT license.
