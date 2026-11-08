# SimplePathWidth · Curved bead inspection

The [project](project.json) measures a bright bead along a curved reference
path. The example uses generated grayscale images. The clean reference
passes. The second image has a gap and a section that is too narrow.

## Setup

Install a local SimplePathWidth checkout or wheel into the Python environment
running DataflowGUI. From the DataflowGUI root, replace the sample path with
your package location. This reuses the host's OpenCV installation.

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps "C:\path\to\SimplePathWidth"
```

## Inspect and train

1. Open `project.json`. Run once for the clean frame and again for the frame
   with the gap. The two images also run in sequence with continuous replay.
2. On SimplePathWidth, select **Result / overlay** for the expected path,
   paired edges, and missing or out-of-tolerance stations. The overlay also
   appears in the **Image results filmstrip** when **Capture result thumbnails**
   is checked. Select **Result / trend** for width and signed offset versus
   distance. The `measurements` port includes coordinates and status per station.
3. Select **Train path width…** to reopen the stored reference and path.
   Drag handles to refine the path, Shift-click to insert a point, or use
   Delete to remove one. **Preview on reference** shows both plots and the
   overlay. Click a plot to locate that station on the image.
4. **Stage recipe** writes a new recipe under the project. **Apply** commits
   that revision to the block. Save the graph to keep it on reopening.

The supplied `assets/trace-mask.png` can be used with **Trace mask…** to
regenerate an editable path from the reference. Mask tracing requires
`scikit-image` and `scipy`. DXF import requires `ezdxf`.
