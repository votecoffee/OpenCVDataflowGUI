# Binary mask boundary and area inspection

Open `project.json` and run once. The 16-bit reference bracket has two holes.
Its sample adds a 4 × 4 pixel tab. **Compare mask boundaries and area** accepts
the tab with a four-pixel boundary tolerance and a 16-pixel extra-area allowance.

Open `defect.json` for the deliberate failure: the same tab plus a 7 × 5 pixel
notch. It removes 35 expected pixels and moves the notch's inner boundary five
pixels. Both the boundary and missing-area judgments fail. `assets/truth.json`
records independent rectangle dimensions and expected results.

| Measurement | Accepted project | Defect project |
| --- | ---: | ---: |
| Maximum bidirectional boundary distance | 4 px | 5 px |
| Missing pixels | 0 | 35 |
| Extra pixels | 16 | 16 |
| Judgment | Pass | Fail |

## Using the block

`inspect.mask_boundary` takes equal-size uint8/uint16 monochrome **Reference**
and **Sample** masks. Nonzero pixels are foreground by default; select **Zero**
to invert that convention. No registration is performed. Matching known frames
are required by default. These independent saved fixtures explicitly select
**Pixel grid**, which compares already aligned pixels by index.

The sample owns output coordinates, ticket, pyramid context and companion data.
A static reference may have a different ticket; **Require match** checks it too.
A connected display image and measurement selection must share the sample's
known ticket. The optional display image keeps its original pixels, encoding,
depth and alpha on **Image**; otherwise Image is the unchanged sample mask.

The block measures inner boundary pixels from a 4- or 8-neighbor erosion,
including holes and disconnected components. Choose precise Euclidean,
Manhattan or chessboard distance. Optional 3/5 Euclidean chamfer masks trade
accuracy for the native approximate method. **Background** assumes exterior
pixels are background, making the image edge a possible boundary. **Ignore**
treats the exterior as unknown; a full mask then has no measurable boundary.

**Both** combines the sample-to-reference and reference-to-sample pixel sets.
Mean, RMS and percentile are weighted by boundary pixel count. A one-way check
can intentionally overlook features visible only in the other direction; both
directed reports remain available. Choose maximum, mean, RMS or a configurable
percentile for the boundary judgment. All distances are pixel-center distances
in the current image grid, including a selected pyramid level. They are not
continuous-curve distances, physical measurements or largest-level values.

An optional mask and region intersect to select measurement centers and area
counts. Neighborhoods and distance targets retain the full masks. The selection
creates no artificial cut boundary. A selected direction with no queries or no
target boundary withholds its distance judgment. Empty selected foreground fails
by default; **Both empty pass** explicitly permits the absence of both masks.
It still withholds undefined distances, rather than emitting zero or infinity.

Boundary tolerance is inclusive with a small float32 roundoff allowance:
`4 * float32_epsilon * max(1, tolerance)` for tolerances below one million
pixels. The same allowance applies to boundary tests, tolerance bands and
beyond-tolerance area. It prevents an exactly specified diagonal distance from
failing solely because the native float32 distance rounded upward.

## Results and graphics

Expose these additional outputs in the block editor as needed:

- **Measurements**: selected areas, intersection/union and IoU, missing/reference
  and extra/sample fractions, both directed reports, combined statistics and validity.
- **Sample distance / Reference distance**: float32 distances at selected query
  boundaries, zero elsewhere. Their companion boundary masks identify valid samples.
  A missing target withholds the corresponding map.
- **Missing / Extra**: selected raw area differences.
- **Missing / Extra beyond tolerance**: differences farther than tolerance from
  the opposite boundary. A wholly missing opposite foreground marks all differences.
- **Tolerance band**: selected pixels within tolerance of the reference boundary.

Area checking can use raw differences or exclude the permitted boundary band.
Independent inclusive limits produce missing/extra failure reasons. IoU is also
reported as the block's score. Missing and extra colors indicate differences,
including permitted differences, rather than replacing the pass/fail judgment.

The preview colors reference boundary cyan, sample boundary green, missing area
red and extra area magenta. Drawable result layers provide full reference/sample
outlines and selected differences. Their vertex budget limits display geometry;
numeric measurements and masks retain all pixels. Disabling graphics skips the
preview and outline construction. The comparison is limited to 4,194,304 pixels
and 16384 pixels per axis.

## Attribution and regeneration

All fixtures, projects and code are original work under this repository's MIT
license. Reference values follow direct half-open rectangle assignments; no
distance-transform implementation generates the expected values.

```powershell
.\.venv\Scripts\python.exe "examples/opencv tools/mask_boundary/generate.py"
```

Algorithm reference: [OpenCV distance transform documentation](https://docs.opencv.org/5.0/main_modules/imgproc_misc.html).
