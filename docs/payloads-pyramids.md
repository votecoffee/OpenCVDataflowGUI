# Port payloads, channels and pyramids

Ports have a primary value and an optional JSON object payload. Primary values
remain ordinary images, numbers and structured data for existing Python blocks.
Payloads travel with graph connections and nested ToolBlocks. Runtime snapshots
copy pixel data and payloads independently between branches.

**Attach port payload** merges a JSON object from its inspector or optional payload
input into a value's existing payload. **Read port payload / coordinates** passes
the primary value through, exposes the payload as data, and supplies a coordinate
transform when spatial information exists. For example, attach a `recipe` object
to an image without adding recipe fields to its pixels or to every block's parameters.
Unknown fields are retained as application data; their meaning is not inferred.

Payloads appear in the inspector's result details and in headless JSON reports.
They are latest-run data, not histories stored in project files. Attach-block
configuration and image-input level selections are saved in projects. Writers
retain their existing primary-data format; connect Read payload's `payload` output
to a data writer to explicitly export the metadata.

## Channel tools

| Tool | Behavior |
| --- | --- |
| Split three channels | Three monochrome outputs in the source encoding's order |
| Merge three channels | Three matching planes; BGR/RGB/HSV/Lab output dropdown |
| Split four channels (alpha) | Three color planes plus alpha |
| Merge four channels (alpha) | Matching color/alpha planes; BGRA/RGBA dropdown |
| Extract image channel | One plane selected by index 0..3; monochrome supports 0 |
| Replace image channel | Replace one plane from a matching monochrome image |

There is no implicit reorder: BGR is B, G, R; RGB is R, G, B; alpha is the fourth
plane. Split outputs carry source encoding and index in their `channel` payload.
Merged outputs drop the single-channel tag. Dimensions, depth and known coordinate
frames must match. Channel operations support uint8, uint16, int16, float32 and
float64; HSV/Lab merging requires uint8 or float32. A depth plane remains a separate
typed physical image; it is not silently treated as an alpha/color component.

## Pyramid tools and level selection

**Pyramid down** and **Pyramid up** perform Gaussian sampling using native OpenCV.
The older combined up/down block remains compatible. They expose levels, applicable
borders and final output dimensions. Geometry and payloads record the true factor
of two per step, including odd dimensions; width ratios are not used as substitutes.
PyrUp interpolates and cannot recover details lost by PyrDown.

**Gaussian image pyramid set** creates 1..9 images, counting level 0 as the largest
source image. Each subsequent image is native `pyrDown` of its predecessor, including
odd-sized dimensions. The source is limited to 64 million pixels, and requests for
levels beyond a 1×1 image report an error. Supported borders are dropdowns.

Its `images` output is one typed `ImagePyramid` value containing all levels. Its
`image` output is level 0; `info` lists every level's size and mapping to level 0.
The node thumbnail and bundle preview display the largest image. A set can branch,
pass through payload tools, and enter nested ToolBlocks without losing its levels.

An ordinary `image` input accepts either an image or a pyramid. Its inspector
**Pyramid level: name** dropdown defaults to `largest`, equivalent to level 0.
Choose 0..8 to select another level; a missing level is an execution error, not a
silent fallback. The choice applies only to bundled inputs, not an already selected
single image. **Select pyramid level** is an explicit selector with a single-image
output. ToolBlocks preserve sets at public image inputs so internal consumers can
make independent selections.

## Coordinate provenance and results

Image payloads use the reserved `spatial` object:

```json
{
  "spatial": {
    "root_id": "source-block-id:image",
    "root_size": [480, 360],
    "image_size": [120, 90],
    "to_root": [[4, 0, 0], [0, 4, 0], [0, 0, 1]]
  }
}
```

`to_root` maps current pixel centers to the original source image's coordinates.
Crop, resize, rotation and warp geometry compose with it automatically. A plain
array supplied directly to a block has an unknown (`null`) root identity unless
the caller supplies one. Source block outputs have stable identities within a job.
These are logical image frames, not automatically calibrated robot/world frames.
Custom calibration transforms may be attached as application payload data anchored
to the root frame and explicitly consumed by a suitable block.

Selected levels also carry a reserved `pyramid` object containing `set_id`, `level`,
`largest_size` and `to_largest`; bundled levels also declare `level_count`.
`to_largest` maps current pixel centers to that set's largest image. After further
image geometry, the mapping updates while `level` identifies the originating level.
Explicit up/down tools also record `pyramid_operation`. A new pyramid can start
from an already cropped/reduced image: its largest image is that new base, while
`to_root` still identifies the original source coordinate system.

Blobs, corners, centroids, regions and other results **stay in working-image pixels**.
Their output payloads identify the frame and originating level; the run message
also labels selected pyramid levels. A blob's area is measured in working pixels,
and acceptance limits apply at that resolution. No automatic area/radius scaling
is implied, particularly after nonuniform/projective transforms.

For coordinates on the largest image:

1. Connect the points or region to Read port payload's `value` input.
2. Leave its reference dropdown at `largest` (or choose `root` for the original source).
3. For points, connect `value` and `transform` to Transform points.
4. For regions, connect `value` and `transform` to Transform region, with the largest
   or root image connected as its target `image`.

The mapping output includes source/target dimensions. Transform points updates
coordinate payloads when target dimensions are provided; a matrix without target
dimensions records an explicit `coordinate_mapping` instead of claiming an image
frame it cannot identify. Empty point lists preserve this same provenance.

Known frame mismatches in channel merging or mask/region selection are errors,
even when dimensions happen to match. Transform the selection first. Unknown
root identities still receive size checks. Histogram plots and unrelated display
grids do not claim source-image coordinates. Custom operations that alter image
geometry must return a `geometry` output or explicit output payload; arbitrary
custom transformations cannot be inferred from pixel values.

## Python extension contract

`ImageArray.metadata` (and `DepthArray.metadata`) provides the image payload.
`Block.input_payloads[port_name]` provides it for every primary type. `Result.payloads`
can explicitly declare output payloads without wrapping the primary data:

```python
from opencv_dataflow.core import Block, Port, Result


class Example(Block):
    type_id = "example.payload_passthrough"
    inputs = (Port("value", "number"),)
    outputs = (Port("value", "number"),)

    def process(self, inputs):
        payload = self.input_payloads.get("value", {})
        return Result({"value": inputs["value"]}, payloads={"value": payload})
```

`ValuePacket(value, payload)` can supply a primary value with metadata to `execute`.
Graph routing handles these packets internally. Payloads must be finite JSON objects;
reserved spatial/pyramid structures are validated. `ImagePyramid.levels` is a
largest-first tuple of images with per-level spatial payloads. Custom sets must
provide valid mappings and decreasing image sizes; the builder does this for you.
Generic `any` outputs can connect to typed inputs; the receiving block validates
the actual primary type at execution. A packet does not bypass image/data checks.

## Example project

Open [channels and pyramids](<../examples/opencv tools/channels_pyramids/channels_pyramids.json>). It round-trips BGR channels, adds a recipe payload,
creates four image levels, detects two blobs at level 2, maps their centroids into
the largest image, previews level 1 and computes a largest-level histogram.

```powershell
.\.venv\Scripts\opencv-dataflow-run "examples/opencv tools/channels_pyramids/channels_pyramids.json" --repeat 3
```

Native reference: [OpenCV image pyramids](https://docs.opencv.org/4.x/d4/d1f/tutorial_pyramids.html).
