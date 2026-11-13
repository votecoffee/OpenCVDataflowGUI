# OpenCVDataflowGUI

Build and run machine vision pipelines in a desktop node editor. Connect OpenCV
tools, inspect intermediate results, and extend the toolbox with Python plugins.

## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [Architecture](#architecture)
- [Plugins](#plugins)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Install

Requires **Python 3.14 or newer** and **OpenCV 5 contrib**. The Windows development
environment uses Python **3.14.3** and OpenCV **5.0.0**
(`opencv-contrib-python` **5.0.0.93**).

From the repository root on Windows:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\opencv-dataflow.exe
```

On macOS or Linux, create the environment with `python3.14 -m venv .venv`,
install with `.venv/bin/python -m pip install -e .`, and launch
`.venv/bin/opencv-dataflow`.

## Usage

The editor opens a synthetic inspection demo. Press **Run once** (F5) to see its
results. To build your own small project:

1. Choose **File → New**. Search for **Demo image**, **Grayscale**, **Threshold**,
   and **Find contours** in the left toolbox and drag them onto the canvas.
2. Connect the image outputs to the next block's image input in that order. Choose
   **Workspace → Auto arrange blocks** to organize the graph.
3. Set **Find contours > Min area** to `1000` and click **Apply configuration**.
   Run the graph, then adjust **Threshold**. Use **Live results** or **Run block**
   to preview eligible edits using the last input. Apply the changes to keep them.
4. Run again and select **Find contours**. Use the **Results** tab to inspect
   images and measurements. Double-click a result image or use **Pop out…** for
   closer review. Save the graph with **File → Save**.

**Run all** replays recorded sequences once from the start, or runs the job once
when there is no sequence. **Continuous** loops the recordings until **Stop**.
The status bar shows the active mode.

Explore the [example catalog](examples/README.md) for runnable projects and the
[workflow guide](docs/workflow-guide.md) for editor, inspection, and runtime details.

## Architecture

- **Graph and scheduling.** Blocks declare typed input and output ports. The
  scheduler runs a block when its dependencies are ready. Independent branches
  can run concurrently, with serial execution available for predictable ordering.
  Buffered stream blocks drain their remaining output before finite playback stops.
- **Branching and merging.** One output can feed multiple consumers. Split and
  decision blocks route work, while Join and input coordination can match
  ticketed results before a downstream block runs. A ToolBlock packages a
  subgraph behind reusable public ports.
- **Connections and failures.** The editor checks port compatibility and rejects
  cycles and duplicate incoming wires. Supported conversions are applied
  automatically, with runtime checks for constraints such as numeric range or
  image format. Block exceptions are recorded as errors and skip dependent
  work. Inspection failures can still emit results for downstream decisions.
- **Extensions and persistence.** Trusted Python plugins register blocks against
  the same runtime contract as built-in tools. Projects save graph configuration,
  layout, and ToolBlock definitions. The Qt editor receives result snapshots
  from a separate execution graph.

[Architecture and extension contracts](docs/architecture.md) · [Control flow](docs/control-flow.md)

## Plugins

Save this example as `count_pixels.py` to define a block with an image input
and a numeric output.

```python
from opencv_dataflow.core import Block, Port, Result


class CountPixels(Block):
    type_id = "example.count_pixels"
    title = "Count pixels"
    inputs = (Port("image"),)
    outputs = (Port("count", "number"),)

    def process(self, inputs):
        height, width = inputs["image"].shape[:2]
        return Result({"count": height * width})


def register(registry):
    registry.register(CountPixels)
```

Choose **File → Load Python plugin…** and select the file. Add **Count pixels**
from **User**, connect an image, and run the graph. Load the plugin before opening
a saved project that uses it. The headless runner accepts `--plugin count_pixels.py`.
Load Python code only from sources you trust.

Projects using installed **Simple Plugins**, including **Simple3DLocate**,
open directly after setup. **SimpleClassify** is bundled with DataflowGUI. See the
[Simple Plugins examples and setup](<examples/Simple Plugins/README.md>).

[Runnable plugin example](examples/user_tools/custom_invert/README.md) · [Plugin dialog API](docs/plugin-dialogs.md)

Plugins can provide their own configuration or training dialogs. The dialog API
can capture several processed inputs together, preserving their images, point
clouds and metadata for training in a plugin worker.

## Maintainers

[@votecoffee](https://github.com/votecoffee)

## Contributing

See the [contribution guide](CONTRIBUTING.md) for local checks and extension
conventions, and the [roadmap](docs/roadmap.md) for planned work.

## License

[MIT](LICENSE) © 2026 votecoffee.
