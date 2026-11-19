# Simple Plugin loading

The editor, headless runner and saved-project API load known **Simple Plugins**
automatically. Open a project or add a tool from its toolbox category; **Load Python
plugin** and `--plugin` are unnecessary for the shipped examples. This applies to
local private projects using the same stable block types, too.

The app includes checksum-pinned pure-Python wheels. These also ship in the installed application wheel.
There is no download, `pip install`, OpenCV replacement or dependency resolution
when opening a project. Embedded package licenses and notices remain intact;
[versions and sources](../src/opencv_dataflow/simple_plugin_wheels/README.md)
identify each selected artifact. SimpleClassify remains the existing bundled tool.

## Included providers

- PatternMatch, Segmentation, PathWidth and ReferenceInspect.
- 3DLocate, 3DMeasure and BinPicking, including jaw clearance and part-instance conversion.
- LaserProfile, SurfaceEnhance, FringeScan and Deflectometry.
- FoundationPose and GDRNPP host adapters.
- Simulation grid/pick/summary and recorded motion review.
- Recorded jaw/depth/pose, RGB-D, lighting, fringe and reflected-pattern capture inputs.

Recorded inputs retain their existing type IDs, fields, formats, source identity
and Save As assets. Their data is read only on explicit execution. No customer
images or private projects are packaged in the application.

## Versions and explicit custom plugins

Discovery and toolbox population do not import optional packages. On first use,
an application-selected provider checks its bundled wheel and makes it importable.
It stages registrations before publishing them to the registry. A partial import
or registration error cannot leave half a set of blocks registered.

The package selected first in a Python session remains selected. The app never
removes modules from `sys.modules` to replace live classes. Restart after updating
the app, or before deliberately selecting an older example's explicit plugin.
Explicit custom plugins continue to use **File → Load Python plugin** or `--plugin`.
If an older, explicitly loaded package cannot supply a newer type, restart to use
the shipped provider. An incompatible declared bundle fails with reinstall guidance
rather than falling through to an unrelated package.

Only fixed application code selects these providers. Project JSON cannot select
a Python plugin, wheel, module or search directory. An adjacent `plugin.py` is
never executed just because a project is opened. Unknown custom block types still
require their trusted plugin to be loaded explicitly.

## Execution setup remains separate

Opening and static validation do not launch workers, install models, start ROS or
connect to equipment. FoundationPose and GDRNPP keep their isolated Python runtime
in machine-local settings, separate from portable project recipes. FoundationPose
also remembers its pinned weights folder in Runtime. GDRNPP remembers a default
verified model folder in Setup; an explicit block model folder takes precedence.
See [pose runtime setup](simple-pose-runtimes.md). Their saved-result reviews remain
available without inference. Simulation motion still requires its explicitly enabled
external services. A loaded block does not establish hardware or model accuracy.
