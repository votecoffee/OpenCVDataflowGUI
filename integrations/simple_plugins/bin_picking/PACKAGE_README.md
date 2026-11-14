# SimpleBinPicking

Observed-surface suction planning and an optional, explicitly assisted
parallel-jaw simulation example for OpenCVDataflowGUI. Numerical core dependencies
are NumPy and SciPy. The host supplies OpenCV and Qt for graphics/training.

## Install

```powershell
python -m pip install --no-deps .
```

Use an environment that already provides the declared compatible dependencies.
The optional native ROS packages run in their own matched interpreter; do not
mix ROS binary extensions into the GUI environment.

## Usage

`Camera`, `Recipe`, `SuctionSettings` and `Picker` form the retained suction API.
The host block `3d.simple_bin_picking` opens its own training editor and reports
ranked contact, collision and unknown-space evidence. `parallel_jaw` proposes
box tops from measured first-return depth without object-pose truth.

The explicitly loaded `sim_dataflow` plugin composes grid tasks, stock serial
Unbatch/Batch, one recapturing simulated pick per cell, summary and recorded
image/cloud review. It uses a separate native primitive backend. Six authored
boxes were picked into a table grid in the included host example. Assistance,
privileged collision/evaluation geometry and physical-grasp limits are explicit.

## Maintainers

The project maintainers.

## Contributing

Preserve missing/occluded observations, physical units/frames, ticket/capture
identity, outcome holds and independent numerical/native evidence.

## License

MIT, included in LICENSE. Runtime dependencies retain their respective licenses.
