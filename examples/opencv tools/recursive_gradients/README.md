# Recursive gradient comparison

Open [recursive_gradients.json](recursive_gradients.json) and choose
**Run Once**. The two rectangles provide known rising and falling boundaries
on both axes. Each branch publishes signed float32 **Dx** and **Dy** fields,
gradient magnitude and an 8-bit magnitude preview.

| Filter | X peak at rising and falling edges | Y peak at rising and falling edges |
| --- | ---: | ---: |
| Deriche | +121.00 / −121.00 | +141.00 / −141.00 |
| Paillou | +75.43 / −75.43 | +87.90 / −87.90 |

Every measured peak was within **0.5 pixels** of its boundary, and the
perpendicular response stayed below 0.01 at all four probes. The X
measurements use row 39 at the left rectangle's sides. The Y measurements
use column 115 at the right rectangle's top and bottom. Each filter has its
own response scale. Apply image thresholds in that filter's units.

The Deriche block is under **OpenCV → Extended Image Processing → Edges**.
It adds replicated borders around the native recurrences, then crops back to
the original image grid. [Deriche's recursive filter paper](https://www.fit.vut.cz/person/ibarina/public/papers/Deriche1987.pdf)

The Paillou block is under **Custom → Image processing → Edges**. It uses
SciPy to implement the Paillou recurrence with stable constant continuation.
The tested OpenCV 5.0.0 wheel exposes the Deriche axes and leaves the Paillou
axes in its C++ API. [Paillou's step-edge paper](https://perso.telecom-paristech.fr/angelini/SI241/papers_for_project/Paillou.pdf), [OpenCV Paillou source](https://github.com/opencv/opencv_contrib/blob/5.0.0/modules/ximgproc/src/paillou_filter.cpp)

Color images produce a signed response per color channel. The derivatives
ignore alpha. The output mask and region select where to publish results
and leave the neighborhood calculation intact. The rectangle branch shows
zero response outside its selection.

The graph, generator and step image are original MIT assets. Rebuild them
from the repository root with the example builder.

    python "examples/opencv tools/recursive_gradients/build_example.py" --output "examples/opencv tools/recursive_gradients"
