# Dependency and data notices

Simple3DMeasure source uses the MIT license in [LICENSE](LICENSE).
Dependencies and dataset material retain their own terms.

- **NumPy** is the base numerical dependency; its package includes its BSD
  license and notices.
- **SciPy** is an optional `flatness` dependency. Its upstream license is
  BSD 3-Clause, copyright Enthought, Inc. and SciPy Developers. See the
  [official license](https://github.com/scipy/scipy/blob/v1.18.1/LICENSE.txt).
  SciPy provides the Qhull-backed `ConvexHull` implementation and ships its
  third-party license notices. This package does not redistribute or copy
  SciPy/Qhull code or binaries.
- **OpenCVDataflowGUI / Qt** provide optional host training and review. Their
  installed packages retain their respective licenses. The numerical API does
  not import the GUI.
- **T-LESS** raw and derived example data use CC BY 4.0 with author attribution,
  selected source hashes and explicit transformations in the public example
  collection. Generated controls and builder code use MIT. Dataset licensing
  is separate from this package's source license; no endorsement is implied.

The minimum-width relationship is mathematical background, not copied source:
Xiangyang Zhu and Han Ding, *Flatness tolerance evaluation: an approximate
minimum zone solution*, Computer-Aided Design 34(9), 655–664 (2002),
[DOI](https://doi.org/10.1016/S0010-4485(01)00137-3).
