# Deflectometry data/license audit — 2026-11-16

Primary-source inspection and bounded first real-image use. OpenCSP's separate
data license, archive identity and selected member bytes are verified. No
upstream executable code was run or bundled. Metadata/content availability does
not establish measurement accuracy. Other candidate datasets remain unused.

## Used: OpenCSP / SOFAST, CC BY-SA 4.0

The [Sandia OpenCSP access page](https://opencsp.sandia.gov/how-to-access/) links
its separate code and sample-data distributions. The
[SOFAST single-facet example](https://github.com/sandialabs/OpenCSP/blob/main/example/sofast_fringe/example_process_single_facet.py)
names measurement/camera/display/orientation and response-calibration files.
The separately published **NSTTF Facet SOFAST Landscape** ZIP instead contains
exported camera/projected PNGs and a processed HDF5. Its README identifies actual
SOFAST Landscape camera images of an NSTTF heliostat facet. The original capture
exposure, response calibration and full reconstruction calibration are not
provided in the inspected bundle. It is used for code-value phase-fit diagnostics
and self-comparison only, with no metric reconstruction or physical accuracy claim.

The repository [code license](https://github.com/sandialabs/OpenCSP/blob/main/OpenCSP_code_license.txt)
is **Apache 2.0**. That does not automatically license the separate Box data.
The authors' [OpenCSP publication](https://d-nb.info/1387908790/34) describes
non-code assets as CC BY-SA, while the
[official acknowledgment page](https://opencsp.sandia.gov/how-to-participate/how-to-acknowledge/)
links a distinct [data license file](https://sandia-csp.app.box.com/s/iftmhdkhgjnmfgefsfmkid011dtv8n3g/file/1341437503062).
Although the web text renderer required JavaScript, the official shared-file
download supplies the 145-byte license. Its SHA1 matches the public file metadata:
`63b5eeee74062d1ea1fb194bfcce0a81afe58f74`. It explicitly applies **CC BY-SA 4.0** to
the directory tree. Original license bytes are retained without newline changes.
The code license is not substituted for these data terms.

Attribution: OpenCSP Team, *OpenCSP: An Environment for
Collaborative CSP Optical Technology Development*, Sandia National Laboratories,
[official project](https://opencsp.sandia.gov/). The full bounded 83,154,541-byte
[sample ZIP](https://sandia-csp.app.box.com/s/iftmhdkhgjnmfgefsfmkid011dtv8n3g/file/1951928463619)
matches publisher SHA1 `61485298ea4fda78425d0e70f0d59c87bc9a58aa` and retained SHA256
`ce027803717c3e97cc4e4481b53e7ac6a90e822c151d1c850f4023233e24b5ff`.
ZIP CRC and member SHA256 checks cover 18 retained original PNGs and README.
Only bounded numeric HDF5 inspection extracted the upstream raw region mask;
the processed HDF5 is not redistributed and its computed shape is not ground truth.
Exact transformations, command-order checks and data terms are in the
[example credits](DATASET_CREDITS.md).

## KIT multiview deflectometry: useful decoded data, license unresolved

**Jonathan Balzer, Sebastian Höfer, Stefan Werling and Jürgen Beyerer**, KIT,
[Multiview Deflectometry Dataset](https://ies.iar.kit.edu/multiviewdeflectometry.php).
The primary page supplies multiview camera parameters and vector-valued VTK
images: observed screen/scene locations in camera millimeters, not surface XYZ.
It describes bowling-ball and automotive-hood data; the latter excludes the
copyrighted CAD reference. The bowling-ball reference is a nominal manufacturer
diameter, which is not an independently measured ground-truth surface.

No explicit redistribution license was found on the inspected dataset page.
Do not import or publish these pixels/VTK/CAD as a permissive dataset until terms
are verified. This is potentially useful for an explicit decoded-correspondence
adapter, not evidence of a raw N-step phase sequence. No archive was fetched.

## Arizona instantaneous deflectometry: licensed visualization assets only

**Isaac Trumper, Heejoo Choi and Dae Wook Kim**, *Instantaneous phase shifting
deflectometry*, 2016. The [University of Arizona record](https://experts.arizona.edu/en/datasets/instantaneous-phase-shifting-deflectometry/)
points to [Figshare collection 3766289](https://doi.org/10.6084/m9.figshare.c.3766289.v1).
The primary [collection API](https://api.figshare.com/v2/collections/3766289/articles)
lists two articles. Each article's primary metadata explicitly states
**CC BY 4.0**:

| Item | DOI | Exposed file |
| --- | --- | --- |
| [Single actuator visualization](https://api.figshare.com/v2/articles/4950230) | 10.1364/OE.24.027993.v001 | AVI, 1,168,810 bytes |
| [Trefoil mode visualization](https://api.figshare.com/v2/articles/4950233) | 10.1364/OE.24.027993.v002 | AVI, 1,072,886 bytes |

These are described as metrology visualization videos. This inspected collection
does not expose ordered original phase frames, a camera/screen calibration or
independent shape truth. Therefore it is **not selected as an estimator benchmark**.
No videos were downloaded or reinterpreted as phase-shift inputs. The retained
[metadata receipt](candidate_metadata.json) records exact licenses, file names,
sizes and metadata hashes; those hashes describe metadata, not unacquired pixels.

## First examples and remaining audit

Independent phase and planar/tilted reflection controls now verify circular
differences, half-angle behavior, slope sign and source/mask reasons. Three
generated phase comparisons and the real code-diagnostic self-review are retained.
Next find a separate registered real reference/test pair plus original response,
exposure and camera/screen calibration. Physical calibration/repeatability/shape
qualification remains open; rendered agreement and upstream computed shape do not
qualify an instrument.
