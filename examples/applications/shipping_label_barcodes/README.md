# Shipping label barcode reading

Read tracking and routing barcodes on photographs of synthetic shipping labels.
The projects use the **Read barcodes (ZXing)** block.

Install the optional barcode dependency from the repository root.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[barcodes]"
```

1. Open [shipping_label_barcodes.json](shipping_label_barcodes.json) and click
   **Run all**. The project reads two Code 128 symbols in each of four photos.
   The filename is the ticket. The result filmstrip shows the decoded text,
   outlines, and pose axes. Double-click a thumbnail to inspect it at a larger
   size. Select the reader and inspect **Codes** for text and geometry.
2. Open [maxicode_crop.json](maxicode_crop.json) and click **Run once**.
   A fixed crop isolates the MaxiCode in `IMG_8745.jpg`. The reader uses
   **Is pure** and **GlobalHistogram** for that crop. The positions in its
   **Codes** output are in crop pixels. The crop is specific to this photo.

| Photo | Tracking code | Routing code |
| --- | --- | --- |
| `IMG_8745.jpg` | `1Z33344N5051525354` | `CA 859 2-40` |
| `IMG_8747.jpg` | `1Z88899K2324252627` | `CA 859 2-37` |
| `IMG_9568.jpg` | `1Z12345E0205271688` | `CA 859 2-32` |
| `IMG_9591.jpg` | `1Z99900P6061626364` | `CA 859 2-41` |

[Expected reads](expected_reads.json) records the selected upstream payloads,
including the MaxiCode. The saved projects check the read count. The example
test also compares each payload with the source annotations. ZXing-C++ 3.1.1
did not read the MaxiCode in the full-label trials. The isolated-crop project
demonstrates one successful read.

```powershell
.\.venv\Scripts\opencv-dataflow-run.exe "examples/applications/shipping_label_barcodes/shipping_label_barcodes.json" --all-images
.\.venv\Scripts\opencv-dataflow-run.exe "examples/applications/shipping_label_barcodes/maxicode_crop.json"
```

## Dataset credit

**UPS Synthetic Shipping Label Dataset, Dynamsoft, 2025.** Creator G
(Dynamsoft). Each label in this subset contains two Code 128 symbols and one
MaxiCode. The upstream repository's broader collections contain other formats.
[Dataset and attribution](https://github.com/Dynamsoft/datasets-from-dynamsoft/tree/ba77c0c64aedddb4553c91ca7e9db46e1ab86af7/UPS-Synthetic-Labels) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

The four bundled JPEGs were downsampled from 3024 × 4032 using OpenCV area
interpolation, re-encoded at JPEG quality 90, and given lowercase extensions.
The project crops the MaxiCode at runtime. Expected payloads were selected
from the [upstream annotations](https://github.com/Dynamsoft/datasets-from-dynamsoft/blob/ba77c0c64aedddb4553c91ca7e9db46e1ab86af7/UPS-Synthetic-Labels/annotation/annotations_min.json).
The dataset author describes these as synthetic labels.

| Bundled photo | Dimensions | Original SHA-256 | Bundled SHA-256 |
| --- | --- | --- | --- |
| [IMG_8745.jpg](assets/IMG_8745.jpg) | 1512 × 2016 | `b5357fde274310fc6a7bda51a637982c8ce6c4cdeed853d4a9eb4d1f0a82069e` | `b3978a7bc56d42d4f24cde0261d8b9a359b2be62021cf5bbb8ffff32b0c9b8a6` |
| [IMG_8747.jpg](assets/IMG_8747.jpg) | 900 × 1200 | `a06af99399cc2da5d13b0fb24ceee53e0a7c1e22ccd5b818c3428ca3bb381189` | `072a67fbebce3d0bf9e8d2f77434e0a3134fe196b14f123c3a195d2694a6ee2f` |
| [IMG_9568.jpg](assets/IMG_9568.jpg) | 1500 × 2000 | `0e936487d74b49bdde73afa530e901b3e60b608570de611a852f9921e3f77244` | `8418fbf45a87d87661cc5ca1ebf32783156afcf4fb73f9dfd42dcaf7259206ac` |
| [IMG_9591.jpg](assets/IMG_9591.jpg) | 1500 × 2000 | `c1375a39b563ea2a69c43d69e9b923f70d2488f59ec1a3bdf73e5c012d1cbdab` | `ae3ee4d63aebb574cc74611f1c1a998979bfc6475f2888ba9a431e09b7b7b08f` |

[All examples](../../README.md)
