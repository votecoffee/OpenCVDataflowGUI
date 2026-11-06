# TIFF page sequence

Open [tiff_page_sequence.json](tiff_page_sequence.json) with **File → Open…**.
Click **Run once** for the next page or start continuous execution to replay
the file. The example measures three 16-bit grayscale pages with means of
1,000, 20,000, and 45,000. Each page has its own ticket and pixel frame.

**TIFF page sequence** is under **Input → Files**. **Start page** is
zero-based, **End page** is exclusive, and **Stride** selects every Nth
page starting from the first requested page. Set **End page** to −1 to
replay through the last decodable page. With an explicit end, a missing
requested page reports an error. **Repeat** controls Stop or Loop.
Reset flow state to return to the start page.

The source preserves decoded depth and channel layout by default. The
inspector can select an output format and depth. Files up to 128 MiB
are read through Python's file API, which supports Unicode filenames.
[OpenCV multipage decode reference](https://docs.opencv.org/5.0/main_modules/imgcodecs.html#imdecodemulti)

```powershell
opencv-dataflow-run examples/inputs/tiff_page_sequence/tiff_page_sequence.json --all-images
```

[All examples](../../README.md)
