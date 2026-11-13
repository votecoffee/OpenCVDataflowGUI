# Data Matrix reading

**Data:** controlled synthetic fixture with explicit reference geometry or labels.

Read the bundled synthetic Data Matrix and verify the decoded text.

Open [`datamatrix_reading.json`](datamatrix_reading.json) with **File → Open…**. Install the optional `barcodes` extra, then run once. The expected text is `LOT-0042`.

The [synthetic code](assets/datamatrix_lot.png) was generated with ZXing-C++ at
scale 8 and padded with a white border.

[All examples](../../README.md)


## Controlled fixture

A generated standards label provides a known payload. The [shipping-label application](../shipping_label_barcodes/README.md) uses real photographs for supported formats.

New application showcase variants should prefer suitable, attributed real captures.
