"""Exercise native code readers with generated linear and matrix symbols."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.graphics_layers import RESULT_GRAPHICS
from opencv_dataflow.payloads import NO_VALUE


@pytest.mark.parametrize("mode", ["multi", "single"])
def test_qr_decode_and_mask(mode):
    reader = builtin_registry().create("inspect.read_qr", parameters={"mode": mode})
    qr = cv2.QRCodeEncoder_create().encode("dataflow-42")
    image = np.pad(np.repeat(np.repeat(qr, 8, axis=0), 8, axis=1), 24, constant_values=255)
    result = reader.execute({"image": image})
    assert result.passed
    assert result.outputs["count"] == 1
    code = result.outputs["codes"][0]
    assert (code["text"], code["symbology"]) == ("dataflow-42", "QR_CODE")
    assert np.asarray(code["corners"]).shape == (4, 2)
    assert np.allclose(code["center"], np.asarray(image.shape[::-1]) / 2, atol=2)
    assert result.outputs["overlay"].shape == (*image.shape, 3)
    outline, axis_x, axis_y, label = result.drawings["decoded codes"].objects
    assert outline.closed
    assert axis_x.kind == axis_y.kind == "arrow"
    assert label.kind == "label" and label.text == code["text"]
    assert not reader.execute({"image": image, "mask": np.zeros_like(image)}).passed
    blank = reader.execute({"image": np.full_like(image, 255)})
    assert not blank.passed
    assert blank.fail_reasons[0]["code"] == "code_read_count_below_minimum"


def ean13(check_digit, module_width):
    # 5901234123457, left digits use L/G/G/L/L/G for leading digit 5.
    modules = (
        "101"
        "0001011"
        "0100111"
        "0110011"
        "0010011"
        "0111101"
        "0011101"
        "01010"
        "1100110"
        "1101100"
        "1000010"
        "1011100"
        "1001110" + {"7": "1000100", "8": "1001000"}[check_digit] + "101"
    )
    row = np.repeat(np.array([0 if bit == "1" else 255 for bit in modules], np.uint8), module_width)
    return np.pad(np.tile(row, (120, 1)), ((100, 100), (80, 80)), constant_values=255)


def test_ean13_decode_with_geometry():
    image = ean13("7", 2)
    reader = builtin_registry().create("inspect.read_barcode", parameters={"draw_overlay": False})
    reader.exposed_outputs.append(RESULT_GRAPHICS)
    reader.set_graphics_mode("off")
    result = reader.execute({"image": image})
    assert result is not None, reader.error
    assert result.passed, result.message
    assert result.outputs["count"] == 1
    assert result.outputs["overlay"] is NO_VALUE
    layers = result.outputs[RESULT_GRAPHICS].layers
    assert len(layers) == 1 and layers[0].objects[0].closed
    _, axis_x, axis_y, label = layers[0].objects
    assert label.text == "5901234123457"
    assert abs(axis_x.points[1][0] - axis_x.points[0][0]) > 40
    assert abs(axis_y.points[1][1] - axis_y.points[0][1]) > 25
    assert not result.graphics
    code = result.outputs["codes"][0]
    assert code["text"] == "5901234123457"
    assert code["symbology"] == "EAN_13"
    assert np.asarray(code["corners"]).shape == (4, 2)


def test_zxing_decodes_mixed_symbols_and_respects_region():
    zxing = pytest.importorskip("zxingcpp")
    matrix = np.asarray(
        zxing.write_barcode_to_image(
            zxing.create_barcode("LOT-0042", zxing.BarcodeFormat.DataMatrix), scale=6
        )
    )
    linear = np.asarray(
        zxing.write_barcode_to_image(
            zxing.create_barcode("SERIAL42", zxing.BarcodeFormat.Code128), scale=3
        )
    )
    h, w = matrix.shape
    rows, cols = linear.shape
    scene = np.full((max(h, rows) + 40, w + cols + 80), 255, np.uint8)
    scene[20 : 20 + h, 20 : 20 + w] = matrix
    scene[20 : 20 + rows, 60 + w : 60 + w + cols] = linear
    block = builtin_registry().create("inspect.read_zxing", parameters={"format": "AllReadable"})
    result = block.execute({"image": scene})
    assert result is not None, block.error
    assert {(c["text"], c["symbology"]) for c in result.outputs["codes"]} == {
        ("LOT-0042", "DataMatrix"),
        ("SERIAL42", "Code128"),
    }
    selection = np.zeros_like(scene)
    selection[:, : w + 40] = 255
    filtered = block.execute({"image": scene, "mask": selection})
    assert filtered.outputs["count"] == 1
    code = filtered.outputs["codes"][0]
    assert code["text"] == "LOT-0042"
    assert code["bytes_hex"] == b"LOT-0042".hex()
    np.testing.assert_allclose(code["center"], [20 + w / 2, 20 + h / 2], atol=1)


def test_zxing_reads_gs1_and_keeps_checksum_errors_out_of_codes():
    zxing = pytest.importorskip("zxingcpp")
    text = "(01)09506000134352(10)LOT0042"
    code = zxing.create_barcode(text, zxing.BarcodeFormat.DataMatrix, gs1=True)
    image = np.asarray(zxing.write_barcode_to_image(code, scale=5))
    block = builtin_registry().create("inspect.read_zxing")
    result = block.execute({"image": image})
    assert result is not None, block.error
    item = result.outputs["codes"][0]
    assert item["text"] == text
    assert item["content_type"] == "GS1"
    assert item["symbology_identifier"] == "]d2"
    block.parameters.update(format="EAN13", report_decode_errors=True)
    invalid = block.execute({"image": ean13("8", 4)})
    assert invalid is not None, block.error
    assert not invalid.passed
    assert invalid.outputs["codes"] == []
    assert "ChecksumError" in invalid.outputs["errors"][0]["error"]


@pytest.mark.local_examples
def test_zxing_example_verifies_ticketed_lot_code():
    pytest.importorskip("zxingcpp")
    project = (
        Path(__file__).resolve().parents[1]
        / "examples/applications/datamatrix_reading/datamatrix_reading.json"
    )
    graph = Graph.load(project, builtin_registry())
    try:
        graph.run()
        verifier = graph.blocks["verify"]
        assert verifier.state == "pass", verifier.error
        assert verifier.last_result.outputs["matches"] == ["LOT-0042"]
        assert verifier.last_result.payloads["matches"]["ticket"]["value"] == "datamatrix_lot.png"
    finally:
        graph.close()


@pytest.mark.local_examples
@pytest.mark.parametrize("isolated", [False, True])
def test_shipping_label_examples_match_dynamsoft_annotations(isolated, tmp_path, monkeypatch):
    pytest.importorskip("zxingcpp")
    root = Path(__file__).resolve().parents[1] / "examples/applications/shipping_label_barcodes"
    expected = json.loads((root / "expected_reads.json").read_bytes())
    if isolated:
        project = "maxicode_crop.json"
        symbology = "MaxiCode"
        crop = expected["maxicode_crop"]
        cases = [(crop["file"], [crop["text"]])]
    else:
        project = "shipping_label_barcodes.json"
        symbology = "Code128"
        cases = [(row["file"], row["code128"]) for row in expected["code128_replay"]]
    monkeypatch.chdir(tmp_path)
    graph = Graph.load(root / project, builtin_registry())
    graph.filmstrip_enabled = True
    try:
        reader = graph.blocks["read"]
        for filename, texts in cases:
            graph.run()
            assert reader.state == "pass", reader.error
            result = reader.last_result
            assert result.outputs["count"] == len(texts)
            assert sorted(code["text"] for code in result.outputs["codes"]) == texts
            assert {code["symbology"] for code in result.outputs["codes"]} == {symbology}
            assert result.payloads["codes"]["ticket"]["value"] == filename
            assert np.any(result.outputs["overlay"] != reader.last_inputs["image"])
        assert isolated or graph.sources_exhausted()
        rows = graph.filmstrip.snapshot()
        assert [row["ticket"]["value"] for row in rows] == [filename for filename, _ in cases]
        assert all(row["override"] is not None for row in rows)
    finally:
        graph.close()


def test_buffered_barcode_filmstrip_reprojects_each_ticket_and_updates_popout(app, tmp_path):
    from opencv_dataflow.filmstrip_viewer import FilmstripViewer
    from opencv_dataflow.toolblocks import encapsulate

    texts = ["lot-001", "lot-002"]
    files = []
    for text in texts:
        code = cv2.QRCodeEncoder_create().encode(text)
        image = np.pad(np.repeat(np.repeat(code, 8, axis=0), 8, axis=1), 50, constant_values=255)
        path = tmp_path / f"{text}.png"
        assert cv2.imwrite(str(path), image)
        files.append(str(path))

    registry = builtin_registry()
    graph = Graph()
    graph.filmstrip_enabled = True
    source = graph.add(registry.create("input.file", parameters={
        "source_mode": "files", "paths": files, "color_format": "gray",
    }))
    batch = graph.add(registry.create("flow.batch", parameters={"size": 2}))
    serial = graph.add(registry.create("flow.unbatch"))
    crop = graph.add(registry.create("cv.crop", parameters={"x": 24, "y": 30}))
    reader = graph.add(registry.create("inspect.read_qr", parameters={"mode": "single"}))
    graph.connect(source.id, "image", batch.id, "item_1")
    graph.connect(batch.id, "group", serial.id, "group")
    graph.connect(serial.id, "value", crop.id, "image")
    graph.connect(crop.id, "image", reader.id, "image")
    encapsulate(graph, [crop.id, reader.id], "Read cropped QR code", registry)

    viewer = FilmstripViewer(lambda: graph.filmstrip)
    try:
        graph.run()
        viewer.refresh()
        viewer.open_image(viewer.items.item(0))
        dialog = viewer.popout
        assert "with result graphics" not in dialog.source_label.text()
        dialog.view.actual_size()

        graph.run()
        viewer.refresh()
        rows = graph.filmstrip.snapshot()
        assert [row["state"] for row in rows] == ["pass", "pass"]
        assert viewer.popout is dialog
        assert "Reloaded source image with result graphics" in dialog.source_label.text()
        assert dialog.view.transform().m11() == 1
        for row, text in zip(rows, texts, strict=True):
            assert row["thumbnail"] != row["source_thumbnail"]
            (layer, _), = row["layers"].values()
            outline, axis_x, axis_y, label = layer.objects
            assert (axis_x.kind, axis_y.kind, label.kind, label.text) == (
                "arrow", "arrow", "label", text
            )
            assert axis_x.points[0] == axis_y.points[0]
            x, y = np.rint(outline.points[0]).astype(int) + [24, 30]
            if row is rows[0]:
                image = dialog.view.image_item.pixmap().toImage()
                assert image.width() > 160
                pixel = image.pixelColor(int(x), int(y))
                assert pixel.green() > pixel.red() > pixel.blue()

        dialog.close()
        Path(files[0]).unlink()
        viewer.open_image(viewer.items.item(0))
        assert "Captured thumbnail with result graphics" in viewer.popout.source_label.text()
        assert viewer.popout.view.image_item.pixmap().height() == 100
    finally:
        if viewer.popout is not None:
            viewer.popout.close()
        viewer.close()
        graph.close()
