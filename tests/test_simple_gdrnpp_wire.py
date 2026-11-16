"""Legacy NPZ interoperability, bounded streaming and independently owned arrays."""

import io
import json
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile

import numpy as np
import pytest

from opencv_dataflow.simple_gdrnpp import wire


def values():
    return dict(
        rgb=np.arange(360, dtype=np.uint8).reshape(10, 12, 3)[::-1, ::2],
        camera=np.eye(3, dtype=np.float64),
        depth=np.asfortranarray(np.arange(120, dtype=np.float32).reshape(10, 12)),
        index=np.arange(15, dtype=np.int32),
        mask=np.ones((3, 4), bool),
        empty=np.empty((0, 3), np.float32),
    )


@pytest.mark.parametrize("compression", ["stored", "deflated_fast", "legacy"])
def test_streams_legacy_and_new_packets_with_same_dtype_order_and_mutability(tmp_path, compression):
    expected, metadata = values(), dict(capture="same", score=None)
    if compression == "legacy":
        np.savez_compressed(tmp_path / "arrays.npz", **expected)
        (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    else:
        wire.write_packet(tmp_path, metadata, expected, compression=compression)
    actual_metadata, arrays = wire.read_packet(tmp_path)
    assert actual_metadata == metadata
    assert arrays.keys() == expected.keys()
    for name in expected:
        np.testing.assert_array_equal(arrays[name], expected[name])
        assert arrays[name].dtype == expected[name].dtype
        assert arrays[name].flags.writeable and arrays[name].flags.owndata
    assert arrays["depth"].flags.f_contiguous
    arrays["depth"][0, 0] = -1
    assert expected["depth"][0, 0] == 0
    # Older NumPy readers can still read the updated writer's standard NPZ.
    with np.load(tmp_path / "arrays.npz", allow_pickle=False) as old:
        for name in expected:
            np.testing.assert_array_equal(old[name], expected[name])


def packet(tmp_path, payload, *, version=(1, 0), duplicate=False):
    stream = io.BytesIO()
    np.lib.format.write_array(stream, payload, version=version)
    with ZipFile(tmp_path / "arrays.npz", "w") as archive:
        archive.writestr("a.npy", stream.getvalue())
        if duplicate:
            with pytest.warns(UserWarning, match="Duplicate"):
                archive.writestr("a.npy", stream.getvalue())
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")


def test_npy_v2_and_empty_packets_remain_supported(tmp_path):
    packet(tmp_path, np.arange(10, dtype=np.float32), version=(2, 0))
    np.testing.assert_array_equal(wire.read_packet(tmp_path)[1]["a"], np.arange(10))
    wire.write_packet(tmp_path, {}, {})
    assert wire.read_packet(tmp_path) == ({}, {})


@pytest.mark.parametrize("dtype", [object, "S5", "<i8", ">f4", [("x", "f4")]])
def test_bad_types_rejected_before_array_allocation(tmp_path, monkeypatch, dtype):
    packet(tmp_path, np.zeros((2, 3), dtype=dtype))
    monkeypatch.setattr(
        wire.np, "empty", lambda *a, **k: pytest.fail("Allocated before validation")
    )
    with pytest.raises(ValueError, match="header"):
        wire.read_packet(tmp_path)


def test_byte_count_and_duplicate_member_fail_before_allocation(tmp_path, monkeypatch):
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        stream, dict(descr="<f4", fortran_order=False, shape=(2_000_000, 2_000_000, 3))
    )
    with ZipFile(tmp_path / "arrays.npz", "w") as archive:
        archive.writestr("a.npy", stream.getvalue() + b"\0")
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        wire.np, "empty", lambda *a, **k: pytest.fail("Allocated before validation")
    )
    with pytest.raises(ValueError, match="header"):
        wire.read_packet(tmp_path)
    packet(tmp_path, np.zeros((2, 3), np.float32), duplicate=True)
    with pytest.raises(ValueError, match="members"):
        wire.read_packet(tmp_path)


def test_file_framing_near_legacy_limit_falls_back_to_fast_compression(tmp_path, monkeypatch):
    monkeypatch.setattr(wire, "LIMIT", 4096)
    wire.write_packet(tmp_path, {}, dict(a=np.zeros(3856, np.uint8)))
    with ZipFile(tmp_path / "arrays.npz") as archive:
        assert archive.infolist()[0].compress_type == ZIP_DEFLATED
    assert wire.read_packet(tmp_path)[1]["a"].shape == (3856,)
    with pytest.raises(ValueError, match="including headers"):
        wire.write_packet(tmp_path, {}, dict(a=np.zeros(4000, np.uint8)))


def test_chunked_reads_cover_large_arrays(tmp_path, monkeypatch):
    monkeypatch.setattr(wire, "CHUNK_BYTES", 29)
    expected = np.arange(300, dtype=np.float32).reshape(20, 15)
    wire.write_packet(tmp_path, {}, dict(a=expected))
    with ZipFile(tmp_path / "arrays.npz") as archive:
        assert archive.infolist()[0].compress_type == ZIP_STORED
    np.testing.assert_array_equal(wire.read_packet(tmp_path)[1]["a"], expected)


def test_crc_corruption_is_not_returned_as_an_array(tmp_path):
    wire.write_packet(tmp_path, {}, dict(a=np.arange(100, dtype=np.float32)))
    path = tmp_path / "arrays.npz"
    raw = bytearray(path.read_bytes())
    with ZipFile(path) as archive:
        member = archive.infolist()[0]
        offset = member.header_offset + 30 + len(member.filename) + 20 + 128
    raw[offset] ^= 1
    path.write_bytes(raw)
    with pytest.raises(BadZipFile, match="CRC"):
        wire.read_packet(tmp_path)


def test_writer_matches_reader_member_and_type_contract(tmp_path):
    with pytest.raises(ValueError, match="members"):
        wire.write_packet(tmp_path, {}, {str(i): np.ones(1, np.uint8) for i in range(17)})
    with pytest.raises(ValueError, match="header"):
        wire.write_packet(tmp_path, {}, dict(a=np.ones(1, np.int64)))
