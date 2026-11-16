"""Conservative structured-append headers from native rectified QR Model 2 modules.

This verifies untouched format copies and complete Reed-Solomon codewords. It does
not repair module errors or replace the native payload decoder. Layout tables are
attributed in qr_rs_tables.py; traversal/validation code is local implementation.
"""

import numpy as np

from .parameter_validation import ValidationError
from .qr_rs_tables import ECC_CODEWORDS_PER_BLOCK, NUM_ERROR_CORRECTION_BLOCKS

LEVELS = ("L", "M", "Q", "H")
FORMAT_LEVELS = (1, 0, 3, 2)


def format_word(value):
    remainder = value << 10
    for bit in range(14, 9, -1):
        if remainder & (1 << bit):
            remainder ^= 0x537 << (bit - 10)
    return ((value << 10) | remainder) ^ 0x5412


def read_format(modules):
    size = len(modules)
    first = [(8, y) for y in range(6)] + [(8, 7), (8, 8), (7, 8)]
    first += [(14 - i, 8) for i in range(9, 15)]
    second = [(size - 1 - i, 8) for i in range(8)]
    second += [(8, size - 15 + i) for i in range(8, 15)]
    words = [
        sum(int(modules[y, x]) << i for i, (x, y) in enumerate(points))
        for points in (first, second)
    ]
    matches = [value for value in range(32) if format_word(value) == words[0]]
    if words[0] != words[1] or len(matches) != 1:
        raise ValidationError("Rectified QR format copies are damaged or disagree")
    value = matches[0]
    return FORMAT_LEVELS.index(value >> 3), value & 7


def function_mask(version):
    size = version * 4 + 17
    reserved = np.zeros((size, size), bool)
    reserved[:9, :9] = reserved[:9, -8:] = reserved[-8:, :9] = True
    reserved[6, :] = reserved[:, 6] = True
    if version >= 7:
        reserved[:6, -11:-8] = reserved[-11:-8, :6] = True
    if version >= 2:
        count = version // 7 + 2
        step = 26 if version == 32 else ((version * 4 + count * 2 + 1) // (count * 2 - 2)) * 2
        centers = [6] + [size - 7 - step * i for i in reversed(range(count - 1))]
        for i, x in enumerate(centers):
            for j, y in enumerate(centers):
                if (i, j) not in ((0, 0), (0, count - 1), (count - 1, 0)):
                    reserved[y - 2 : y + 3, x - 2 : x + 3] = True
    return reserved


def raw_codewords(modules, version, mask):
    reserved = function_mask(version)
    y, x = np.indices(modules.shape)
    masks = (
        (x + y) % 2 == 0,
        y % 2 == 0,
        x % 3 == 0,
        (x + y) % 3 == 0,
        (x // 3 + y // 2) % 2 == 0,
        (x * y) % 2 + (x * y) % 3 == 0,
        ((x * y) % 2 + (x * y) % 3) % 2 == 0,
        ((x + y) % 2 + (x * y) % 3) % 2 == 0,
    )
    data = np.logical_xor(modules, masks[mask])
    bits, right, upward = [], len(modules) - 1, True
    while right > 0:
        if right == 6:
            right = 5
        for row in range(len(modules) - 1, -1, -1) if upward else range(len(modules)):
            for col in (right, right - 1):
                if not reserved[row, col]:
                    bits.append(data[row, col])
        upward = not upward
        right -= 2
    return np.packbits(np.asarray(bits[: len(bits) // 8 * 8], np.uint8))


def gf_tables():
    exponent = np.empty(255, np.uint16)
    logarithm = np.zeros(256, np.int16)
    value = 1
    for power in range(255):
        exponent[power], logarithm[value] = value, power
        value <<= 1
        if value & 256:
            value ^= 0x11D
    return exponent, logarithm


EXP, LOG = gf_tables()


def data_stream(raw, version, level):
    count = NUM_ERROR_CORRECTION_BLOCKS[level][version]
    ecc = ECC_CODEWORDS_PER_BLOCK[level][version]
    short, longer = divmod(len(raw), count)
    sizes = [short - ecc + int(i >= count - longer) for i in range(count)]
    if min(sizes) < 1:
        raise ValidationError("QR codeword layout is inconsistent")
    blocks = [np.empty(n + ecc, np.uint8) for n in sizes]
    cursor = 0
    for col in range(max(sizes)):
        for block, n in zip(blocks, sizes, strict=True):
            if col < n:
                block[col] = raw[cursor]
                cursor += 1
    for col in range(ecc):
        for block, n in zip(blocks, sizes, strict=True):
            block[n + col] = raw[cursor]
            cursor += 1
    if cursor != len(raw):
        raise ValidationError("QR codeword count is inconsistent")
    for block in blocks:
        powers = np.arange(len(block) - 1, -1, -1, dtype=np.int32)
        for root in range(ecc):
            terms = np.where(block != 0, EXP[(LOG[block] + root * powers) % 255], 0)
            if np.bitwise_xor.reduce(terms):
                raise ValidationError(
                    "Rectified QR codewords need error correction; header not trusted"
                )
    return bytes(np.concatenate([b[:n] for b, n in zip(blocks, sizes, strict=True)]))


def structured_header(rectified):
    if (
        not isinstance(rectified, np.ndarray)
        or rectified.ndim != 2
        or rectified.shape[0] != rectified.shape[1]
        or not 21 <= len(rectified) <= 177
        or (len(rectified) - 17) % 4
        or not np.isin(rectified, [0, 255]).all()
    ):
        raise ValidationError("Need a native binary QR Model 2 module grid, without quiet zone")
    version = (len(rectified) - 17) // 4
    modules = rectified == 0
    level, mask = read_format(modules)
    stream = data_stream(raw_codewords(modules, version, mask), version, level)
    bits = np.unpackbits(np.frombuffer(stream, np.uint8))

    def number(start, count):
        return sum(int(v) << (count - i - 1) for i, v in enumerate(bits[start : start + count]))

    if number(0, 4) != 3:
        raise ValidationError("Decoded QR symbol has no leading structured-append header")
    index, count, parity = number(4, 4), number(8, 4) + 1, number(12, 8)
    if index >= count or count < 2:
        raise ValidationError("Structured-append index/count must describe a multipart sequence")
    return {
        "index": index,
        "count": count,
        "parity": parity,
        "version": version,
        "correction_level": LEVELS[level],
        "mask": mask,
    }
