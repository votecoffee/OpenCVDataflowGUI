"""Bounded classification assessment charts; not source-image coordinates."""

import cv2
import numpy as np

from .model import image_pixels


def _text(image, text, point, *, width=590, scale=.55, color=(230, 234, 240)):
    # Hershey fonts support ASCII. Full labels remain in the numeric output
    # and Unicode Qt review title; never alter the model's labels.
    text = str(text).encode("ascii", errors="replace").decode("ascii")[:256]
    if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > width:
        while text and cv2.getTextSize(
            text + "...", cv2.FONT_HERSHEY_SIMPLEX, scale, 1
        )[0][0] > width:
            text = text[:-1]
        text += "..."
    cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_assessment(image, prediction):
    """Show the source and up to twenty ranked classes without hiding the part.

    Scores are the unchanged normalized model evidence, not calibrated
    confidence. This display uses its own grid; it is not a spatial overlay.
    """
    rows = prediction["probabilities"][:20]
    height = max(480, 190 + 46 * len(rows))
    output = np.full((height, 1120, 3), (28, 25, 23), np.uint8)
    _text(output, "SimpleClassify | Ranked class scores", (22, 31), width=1080, scale=.7)
    _text(output, f"Winner: {prediction['label']}", (510, 65))
    _text(output, f"Score {prediction['score']:.2f}%   Confusion {prediction['confusion']:.2f}%",
          (510, 92))
    _text(output, "Lower confusion = greater separation from runner-up", (510, 116), scale=.47)
    pixels = image_pixels(image)
    scale = min(460 / pixels.shape[1], (height - 120) / pixels.shape[0])
    size = tuple(max(1, round(v * scale)) for v in pixels.shape[1::-1])
    thumbnail = cv2.resize(pixels, size, interpolation=cv2.INTER_AREA)
    x, y = 22 + (460 - size[0]) // 2, 70 + (height - 120 - size[1]) // 2
    output[y:y + size[1], x:x + size[0]] = thumbnail
    _text(output, f"Input: {pixels.shape[1]} x {pixels.shape[0]} | display only",
          (22, height - 22), width=460, scale=.47)
    for index, row in enumerate(rows):
        top = 142 + index * 46
        _text(output, f"{index + 1}. {row['label']}", (510, top), width=485)
        _text(output, f"{row['score']:.2f}%", (1005, top), width=100)
        cv2.rectangle(output, (510, top + 9), (1094, top + 17), (62, 57, 53), -1)
        length = round(584 * row["score"] / 100)
        if length:
            color = (100, 220, 155) if index == 0 else (210, 160, 80)
            cv2.rectangle(output, (510, top + 9), (509 + length, top + 17), color, -1)
    hidden = len(prediction["probabilities"]) - len(rows)
    footer = (f"Top {len(rows)} shown; {hidden} more in probabilities output" if hidden else
              "Class scores sum to 100%; scores are not calibrated confidence")
    _text(output, footer, (510, height - 22), scale=.44)
    return output
