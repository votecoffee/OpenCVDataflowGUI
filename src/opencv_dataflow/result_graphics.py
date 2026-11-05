"""Shared colors and drawing for inspection review graphics."""

import cv2

ACCEPTED = (60, 230, 120)
REJECTED = (60, 90, 240)


def review_label(image, text, point, color):
    cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (15, 15, 15), 3)
    cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def review_legend(image, accepted, rejected):
    review_label(image, f"Accepted: {accepted}", (5, 15), ACCEPTED)
    review_label(image, f"Rejected: {rejected}", (5, 32), REJECTED)
