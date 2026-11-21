"""Current assignment checks using the classifier's existing decoded-pixel split rules."""

from concurrent.futures import CancelledError
from dataclasses import dataclass, replace
from pathlib import Path

import cv2

from .model import pixel_digest, read_image
from .training_split import plan_split


@dataclass(frozen=True)
class AuditSample:
    class_index: int
    image_index: int
    path: Path
    shape: tuple | None
    fingerprint: str | None
    validation: bool
    error: str | None = None


@dataclass(frozen=True)
class AuditClass:
    label: str
    assigned: int
    readable: int
    training: int
    validation: int


@dataclass(frozen=True)
class AuditIssue:
    codes: tuple[str, ...]
    class_index: int | None
    image_index: int | None
    message: str


@dataclass(frozen=True)
class DatasetAudit:
    split: str
    classes: tuple[AuditClass, ...]
    samples: tuple[AuditSample, ...]
    issues: tuple[AuditIssue, ...]

    @property
    def ready(self):
        return not self.issues


def audit_assignments(assignments, *, cancelled=lambda: False):
    """Read all assignments without extracting features, fitting or predicting.

    Split counts are provisional if any images are invalid or duplicated.
    Group checks use the specimen and recording IDs assigned by the user.
    """

    def check():
        if cancelled():
            raise CancelledError("Dataset check cancelled")

    labels = list(assignments)
    issues, samples, classes = [], [], []
    if len(labels) < 2:
        issues.append(AuditIssue(("classes",), None, None, "Create at least two classes."))
    if any(not label or label != label.strip() for label in labels) or len(
        {label.casefold() for label in labels}
    ) != len(labels):
        issues.append(
            AuditIssue(("classes",), None, None, "Class names must be trimmed and unique.")
        )
    for class_index, (label, examples) in enumerate(assignments.items()):
        for image_index, example in enumerate(examples):
            check()
            try:
                pixels = read_image(example.path)
                fingerprint = pixel_digest(pixels).hex()
                row = AuditSample(
                    class_index,
                    image_index,
                    example.path,
                    tuple(pixels.shape),
                    fingerprint,
                    example.validation,
                )
            except (OSError, ValueError, cv2.error) as error:
                row = AuditSample(
                    class_index,
                    image_index,
                    example.path,
                    None,
                    None,
                    example.validation,
                    str(error),
                )
                issues.append(
                    AuditIssue(
                        ("unreadable",),
                        class_index,
                        image_index,
                        "Cannot read/decode this image. Remove its assignment "
                        "or add a repaired image.",
                    )
                )
            samples.append(row)
            check()
    split = plan_split(assignments, {
        (row.class_index, row.image_index): row.fingerprint
        for row in samples if row.fingerprint is not None
    })
    held_out = [set(indices) for indices in split.validation]
    samples = [
        replace(row, validation=row.image_index in held_out[row.class_index])
        if row.error is None else row
        for row in samples
    ]
    for class_index, (label, examples) in enumerate(assignments.items()):
        training = len(split.training[class_index])
        validation = len(split.validation[class_index])
        classes.append(
            AuditClass(label, len(examples), training + validation, training, validation)
        )
        if not training or not validation:
            issues.append(
                AuditIssue(
                    ("split",),
                    class_index,
                    None,
                    "Each class needs at least one readable training and one held-out image.",
                )
            )
    if split.error:
        issues.append(AuditIssue(("group_split",), None, None, split.error))
    for overlap in split.overlaps:
        check()
        for class_index, image_index in overlap.members:
            issues.append(AuditIssue(("group_overlap",), class_index, image_index, overlap.message))

    groups = {}
    for sample in samples:
        if sample.fingerprint is not None:
            groups.setdefault(sample.fingerprint, []).append(sample)
    for group in groups.values():
        if len(group) < 2:
            continue
        codes = ["duplicate"]
        if len({row.class_index for row in group}) > 1:
            codes.append("label_conflict")
        if len({row.validation for row in group}) > 1:
            codes.append("split_leakage")
        members = "; ".join(
            f"{labels[row.class_index]}/{row.path.name} "
            f"({'held out' if row.validation else 'training'})"
            for row in group[:6]
        )
        if len(group) > 6:
            members += f"; and {len(group) - 6} more assignments"
        for sample in group:
            issues.append(
                AuditIssue(
                    tuple(codes),
                    sample.class_index,
                    sample.image_index,
                    "Identical normalized pixels: " + members,
                )
            )
    check()
    return DatasetAudit(split.mode, tuple(classes), tuple(samples), tuple(issues))
