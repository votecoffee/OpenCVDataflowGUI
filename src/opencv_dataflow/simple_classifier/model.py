"""Train and run whole-image classifiers with OpenCV color and texture features."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .surface_texture import SurfaceModel, describe_surface

MODEL_SCHEMA = 1
APPEARANCE = "appearance"
SURFACE_TEXTURE = "surface_texture"
PROFILES = {APPEARANCE: "Color and layout", SURFACE_TEXTURE: "Surface texture"}


@dataclass(frozen=True)
class Example:
    path: Path
    validation: bool = False
    display_name: str = ""

    @property
    def name(self):
        return self.display_name or self.path.name


def image_pixels(image):
    pixels = np.asarray(image)
    if pixels.size == 0:
        raise ValueError("Image is empty")
    if pixels.dtype == np.uint16:
        pixels = np.rint(pixels / 257).astype(np.uint8)
    elif pixels.dtype in (np.float32, np.float64):
        if not np.isfinite(pixels).all():
            raise ValueError("Image contains nonfinite pixel values")
        pixels = np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8)
    elif pixels.dtype != np.uint8:
        raise ValueError(f"Unsupported image pixel depth: {pixels.dtype}")
    if pixels.ndim == 2:
        return cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
    if pixels.ndim == 3 and pixels.shape[2] == 4:
        return cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("Expected grayscale, BGR or BGRA image pixels")
    return pixels


def read_image(path):
    data = Path(path).read_bytes()
    if not data:
        raise ValueError(f"Cannot decode empty image {path}")
    pixels = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if pixels is None:
        raise ValueError(f"Cannot decode image {path}")
    return image_pixels(pixels)


def pixel_digest(pixels):
    """Identify the canonical decoded pixels and their shape."""
    digest = hashlib.sha256(str(pixels.shape).encode())
    digest.update(pixels.tobytes())
    return digest.digest()


def image_features(image):
    pixels = image_pixels(image)
    height, width = pixels.shape[:2]
    scaled = cv2.resize(pixels, (96, 96), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2Lab)
    gray = lab[..., 0]
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.minimum(cv2.magnitude(dx, dy) / 2, 255).astype(np.uint8)
    bands = cv2.merge((*cv2.split(lab), edges))
    features = [math.log(width / height)]
    # Whole-image appearance and coarse spatial layout use the same descriptor.
    quadrants = [bands[y:y + 48, x:x + 48] for y in (0, 48) for x in (0, 48)]
    for region in (bands, bands[24:72, 24:72], *quadrants):
        for channel in cv2.split(region):
            hist = cv2.calcHist([channel], [0], None, [16], [0, 256]).ravel()
            features.extend(hist / channel.size)
            features.extend((channel.mean() / 255, channel.std() / 255))
    return np.asarray(features, dtype=np.float32)


def _project_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or path.anchor or ".." in path.parts:
        raise ValueError("Classifier asset paths must be project-relative")
    path = (root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Classifier asset is outside the project folder")
    return path


def ranked_scores(labels, weights):
    """Normalize class evidence into percentages, preserving ties."""
    total = math.fsum(weights)
    if (
        not math.isfinite(total) or total <= 0 or len(labels) != len(weights)
        or len(labels) < 2 or any(weight < 0 for weight in weights)
    ):
        raise ValueError("Classifier did not produce valid class weights")
    ranked = sorted(range(len(weights)), key=lambda i: (-weights[i], i))
    first, second = ranked[:2]
    confusion = 100.0 * weights[second] / weights[first]
    candidates = [{"label": labels[i], "score": 100.0 * weights[i] / total} for i in ranked]
    return {**candidates[0], "confusion": confusion, "probabilities": candidates}


@dataclass(frozen=True)
class ValidationPrediction:
    class_index: int
    image_index: int
    predicted_index: int
    score: float
    confusion: float


@dataclass(frozen=True)
class TrainingHealth:
    train_counts: list[int]
    validation_counts: list[int]
    matrix: list[list[int]]
    split: str
    predictions: list[ValidationPrediction] = field(default_factory=list)
    profile: str = APPEARANCE

    @property
    def recalls(self):
        return [100.0 * row[i] / sum(row) for i, row in enumerate(self.matrix)]

    @property
    def score(self):
        return sum(self.recalls) / len(self.matrix)


class Classifier:
    def __init__(self, labels, forest=None, *, surface=None):
        self.labels = tuple(labels)
        self.forest = forest
        self.surface = surface
        self.profile = SURFACE_TEXTURE if surface is not None else APPEARANCE

    def _weights(self, features):
        if self.surface is not None:
            return self.surface.weights(features)
        votes = self.forest.getVotes(features, 0)
        ids = votes[0].tolist()
        if sorted(ids) != list(range(len(self.labels))):
            raise ValueError("Classifier model classes do not match its manifest")
        return votes[1:, np.argsort(ids)]

    def predict(self, image):
        row = _features(image, self.profile)[None, :]
        return ranked_scores(self.labels, self._weights(row)[0].tolist())


def _features(image, profile):
    if profile == SURFACE_TEXTURE:
        return describe_surface(image_pixels(image))
    return image_features(image)


def _fit_classifier(labels, features, class_ids, profile):
    if profile == SURFACE_TEXTURE:
        return Classifier(labels, surface=SurfaceModel.train(features, class_ids, len(labels)))
    forest = cv2.ml.RTrees_create()
    forest.setMaxDepth(16)
    forest.setMinSampleCount(1)
    forest.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 0))
    if not forest.train(features, cv2.ml.ROW_SAMPLE, class_ids):
        raise ValueError("OpenCV could not train the classifier")
    return Classifier(labels, forest)


def train_classifier(assignments, *, profile=APPEARANCE):
    """Fit on training images and measure balanced accuracy on held-out images."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown classifier profile: {profile}")
    labels = list(assignments)
    if len(labels) < 2 or len({label.casefold() for label in labels}) != len(labels):
        raise ValueError("Create at least two uniquely named classes")
    if any(not label or label != label.strip() for label in labels):
        raise ValueError("Class labels must be nonempty and trimmed")
    explicit = any(example.validation for samples in assignments.values() for example in samples)
    seen = {}
    train_x, train_y, val_x, val_y = [], [], [], []
    val_image_indices = []
    train_counts, val_counts = [], []
    for index, (label, samples) in enumerate(assignments.items()):
        rows = []
        for image_index, example in enumerate(samples):
            pixels = read_image(example.path)
            key = pixel_digest(pixels)
            if key in seen:
                raise ValueError(f"Duplicate image assignment in {label!r} and {seen[key]!r}")
            seen[key] = label
            rows.append((key, example.validation, image_index, _features(pixels, profile)))
        if explicit:
            train = [row for row in rows if not row[1]]
            valid = [row for row in rows if row[1]]
        else:
            holdout = max(1, len(rows) // 5)
            ordered = sorted(rows, key=lambda row: row[0])
            valid, train = ordered[:holdout], ordered[holdout:]
        if not train or not valid:
            raise ValueError(f"{label!r} needs at least one training and one validation image")
        train_counts.append(len(train))
        val_counts.append(len(valid))
        train_x.extend(row[3] for row in train)
        train_y.extend([index] * len(train))
        val_x.extend(row[3] for row in valid)
        val_y.extend([index] * len(valid))
        val_image_indices.extend(row[2] for row in valid)

    model = _fit_classifier(labels, np.asarray(train_x), np.asarray(train_y, np.int32), profile)
    weights = model._weights(np.asarray(val_x))
    predicted_indices = weights.argmax(axis=1).tolist()
    matrix = np.zeros((len(labels), len(labels)), np.int32)
    np.add.at(matrix, (val_y, predicted_indices), 1)
    predictions = []
    for actual, image_index, predicted, row in zip(
        val_y, val_image_indices, predicted_indices, weights.tolist(), strict=True
    ):
        scores = ranked_scores(labels, row)
        predictions.append(ValidationPrediction(
            actual, image_index, predicted, scores["score"], scores["confusion"]
        ))
    health = TrainingHealth(
        train_counts, val_counts, matrix.tolist(), "explicit" if explicit else "automatic",
        predictions, profile,
    )
    return model, health


def stage_classifier(project_dir, assignments, model, health, *, revision=None):
    """Write an immutable, project-relative model and its training assignments."""
    root = Path(project_dir).resolve()
    if tuple(assignments) != model.labels:
        raise ValueError("Classifier labels do not match the training assignments")
    if health.profile != model.profile:
        raise ValueError("Classifier profile does not match the training report")
    base = _project_path(root, "assets/image_classification")
    base.mkdir(parents=True, exist_ok=True)
    target = _project_path(root, f"assets/image_classification/{revision or uuid.uuid4().hex}")
    if target.parent != base or target.exists():
        raise ValueError("Classifier revision already exists or has an invalid name")
    with tempfile.TemporaryDirectory(dir=base) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        classes = []
        for label, samples in assignments.items():
            rows = []
            for example in samples:
                source = example.path.resolve()
                if source.is_relative_to(root) and not source.is_relative_to(base):
                    relative = source.relative_to(root)
                else:
                    suffix = source.suffix.lower()
                    content = source.read_bytes()
                    name = hashlib.sha256(content).hexdigest() + suffix
                    stored = bundle / "images" / name
                    stored.parent.mkdir(exist_ok=True)
                    stored.write_bytes(content)
                    relative = (target / "images" / name).relative_to(root)
                row = {"path": relative.as_posix(), "validation": example.validation}
                if example.display_name:
                    row["display_name"] = example.display_name
                rows.append(row)
            classes.append({"label": label, "images": rows})
        if model.surface is not None:
            model.surface.save(bundle / "surface.npz")
        else:
            # Memory mode keeps Unicode paths out of OpenCV's native file APIs.
            storage = cv2.FileStorage(
                "forest.yml", cv2.FILE_STORAGE_WRITE | cv2.FILE_STORAGE_MEMORY
            )
            model.forest.write(storage, "forest")
            (bundle / "forest.yml").write_text(storage.releaseAndGetString(), encoding="utf-8")
        report = asdict(health)
        profile = report.pop("profile")
        manifest = {
            "schema": MODEL_SCHEMA, "profile": profile, "classes": classes, "health": report
        }
        (bundle / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        bundle.rename(target)
    return (target / "manifest.json").relative_to(root).as_posix()


def read_manifest(project_dir, manifest):
    root = Path(project_dir).resolve()
    path = _project_path(root, manifest)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["schema"] != MODEL_SCHEMA:
        raise ValueError("Unsupported classifier model schema")
    data.setdefault("profile", APPEARANCE)
    if data["profile"] not in PROFILES:
        raise ValueError(f"Unknown classifier profile: {data['profile']}")
    return path, root, data


def load_assignments(project_dir, manifest):
    _, root, data = read_manifest(project_dir, manifest)
    assignments = {
        entry["label"]: [
            Example(
                _project_path(root, row["path"]), row["validation"], row.get("display_name", "")
            )
            for row in entry["images"]
        ]
        for entry in data["classes"]
    }
    health = data["health"]
    predictions = [ValidationPrediction(**row) for row in health.get("predictions", [])]
    return assignments, TrainingHealth(**{
        **health, "predictions": predictions, "profile": data["profile"],
    })


def load_classifier(project_dir, manifest):
    path, _, data = read_manifest(project_dir, manifest)
    labels = [entry["label"] for entry in data["classes"]]
    if data["profile"] == SURFACE_TEXTURE:
        surface = SurfaceModel.load(path.parent / "surface.npz", len(labels))
        return Classifier(labels, surface=surface)
    text = (path.parent / "forest.yml").read_text(encoding="utf-8")
    storage = cv2.FileStorage(text, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    forest = cv2.ml.RTrees_create()
    try:
        forest.read(storage.getNode("forest"))
    finally:
        storage.release()
    if not forest.isTrained():
        raise ValueError("Classifier model has no trained forest")
    return Classifier(labels, forest)
