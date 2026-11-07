"""Filmstrip trainer for the DataflowGUI image classifier."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QIcon, QImageReader, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
)

from .model import PROFILES, Example, load_assignments, stage_classifier, train_classifier

_TRAINERS = ThreadPoolExecutor(max_workers=1, thread_name_prefix="image-classifier")
_THUMBNAIL = QSize(112, 84)
_EXAMPLE = Qt.ItemDataRole.UserRole
_ALL_IMAGES, _HELD_OUT, _MISCLASSIFIED = range(3)


def _thumbnail(path):
    reader = QImageReader(str(path))
    size = reader.size()
    if size.isValid():
        reader.setScaledSize(size.scaled(_THUMBNAIL, Qt.AspectRatioMode.KeepAspectRatio))
    image = reader.read()
    return QIcon(QPixmap.fromImage(image)) if not image.isNull() else QIcon()


class ClassifierTrainer(QDialog):
    def __init__(self, parent, parameters, context):
        super().__init__(parent)
        self.parameters = parameters
        self.project_dir = context.project_dir
        self.candidate = None
        self.future = None
        self.predictions = {}
        self.timer = QTimer(self, interval=60)
        self.timer.timeout.connect(self._training_finished)
        self.setWindowTitle("Image classifier training")
        self.resize(900, 590)

        layout = QVBoxLayout(self)
        instructions = QLabel(
            "Create a class and add its labeled images. Select a class tab to view its filmstrip. "
            "Delete removes selected assignments from that class."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        review = QHBoxLayout()
        review.addWidget(QLabel("Show"))
        self.review_filter = QComboBox()
        self.review_filter.addItems(
            ["All images", "Held-out images", "Misclassified held-out images"]
        )
        self.review_filter.currentIndexChanged.connect(self._filter_images)
        review.addWidget(self.review_filter)
        self.review_counts = QLabel()
        review.addWidget(self.review_counts)
        review.addStretch()
        review.addWidget(QLabel("Training profile"))
        self.profile = QComboBox()
        for key, label in PROFILES.items():
            self.profile.addItem(label, key)
        self.profile.setToolTip(
            "Color and layout uses a random forest. Surface texture compares local appearance "
            "against labeled examples."
        )
        self.profile.currentIndexChanged.connect(self._invalidate)
        review.addWidget(self.profile)
        layout.addLayout(review)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.currentChanged.connect(self._update_review_counts)
        layout.addWidget(self.tabs, 1)
        edits = QHBoxLayout()
        layout.addLayout(edits)
        self.new_button = self._button(edits, "New class…", self._new_class)
        self.rename_button = self._button(edits, "Rename class…", self._rename_class)
        self.drop_button = self._button(edits, "Remove class…", self._remove_class)
        edits.addStretch()
        self.import_button = self._button(edits, "Add images…", self._import_images)
        self.split_button = self._button(edits, "Toggle validation", self._toggle_validation)
        self.split_button.setToolTip(
            "Mark selected images as held out. Mark at least one image per class and "
            "leave at least one per class for training. "
            "With no marked images, splitting is automatic."
        )
        self.move_button = self._button(edits, "Move to class…", self._move_selected)
        self.remove_button = self._button(edits, "Remove selected", self.remove_selected)

        self.status = QLabel("Training health is measured on held-out images.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.report = QTableWidget()
        self.report.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.report.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.report, 1)

        actions = QHBoxLayout()
        layout.addLayout(actions)
        actions.addStretch()
        self.train_button = self._button(actions, "Train", self.start_training)
        self.stage_button = self._button(actions, "Stage trained model", self._stage)
        self._button(actions, "Close", self.reject)
        self.finished.connect(self._stop_polling)

        if parameters["model"]:
            try:
                assignments, health = load_assignments(self.project_dir, parameters["model"])
                for label, samples in assignments.items():
                    self.add_class(label, samples)
                self.profile.setCurrentIndex(self.profile.findData(health.profile))
                self._show_health(health)
            except (KeyError, OSError, TypeError, ValueError) as exc:
                self.status.setText(f"Could not open training assignments. {exc}")
        elif self.project_dir is None:
            self.status.setText("Save the project before staging a trained model.")
        self._refresh_controls()

    @staticmethod
    def _button(layout, title, callback):
        button = QPushButton(title)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def _refresh_controls(self):
        if not hasattr(self, "stage_button"):
            return
        idle = self.future is None
        strip = self.tabs.currentWidget()
        selected = strip is not None and bool(strip.selectedItems())
        self.new_button.setEnabled(idle)
        self.profile.setEnabled(idle)
        self.review_filter.setEnabled(idle and bool(self.predictions))
        for button in (self.rename_button, self.drop_button, self.import_button):
            button.setEnabled(idle and strip is not None)
        for button in (self.split_button, self.remove_button):
            button.setEnabled(idle and selected)
        self.move_button.setEnabled(idle and selected and self.tabs.count() > 1)
        self.train_button.setEnabled(idle and self.tabs.count() >= 2)
        self.stage_button.setEnabled(
            idle and self.candidate is not None and self.project_dir is not None
        )

    def _invalidate(self):
        self.candidate = None
        if self.predictions:
            self.predictions.clear()
            self.review_filter.setCurrentIndex(_ALL_IMAGES)
            self._annotate_filmstrips()
            self._filter_images()
        self.report.clear()
        self.report.setRowCount(0)
        self.report.setColumnCount(0)
        message = "Train to update health. Keep at least two distinct images per class."
        if self.project_dir is None:
            message += " Save the project before staging a model."
        self.status.setText(message)
        self._refresh_controls()

    def _class_name(self, label, *, current=-1):
        label = label.strip()
        names = {
            self.tabs.tabText(i).casefold() for i in range(self.tabs.count()) if i != current
        }
        if not label or label.casefold() in names:
            raise ValueError("Enter a nonempty, unique class label.")
        return label

    def add_class(self, label, examples=()):
        label = self._class_name(label)
        strip = QListWidget()
        strip.setViewMode(QListView.ViewMode.IconMode)
        strip.setFlow(QListView.Flow.LeftToRight)
        strip.setWrapping(False)
        strip.setMovement(QListView.Movement.Static)
        strip.setResizeMode(QListView.ResizeMode.Adjust)
        strip.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        strip.setHorizontalScrollMode(QListView.ScrollMode.ScrollPerPixel)
        strip.setIconSize(_THUMBNAIL)
        strip.setGridSize(QSize(150, 122))
        strip.itemSelectionChanged.connect(self._refresh_controls)
        delete = QShortcut(QKeySequence(Qt.Key.Key_Delete), strip)
        delete.setContext(Qt.ShortcutContext.WidgetShortcut)
        delete.activated.connect(self.remove_selected)
        for example in examples:
            self._append_item(strip, example)
        self.tabs.setCurrentIndex(self.tabs.addTab(strip, label))
        self._invalidate()

    def _new_class(self):
        label, ok = QInputDialog.getText(self, "New class", "Class label")
        if ok:
            try:
                self.add_class(label)
            except ValueError as exc:
                self.status.setText(str(exc))

    def _rename_class(self):
        index = self.tabs.currentIndex()
        previous = self.tabs.tabText(index)
        label, ok = QInputDialog.getText(self, "Rename class", "Class label", text=previous)
        if ok:
            try:
                label = self._class_name(label, current=index)
            except ValueError as exc:
                self.status.setText(str(exc))
                return
            if label != previous:
                self.tabs.setTabText(index, label)
                self._invalidate()

    def _remove_class(self):
        index = self.tabs.currentIndex()
        strip = self.tabs.widget(index)
        self.tabs.removeTab(index)
        strip.deleteLater()
        self._invalidate()

    @staticmethod
    def _set_example(item, example):
        item.setData(_EXAMPLE, example)
        split = "Validation" if example.validation else "Labeled image"
        marker = " · Validation" if example.validation else ""
        item.setText(f"{example.path.name}{marker}")
        item.setToolTip(f"{split}\n{example.path}")

    def _append_item(self, strip, example):
        item = QListWidgetItem(_thumbnail(example.path), "")
        self._set_example(item, example)
        strip.addItem(item)

    def _import_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add images to class", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.pgm *.ppm);;All files (*)",
        )
        self.add_images(paths)

    def add_images(self, paths):
        strip = self.tabs.currentWidget()
        known = {example.path.resolve() for rows in self.assignments().values() for example in rows}
        duplicates = 0
        added = 0
        for path in map(Path, paths):
            path = path.resolve()
            if path in known:
                duplicates += 1
                continue
            self._append_item(strip, Example(path))
            known.add(path)
            added += 1
        if added:
            self._invalidate()
        if duplicates:
            self.status.setText(f"Skipped {duplicates} image path(s) already assigned to a class.")

    def _toggle_validation(self):
        for item in self.tabs.currentWidget().selectedItems():
            example = item.data(_EXAMPLE)
            self._set_example(item, Example(example.path, not example.validation))
        self._invalidate()

    def _move_selected(self):
        current = self.tabs.currentIndex()
        choices = [self.tabs.tabText(i) for i in range(self.tabs.count()) if i != current]
        label, ok = QInputDialog.getItem(
            self, "Move selected images", "Class label", choices, editable=False
        )
        if not ok:
            return
        target = next(i for i in range(self.tabs.count()) if self.tabs.tabText(i) == label)
        source_strip, target_strip = self.tabs.widget(current), self.tabs.widget(target)
        for item in source_strip.selectedItems():
            target_strip.addItem(source_strip.takeItem(source_strip.row(item)))
        self._invalidate()
        self.tabs.setCurrentIndex(target)

    def remove_selected(self):
        strip = self.tabs.currentWidget()
        if self.future is not None or strip is None:
            return
        selected = strip.selectedItems()
        for item in selected:
            strip.takeItem(strip.row(item))
        if selected:
            self._invalidate()

    def assignments(self):
        result = {}
        for index in range(self.tabs.count()):
            strip = self.tabs.widget(index)
            result[self.tabs.tabText(index)] = [
                strip.item(row).data(_EXAMPLE) for row in range(strip.count())
            ]
        return result

    def start_training(self):
        if self.future is not None:
            return
        self._invalidate()
        self.status.setText("Training and evaluating held-out images…")
        self.training_assignments = self.assignments()
        self.future = _TRAINERS.submit(
            train_classifier, self.training_assignments, profile=self.profile.currentData()
        )
        self.timer.start()
        self._refresh_controls()

    def _training_finished(self):
        if not self.future.done():
            return
        self.timer.stop()
        future, self.future = self.future, None
        try:
            model, health = future.result()
        except Exception as exc:
            self.status.setText(f"Training failed. {exc}")
        else:
            self.candidate = self.training_assignments, model, health
            self._show_health(health)
        self._refresh_controls()

    def _stop_polling(self):
        self.timer.stop()
        if self.future is not None:
            self.future.cancel()

    def _show_health(self, health):
        labels = [self.tabs.tabText(i) for i in range(self.tabs.count())]
        self.status.setText(
            f"Training health {health.score:.1f}% · Average class recall · "
            f"{sum(health.validation_counts)} held-out images ({health.split} split)"
        )
        headers = ["Class", "Training", "Held out", "Recall", *[f"Predicted {s}" for s in labels]]
        self.report.setColumnCount(len(headers))
        self.report.setHorizontalHeaderLabels(headers)
        self.report.setRowCount(len(labels))
        for row, label in enumerate(labels):
            cells = (
                label, health.train_counts[row], health.validation_counts[row],
                f"{health.recalls[row]:.1f}%", *health.matrix[row],
            )
            for column, value in enumerate(cells):
                self.report.setItem(row, column, QTableWidgetItem(str(value)))
        self.report.resizeColumnsToContents()
        self.predictions = {
            (row.class_index, row.image_index): row for row in health.predictions
        }
        self._annotate_filmstrips()
        self._filter_images()

    def _annotate_filmstrips(self):
        for class_index in range(self.tabs.count()):
            strip = self.tabs.widget(class_index)
            for index in range(strip.count()):
                item = strip.item(index)
                example = item.data(_EXAMPLE)
                self._set_example(item, example)
                prediction = self.predictions.get((class_index, index))
                if prediction is None:
                    continue
                expected = self.tabs.tabText(class_index)
                predicted = self.tabs.tabText(prediction.predicted_index)
                status = "Correct" if class_index == prediction.predicted_index else "Wrong"
                item.setText(f"{example.path.name} · {status} ({prediction.score:.0f}%)")
                item.setToolTip(
                    f"Held out · {status}\n"
                    f"Expected {expected}\n"
                    f"Predicted {predicted} ({prediction.score:.1f}%)\n"
                    f"Confusion {prediction.confusion:.1f}%\n{example.path}"
                )

    def _filter_images(self):
        mode = self.review_filter.currentIndex()
        for class_index in range(self.tabs.count()):
            strip = self.tabs.widget(class_index)
            for index in range(strip.count()):
                prediction = self.predictions.get((class_index, index))
                visible = mode == _ALL_IMAGES or (
                    prediction is not None and (
                        mode == _HELD_OUT or prediction.predicted_index != class_index
                    )
                )
                item = strip.item(index)
                if not visible:
                    item.setSelected(False)
                item.setHidden(not visible)
        self._update_review_counts()

    def _update_review_counts(self):
        current = self.tabs.currentIndex()
        rows = [row for row in self.predictions.values() if row.class_index == current]
        errors = sum(row.class_index != row.predicted_index for row in rows)
        self.review_counts.setText(f"{len(rows)} held out · {errors} misclassified" if rows else "")
        self._refresh_controls()

    def _stage(self):
        if self.candidate is None or self.project_dir is None:
            return
        try:
            self.parameters["model"] = stage_classifier(self.project_dir, *self.candidate)
        except (OSError, RuntimeError, ValueError) as exc:
            self.status.setText(f"Could not stage the model. {exc}")
            return
        self.accept()
