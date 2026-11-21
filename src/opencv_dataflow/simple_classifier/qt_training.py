"""Filmstrip trainer for the DataflowGUI image classifier."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

import cv2
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QIcon, QImageReader, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..images import opencv_image
from ..plugin_image_picker import PluginImagePicker
from .dataset_audit import audit_assignments
from .graphics import draw_assessment
from .group_review import summarize_groups
from .model import (
    PROFILES,
    Example,
    image_pixels,
    load_assignments,
    load_classifier,
    pixel_digest,
    read_image,
    stage_classifier,
    train_classifier,
)
from .qt_review import SampleReview
from .training_split import GROUP_FIELDS

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
        self._capture_groups = []
        self._focused_group = None
        self._review_model = None
        self._review_window = None
        self._audit = None
        self._dataset_blocked = False
        self._revision = 0
        self._job = None
        self._closing = False
        self._cancelled = Event()
        self.capture_directory = None
        self.capture_paths = {}
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
        self.capture_ids_button = self._button(review, "Capture IDs…", self._edit_capture_ids)
        self.capture_ids_button.setToolTip("Edit specimen and recording IDs for selected images.")
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
        self.capture_panel_button = QPushButton("Capture from graph")
        self.capture_panel_button.setCheckable(True)
        edits.addWidget(self.capture_panel_button)
        self.split_button = self._button(edits, "Toggle validation", self._toggle_validation)
        self.split_button.setToolTip(
            "Mark selected images as held out. Mark at least one image per class and "
            "leave at least one per class for training. "
            "Images with the same capture ID belong together. "
            "With no marked images, splitting is automatic."
        )
        self.move_button = self._button(edits, "Move to class…", self._move_selected)
        self.review_button = self._button(edits, "Review selected…", self._review_selected)
        self.remove_button = self._button(edits, "Remove selected", self.remove_selected)

        self.image_picker = PluginImagePicker(context, self, input_port="image")
        self.image_picker.setVisible(False)
        self.image_picker.captured.connect(self._capture_image)
        self.capture_panel_button.toggled.connect(self.image_picker.setVisible)
        layout.addWidget(self.image_picker)

        self.status = QLabel("Training health is measured on held-out images.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.report = QTableWidget()
        self.report.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.report.horizontalHeader().setStretchLastSection(True)
        self.feedback = QTabWidget()
        self.feedback.addTab(self.report, "Training health")
        check_page = QWidget()
        check_layout = QVBoxLayout(check_page)
        self.audit_summary = QLabel(
            "Check dataset for unreadable images, exact duplicates and capture group overlap."
        )
        self.audit_summary.setWordWrap(True)
        self.audit_summary.setTextFormat(Qt.TextFormat.PlainText)
        check_layout.addWidget(self.audit_summary)
        self.audit_table = QTableWidget(0, 4)
        self.audit_table.setHorizontalHeaderLabels(("Problem", "Class", "Image", "Details"))
        self.audit_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.audit_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.audit_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.audit_table.horizontalHeader().setStretchLastSection(True)
        self.audit_table.currentCellChanged.connect(self._focus_issue)
        self.audit_table.cellClicked.connect(self._focus_issue)
        check_layout.addWidget(self.audit_table)
        self.feedback.addTab(check_page, "Dataset checks")
        self.feedback.addTab(self._group_review_page(), "Capture groups")
        layout.addWidget(self.feedback, 1)

        actions = QHBoxLayout()
        layout.addLayout(actions)
        actions.addStretch()
        self.audit_button = self._button(actions, "Check dataset", self.start_audit)
        self.train_button = self._button(actions, "Train", self.start_training)
        self.stage_button = self._button(actions, "Stage trained model", self._stage)
        self._button(actions, "Close", self.reject)
        self.finished.connect(self._close_training)

        if parameters["model"]:
            try:
                assignments, health = load_assignments(self.project_dir, parameters["model"])
                for label, samples in assignments.items():
                    self.add_class(label, samples)
                self.profile.setCurrentIndex(self.profile.findData(health.profile))
                self._show_health(health)
                self._review_model = load_classifier(self.project_dir, parameters["model"])
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

    def _group_review_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        actions = QHBoxLayout()
        actions.addWidget(QLabel("Group held-out images by"))
        self.group_field = QComboBox()
        for field, label in reversed(GROUP_FIELDS):
            self.group_field.addItem(label, field)
        self.group_field.currentIndexChanged.connect(self._refresh_group_review)
        actions.addWidget(self.group_field)
        actions.addStretch()
        self.focus_group_button = self._button(actions, "Show group images", self._focus_group)
        self.clear_group_button = self._button(actions, "Clear group filter", self._clear_group)
        layout.addLayout(actions)
        self.group_summary = QLabel("Train to review held-out images by capture group.")
        self.group_summary.setWordWrap(True)
        self.group_summary.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.group_summary)
        self.group_table = QTableWidget()
        self.group_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.group_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.group_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.group_table.horizontalHeader().setStretchLastSection(True)
        self.group_table.currentCellChanged.connect(self._refresh_controls)
        self.group_table.cellDoubleClicked.connect(self._focus_group)
        layout.addWidget(self.group_table)
        return page

    def _refresh_controls(self):
        if not hasattr(self, "stage_button"):
            return
        idle = self.future is None
        strip = self.tabs.currentWidget()
        selected = strip is not None and bool(strip.selectedItems())
        self.new_button.setEnabled(idle)
        self.profile.setEnabled(idle)
        self.review_filter.setEnabled(idle and bool(self.predictions))
        for widget in (
            self.rename_button, self.drop_button, self.import_button, self.capture_panel_button,
            self.image_picker,
        ):
            widget.setEnabled(idle and strip is not None)
        for button in (self.split_button, self.remove_button, self.capture_ids_button):
            button.setEnabled(idle and selected)
        self.move_button.setEnabled(idle and selected and self.tabs.count() > 1)
        self.review_button.setEnabled(idle and selected)
        self.train_button.setEnabled(idle and self.tabs.count() >= 2)
        self.audit_button.setEnabled(idle and self.tabs.count() > 0)
        self.audit_table.setEnabled(idle and self._audit is not None)
        self.group_field.setEnabled(idle and bool(self.predictions))
        self.group_table.setEnabled(idle and bool(self._capture_groups))
        current_group = self.group_table.currentRow()
        self.focus_group_button.setEnabled(idle and 0 <= current_group < len(self._capture_groups))
        self.clear_group_button.setEnabled(idle and self._focused_group is not None)
        self.stage_button.setEnabled(
            idle and self.candidate is not None and self.project_dir is not None
            and not self._dataset_blocked
        )

    def _invalidate(self):
        self._revision += 1
        self._audit = None
        self._dataset_blocked = False
        self.audit_summary.setText(
            "Assignments/profile changed. Check dataset again to refresh diagnostics."
        )
        self.candidate = None
        self._review_model = None
        if self._review_window is not None:
            self._review_window.close()
            self._review_window = None
        if self.predictions:
            self.predictions.clear()
            self.review_filter.setCurrentIndex(_ALL_IMAGES)
            self._annotate_filmstrips()
        self.report.clear()
        self.report.setRowCount(0)
        self.report.setColumnCount(0)
        self._refresh_group_review()
        message = "Train to update health. Keep at least two distinct images per class."
        if self.project_dir is None:
            message += " Save the project before staging a model."
        self.status.setText(message)
        self._update_review_counts()
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
        strip.itemDoubleClicked.connect(lambda item: self._review_sample(item))
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
    def _capture_summary(example):
        return "\n".join(
            f"{label} ID · {getattr(example, field) or 'Unspecified'}"
            for field, label in GROUP_FIELDS
        )

    @staticmethod
    def _set_example(item, example):
        item.setData(_EXAMPLE, example)
        split = "Validation" if example.validation else "Labeled image"
        marker = " · Validation" if example.validation else ""
        item.setText(f"{example.name}{marker}")
        item.setToolTip(f"{split}\n{example.name}\n{ClassifierTrainer._capture_summary(example)}\n{example.path}")

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
        known = self._assigned_paths()
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

    def _assigned_paths(self):
        return {example.path.resolve() for rows in self.assignments().values() for example in rows}

    def _capture_image(self, packet):
        strip = self.tabs.currentWidget()
        if strip is None or self.future is not None:
            return
        notice = self.image_picker.status_label
        try:
            pixels = image_pixels(opencv_image(packet.value))
            digest = pixel_digest(pixels)
            path = self.capture_paths.get(digest)
            if path is not None and path in self._assigned_paths():
                notice.setText("This capture is already assigned to a class.")
                return
            if path is None:
                if self.capture_directory is None:
                    self.capture_directory = TemporaryDirectory(prefix="dataflow-classifier-")
                name = f"Capture {len(self.capture_paths) + 1:03d}"
                path = Path(self.capture_directory.name).resolve() / f"{name}.png"
                ok, encoded = cv2.imencode(".png", pixels)
                if not ok:
                    raise ValueError("Could not encode capture")
                path.write_bytes(encoded.tobytes())
                self.capture_paths[digest] = path
        except (OSError, TypeError, ValueError, cv2.error) as exc:
            notice.setText(f"Capture failed. {exc}")
            return
        self._append_item(strip, Example(path, display_name=path.stem))
        self._invalidate()
        height, width = pixels.shape[:2]
        ticket = (packet.payload.get("ticket") or {}).get("value")
        ticket_text = f" · ticket {ticket}" if ticket is not None else ""
        label = self.tabs.tabText(self.tabs.currentIndex())
        notice.setText(f"{path.stem} added to {label} · {width} × {height}{ticket_text}")

    def _toggle_validation(self):
        for item in self.tabs.currentWidget().selectedItems():
            example = item.data(_EXAMPLE)
            self._set_example(item, replace(example, validation=not example.validation))
        self._invalidate()

    def _edit_capture_ids(self):
        strip = self.tabs.currentWidget()
        if self.future is not None or strip is None:
            return
        selected = strip.selectedItems()
        if not selected:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Capture IDs")
        layout = QVBoxLayout(dialog)
        intro = QLabel(
            f"Edit IDs for {len(selected)} selected image(s). Match IDs across related views "
            "and classes. Check a field to replace its value. Clear a checked field when "
            "the ID is unknown."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        fields = []
        for field_name, label in GROUP_FIELDS:
            values = {getattr(item.data(_EXAMPLE), field_name) for item in selected}
            row = QHBoxLayout()
            enabled = QCheckBox(f"{label} ID")
            enabled.setChecked(len(values) == 1)
            entry = QLineEdit(next(iter(values)) if len(values) == 1 else "")
            entry.setAccessibleName(f"{label} ID")
            entry.setPlaceholderText("Mixed values" if len(values) != 1 else "Unspecified")
            entry.setEnabled(enabled.isChecked())
            enabled.toggled.connect(entry.setEnabled)
            row.addWidget(enabled)
            row.addWidget(entry, 1)
            layout.addLayout(row)
            fields.append((field_name, enabled, entry))
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.resize(440, dialog.sizeHint().height())
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        updates = {key: entry.text() for key, enabled, entry in fields if enabled.isChecked()}
        try:
            replacements = [
                replace(item.data(_EXAMPLE), **updates) for item in selected
            ]
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        if any(item.data(_EXAMPLE) != example
               for item, example in zip(selected, replacements, strict=True)):
            for item, example in zip(selected, replacements, strict=True):
                self._set_example(item, example)
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
        if self._audit is not None and not self._audit.ready:
            self.status.setText("Fix the listed dataset problems, then train again.")
            self.feedback.setCurrentIndex(1)
            return
        self._invalidate()
        self.status.setText("Training and evaluating held-out images…")
        self.training_assignments = self.assignments()
        self.future = _TRAINERS.submit(
            train_classifier, self.training_assignments, profile=self.profile.currentData()
        )
        self._job = ("Training", self._revision)
        self.timer.start()
        self._refresh_controls()

    def _training_finished(self):
        if self._closing or self.future is None or not self.future.done():
            return
        self.timer.stop()
        future, self.future = self.future, None
        job, revision = self._job
        self._job = None
        try:
            result = future.result()
            if revision != self._revision:
                self.status.setText(f"{job} result discarded because assignments changed.")
                self._refresh_controls()
                return
        except Exception as exc:
            self.status.setText(f"{job} failed. {exc}")
        else:
            if job == "Dataset check":
                self._show_audit(result)
            else:
                model, health = result
                self.candidate = self.training_assignments, model, health
                self._review_model = model
                self._show_health(health)
                self.feedback.setCurrentIndex(0)
        self._refresh_controls()

    def _close_training(self):
        self._closing = True
        self._cancelled.set()
        if self._review_window is not None:
            self._review_window.close()
        self.timer.stop()
        directory, self.capture_directory = self.capture_directory, None
        if self.future is not None:
            self.future.cancel()
            if directory is not None:
                self.future.add_done_callback(lambda _: directory.cleanup())
        elif directory is not None:
            directory.cleanup()

    def start_audit(self):
        if self.future is not None:
            return
        self._cancelled.clear()
        self._audit = None
        self._dataset_blocked = True
        self._job = ("Dataset check", self._revision)
        self.future = _TRAINERS.submit(
            audit_assignments, self.assignments(), cancelled=self._cancelled.is_set,
        )
        self.audit_summary.setText(
            "Reading current images and checking normalized pixels and capture groups…"
        )
        self.feedback.setCurrentIndex(1)
        self.timer.start()
        self._refresh_controls()

    def _show_audit(self, audit):
        self._audit = audit
        self._dataset_blocked = not audit.ready
        message = ("Dataset checks passed." if audit.ready else
                   f"{len(audit.issues)} problems. Select a row to locate its assignment; "
                   "correct it and recheck.")
        self.audit_summary.setText(
            message + f" Split: {audit.split}. Counts use readable images; "
            "they are provisional until all problems are fixed. Checks do not train or predict."
        )
        titles = dict(unreadable="Unreadable image", duplicate="Duplicate pixels",
                      label_conflict="Label conflict", split_leakage="Split leakage",
                      split="Missing split member", classes="Class setup",
                      group_overlap="Capture group overlap", group_split="Capture group split")
        self.audit_table.setRowCount(len(audit.issues))
        samples = {(sample.class_index, sample.image_index): sample for sample in audit.samples}
        image_names = {
            (class_index, image_index): example.name
            for class_index, examples in enumerate(self.assignments().values())
            for image_index, example in enumerate(examples)
        }
        for row, issue in enumerate(audit.issues):
            sample = samples.get((issue.class_index, issue.image_index))
            label = ("All classes" if issue.class_index is None else
                     audit.classes[issue.class_index].label)
            name = "" if sample is None else image_names[(sample.class_index, sample.image_index)]
            values = (" · ".join(titles[code] for code in issue.codes), label, name, issue.message)
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                if sample is not None:
                    item.setToolTip(sample.error or str(sample.path))
                self.audit_table.setItem(row, column, item)
        self.audit_table.resizeColumnsToContents()
        self.status.setText(message)
        self._update_review_counts()

    def _focus_issue(self, row, *_args):
        if self._audit is None or self.future is not None or not 0 <= row < len(self._audit.issues):
            return
        issue = self._audit.issues[row]
        if issue.class_index is None:
            return
        self.review_filter.setCurrentIndex(_ALL_IMAGES)
        self.tabs.setCurrentIndex(issue.class_index)
        strip = self.tabs.currentWidget()
        strip.clearSelection()
        if issue.image_index is not None:
            item = strip.item(issue.image_index)
            item.setSelected(True)
            strip.scrollToItem(item)
        self.status.setText(issue.message)

    def _review_selected(self):
        strip = self.tabs.currentWidget()
        if strip is not None and strip.selectedItems():
            self._review_sample(strip.selectedItems()[0])

    def _review_sample(self, item):
        if self.future is not None:
            return
        example = item.data(_EXAMPLE)
        expected = self.tabs.tabText(self.tabs.currentIndex())
        description = f"Assigned class: {expected}\n{item.toolTip()}"
        try:
            image = read_image(example.path)
            chart = None
            if self._review_model is not None:
                prediction = self._review_model.predict(image)
                chart = draw_assessment(image, prediction)
                description += (f"\nPredicted {prediction['label']} ({prediction['score']:.2f}%)"
                                f"; confusion {prediction['confusion']:.2f}%")
            else:
                description += "\nTrain current assignments to review candidate scores."
            if self._review_window is not None:
                self._review_window.close()
            self._review_window = SampleReview(self, image, chart, description)
            window = self._review_window

            def forget_window(_):
                if self._review_window is window:
                    self._review_window = None

            window.finished.connect(forget_window)
            self._review_window.show()
        except (OSError, ValueError) as exc:
            self.status.setText(f"Could not review sample. {exc}")

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
        self._refresh_group_review()

    def _refresh_group_review(self):
        self._focused_group = None
        field = self.group_field.currentData()
        label = self.group_field.currentText()
        assigned = tuple(self.assignments().values()) if self.predictions else ()
        self._capture_groups = summarize_groups(assigned, self.predictions, field)
        self.group_table.setCurrentCell(-1, -1)
        self.group_table.setColumnCount(6)
        self.group_table.setHorizontalHeaderLabels(
            (f"{label} ID", "Held out", "Correct", "Errors", "Correct %", "Most common error")
        )
        self.group_table.setRowCount(len(self._capture_groups))
        for row, group in enumerate(self._capture_groups):
            error = ""
            if group.common_error is not None:
                expected, predicted, count = group.common_error
                error = f"{self.tabs.tabText(expected)} → {self.tabs.tabText(predicted)} ({count})"
            cells = (
                group.identifier or "Unspecified ID", len(group.members), group.correct,
                group.errors, f"{100 * group.correct / len(group.members):.1f}%", error,
            )
            for column, value in enumerate(cells):
                self.group_table.setItem(row, column, QTableWidgetItem(str(value)))
        self.group_table.resizeColumnsToContents()
        if self._capture_groups:
            unknown = sum(
                len(group.members) for group in self._capture_groups if not group.identifier
            )
            self.group_summary.setText(
                f"{len(self.predictions)} held-out images · "
                f"{unknown} without a {label.lower()} ID. "
                "Choose a row to show its images across the class tabs."
            )
        elif self.report.rowCount():
            self.group_summary.setText(
                "Per-image predictions are unavailable in this saved report. Train to refresh."
            )
        else:
            self.group_summary.setText("Train to review held-out images by capture group.")
        self._filter_images()

    def _focus_group(self, *_args):
        row = self.group_table.currentRow()
        if self.future is not None or not 0 <= row < len(self._capture_groups):
            return
        self._focused_group = self._capture_groups[row]
        mode = self.review_filter.currentIndex()
        if mode == _ALL_IMAGES or (mode == _MISCLASSIFIED and not self._focused_group.errors):
            self.review_filter.setCurrentIndex(_HELD_OUT)
        else:
            self._filter_images()
        for class_index, image_index in self._focused_group.members:
            strip = self.tabs.widget(class_index)
            item = strip.item(image_index)
            if not item.isHidden():
                self.tabs.setCurrentIndex(class_index)
                strip.clearSelection()
                item.setSelected(True)
                strip.scrollToItem(item)
                break

    def _clear_group(self):
        self._focused_group = None
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
                item.setText(f"{example.name} · {status} ({prediction.score:.0f}%)")
                item.setToolTip(
                    f"Held out · {status}\n"
                    f"Expected {expected}\n"
                    f"Predicted {predicted} ({prediction.score:.1f}%)\n"
                    f"Confusion {prediction.confusion:.1f}%\n"
                    f"{example.name}\n{self._capture_summary(example)}\n{example.path}"
                )

    def _filter_images(self):
        mode = self.review_filter.currentIndex()
        if mode == _ALL_IMAGES:
            self._focused_group = None
        members = (
            None if self._focused_group is None else set(self._focused_group.members)
        )
        for class_index in range(self.tabs.count()):
            strip = self.tabs.widget(class_index)
            for index in range(strip.count()):
                prediction = self.predictions.get((class_index, index))
                visible = mode == _ALL_IMAGES or (
                    prediction is not None and (
                        mode == _HELD_OUT or prediction.predicted_index != class_index
                    )
                )
                visible = visible and (members is None or (class_index, index) in members)
                item = strip.item(index)
                if not visible:
                    item.setSelected(False)
                item.setHidden(not visible)
        self._update_review_counts()

    def _update_review_counts(self):
        current = self.tabs.currentIndex()
        strip = self.tabs.currentWidget()
        assigned = 0 if strip is None else strip.count()
        rows = [row for row in self.predictions.values() if row.class_index == current]
        errors = sum(row.class_index != row.predicted_index for row in rows)
        if self._audit is not None and 0 <= current < len(self._audit.classes):
            counts = self._audit.classes[current]
            text = (f"{assigned} assigned · {counts.training} training · "
                    f"{counts.validation} held out ({self._audit.split} check)")
        else:
            marked = 0 if strip is None else sum(
                strip.item(i).data(_EXAMPLE).validation for i in range(strip.count())
            )
            text = f"{assigned} assigned · {marked} marked validation"
        if rows:
            text += f" · {len(rows)} held out · {errors} misclassified"
        if self._focused_group is not None:
            name = self._focused_group.identifier or "Unspecified ID"
            visible = sum(not strip.item(i).isHidden() for i in range(assigned))
            text += f" · {self.group_field.currentText()} {name} · {visible} shown"
        self.review_counts.setText(text)
        self._refresh_controls()

    def _stage(self):
        if (self._closing or self.future is not None or self._dataset_blocked
                or self.candidate is None or self.project_dir is None):
            return
        try:
            self.parameters["model"] = stage_classifier(self.project_dir, *self.candidate)
        except (OSError, RuntimeError, ValueError) as exc:
            self.status.setText(f"Could not stage the model. {exc}")
            return
        self.accept()
