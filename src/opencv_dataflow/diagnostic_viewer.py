"""Selection-aware diagnostic list and detail panes for the bottom application dock."""

import json

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFontDatabase
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .diagnostics import COLORS, LEVELS


class DiagnosticViewer(QWidget):
    def __init__(self, log, parent=None):
        super().__init__(parent)
        self.log = log
        self.block = None
        self.frozen = None
        self.revision = None
        self.selected_id = None
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.scope = QLabel("All blocks")
        self.scope.setWordWrap(True)
        controls.addWidget(self.scope, 1)
        controls.addWidget(QLabel("Minimum severity"))
        self.severity = QComboBox()
        self.severity.addItems(LEVELS)
        self.severity.setCurrentText("INFO")
        self.severity.currentIndexChanged.connect(self.refresh)
        controls.addWidget(self.severity)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search messages, parameters, tickets or traceback")
        self.search.textChanged.connect(self.refresh)
        controls.addWidget(self.search, 1)
        self.follow = QCheckBox("Follow latest")
        self.follow.setChecked(True)
        controls.addWidget(self.follow)
        self.pause = QPushButton("Pause")
        self.pause.setCheckable(True)
        self.pause.toggled.connect(self.set_paused)
        controls.addWidget(self.pause)
        layout.addLayout(controls)
        self.summary = QLabel()
        layout.addWidget(self.summary)
        self.table = QTreeWidget()
        self.table.setHeaderLabels(["UTC time", "Severity", "Block", "Ticket", "Message"])
        self.table.setUniformRowHeights(True)
        self.table.setStyleSheet("QTreeWidget { background: #142031; color: #d9e3f0; }")
        self.table.header().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self.show_selected)
        self.tabs = QTabWidget()
        self.details, self.parameters, self.traceback = (QTextEdit() for _ in range(3))
        for widget in (self.parameters, self.traceback):
            widget.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
            widget.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        for title, widget in (
            ("Details", self.details),
            ("Parameters", self.parameters),
            ("Exception", self.traceback),
        ):
            widget.setReadOnly(True)
            self.tabs.addTab(widget, title)
        split = QSplitter()
        split.addWidget(self.table)
        split.addWidget(self.tabs)
        split.setSizes([850, 500])
        layout.addWidget(split)
        self.setMinimumHeight(180)
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.refresh)
        self.refresh()

    def set_block(self, block):
        self.block = block
        self.revision = None
        self.refresh()

    def set_paused(self, paused):
        self.frozen = self.log().snapshot() if paused else None
        self.pause.setText("Resume" if paused else "Pause")
        self.revision = None
        self.refresh()

    def refresh(self, *_):
        log = self.log()
        path = self.block.logger.name if self.block else None
        revision = (
            id(log),
            self.frozen["sequence"] if self.frozen else log.revision(),
            path,
            self.severity.currentText(),
            self.search.text(),
            self.frozen is not None,
        )
        if revision == self.revision:
            return
        self.revision = revision
        snapshot = self.frozen if self.frozen is not None else log.snapshot()
        minimum = LEVELS[self.severity.currentText()]
        query = self.search.text().casefold()
        matching = []
        for event in snapshot["events"]:
            source = event["source"]
            if path is not None and source.get("logical_path", source["path"]) != path:
                continue
            if LEVELS[event["severity"]] < minimum:
                continue
            if query and query not in json.dumps(event, ensure_ascii=False).casefold():
                continue
            matching.append(event)
        current = self.table.currentItem()
        scroll_value = self.table.verticalScrollBar().value()
        selected = current.data(0, Qt.ItemDataRole.UserRole)["id"] if current else None
        self.table.setUpdatesEnabled(False)
        self.table.blockSignals(True)
        self.table.clear()
        for event in matching[-2000:]:
            item = QTreeWidgetItem(
                self.table,
                [
                    event["time"]["utc"],
                    event["severity"],
                    event["source"]["name"],
                    ", ".join(str(t["value"]) for t in event["tickets"])[:100],
                    event["message"].split("\n", 1)[0][:300],
                ],
            )
            item.setData(0, Qt.ItemDataRole.UserRole, event)
            item.setToolTip(2, event["source"]["path"])
            item.setToolTip(4, event["message"])
            for column in range(5):
                item.setForeground(column, QBrush(QColor(COLORS[event["severity"]])))
            if event["id"] == selected:
                self.table.setCurrentItem(item)
        self.table.blockSignals(False)
        self.table.setUpdatesEnabled(True)
        if self.table.currentItem() is None and matching:
            self.table.setCurrentItem(self.table.topLevelItem(self.table.topLevelItemCount() - 1))
        self.show_selected()
        if self.follow.isChecked():
            self.table.scrollToBottom()
        else:
            self.table.verticalScrollBar().setValue(scroll_value)
        self.scope.setText(f"Block: {self.block.name} · {path}" if self.block else "All blocks")
        self.summary.setText(
            f"{min(len(matching), 2000)} displayed / {len(matching)} matching · "
            f"{len(snapshot['events'])} events retained · {snapshot['evicted_events']} evicted. "
            "Filters affect display only; retention limits still apply."
        )

    def show_selected(self):
        item = self.table.currentItem()
        event = item.data(0, Qt.ItemDataRole.UserRole) if item else None
        if event is None:
            self.selected_id = None
            for widget in (self.details, self.parameters, self.traceback):
                widget.clear()
            return
        if event["id"] == self.selected_id:
            return
        self.selected_id = event["id"]
        self.details.setPlainText(
            event["message"]
            + "\n\n"
            + json.dumps(
                {k: v for k, v in event.items() if k not in ("parameters", "exception")},
                indent=2,
                ensure_ascii=False,
            )
        )
        self.parameters.setPlainText(json.dumps(event["parameters"], indent=2, ensure_ascii=False))
        error = event["exception"]
        self.traceback.setPlainText(error["traceback"] if error else "No exception recorded")
        self.tabs.setCurrentIndex(2 if error else 0)

    def showEvent(self, event):
        self.timer.start()
        self.refresh()
        super().showEvent(event)

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)
