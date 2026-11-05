"""Ticket Gantt and rolling block concurrency views using Qt graphics, without QtCharts."""

import json
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
)

from .activity import spans
from .tickets import ticket_key
from .timing import CLOCK_ID, mark

COLORS = {
    "pass": "#36b978",
    "fail": "#ed645b",
    "error": "#d76aeb",
    "waiting": "#e4b04b",
    "inactive": "#8393a7",
    "running": "#56b8ee",
    "skipped": "#8393a7",
}


def row_key(source):
    return source.get("logical_path", source["path"])


def ticket_id(ticket):
    return json.dumps(ticket_key(ticket))


class ActivityWindow(QDialog):
    def __init__(self, log, mode="ticket", parent=None):
        super().__init__(parent)
        self.log, self.mode = log, mode
        self.frozen = None
        self.first_tickets = True
        self.setWindowTitle("Ticket activity timeline" if mode == "ticket" else "Block concurrency")
        self.resize(1250, 760)
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.ticket = QComboBox()
        self.ticket.setMinimumWidth(250)
        self.ticket.currentIndexChanged.connect(self.refresh)
        controls.addWidget(self.ticket)
        self.clock = QComboBox()
        self.clock.currentIndexChanged.connect(self.refresh)
        controls.addWidget(self.clock)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter block name or path")
        self.search.textChanged.connect(self.refresh)
        controls.addWidget(self.search)
        self.window = QDoubleSpinBox()
        self.window.setRange(0.1, 600)
        self.window.setValue(10)
        self.window.setSuffix(" s window")
        self.window.valueChanged.connect(self.refresh)
        self.window.setVisible(mode == "concurrency")
        controls.addWidget(self.window)
        self.pause = QPushButton("Pause")
        self.pause.setCheckable(True)
        self.pause.toggled.connect(self.set_paused)
        controls.addWidget(self.pause)
        export = QPushButton("Export events…")
        export.clicked.connect(self.export_events)
        controls.addWidget(export)
        layout.addLayout(controls)
        zoom = QHBoxLayout()
        for title, factor in (("Zoom in", 1.5), ("Zoom out", 1 / 1.5), ("Reset zoom", None)):
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, f=factor: self.zoom(f))
            zoom.addWidget(button)
        zoom.addWidget(QLabel("Drag to pan · select a bar/flag for event details"), 1)
        layout.addLayout(zoom)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setBackgroundBrush(QColor("#152031"))
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.view)
        split.addWidget(self.details)
        split.setSizes([470, 160])
        layout.addWidget(split)
        legend = " · ".join(f"<span style='color:{c}'>{s}</span>" for s, c in COLORS.items())
        layout.addWidget(
            QLabel(legend + " · ◆ diagnostic flag · parallel parts share stacked rows")
        )
        self.scene.selectionChanged.connect(self.show_selected)
        self.timer = QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.bar_count = self.flag_count = 0
        self.refresh()

    def zoom(self, factor):
        if factor is None:
            self.view.resetTransform()
        else:
            current = self.view.transform().m11()
            if 0.2 <= current * factor <= 20:
                self.view.scale(factor, 1)

    def set_paused(self, paused):
        self.frozen = (self.log().snapshot(), mark()) if paused else None
        self.pause.setText("Resume" if paused else "Pause")
        self.refresh()

    def choices(self, snapshot):
        if self.ticket.view().isVisible() or self.clock.view().isVisible():
            return
        selected = self.ticket.currentData()
        tickets = {}
        parent = None
        clocks = []
        for event in snapshot["events"]:
            clock = event["time"]["clock"]
            if clock not in clocks:
                clocks.append(clock)
            for ticket in (*event["tickets"], *event["parent_tickets"]):
                tickets[ticket_id(ticket)] = ticket
            if event["parent_tickets"]:
                parent = ticket_id(event["parent_tickets"][-1])
        self.ticket.blockSignals(True)
        self.ticket.clear()
        self.ticket.addItem("All tickets", None)
        for key, ticket in reversed(list(tickets.items())):
            self.ticket.addItem(
                f"{str(ticket['value'])[:60]} · {ticket['kind']} · {ticket['session'][:12]}", key
            )
            self.ticket.setItemData(
                self.ticket.count() - 1, json.dumps(ticket), Qt.ItemDataRole.ToolTipRole
            )
        index = self.ticket.findData(selected)
        if self.first_tickets and tickets and self.mode == "ticket":
            index = self.ticket.findData(parent) if parent else 1
            self.first_tickets = False
        self.ticket.setCurrentIndex(max(0, index))
        self.ticket.blockSignals(False)
        current = self.clock.currentData()
        self.clock.blockSignals(True)
        self.clock.clear()
        for value in clocks:
            self.clock.addItem(f"Clock {value[:10]}", value)
        index = self.clock.findData(current or CLOCK_ID)
        self.clock.setCurrentIndex(index if index >= 0 else max(0, len(clocks) - 1))
        self.clock.blockSignals(False)

    def refresh(self, *_):
        snapshot, now = self.frozen or (self.log().snapshot(), mark())
        self.choices(snapshot)
        key, clock = self.ticket.currentData(), self.clock.currentData()
        calls = [s for s in spans(snapshot) if s["start"]["clock"] == clock]
        if key is not None:
            calls = [
                s
                for s in calls
                if key in {ticket_id(t) for t in (*s["tickets"], *s["parent_tickets"])}
            ]
        invocations = {s["invocation"] for s in calls}
        events = [e for e in snapshot["events"] if e["time"]["clock"] == clock]
        sources = {row_key(e["source"]): e["source"] for e in events}
        if key is not None:
            events = [
                e
                for e in events
                if e["invocation"] in invocations
                or key in {ticket_id(t) for t in (*e["tickets"], *e["parent_tickets"])}
            ]
            sources = {row_key(e["source"]): e["source"] for e in events}
        domain_events = [e for e in events if e["kind"] != "registered"]
        latest = max((e["time"]["ns"] for e in domain_events), default=now["ns"])
        end = max(latest, now["ns"]) if clock == now["clock"] else latest
        if self.mode == "concurrency":
            start = max(0, end - int(self.window.value() * 1e9))
        else:
            start = min(
                [s["start"]["ns"] for s in calls] + [e["time"]["ns"] for e in domain_events],
                default=latest,
            )
            if not any(s["finish"] is None for s in calls):
                end = max(latest, max((s["finish"]["ns"] for s in calls), default=latest))
        end = max(start + 1_000_000, end)
        calls = [
            s
            for s in calls
            if s["start"]["ns"] <= end and (s["finish"] is None or s["finish"]["ns"] >= start)
        ]
        omitted_bars = max(0, len(calls) - 4000)
        calls = sorted(calls, key=lambda s: s["start"]["ns"])[-4000:]
        query = self.search.text().casefold()
        sources = {k: v for k, v in sources.items() if query in (v["name"] + " " + k).casefold()}
        paths = sorted(sources)[:400]
        grouped = {p: [] for p in paths}
        for span in calls:
            if row_key(span["source"]) in grouped:
                grouped[row_key(span["source"])].append(span)
        self.scene.blockSignals(True)
        self.scene.clear()
        self.bar_count = self.flag_count = 0
        left, width = 280, max(800, self.view.viewport().width() - 310)
        scale = width / (end - start)
        y = 38
        for tick in range(6):
            x = left + width * tick / 5
            ns = start + (end - start) * tick / 5
            text = (
                f"{(ns - start) / 1e6:.1f} ms"
                if self.mode == "ticket"
                else f"{(ns - end) / 1e9:.2f} s"
            )
            label = self.scene.addSimpleText(text)
            label.setBrush(QColor("#d9e6f5"))
            label.setPos(x - label.boundingRect().width() if tick == 5 else x, 0)
        row_positions = {}
        for path in paths:
            lanes = []
            label = self.scene.addSimpleText(sources[path]["name"][:32])
            label.setBrush(QColor("#d9e6f5"))
            label.setPos(2, y)
            label.setToolTip(path)
            row_positions[path] = y
            for span in sorted(grouped[path], key=lambda s: s["start"]["ns"]):
                a, b = (
                    max(start, span["start"]["ns"]),
                    min(end, span["finish"]["ns"] if span["finish"] else end),
                )
                lane = next((i for i, until in enumerate(lanes) if until <= a), len(lanes))
                if lane == len(lanes):
                    lanes.append(b)
                else:
                    lanes[lane] = b
                color = QColor(COLORS.get(span["judgment"], "#8393a7"))
                bar = self.scene.addRect(
                    left + (a - start) * scale,
                    y + 16 * min(lane, 7),
                    max(2, (b - a) * scale),
                    13,
                    QPen(color),
                    QBrush(color),
                )
                self.decorate(
                    bar,
                    span,
                    f"{path}\n{span['judgment']}\n{span['event'].get('duration_ms', 'active')} ms",
                )
                self.bar_count += 1
            height = 26 + min(7, max(0, len(lanes) - 1)) * 16
            self.scene.addLine(
                0, y + height - 3, left + width, y + height - 3, QPen(QColor("#304159"))
            )
            y += height
        flags = [
            e
            for e in events
            if e["kind"] not in ("input", "start", "finish", "registered", "ticket_assigned")
            and start <= e["time"]["ns"] <= end
            and row_key(e["source"]) in row_positions
        ]
        for event in flags[-2000:]:
            path = row_key(event["source"])
            x, row_y = left + (event["time"]["ns"] - start) * scale, row_positions[path]
            color = QColor("#ed645b" if event["kind"] == "exception" else "#f0c25d")
            flag = self.scene.addRect(x - 3, row_y - 3, 6, 6, QPen(color), QBrush(color))
            flag.setTransformOriginPoint(x, row_y)
            flag.setRotation(45)
            flag.setZValue(5)
            self.decorate(flag, event, f"{event['kind']}\n{path}\n{json.dumps(event['data'])}")
            self.flag_count += 1
        self.scene.setSceneRect(0, 0, left + width + 35, max(100, y))
        self.scene.blockSignals(False)
        self.summary.setText(
            f"{self.bar_count} activations · {len(paths)} block rows · {self.flag_count} flags · "
            f"{snapshot['evicted_events']} events evicted · "
            f"Profile: {snapshot.get('capture_profile', 'debug')} · "
            f"{max(0, len(sources) - 400)} rows hidden by display limit. "
            f"{omitted_bars} bars and {max(0, len(flags) - 2000)} flags hidden by display limits. "
            "Blue bars are still active; finish colors show the tool's own result."
        )

    def decorate(self, item, data, tooltip):
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        item.setData(0, data)
        item.setToolTip(tooltip)

    def show_selected(self):
        items = self.scene.selectedItems()
        if items:
            self.details.setPlainText(json.dumps(items[0].data(0), indent=2, ensure_ascii=False))

    def export_events(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export activity", filter="JSON (*.json)")
        if path:
            try:
                snapshot = self.frozen[0] if self.frozen else self.log().snapshot()
                with Path(path).open("x", encoding="utf-8") as stream:
                    stream.write(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
            except (ValueError, OSError) as exc:
                QMessageBox.warning(self, "Export activity", str(exc))

    def showEvent(self, event):
        self.timer.start()
        self.refresh()
        super().showEvent(event)

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)
