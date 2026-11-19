"""Operator window for saved projects, with no graph editing or live code loading."""

import argparse
import sqlite3
import sys
from collections import deque
from pathlib import Path
from threading import Event, Thread

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from .diagnostics import exception_message
from .operator_runtime import RunOptions, default_history, run
from .ticket_history import recent


class OperatorWindow(QWidget):
    def __init__(self, project, history, plugins=()):
        super().__init__()
        self.history = Path(history)
        self.plugins = plugins
        self.worker = None
        self.closing = False
        self.cancel = Event()
        self.messages = deque(maxlen=200)
        self.setWindowTitle("OpenCV Dataflow Operator")
        self.resize(840, 570)
        form = QFormLayout(self)
        self.project = QLineEdit(str(project or ""))
        self.browse = QPushButton("Open…")
        self.browse.clicked.connect(self.choose_project)
        project_row = QHBoxLayout()
        project_row.addWidget(self.project)
        project_row.addWidget(self.browse)
        form.addRow("Project", project_row)
        self.mode = QComboBox()
        self.mode.addItems(["Run Once", "Run All", "Continuous"])
        form.addRow("Mode", self.mode)
        self.deadline = QSpinBox()
        self.deadline.setRange(1, 7200)
        self.deadline.setValue(RunOptions().cycle_seconds)
        self.deadline.setSuffix(" s")
        form.addRow("Acquisition and processing deadline", self.deadline)
        self.start = QPushButton("Start")
        self.start.clicked.connect(self.start_run)
        self.stop = QPushButton("Stop")
        self.stop.setEnabled(False)
        self.stop.clicked.connect(self.cancel.set)
        buttons = QHBoxLayout()
        buttons.addWidget(self.start)
        buttons.addWidget(self.stop)
        form.addRow(buttons)
        form.addRow("History", QLabel(str(self.history)))
        self.log = QPlainTextEdit(readOnly=True)
        self.log.setMaximumBlockCount(200)
        form.addRow("Run activity", self.log)
        self.results = QPlainTextEdit(readOnly=True)
        form.addRow("Recent ticket results", self.results)
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()

    def choose_project(self):
        name, _ = QFileDialog.getOpenFileName(
            self, "Open trusted project", filter="Project (*.json)"
        )
        if name:
            self.project.setText(name)

    def start_run(self):
        if self.worker is not None:
            return
        project = self.project.text().strip()
        if not project:
            self.log.appendPlainText("Select a saved project.")
            return
        options = RunOptions(
            mode=("once", "all", "continuous")[self.mode.currentIndex()],
            cycle_seconds=self.deadline.value(),
        )
        self.cancel.clear()
        self.set_active(True)

        def execute():
            try:
                run(
                    project,
                    self.history,
                    options=options,
                    plugins=self.plugins,
                    cancelled=self.cancel.is_set,
                    progress=self.messages.append,
                )
            except Exception as exc:
                self.messages.append(exception_message(exc))

        self.worker = Thread(target=execute, name="operator-runtime")
        self.worker.start()

    def set_active(self, active):
        for widget in (self.start, self.project, self.browse, self.mode, self.deadline):
            widget.setEnabled(not active)
        self.stop.setEnabled(active)

    def refresh(self):
        while self.messages:
            self.log.appendPlainText(self.messages.popleft())
        if self.worker is not None and not self.worker.is_alive():
            self.worker.join()
            self.worker = None
            self.set_active(False)
            if self.closing:
                self.close()
                return
        if not self.history.is_file():
            return
        try:
            lines = []
            for row in recent(self.history, 20):
                detail = row["detail"]
                reasons = detail.get("final_fail_reasons", detail.get("fail_reasons", []))
                reason_text = ", ".join(r["code"] for r in reasons)
                elapsed = detail.get("completion", {}).get("elapsed_ms")
                duration = f" · {elapsed:.1f} ms" if elapsed is not None else ""
                reason_text = f" · {reason_text}" if reason_text else ""
                lines.append(
                    f"{row['updated']} · {detail['ticket']['value']} · "
                    f"{row['status'].upper()}{duration}{reason_text}"
                )
            value = "\n".join(lines)
        except sqlite3.Error as exc:
            value = exception_message(exc)
        if value != self.results.toPlainText():
            self.results.setPlainText(value)

    def closeEvent(self, event):
        if self.worker is not None and self.worker.is_alive():
            self.closing = True
            self.cancel.set()
            event.ignore()
            self.log.appendPlainText("Stopping the runtime…")
            return
        super().closeEvent(event)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path, nargs="?")
    parser.add_argument("--history", type=Path, default=default_history())
    parser.add_argument("--plugin", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    app = QApplication(sys.argv[:1])
    window = OperatorWindow(args.project, args.history, args.plugin)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
