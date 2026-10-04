"""
gui/score_widget.py — In-app score preview (notation + TAB).

Shows the SVG pages MuseScore renders from the transcription, so the score can
be read in the window as well as exported to PDF. The panel has three states:
a placeholder message, an indeterminate "rendering" indicator, and the
rendered pages (zoomable, scrollable).
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

_ZOOM_LEVELS = [50, 75, 100, 125, 150, 200]
_PAGE_MESSAGE, _PAGE_RENDERING, _PAGE_SCORE = 0, 1, 2


class ScorePanel(QFrame):
    """
    Card showing the rendered score pages.

    Parameters
    ----------
    parent : QWidget | None
        Parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self._pages: list[Path] = []
        self._zoom_index = _ZOOM_LEVELS.index(100)
        self._setup_ui()
        self.show_message("Record and press Stop to render the score.")

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 16)
        root.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("Score preview")
        title.setObjectName("CardTitle")
        self._subtitle = QLabel("notation + TAB")
        self._subtitle.setObjectName("CardSub")
        header.addWidget(title)
        header.addWidget(self._subtitle)
        header.addStretch()

        self._zoom_out = QPushButton("−")
        self._zoom_out.setObjectName("Zoom")
        self._zoom_out.setToolTip("Zoom out")
        self._zoom_out.clicked.connect(lambda: self._step_zoom(-1))
        self._zoom_label = QLabel("100%")
        self._zoom_label.setObjectName("CardSub")
        self._zoom_label.setMinimumWidth(44)
        self._zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._zoom_in = QPushButton("+")
        self._zoom_in.setObjectName("Zoom")
        self._zoom_in.setToolTip("Zoom in")
        self._zoom_in.clicked.connect(lambda: self._step_zoom(1))
        for w in (self._zoom_out, self._zoom_label, self._zoom_in):
            header.addWidget(w)
        root.addLayout(header)

        self._stack = QStackedWidget()
        root.addWidget(self._stack, 1)

        # Page 0: message
        self._message = QLabel()
        self._message.setObjectName("Hint")
        self._message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._message.setWordWrap(True)
        self._stack.addWidget(self._message)

        # Page 1: rendering
        busy = QWidget()
        busy_col = QVBoxLayout(busy)
        busy_col.setAlignment(Qt.AlignmentFlag.AlignCenter)
        busy_title = QLabel("Rendering score…")
        busy_title.setStyleSheet("font-size: 16px; font-weight: 700;")
        busy_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        busy_sub = QLabel("Building notation and tablature with MuseScore")
        busy_sub.setObjectName("Hint")
        busy_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bar = QProgressBar()
        bar.setRange(0, 0)
        bar.setTextVisible(False)
        bar.setFixedWidth(260)
        busy_col.addWidget(busy_title)
        busy_col.addWidget(busy_sub)
        busy_col.addSpacing(8)
        busy_col.addWidget(bar, 0, Qt.AlignmentFlag.AlignCenter)
        self._stack.addWidget(busy)

        # Page 2: scrollable pages
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._pages_host = QWidget()
        self._pages_layout = QVBoxLayout(self._pages_host)
        self._pages_layout.setContentsMargins(20, 20, 20, 20)
        self._pages_layout.setSpacing(20)
        self._pages_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        self._scroll.setWidget(self._pages_host)
        self._stack.addWidget(self._scroll)

        self._set_zoom_enabled(False)

    # ------------------------------------------------------------------
    # State changes
    # ------------------------------------------------------------------

    def show_message(self, text: str) -> None:
        """Show a centred informational message (no score)."""
        self._clear_pages()
        self._message.setText(text)
        self._stack.setCurrentIndex(_PAGE_MESSAGE)
        self._set_zoom_enabled(False)

    def show_rendering(self) -> None:
        """Show the indeterminate 'rendering' indicator."""
        self._clear_pages()
        self._stack.setCurrentIndex(_PAGE_RENDERING)
        self._set_zoom_enabled(False)

    def show_pages(self, pages: list[Path]) -> None:
        """Display the rendered SVG *pages*."""
        self._pages = list(pages)
        self._subtitle.setText(
            f"notation + TAB · {len(pages)} page{'s' if len(pages) != 1 else ''}"
        )
        self._rebuild_pages()
        self._stack.setCurrentIndex(_PAGE_SCORE)
        self._set_zoom_enabled(True)

    # ------------------------------------------------------------------
    # Pages / zoom
    # ------------------------------------------------------------------

    def _clear_pages(self) -> None:
        self._pages = []
        while self._pages_layout.count():
            item = self._pages_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

    def _rebuild_pages(self) -> None:
        while self._pages_layout.count():
            item = self._pages_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        zoom = _ZOOM_LEVELS[self._zoom_index] / 100.0
        for path in self._pages:
            widget = QSvgWidget(str(path))
            default: QSize = QSvgRenderer(str(path)).defaultSize()
            widget.setFixedSize(
                max(1, int(default.width() * zoom)), max(1, int(default.height() * zoom))
            )
            widget.setStyleSheet("background: #fbfaf6; border-radius: 4px;")
            self._pages_layout.addWidget(widget, 0, Qt.AlignmentFlag.AlignHCenter)
        self._zoom_label.setText(f"{_ZOOM_LEVELS[self._zoom_index]}%")

    def _step_zoom(self, delta: int) -> None:
        new = min(max(self._zoom_index + delta, 0), len(_ZOOM_LEVELS) - 1)
        if new != self._zoom_index:
            self._zoom_index = new
            self._rebuild_pages()
            self._set_zoom_enabled(True)

    def _set_zoom_enabled(self, enabled: bool) -> None:
        self._zoom_out.setEnabled(enabled and self._zoom_index > 0)
        self._zoom_in.setEnabled(enabled and self._zoom_index < len(_ZOOM_LEVELS) - 1)
        self._zoom_label.setEnabled(enabled)
