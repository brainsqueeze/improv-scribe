"""
gui/theme.py — Dark theme: palette, Qt stylesheet and icon helpers.

All colours used by the GUI live here so the widgets (including the pyqtgraph
plots) stay visually consistent. Call :func:`apply_theme` once on the
``QApplication`` before showing the main window.
"""

# ruff: noqa: E501  (long QSS lines)

from __future__ import annotations

import tempfile
from pathlib import Path

from PyQt6.QtCore import QByteArray, Qt
from PyQt6.QtGui import QIcon, QImage, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QApplication

# -- Palette ---------------------------------------------------------------
BG = "#0d0f1c"
SURFACE = "#141829"
SURFACE_ALT = "#1b1f33"
PLOT_BG = "#0f1224"
BORDER = "#242946"
BORDER_STRONG = "#2a2f4a"
TEXT = "#eef0ff"
TEXT_MUTED = "#8a90b4"
TEXT_DIM = "#6b7197"
ACCENT = "#7b6cff"
ACCENT_LIGHT = "#8f8bff"
CYAN = "#2fd4ff"
RECORD = "#e0344d"
RECORD_LIGHT = "#ff5a6e"
OK = "#3fd6a0"
WARN = "#ffc857"

_ICON_PATHS = {
    "mic": '<path d="M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3Z"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/>',
    "guitar": '<path d="m14 4 6 6M13 8l3 3M9.5 11.5a4 4 0 0 0-5 5 3 3 0 0 0 3 3 4 4 0 0 0 5-5l5-8-8 5Z"/>',
    "wave": '<path d="M3 12h2M7 7v10M11 4v16M15 8v8M19 10v4M21 12h0"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "pdf": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z"/><path d="M14 3v5h5M9 14h6M9 17h4"/>',
    "midi": '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M8 5v8M12 5v8M16 5v8"/>',
    "stop": '<rect x="6" y="6" width="12" height="12" rx="2.5" fill="currentColor"/>',
    "record": '<circle cx="12" cy="12" r="6" fill="currentColor"/>',
    "chevron": '<path d="m6 9 6 6 6-6"/>',
}


def _svg(name: str, color: str) -> bytes:
    body = _ICON_PATHS[name].replace("currentColor", color)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round">{body}</svg>'
    ).encode()


def make_pixmap(name: str, color: str = TEXT, size: int = 18) -> QPixmap:
    """Render a named line icon to a HiDPI-aware pixmap."""
    scale = 2
    image = QImage(size * scale, size * scale, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    QSvgRenderer(QByteArray(_svg(name, color))).render(painter)
    painter.end()
    pix = QPixmap.fromImage(image)
    pix.setDevicePixelRatio(scale)
    return pix


def make_icon(name: str, color: str = TEXT, size: int = 18) -> QIcon:
    """Build a QIcon from a named line icon (normal + muted disabled state)."""
    icon = QIcon()
    icon.addPixmap(make_pixmap(name, color, size), QIcon.Mode.Normal)
    icon.addPixmap(make_pixmap(name, TEXT_DIM, size), QIcon.Mode.Disabled)
    return icon


def _write_chevron() -> str:
    path = Path(tempfile.gettempdir()) / "ats_chevron.svg"
    path.write_bytes(_svg("chevron", TEXT_DIM))
    return path.as_posix()


def build_stylesheet() -> str:
    """Return the application-wide Qt stylesheet."""
    chevron = _write_chevron()
    return f"""
* {{ font-family: "Helvetica Neue", "SF Pro Text", sans-serif; }}
QMainWindow, QWidget#Root, QDialog, QMessageBox {{ background: {BG}; color: {TEXT}; }}
QLabel {{ color: {TEXT}; background: transparent; }}
QToolTip {{ background: {SURFACE_ALT}; color: {TEXT}; border: 1px solid {BORDER_STRONG}; padding: 4px 6px; }}

QFrame#Field {{ background: {SURFACE_ALT}; border: 1px solid {BORDER_STRONG}; border-radius: 12px; }}
QFrame#Field[disabledField="true"] {{ background: #171a2c; }}
QLabel#FieldLabel {{ color: #7a80a6; font-size: 10px; font-weight: 700; letter-spacing: 1px; }}
QFrame#Field QComboBox {{
    background: transparent; border: 0; color: {TEXT}; font-size: 14px; font-weight: 600;
    padding: 0 18px 0 0; min-height: 22px;
}}
QFrame#Field QComboBox:disabled {{ color: {TEXT_DIM}; }}
QComboBox::drop-down {{ border: 0; width: 18px; subcontrol-origin: padding; subcontrol-position: center right; }}
QComboBox::down-arrow {{ image: url({chevron}); width: 14px; height: 14px; }}
QComboBox QAbstractItemView {{
    background: {SURFACE_ALT}; color: {TEXT}; border: 1px solid {BORDER_STRONG};
    selection-background-color: {ACCENT}; selection-color: white; outline: 0; padding: 4px;
}}

QPushButton {{
    background: #23284a; color: {TEXT}; border: 1px solid #3a4066; border-radius: 12px;
    padding: 0 14px; min-height: 48px; font-size: 14px; font-weight: 600;
}}
QPushButton:hover {{ background: #2b315a; }}
QPushButton:disabled {{ color: {TEXT_DIM}; background: transparent; border-color: {BORDER_STRONG}; }}
QPushButton:focus {{ border-color: {ACCENT_LIGHT}; }}
QPushButton#Record {{
    color: white; font-size: 15px; font-weight: 700; padding: 0 20px; border: 0;
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {RECORD_LIGHT}, stop:1 {RECORD});
}}
QPushButton#Record:hover {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #ff7283, stop:1 #ea3f58); }}
QPushButton#Record:disabled {{ background: #3a2a3a; color: #d99aa6; }}
QPushButton#Export:enabled {{
    border: 1px solid {ACCENT_LIGHT};
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {ACCENT}, stop:1 #5b4fe6);
}}
QPushButton#Export:enabled:hover {{ background: #6c5cf5; }}

QPushButton#Seg {{
    min-height: 32px; padding: 0 16px; border: 0; border-radius: 9px; background: transparent;
    color: {TEXT_MUTED}; font-size: 13px; font-weight: 700;
}}
QPushButton#Seg:checked {{ background: #2a2f55; color: white; border: 1px solid #3d4478; }}
QPushButton#Seg:disabled {{ color: #4d5378; background: transparent; border: 0; }}
QFrame#SegBar {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px; }}
QPushButton#Zoom {{ min-height: 30px; min-width: 30px; max-width: 30px; padding: 0; border-radius: 8px; background: {SURFACE_ALT}; border: 1px solid {BORDER_STRONG}; font-size: 16px; }}

QFrame#Card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px; }}
QLabel#CardTitle {{ font-size: 13px; font-weight: 700; }}
QLabel#CardSub {{ font-size: 12px; color: #7a80a6; }}
QLabel#Badge {{ font-size: 11px; font-weight: 600; letter-spacing: 1px; padding: 3px 10px; border-radius: 10px; }}
QLabel#Badge[kind="live"] {{ color: #ff8a98; background: rgba(255,90,110,0.14); }}
QLabel#Badge[kind="idle"] {{ color: #8a90b4; background: #1d2138; }}
QLabel#Hint {{ color: #7a80a6; font-size: 13px; }}

QSplitter::handle {{ background: transparent; }}
QSplitter::handle:vertical {{ height: 14px; }}
QStatusBar {{ padding: 0 14px; background: #0a0c18; color: #dfe2ff; border-top: 1px solid #1b1f36; font-size: 13px; min-height: 30px; }}
QStatusBar QLabel {{ color: {TEXT_DIM}; font-family: Menlo, monospace; font-size: 12px; }}
QStatusBar::item {{ border: 0; }}

QScrollArea {{ border: 0; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 12px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #2f3556; border-radius: 4px; min-height: 30px; }}
QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #2f3556; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QProgressBar {{ background: #232842; border: 0; border-radius: 3px; max-height: 6px; min-height: 6px; text-align: center; }}
QProgressBar::chunk {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {ACCENT}, stop:1 {CYAN}); border-radius: 3px; }}

QMessageBox QPushButton {{ min-height: 30px; min-width: 72px; }}
"""


def style_plot(plot) -> None:  # noqa: ANN001 - pyqtgraph PlotWidget
    """Give a pyqtgraph PlotWidget the dark theme's axis and frame styling."""
    plot.setFrameShape(plot.Shape.NoFrame)
    item = plot.getPlotItem()
    for axis_name in ("left", "bottom"):
        axis = item.getAxis(axis_name)
        axis.setPen(BORDER_STRONG)
        axis.setTextPen(TEXT_DIM)
        axis.enableAutoSIPrefix(False)
    item.hideButtons()


def apply_theme(app: QApplication) -> None:
    """Apply the Fusion base style and the dark stylesheet to *app*."""
    app.setStyle("Fusion")
    app.setStyleSheet(build_stylesheet())
