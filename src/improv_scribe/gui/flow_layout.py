"""
gui/flow_layout.py — A QLayout that wraps its items onto new rows.

Qt has no built-in wrapping layout; this is the standard flow-layout pattern
used so the transport bar reflows on narrow windows instead of clipping.
"""

from __future__ import annotations

from PyQt6.QtCore import QMargins, QPoint, QRect, QSize, Qt
from PyQt6.QtWidgets import QLayout, QLayoutItem, QWidget


class FlowLayout(QLayout):
    """
    Left-to-right layout that wraps onto additional rows when out of width.

    Parameters
    ----------
    parent : QWidget | None
        Widget that owns the layout.
    margin : int
        Outer margin in pixels.
    h_spacing, v_spacing : int
        Horizontal / vertical gap between items.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        margin: int = 0,
        h_spacing: int = 10,
        v_spacing: int = 10,
    ) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._h = h_spacing
        self._v = v_spacing
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._do_layout(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, apply=True)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m: QMargins = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _do_layout(self, rect: QRect, apply: bool) -> int:
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, row_h = area.x(), area.y(), 0
        for item in self._items:
            hint = item.sizeHint()
            w = min(hint.width(), area.width())
            if x + w > area.right() + 1 and row_h > 0:
                x = area.x()
                y += row_h + self._v
                row_h = 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), QSize(w, hint.height())))
            x += w + self._h
            row_h = max(row_h, hint.height())
        return y + row_h - rect.y() + m.bottom()

