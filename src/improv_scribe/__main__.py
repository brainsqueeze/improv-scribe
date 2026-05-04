"""
__main__.py — GUI entry point.

Run with:
    python -m improv_scribe
"""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from improv_scribe.config import AppConfig
from improv_scribe.gui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Audio to Sheet Music")
    app.setOrganizationName("AudioToSheet")

    config = AppConfig()
    window = MainWindow(config)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
