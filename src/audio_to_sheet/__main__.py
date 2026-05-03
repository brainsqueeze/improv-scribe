"""
__main__.py — GUI entry point.

Run with:
    python -m audio_to_sheet
"""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from audio_to_sheet.config import AppConfig
from audio_to_sheet.gui.main_window import MainWindow


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
