cd audio_to_sheet

# System deps (macOS)
brew install musescore portaudio

# Python env
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Generate test fixtures (synthetic WAV files for the pitch tests)
python scripts/generate_fixtures.py

# Run the full test suite (80 tests)
pytest

# Launch the GUI
python -m audio_to_sheet