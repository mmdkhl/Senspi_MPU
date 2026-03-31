from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path

# Make sure the 'src' directory is on sys.path so 'sensepi' can be imported
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sensepi.gui.application import main as run_gui_main


def main(argv: Sequence[str] | None = None) -> None:
    """
    Entry point for the SensePi desktop GUI.

    Parameters
    ----------
    argv:
        Command-line arguments to pass through to the GUI. If None, uses sys.argv.
    """
    if argv is None:
        argv = sys.argv
    # Ensure we pass a list, not a generic Sequence
    run_gui_main(list(argv))


def _run_with_cprofile(argv: Sequence[str] | None = None) -> None:
    """Run the GUI under cProfile and print top cumulative functions."""
    import cProfile
    import io
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        main(argv)
    finally:
        profiler.disable()
        buffer = io.StringIO()
        stats = pstats.Stats(profiler, stream=buffer).sort_stats("cumulative")
        stats.print_stats(50)
        print(buffer.getvalue())


if __name__ == "__main__":
    if os.getenv("SENSEPI_PROFILE", ""):
        _run_with_cprofile(sys.argv)
    else:
        main(sys.argv)
