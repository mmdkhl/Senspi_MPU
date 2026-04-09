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


def _read_broken_venv_base_python() -> str | None:
    """Return the missing base interpreter configured in .venv, if detectable."""
    pyvenv_cfg = REPO_ROOT / ".venv" / "pyvenv.cfg"
    if not pyvenv_cfg.exists():
        return None

    for line in pyvenv_cfg.read_text(encoding="utf-8").splitlines():
        if line.startswith("executable = "):
            candidate = line.split("=", 1)[1].strip()
            if candidate:
                try:
                    exists = Path(candidate).exists()
                except OSError:
                    return candidate
                if not exists:
                    return candidate
    return None


def _import_gui_main():
    try:
        from sensepi.gui.application import main as run_gui_main
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise

        lines = [
            "SensePi GUI could not start because 'matplotlib' is not installed",
            f"for the current interpreter: {sys.executable}",
        ]

        broken_base = _read_broken_venv_base_python()
        if broken_base is not None:
            lines.extend(
                [
                    "",
                    "The project virtual environment is also broken:",
                    f"  .venv expects base Python at: {broken_base}",
                    "  That interpreter does not exist on this machine anymore.",
                ]
            )

        lines.extend(
            [
                "",
                "Fix options:",
                "  1. Recreate .venv with a working Python 3.11+ installation.",
                "  2. Install project dependencies:",
                "     python -m pip install -r requirements.txt",
                "     or",
                "     python -m pip install -e .",
            ]
        )
        raise SystemExit("\n".join(lines)) from exc

    return run_gui_main


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
    run_gui_main = _import_gui_main()
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
