"""Legacy helper for spawning the standalone plotter process."""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from typing import List, Optional


class LocalPlotRunner:
    """
    Small helper that starts/stops your existing live plotting / FFT script
    (e.g. plotter.py) as a separate process.

    By default it runs:  sys.executable src/sensepi/tools/plotter.py
    in the given project_root.
    """

    def __init__(self, project_root: Path | str, script_name: str = "src/sensepi/tools/plotter.py") -> None:
        self.project_root = Path(project_root).resolve()
        self.script_name = script_name
        self._proc: Optional[subprocess.Popen] = None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, extra_args: Optional[List[str]] = None) -> None:
        """
        Start the local plotting / FFT pipeline if it's not already running.
        extra_args can be used to pass CLI options to plotter.py if you add them later.
        """
        if self.is_running:
            # Already running; nothing to do.
            return

        script_path = self.project_root / self.script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Plot script not found: {script_path}")

        cmd = [sys.executable, str(script_path)]
        if extra_args:
            cmd.extend(extra_args)

        # Launch in the project root so relative paths (e.g. logs/) still work
        self._proc = subprocess.Popen(cmd, cwd=str(self.project_root))

    def stop(self) -> None:
        """
        Stop the plotting process if it is running.
        """
        if not self.is_running:
            self._proc = None
            return

        try:
            self._proc.terminate()
            self._proc.wait(timeout=5.0)
        except Exception:
            # If terminate failed or timed out, force kill
            try:
                self._proc.kill()
            except Exception:
                pass
        finally:
            self._proc = None
