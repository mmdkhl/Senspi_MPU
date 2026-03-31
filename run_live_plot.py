"""Standalone demo that drives LivePlot with a synthetic signal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sensepi.config import SensePiConfig, load_config

from live_plot import LivePlot


def fake_decimated_stream(
    chunk_size: int = 2,
    dt: float = 0.02,
    freq_hz: float = 1.2,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Yield synthetic decimated data roughly resembling Plotter output.

    ``dt`` simulates the decimated sampling interval (50 Hz by default).  Each
    chunk provides ``chunk_size`` points so callers can mimic batched updates.
    """
    t = 0.0
    rand = np.random.default_rng()
    while True:
        t_vals = t + np.arange(chunk_size, dtype=float) * dt
        base = np.sin(2.0 * np.pi * freq_hz * t_vals)
        noise = 0.1 * rand.standard_normal(size=chunk_size)
        mean = base + noise
        width = 0.15 + 0.05 * rand.standard_normal(size=chunk_size)
        y_min = mean - np.abs(width)
        y_max = mean + np.abs(width)
        yield t_vals, mean, y_min, y_max
        t = float(t_vals[-1] + dt)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SensePi LivePlot demo")
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional YAML file describing SensePiConfig overrides",
    )
    parser.add_argument(
        "--plot-window",
        type=float,
        help="Override plot_window_seconds without editing the YAML",
    )
    parser.add_argument(
        "--spike-threshold",
        type=float,
        help="Override spike_threshold without editing the YAML",
    )
    parser.add_argument(
        "--plot-fs",
        type=float,
        help="Decimated frequency (Hz) for the synthetic input",
    )
    return parser


def _resolve_config(args: argparse.Namespace) -> SensePiConfig:
    cfg = load_config(args.config) if args.config else SensePiConfig()
    if args.plot_window is not None:
        cfg.plot_window_seconds = float(args.plot_window)
    if args.spike_threshold is not None:
        cfg.spike_threshold = float(args.spike_threshold)
    if args.plot_fs is not None:
        cfg.plot_fs = float(args.plot_fs)
    return cfg.sanitized()


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    cfg = _resolve_config(args)

    lp = LivePlot.from_config(cfg)

    dt = 1.0 / float(cfg.plot_fs)
    stream = fake_decimated_stream(dt=dt)

    def fetch():
        # In production replace with queue-draining logic that pulls PlotUpdate
        # objects from Plotter.queue. Returning ``None`` tells the animation loop
        # that no new data is available for this frame.
        try:
            return next(stream)
        except StopIteration:
            return None

    refresh_hz = max(1.0, min(60.0, cfg.plot_fs))
    interval_ms = max(5, int(round(1000.0 / refresh_hz)))
    lp.start_animation(fetch, interval_ms=interval_ms)
    plt.show()


if __name__ == "__main__":
    main()
