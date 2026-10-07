"""Qt application entry point for the SensePi desktop GUI.

This module wires up argument parsing, configures matplotlib for
interactive use, builds the :class:`~sensepi.gui.main_window.MainWindow`, and
starts the Qt event loop. All GUI launches—whether through ``python main.py``
or ``python -m sensepi.gui.application``—flow through ``main()`` here.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import matplotlib as mpl

from PySide6.QtCore import QLoggingCategory
from PySide6.QtWidgets import QApplication, QMainWindow

from .benchmark import BenchmarkDriver, BenchmarkOptions
from .hang_watchdog import HangWatchdog
from .main_window import MainWindow
from ..config.app_config import AppConfig, AppPaths, SensorDefaults

_MPL_CONFIGURED = False


def configure_matplotlib_for_realtime() -> None:
    """
    Apply global Matplotlib tweaks that improve interactive / real-time performance.

    This should be called once before any figures are created.
    """
    global _MPL_CONFIGURED
    if _MPL_CONFIGURED:
        return

    try:
        mpl.style.use("fast")
    except Exception:
        pass

    rc = mpl.rcParams
    rc["path.simplify"] = True
    rc["path.simplify_threshold"] = 0.2
    rc["agg.path.chunksize"] = 10000
    rc["axes.grid"] = True
    rc["figure.autolayout"] = True

    _MPL_CONFIGURED = True


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SensePi GUI controller")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the GUI in synthetic benchmark mode (no Pi connection)",
    )
    parser.add_argument(
        "--bench-rate",
        type=float,
        default=200.0,
        help="Synthetic input rate in Hz (default: 200)",
    )
    parser.add_argument(
        "--bench-duration",
        type=float,
        default=30.0,
        help="Benchmark duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--bench-refresh",
        type=float,
        default=20.0,
        help="Plot refresh rate in Hz when benchmarking (default: 20)",
    )
    parser.add_argument(
        "--bench-channels",
        type=int,
        default=18,
        help="Number of charts to show when benchmarking (9 or 18)",
    )
    parser.add_argument(
        "--bench-log-interval",
        type=float,
        default=1.0,
        help="Seconds between benchmark log snapshots (default: 1.0)",
    )
    parser.add_argument(
        "--bench-sensors",
        type=int,
        default=4,
        help="Synthetic sensor count (default: 4)",
    )
    parser.add_argument(
        "--bench-csv",
        type=str,
        default="benchmark_results.csv",
        help="CSV file for benchmark metrics (default: benchmark_results.csv)",
    )
    parser.add_argument(
        "--bench-no-csv",
        action="store_true",
        help="Skip writing benchmark metrics to CSV",
    )
    parser.add_argument(
        "--bench-keep-open",
        action="store_true",
        help="Keep the GUI open after benchmark completes",
    )
    return parser


def _parse_cli_args(
    argv: list[str],
) -> tuple[argparse.Namespace, list[str]]:
    parser = _build_arg_parser()
    args, qt_args = parser.parse_known_args(argv[1:])
    qt_argv = [argv[0], *qt_args]
    return args, qt_argv


def create_app(
    argv: list[str] | None = None,
    *,
    app_config: AppConfig | None = None,
) -> Tuple[QApplication, QMainWindow]:
    """
    Create the QApplication and main SensePi window.

    Parameters
    ----------
    argv:
        Optional argument list to pass to :class:`QApplication`.

    Returns
    -------
    app:
        The QApplication instance (owned by caller).
    window:
        The main window instance with all tabs set up.
    """
    qt_args = argv if argv is not None else sys.argv
    configure_matplotlib_for_realtime()
    app = QApplication.instance() or QApplication(qt_args)

    # Suppress noisy QObject::connect warnings from QStyleHints and similar internals
    QLoggingCategory.setFilterRules("qt.core.qobject.connect=false")

    window = MainWindow(app_config=app_config)
    return app, window


def main(argv: list[str] | None = None) -> None:
    # Basic logging so that GUI + benchmark messages are visible on the console.
    logging.basicConfig(
        level=logging.ERROR,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    # Also keep warnings/errors on disk: a GUI freeze or crash leaves nothing
    # behind otherwise, and the console is usually gone with the process.
    logs_dir = AppPaths().logs
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(logs_dir / "sensepi.log", encoding="utf-8")
        file_handler.setLevel(logging.WARNING)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s"))
        logging.getLogger().addHandler(file_handler)
        logging.getLogger().setLevel(logging.WARNING)
    except OSError:
        pass

    raw_argv = argv if argv is not None else sys.argv
    args, qt_argv = _parse_cli_args(raw_argv)
    sensor_defaults = SensorDefaults()
    sensors_mapping = sensor_defaults.load()
    sampling_cfg = sensor_defaults.load_sampling_config(sensors_mapping)

    app_config = AppConfig(
        sensor_defaults=sensors_mapping,
        sampling_config=sampling_cfg,
    )
    app, win = create_app(qt_argv, app_config=app_config)
    watchdog = HangWatchdog(logs_dir / "hang_traces.log", timeout_s=5.0, parent=app)
    watchdog.start()

    benchmark_driver: BenchmarkDriver | None = None
    if args.benchmark:
        csv_path = None
        if not args.bench_no_csv:
            csv_path = Path(args.bench_csv).expanduser().resolve()
        benchmark_driver = BenchmarkDriver(
            app=app,
            window=win,
            options=BenchmarkOptions(
                rate_hz=float(args.bench_rate),
                duration_s=float(args.bench_duration),
                refresh_hz=float(args.bench_refresh),
                channel_count=int(max(1, args.bench_channels)),
                log_interval_s=max(0.1, float(args.bench_log_interval)),
                csv_path=csv_path,
                sensor_count=max(1, int(args.bench_sensors)),
                keep_open=bool(args.bench_keep_open),
            ),
        )
        setattr(win, "_benchmark_driver", benchmark_driver)

    win.show()
    if benchmark_driver is not None:
        benchmark_driver.start()
    code = app.exec()
    watchdog.stop()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
