"""Helper script to profile the SensePi benchmark mode with cProfile."""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
from pathlib import Path
from typing import List


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile SensePi benchmark mode")
    parser.add_argument(
        "--prof-output",
        type=str,
        default="benchmark_profile.prof",
        help="cProfile output file (default: benchmark_profile.prof)",
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
        help="Plot refresh rate in Hz (default: 20)",
    )
    parser.add_argument(
        "--bench-channels",
        type=int,
        default=18,
        help="Number of charts to render (9 or 18)",
    )
    parser.add_argument(
        "--bench-sensors",
        type=int,
        default=3,
        help="Synthetic sensor count (default: 3)",
    )
    parser.add_argument(
        "--bench-log-interval",
        type=float,
        default=1.0,
        help="Seconds between benchmark log entries (default: 1)",
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
        help="Keep GUI open after benchmark completes",
    )
    parser.add_argument(
        "--print-stats",
        action="store_true",
        help="Print the top cumulative functions after profiling",
    )

    args = parser.parse_args()

    # ---- basic validation / fail-fast checks ----
    if args.bench_rate <= 0:
        parser.error("--bench-rate must be positive")

    if args.bench_duration <= 0:
        parser.error("--bench-duration must be positive")

    # The GUI currently supports 9 or 18 charts; keep users inside that set.
    if args.bench_channels not in (9, 18):
        parser.error("--bench-channels must be 9 or 18")

    if args.bench_sensors <= 0:
        parser.error("--bench-sensors must be positive")

    if args.bench_log_interval <= 0:
        parser.error("--bench-log-interval must be positive")

    return args


def _build_gui_argv(args: argparse.Namespace) -> List[str]:
    argv = ["profile_benchmark.py", "--benchmark"]
    argv.extend(["--bench-rate", str(args.bench_rate)])
    argv.extend(["--bench-duration", str(args.bench_duration)])
    argv.extend(["--bench-refresh", str(args.bench_refresh)])
    argv.extend(["--bench-channels", str(args.bench_channels)])
    argv.extend(["--bench-log-interval", str(args.bench_log_interval)])
    argv.extend(["--bench-sensors", str(args.bench_sensors)])
    if args.bench_no_csv:
        argv.append("--bench-no-csv")
    else:
        argv.extend(["--bench-csv", args.bench_csv])
    if args.bench_keep_open:
        argv.append("--bench-keep-open")
    return argv


def _run_gui(argv: List[str]) -> None:
    from sensepi.gui import application

    try:
        application.main(argv)
    except SystemExit as exc:  # Allow the GUI to request exit without killing profiling
        code = exc.code
        if code not in (0, None):
            raise


def main() -> None:
    _ensure_src_on_path()
    args = _parse_args()
    gui_argv = _build_gui_argv(args)
    prof_path = Path(args.prof_output).expanduser().resolve()
    # Ensure the output directory exists if the user passed a nested path
    prof_path.parent.mkdir(parents=True, exist_ok=True)

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        _run_gui(gui_argv)
    finally:
        profiler.disable()

    profiler.dump_stats(str(prof_path))
    print(f"[profile] cProfile stats written to {prof_path}")

    if args.print_stats:
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative").print_stats(20)


if __name__ == "__main__":
    main()
