from __future__ import annotations

import argparse
from pathlib import Path

from .structural import sonify_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SensePi structural data sonification")
    parser.add_argument("--input", required=True, help="Path to structural CSV file")
    parser.add_argument("--out", required=True, help="Output WAV file")
    parser.add_argument("--mode", choices=["melody", "harmonic"], default="melody")
    parser.add_argument("--joint", type=int, default=28, help="Joint index")

    parser.add_argument("--measurement", default="U1", help="Measurement column for melody mode")
    parser.add_argument("--root-note", type=int, default=60, help="MIDI root note for melody mode")
    parser.add_argument("--note-duration", type=float, default=0.15, help="Seconds per data sample in melody mode")

    parser.add_argument("--r1-measurement", default="R1", help="R1-like column for harmonic spacing modulation")
    parser.add_argument("--u1-measurement", default="U1", help="U1-like column for base frequency modulation")
    parser.add_argument("--base-freq", type=float, default=220.0, help="Base oscillator frequency for harmonic mode")
    parser.add_argument("--num-harmonics", type=int, default=40, help="Number of odd harmonics in harmonic mode")
    parser.add_argument("--harmonic-mod-depth", type=float, default=0.05, help="R1 modulation depth")
    parser.add_argument("--freq-mod-depth", type=float, default=5.0, help="U1 frequency modulation depth")
    parser.add_argument("--sample-rate", type=int, default=None, help="WAV sample rate")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    sample_rate = args.sample_rate or (44100 if args.mode == "melody" else 48000)
    kwargs = {
        "measurement": args.measurement,
        "root_note": args.root_note,
        "note_duration": args.note_duration,
        "r1_measurement": args.r1_measurement,
        "u1_measurement": args.u1_measurement,
        "base_freq": args.base_freq,
        "num_harmonics": args.num_harmonics,
        "harmonic_mod_depth": args.harmonic_mod_depth,
        "freq_mod_depth": args.freq_mod_depth,
        "sample_rate": sample_rate,
    }
    out = sonify_file(
        csv_path=Path(args.input),
        output_path=Path(args.out),
        mode=args.mode,
        joint=args.joint,
        kwargs=kwargs,
    )
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

