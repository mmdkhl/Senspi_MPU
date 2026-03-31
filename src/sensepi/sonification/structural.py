from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import interpolate
from scipy.io import wavfile
from scipy.signal import savgol_filter


def _to_float(value: str) -> float:
    text = value.strip().replace(",", ".")
    if text == "":
        return float("nan")
    return float(text)


def _midi_to_freq(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


@dataclass(frozen=True)
class StructuralDataset:
    """Structured view of semicolon-separated structural response CSV data."""

    units: dict[str, str]
    measurements: list[str]
    by_joint: dict[int, dict[str, np.ndarray]]

    def joints(self) -> list[int]:
        return sorted(self.by_joint.keys())

    def get_measurement(self, joint: int, measurement: str) -> np.ndarray:
        if joint not in self.by_joint:
            raise KeyError(f"Unknown joint: {joint}")
        if measurement not in self.by_joint[joint]:
            raise KeyError(f"Unknown measurement '{measurement}' for joint {joint}")
        return self.by_joint[joint][measurement]

    def get_time_series(self, joint: int, time_column: str = "Time") -> np.ndarray:
        return self.get_measurement(joint, time_column)


def load_structural_csv(csv_path: str | Path) -> StructuralDataset:
    """
    Load structural data with header + units row.

    Expected format: first row columns, second row units, then data rows.
    """
    path = Path(csv_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ";"

        reader = csv.reader(handle, delimiter=delimiter)
        rows = list(reader)

    if len(rows) < 3:
        raise ValueError("CSV must contain header row, units row, and at least one data row")

    headers = [h.strip() for h in rows[0]]
    units = {headers[i]: rows[1][i].strip() if i < len(rows[1]) else "" for i in range(len(headers))}

    if "Joint" not in headers:
        raise ValueError("CSV must contain a 'Joint' column")
    joint_index = headers.index("Joint")

    joint_data: dict[int, dict[str, list[float]]] = {}
    measurement_columns = [h for h in headers if h and h != "Joint"]

    for row in rows[2:]:
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) < len(headers):
            row = row + [""] * (len(headers) - len(row))

        joint = int(round(_to_float(row[joint_index])))
        if joint not in joint_data:
            joint_data[joint] = {m: [] for m in measurement_columns}

        for idx, column in enumerate(headers):
            if column == "Joint" or column == "":
                continue
            try:
                joint_data[joint][column].append(_to_float(row[idx]))
            except ValueError:
                joint_data[joint][column].append(float("nan"))

    by_joint = {joint: {k: np.asarray(v, dtype=np.float64) for k, v in values.items()} for joint, values in joint_data.items()}
    return StructuralDataset(units=units, measurements=measurement_columns, by_joint=by_joint)


def generate_tone(frequency: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    tone = np.sin(2 * np.pi * frequency * t) + 0.3 * np.sin(2 * np.pi * frequency * 2 * t)
    fade = min(max(1, int(0.05 * sample_rate)), max(1, tone.shape[0] // 2))
    tone[:fade] *= np.linspace(0.0, 1.0, fade)
    tone[-fade:] *= np.linspace(1.0, 0.0, fade)
    return tone * 0.3


def build_melody_audio(
    dataset: StructuralDataset,
    joint: int,
    measurement: str = "U1",
    root_note: int = 60,
    note_duration: float = 0.15,
    sample_rate: int = 44100,
) -> np.ndarray:
    values = dataset.get_measurement(joint, measurement)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError(f"No finite values found for joint={joint}, measurement={measurement}")

    min_v = float(np.min(finite_values))
    max_v = float(np.max(finite_values))
    if np.isclose(min_v, max_v):
        normalized = np.zeros_like(values)
    else:
        normalized = 2.0 * (values - min_v) / (max_v - min_v) - 1.0

    midi_notes = np.clip(root_note + (normalized * 12).astype(int), 48, 84)
    chunks = [generate_tone(_midi_to_freq(int(note)), note_duration, sample_rate) for note in midi_notes]
    if not chunks:
        return np.array([], dtype=np.float32)
    audio = np.concatenate(chunks).astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def _smooth_modulation(signal: np.ndarray, source_time: np.ndarray, audio_time: np.ndarray) -> np.ndarray:
    spline = interpolate.CubicSpline(source_time, signal, bc_type="natural")
    interp = spline(audio_time)
    window = min(101, max(5, len(interp) // 10))
    if window % 2 == 0:
        window += 1
    if window < len(interp):
        interp = savgol_filter(interp, window_length=window, polyorder=3)
    return interp


def build_harmonic_audio(
    dataset: StructuralDataset,
    joint: int,
    r1_measurement: str = "R1",
    u1_measurement: str | None = "U1",
    base_freq: float = 220.0,
    num_harmonics: int = 40,
    harmonic_mod_depth: float = 0.05,
    freq_mod_depth: float = 5.0,
    sample_rate: int = 48000,
) -> np.ndarray:
    r1_data = dataset.get_measurement(joint, r1_measurement)
    time_data = dataset.get_time_series(joint)

    r1_min, r1_max = np.nanmin(r1_data), np.nanmax(r1_data)
    r1_norm = np.zeros_like(r1_data) if np.isclose(r1_min, r1_max) else 2 * (r1_data - r1_min) / (r1_max - r1_min) - 1

    u1_norm: np.ndarray | None = None
    if u1_measurement is not None:
        u1_data = dataset.get_measurement(joint, u1_measurement)
        u1_min, u1_max = np.nanmin(u1_data), np.nanmax(u1_data)
        u1_norm = np.zeros_like(u1_data) if np.isclose(u1_min, u1_max) else 2 * (u1_data - u1_min) / (u1_max - u1_min) - 1

    time_norm = time_data - time_data[0]
    duration = float(time_norm[-1]) if time_norm.size else 0.0
    num_samples = int(max(1, duration * sample_rate))
    audio_time = np.linspace(0.0, duration, num_samples, endpoint=False)

    r1_interp = _smooth_modulation(r1_norm, time_norm, audio_time)
    u1_interp = _smooth_modulation(u1_norm, time_norm, audio_time) if u1_norm is not None else np.zeros_like(r1_interp)

    audio = np.zeros(num_samples, dtype=np.float64)
    for harmonic in range(1, num_harmonics + 1, 2):
        mod_base = base_freq + freq_mod_depth * u1_interp
        spacing = 1.0 + (r1_interp * harmonic_mod_depth)
        harmonic_freq = mod_base * harmonic * spacing
        amp = (0.3 / harmonic) * 1.5
        audio += amp * np.sin(2 * np.pi * harmonic_freq * audio_time)

    fade = min(max(1, int(0.1 * sample_rate)), max(1, audio.shape[0] // 2))
    audio[:fade] *= np.linspace(0.0, 1.0, fade)
    audio[-fade:] *= np.linspace(1.0, 0.0, fade)
    return np.clip(audio.astype(np.float32), -1.0, 1.0)


def write_wav(output_path: str | Path, audio: np.ndarray, sample_rate: int) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    wavfile.write(out, sample_rate, pcm)
    return out


def sonify_file(
    csv_path: str | Path,
    output_path: str | Path,
    mode: str,
    joint: int,
    kwargs: dict[str, Any],
) -> Path:
    dataset = load_structural_csv(csv_path)
    if mode == "melody":
        sample_rate = int(kwargs.get("sample_rate", 44100))
        audio = build_melody_audio(
            dataset,
            joint=joint,
            measurement=str(kwargs.get("measurement", "U1")),
            root_note=int(kwargs.get("root_note", 60)),
            note_duration=float(kwargs.get("note_duration", 0.15)),
            sample_rate=sample_rate,
        )
    elif mode == "harmonic":
        sample_rate = int(kwargs.get("sample_rate", 48000))
        audio = build_harmonic_audio(
            dataset,
            joint=joint,
            r1_measurement=str(kwargs.get("r1_measurement", "R1")),
            u1_measurement=kwargs.get("u1_measurement", "U1"),
            base_freq=float(kwargs.get("base_freq", 220.0)),
            num_harmonics=int(kwargs.get("num_harmonics", 40)),
            harmonic_mod_depth=float(kwargs.get("harmonic_mod_depth", 0.05)),
            freq_mod_depth=float(kwargs.get("freq_mod_depth", 5.0)),
            sample_rate=sample_rate,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return write_wav(output_path, audio, sample_rate=sample_rate)

