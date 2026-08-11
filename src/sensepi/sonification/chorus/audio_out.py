"""Optional audio output + WAV capture.

``sounddevice`` is an optional extra (guardrail G8 style). Without it the model
still ticks and the views stay live — the tab reports silent mode rather than
dying.
"""
from __future__ import annotations

import importlib.util
import logging
import threading
import wave
from pathlib import Path

import numpy as np

from .types import BLOCK_SIZE, SAMPLE_RATE

logger = logging.getLogger(__name__)


def is_audio_available() -> bool:
    """True when a real audio backend can be imported."""
    return importlib.util.find_spec("sounddevice") is not None


class AudioOutput:
    """Streams pre-rendered blocks. The callback never computes audio."""

    def __init__(self, sample_rate: int = SAMPLE_RATE,
                 block_size: int = BLOCK_SIZE) -> None:
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self._stream = None
        self._pull = None
        self.underruns = 0
        self._lock = threading.Lock()

    def start(self, pull_fn) -> bool:
        """Begin streaming. ``pull_fn(n)`` must return an (n, 2) float32 block."""
        if not is_audio_available():
            return False
        import sounddevice as sd

        self._pull = pull_fn

        def _callback(outdata, frames, _time, status):
            if status:
                self.underruns += 1
            try:
                block = self._pull(frames)
            except Exception:
                block = None
            if block is None or len(block) != frames:
                outdata[:] = 0.0
                self.underruns += 1
                return
            outdata[:] = block

        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate, channels=2, dtype="float32",
                blocksize=self.block_size, callback=_callback,
            )
            self._stream.start()
        except Exception:
            logger.exception("chorus: could not open the audio device")
            self._stream = None
            return False
        return True

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                logger.debug("chorus: error closing audio stream", exc_info=True)

    @property
    def running(self) -> bool:
        return self._stream is not None


class WavCapture:
    """Accumulates rendered blocks and writes a 16-bit stereo WAV."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = int(sample_rate)
        self._blocks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self.active = False

    def start(self) -> None:
        with self._lock:
            self._blocks.clear()
            self.active = True

    def add(self, block: np.ndarray) -> None:
        if not self.active:
            return
        with self._lock:
            if len(self._blocks) < 60_000:              # ~23 min ceiling
                self._blocks.append(np.asarray(block, dtype=np.float32).copy())

    @property
    def seconds(self) -> float:
        with self._lock:
            n = sum(len(b) for b in self._blocks)
        return n / self.sample_rate

    def save(self, path: Path) -> Path | None:
        with self._lock:
            self.active = False
            blocks, self._blocks = list(self._blocks), []
        if not blocks:
            return None
        audio = np.concatenate(blocks)
        audio = np.clip(audio, -1.0, 1.0)
        pcm = (audio * 32767.0).astype("<i2")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(pcm.tobytes())
        return path
