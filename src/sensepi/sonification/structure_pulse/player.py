"""Playback cursor over a finished render.

Qt-free (guardrail G7). The audio device pulls blocks from here on its own
callback thread while the GUI reads :attr:`position_s` on the GUI thread to
draw the playhead — a lock keeps the two honest, and the hold is a slice copy,
so it never blocks the callback for long.

Deliberately dumb: the buffer is already rendered, so playback is a cursor and
nothing more. No scheduling, no synthesis, nothing that can underrun.
"""
from __future__ import annotations

import threading

import numpy as np

from .types import SAMPLE_RATE, RenderResult


class BufferPlayer:
    """Hands out successive blocks of a rendered buffer."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self._lock = threading.Lock()
        self._buf = np.zeros((0, 2), dtype=np.float32)
        self._pos = 0
        self._loop = False
        self._playing = False
        self.sample_rate = int(sample_rate)
        self.finished = False

    # ---------------------------------------------------------------- loading
    def load(self, result: RenderResult) -> None:
        with self._lock:
            self._buf = np.asarray(result.audio, dtype=np.float32)
            if self._buf.ndim == 1:
                self._buf = np.column_stack([self._buf, self._buf])
            self.sample_rate = int(result.sample_rate or SAMPLE_RATE)
            self._pos = 0
            self._playing = False
            self.finished = False

    def clear(self) -> None:
        with self._lock:
            self._buf = np.zeros((0, 2), dtype=np.float32)
            self._pos = 0
            self._playing = False

    # -------------------------------------------------------------- transport
    def play(self, *, restart: bool = True, loop: bool = False) -> None:
        with self._lock:
            if restart or self._pos >= len(self._buf):
                self._pos = 0
            self._loop = bool(loop)
            self._playing = len(self._buf) > 0
            self.finished = False

    def pause(self) -> None:
        with self._lock:
            self._playing = False

    def stop(self) -> None:
        with self._lock:
            self._playing = False
            self._pos = 0

    def seek_seconds(self, t: float) -> None:
        with self._lock:
            self._pos = int(np.clip(float(t) * self.sample_rate, 0,
                                    max(len(self._buf) - 1, 0)))

    # ----------------------------------------------------------------- status
    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._playing

    @property
    def position_s(self) -> float:
        with self._lock:
            return self._pos / float(self.sample_rate or SAMPLE_RATE)

    @property
    def duration_s(self) -> float:
        with self._lock:
            return len(self._buf) / float(self.sample_rate or SAMPLE_RATE)

    @property
    def has_audio(self) -> bool:
        with self._lock:
            return len(self._buf) > 0

    # ------------------------------------------------------------------ pull
    def pull(self, frames: int) -> np.ndarray:
        """Return exactly ``frames`` stereo samples. Silence when idle.

        Called on the audio callback thread. Must never raise and must never
        return the wrong length, or the device underruns.
        """
        n = int(frames)
        out = np.zeros((n, 2), dtype=np.float32)
        with self._lock:
            if not self._playing or len(self._buf) == 0:
                return out
            end = self._pos + n
            chunk = self._buf[self._pos:end]
            got = len(chunk)
            out[:got] = chunk
            self._pos += got
            if got < n:                       # ran off the end
                if self._loop:
                    self._pos = 0
                    rest = self._buf[:n - got]
                    out[got:got + len(rest)] = rest
                    self._pos = len(rest)
                else:
                    self._playing = False
                    self.finished = True
        return out
