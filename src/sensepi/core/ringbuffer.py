from __future__ import annotations

from collections.abc import Iterable
from typing import Generic, TypeVar

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """
    Fixed-size ring buffer for streaming data.
    Overwrites the oldest entries when full.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._data: list[T | None] = [None] * capacity
        self._start = 0
        self._size = 0

    def append(self, item: T) -> None:
        idx = (self._start + self._size) % self._capacity
        self._data[idx] = item
        if self._size < self._capacity:
            self._size += 1
        else:
            self._start = (self._start + 1) % self._capacity

    def clear(self) -> None:
        self._data = [None] * self._capacity
        self._start = 0
        self._size = 0

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._size

    def __getitem__(self, index: int) -> T:
        """Support buf[i] and buf[-1] indexing over the logical contents (0 = oldest)."""
        size = self._size
        if size == 0:
            raise IndexError("RingBuffer is empty")

        if index < 0:
            index += size

        if index < 0 or index >= size:
            raise IndexError("RingBuffer index out of range")

        physical = (self._start + index) % self._capacity
        item = self._data[physical]
        assert item is not None
        return item

    def __iter__(self) -> Iterable[T]:
        for i in range(self._size):
            idx = (self._start + i) % self._capacity
            item = self._data[idx]
            if item is not None:
                yield item
