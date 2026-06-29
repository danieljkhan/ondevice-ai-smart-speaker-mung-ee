"""Audio input stream lifecycle wrapper for live touchscreen sessions."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

AUDIO_QUEUE_MAXSIZE = 100
CALLBACK_WARNING_INTERVAL_S = 1.0
DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_CHANNELS = 2
DEFAULT_DTYPE = "float32"


class AudioCapture:
    """Own a sounddevice input stream and bounded frame queue."""

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        dtype: str = DEFAULT_DTYPE,
        device: str | int | None = None,
        queue_maxsize: int = AUDIO_QUEUE_MAXSIZE,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a capture component without opening the device."""
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.device = device
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_maxsize)
        self.stop_event = threading.Event()
        self._muted = threading.Event()
        self._stream_lock = threading.Lock()
        self._stream: Any | None = None
        self._stream_paused = False
        self._closed = False
        self._monotonic_clock = monotonic_clock
        self._last_callback_warning_monotonic = float("-inf")
        self._pending_status_count = 0
        self._last_status: str | None = None
        self._pending_dropped_oldest = 0
        self._pending_dropped_newest = 0

    def start(self) -> None:
        """Start the input stream if it is not already running."""
        with self._stream_lock:
            if self._closed:
                msg = "AudioCapture cannot be restarted after close()."
                raise RuntimeError(msg)
            if self._stream is not None:
                return
            self.stop_event.clear()

            import sounddevice as sd  # type: ignore[import-not-found, import-untyped]

            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                device=self.device,
                callback=self._callback,
            )
            self._stream = stream
            self._stream_paused = False
            stream.start()

    def stop(self) -> None:
        """Stop the input stream and drain queued frames."""
        self.stop_event.set()
        with self._stream_lock:
            stream = self._stream
            self._stream = None
            self._stream_paused = False
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()
        self._drain_queue()

    def close(self) -> None:
        """Permanently close the capture component."""
        self.stop()
        self._closed = True

    def pause(self) -> None:
        """Temporarily stop the live input stream without closing it.

        ``sounddevice.InputStream`` supports stop()/start() cycles on the same
        stream object, so playback can release the capture side of a USB audio
        device without rebuilding the stream. This is a no-op when no stream is
        open, or when the current stream is already stopped or closed.
        """
        should_drain = False
        with self._stream_lock:
            stream = self._stream
            if stream is None or self._stream_flag(stream, "closed"):
                return
            if self._stream_flag(stream, "stopped"):
                self._stream_paused = True
                should_drain = True
            elif self._stream_paused and not self._stream_flag(stream, "active"):
                should_drain = True
            else:
                stream.stop()
                self._stream_paused = True
                should_drain = True
        if should_drain:
            self._drain_queue()

    def resume(self) -> None:
        """Restart a paused input stream without recreating it.

        ``pause()`` keeps the PortAudio stream object alive, so ``resume()``
        restarts that object in place. This is a no-op when no stream is open,
        when the stream is closed, or when the stream is already active.
        """
        with self._stream_lock:
            stream = self._stream
            if stream is None or self._stream_flag(stream, "closed"):
                return
            if self._stream_flag(stream, "active"):
                self._stream_paused = False
                return
            stream.start()
            self._stream_paused = False

    def mute(self) -> None:
        """Drop incoming callback frames until capture is unmuted."""
        self._muted.set()

    def unmute(self) -> None:
        """Resume enqueueing incoming callback frames."""
        self._muted.clear()

    def is_muted(self) -> bool:
        """Return whether incoming callback frames are currently dropped."""
        return self._muted.is_set()

    def drain(self) -> None:
        """Remove all queued frames from the public capture queue."""
        self._drain_queue()

    def _callback(
        self,
        indata: Any,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """Copy one sounddevice callback frame into the bounded queue."""
        del frames, time_info
        status_text = str(status) if status else None

        if self.is_muted():
            if status_text is not None:
                self._record_callback_warning(status=status_text)
            return

        frame = np.asarray(indata, dtype=np.float32).copy()
        dropped_oldest = 0
        dropped_newest = 0
        if self.audio_queue.full():
            try:
                self.audio_queue.get_nowait()
                dropped_oldest = 1
            except queue.Empty:
                pass
        try:
            self.audio_queue.put_nowait(frame)
        except queue.Full:
            dropped_newest = 1
        if status_text is not None or dropped_oldest or dropped_newest:
            self._record_callback_warning(
                status=status_text,
                dropped_oldest=dropped_oldest,
                dropped_newest=dropped_newest,
            )

    def _record_callback_warning(
        self,
        *,
        status: str | None = None,
        dropped_oldest: int = 0,
        dropped_newest: int = 0,
    ) -> None:
        """Rate-limit callback warning output and summarize suppressed events."""
        if status is not None:
            self._pending_status_count += 1
            self._last_status = status
        self._pending_dropped_oldest += dropped_oldest
        self._pending_dropped_newest += dropped_newest

        now = self._monotonic_clock()
        if now - self._last_callback_warning_monotonic < CALLBACK_WARNING_INTERVAL_S:
            return

        parts: list[str] = []
        if self._pending_status_count:
            parts.append(
                f"status_count={self._pending_status_count} last_status={self._last_status}"
            )
        if self._pending_dropped_oldest:
            parts.append(f"dropped_oldest={self._pending_dropped_oldest}")
        if self._pending_dropped_newest:
            parts.append(f"dropped_newest={self._pending_dropped_newest}")
        if not parts:
            return

        logger.warning("Audio input callback warnings: %s", "; ".join(parts))
        self._last_callback_warning_monotonic = now
        self._pending_status_count = 0
        self._last_status = None
        self._pending_dropped_oldest = 0
        self._pending_dropped_newest = 0

    def _drain_queue(self) -> None:
        """Remove all queued frames."""
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                return

    @staticmethod
    def _stream_flag(stream: Any, name: str) -> bool:
        """Return a best-effort boolean stream state flag."""
        try:
            return bool(getattr(stream, name, False))
        except (OSError, RuntimeError, ValueError):
            return False
