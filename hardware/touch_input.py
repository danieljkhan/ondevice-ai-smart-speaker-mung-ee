"""Touchscreen input listener based on Linux evdev devices."""

from __future__ import annotations

import logging
import os
import queue
import select
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

logger = logging.getLogger(__name__)

TouchEventType = Literal["tap", "long_press"]
TouchOrientation = Literal[
    "identity",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "flip_x",
    "flip_y",
    "swap_xy",
]

DEBOUNCE_FLOOR_MS = 50
TAP_MAX_MS = 500
CHIME_THRESHOLD_MS = 1500
LONG_PRESS_THRESHOLD_MS = 3000
READER_POLL_S = 0.05
RECONNECT_DELAY_S = 0.5
SCREEN_COORD_MIN = 0
SCREEN_COORD_MAX = 719
TOUCH_ORIENTATION_ENV = "MUNGI_TOUCH_ORIENTATION"

EV_KEY = 1
EV_ABS = 3
BTN_TOUCH = 330
ABS_X = 0
ABS_Y = 1
ABS_MT_POSITION_X = 53
ABS_MT_POSITION_Y = 54
ABS_MT_TRACKING_ID = 57
INPUT_PROP_DIRECT = 1


@dataclass(frozen=True)
class TouchEvent:
    """High-level touchscreen gesture event."""

    type: TouchEventType
    press_duration_ms: int
    timestamp: float
    x: int | None = None
    y: int | None = None


class TouchInputListener:
    """Read tap and long-press gestures from one evdev touchscreen."""

    def __init__(
        self,
        device_path: str | Path | None = None,
        *,
        chime_callback: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        event_queue_size: int = 32,
        orientation: TouchOrientation | None = None,
    ) -> None:
        """Create the listener without starting its background thread."""
        self._device_path = str(device_path) if device_path is not None else None
        self._chime_callback = chime_callback
        self._clock = clock
        self._orientation = _parse_orientation(
            orientation if orientation is not None else os.getenv(TOUCH_ORIENTATION_ENV)
        )
        self._events: queue.Queue[TouchEvent] = queue.Queue(maxsize=event_queue_size)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._device: Any | None = None
        self._primary_active = False
        self._contact_count = 0
        self._press_started_at: float | None = None
        self._latest_x: int | None = None
        self._latest_y: int | None = None
        self._chime_sent = False
        self._long_press_sent = False

    def start(self) -> None:
        """Start the background reader thread if needed."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="mungi-touch-input",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the reader thread, close the device, and drain queued events."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None
        self._close_device()
        self._drain_events()

    def wait_for_event(self, timeout: float | None = None) -> TouchEvent | None:
        """Return the next high-level touch event, or ``None`` on timeout."""
        try:
            return self._events.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain_taps(self) -> None:
        """Remove queued tap events while preserving long-press requests."""
        preserved: list[TouchEvent] = []
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            if event.type != "tap":
                preserved.append(event)
        for event in preserved:
            try:
                self._events.put_nowait(event)
            except queue.Full:
                logger.warning("Touch event queue full; dropped preserved event %s", event.type)

    def discover_device_path(self) -> str | None:
        """Return the best touchscreen input path using env and capabilities."""
        env_path = os.getenv("MUNGI_TOUCH_DEVICE", "").strip()
        if env_path:
            if self._is_touch_device(env_path):
                return env_path
            logger.warning("Configured touch device lacks required capabilities: %s", env_path)
            return None

        candidates = [
            *sorted(str(path) for path in Path("/dev/input/by-id").glob("*event*")),
            *sorted(str(path) for path in Path("/dev/input").glob("event*")),
        ]
        direct_matches: list[str] = []
        indirect_matches: list[str] = []
        for path in candidates:
            if not self._is_touch_device(path):
                continue
            if self._has_direct_input_property(path):
                direct_matches.append(path)
            else:
                indirect_matches.append(path)
        if direct_matches:
            return direct_matches[0]
        if indirect_matches:
            return indirect_matches[0]
        return None

    def _reader_loop(self) -> None:
        """Poll the evdev device until stopped."""
        while not self._stop_event.is_set():
            if self._device is None:
                try:
                    self._device = self._open_device()
                except (OSError, RuntimeError) as exc:
                    logger.warning("Touch device unavailable: %s", exc)
                    self._stop_event.wait(RECONNECT_DELAY_S)
                    continue
            try:
                self._poll_once()
            except OSError as exc:
                logger.warning("Touch device disconnected: %s", exc)
                self._close_device()
                self._reset_touch_state()
                self._stop_event.wait(RECONNECT_DELAY_S)
        self._close_device()

    def _open_device(self) -> Any:
        """Open the configured or discovered evdev input device."""
        evdev = _load_evdev_module()
        path = self._device_path or self.discover_device_path()
        if path is None:
            msg = "No touchscreen input device found"
            raise RuntimeError(msg)
        return evdev.InputDevice(path)

    def _poll_once(self) -> None:
        """Poll and process all currently available evdev events."""
        device = self._device
        if device is None:
            return
        timeout = self._next_poll_timeout()
        ready, _, _ = select.select([_device_fd(device)], [], [], timeout)
        now = self._clock()
        if ready:
            while True:
                event = device.read_one()
                if event is None:
                    break
                self._handle_input_event(event, now)
        self._check_deadlines(self._clock())

    def _handle_input_event(self, event: Any, now: float | None = None) -> None:
        """Update gesture state from one low-level evdev event."""
        event_time = self._clock() if now is None else now
        if (
            int(getattr(event, "type", -1)) == EV_KEY
            and int(getattr(event, "code", -1)) == BTN_TOUCH
        ):
            if int(getattr(event, "value", 0)):
                self._contact_count = max(self._contact_count, 1)
                self._primary_down(event_time)
            else:
                self._contact_count = 0
                self._primary_release(event_time)
            return

        if (
            int(getattr(event, "type", -1)) == EV_ABS
            and int(getattr(event, "code", -1)) == ABS_MT_TRACKING_ID
        ):
            value = int(getattr(event, "value", -1))
            if value >= 0:
                if self._contact_count == 0:
                    self._primary_down(event_time)
                self._contact_count += 1
            else:
                self._contact_count = max(self._contact_count - 1, 0)
                if self._contact_count == 0:
                    self._primary_release(event_time)
            return

        if int(getattr(event, "type", -1)) == EV_ABS:
            code = int(getattr(event, "code", -1))
            value = _clamp_screen_coordinate(int(getattr(event, "value", 0)))
            if code in (ABS_MT_POSITION_X, ABS_X):
                self._latest_x = value
            elif code in (ABS_MT_POSITION_Y, ABS_Y):
                self._latest_y = value

    def _primary_down(self, now: float) -> None:
        """Start tracking the primary contact if no primary is active."""
        if self._primary_active:
            return
        self._primary_active = True
        self._press_started_at = now
        self._chime_sent = False
        self._long_press_sent = False

    def _primary_release(self, now: float) -> None:
        """Finish the primary contact and emit a tap if duration qualifies."""
        if not self._primary_active or self._press_started_at is None:
            self._reset_touch_state()
            return
        duration_ms = int(round((now - self._press_started_at) * 1000))
        if not self._long_press_sent and DEBOUNCE_FLOOR_MS <= duration_ms < TAP_MAX_MS:
            x, y = self._transformed_coordinates()
            self._put_event(
                TouchEvent(
                    type="tap",
                    press_duration_ms=duration_ms,
                    timestamp=now,
                    x=x,
                    y=y,
                )
            )
        self._reset_touch_state()

    def _check_deadlines(self, now: float) -> None:
        """Emit long-press side effects when hold deadlines are crossed."""
        if not self._primary_active or self._press_started_at is None:
            return
        duration_ms = int(round((now - self._press_started_at) * 1000))
        if not self._chime_sent and duration_ms >= CHIME_THRESHOLD_MS:
            self._chime_sent = True
            self._fire_chime()
        if not self._long_press_sent and duration_ms >= LONG_PRESS_THRESHOLD_MS:
            self._long_press_sent = True
            x, y = self._transformed_coordinates()
            self._put_event(
                TouchEvent(
                    type="long_press",
                    press_duration_ms=duration_ms,
                    timestamp=now,
                    x=x,
                    y=y,
                )
            )

    def _next_poll_timeout(self) -> float:
        """Return a bounded timeout that wakes near gesture deadlines."""
        if not self._primary_active or self._press_started_at is None:
            return READER_POLL_S
        now = self._clock()
        deadlines = []
        if not self._chime_sent:
            deadlines.append(self._press_started_at + CHIME_THRESHOLD_MS / 1000.0)
        if not self._long_press_sent:
            deadlines.append(self._press_started_at + LONG_PRESS_THRESHOLD_MS / 1000.0)
        if not deadlines:
            return READER_POLL_S
        return max(0.0, min(READER_POLL_S, min(deadlines) - now))

    def _fire_chime(self) -> None:
        """Run the optional chime callback without blocking the reader."""
        if self._chime_callback is None:
            return
        threading.Thread(
            target=self._run_chime_callback,
            name="mungi-touch-chime",
            daemon=True,
        ).start()

    def _run_chime_callback(self) -> None:
        """Run the chime callback and log failures."""
        try:
            if self._chime_callback is not None:
                self._chime_callback()
        except Exception:
            logger.warning("Touch chime callback failed", exc_info=True)

    def _put_event(self, event: TouchEvent) -> None:
        """Put one high-level event, dropping the oldest event if needed."""
        if self._events.full():
            try:
                self._events.get_nowait()
            except queue.Empty:
                pass
        try:
            self._events.put_nowait(event)
        except queue.Full:
            logger.warning("Touch event queue full; dropped event %s", event.type)

    def _drain_events(self) -> None:
        """Drain queued high-level events."""
        while True:
            try:
                self._events.get_nowait()
            except queue.Empty:
                return

    def _reset_touch_state(self) -> None:
        """Reset contact tracking to idle."""
        self._primary_active = False
        self._contact_count = 0
        self._press_started_at = None
        self._latest_x = None
        self._latest_y = None
        self._chime_sent = False
        self._long_press_sent = False

    def _transformed_coordinates(self) -> tuple[int | None, int | None]:
        """Return the latest screen coordinates after the configured transform."""
        if self._latest_x is None or self._latest_y is None:
            return None, None
        return _transform_coordinates(self._latest_x, self._latest_y, self._orientation)

    def _close_device(self) -> None:
        """Close the current evdev device if one is open."""
        device = self._device
        self._device = None
        if device is not None:
            close = getattr(device, "close", None)
            if close is not None:
                close()

    def _is_touch_device(self, path: str) -> bool:
        """Return whether ``path`` exposes required touch capabilities."""
        try:
            evdev = _load_evdev_module()
            device = evdev.InputDevice(path)
            try:
                return _device_has_required_capabilities(device)
            finally:
                device.close()
        except (OSError, RuntimeError, AttributeError):
            return False

    def _has_direct_input_property(self, path: str) -> bool:
        """Return whether ``path`` advertises INPUT_PROP_DIRECT."""
        try:
            evdev = _load_evdev_module()
            device = evdev.InputDevice(path)
            try:
                input_props: Any = getattr(device, "input_props", lambda: [])()
                return INPUT_PROP_DIRECT in {int(prop) for prop in input_props}
            finally:
                device.close()
        except (OSError, RuntimeError, AttributeError, TypeError):
            return False


def _load_evdev_module() -> Any:
    """Import evdev lazily so tests can run without the package installed."""
    import evdev  # type: ignore[import-not-found]

    return evdev


def _device_fd(device: Any) -> int:
    """Return an integer file descriptor from an evdev InputDevice-like object."""
    fileno = getattr(device, "fileno", None)
    if fileno is not None:
        return int(fileno())
    return int(device.fd)


def _device_has_required_capabilities(device: Any) -> bool:
    """Check for absolute-position and BTN_TOUCH capabilities."""
    capabilities = device.capabilities(absinfo=False)
    abs_codes = _capability_codes(capabilities.get(EV_ABS, []))
    key_codes = _capability_codes(capabilities.get(EV_KEY, []))
    has_abs_xy = ABS_X in abs_codes and ABS_Y in abs_codes
    has_mt_xy = ABS_MT_POSITION_X in abs_codes and ABS_MT_POSITION_Y in abs_codes
    return (has_abs_xy or has_mt_xy) and BTN_TOUCH in key_codes


def _capability_codes(values: Any) -> set[int]:
    """Normalize evdev capability values to integer codes."""
    codes: set[int] = set()
    for value in values:
        if isinstance(value, tuple):
            codes.add(int(value[0]))
        else:
            codes.add(int(value))
    return codes


def _parse_orientation(raw: str | None) -> TouchOrientation:
    """Return a supported touch orientation transform."""
    if raw is None or not raw.strip():
        return "identity"
    value = raw.strip().lower()
    allowed: set[TouchOrientation] = {
        "identity",
        "rotate_90",
        "rotate_180",
        "rotate_270",
        "flip_x",
        "flip_y",
        "swap_xy",
    }
    if value not in allowed:
        logger.warning("Ignoring unsupported touch orientation: %s", raw)
        return "identity"
    return cast(TouchOrientation, value)


def _clamp_screen_coordinate(value: int) -> int:
    """Clamp a raw evdev coordinate to the 720 px touchscreen canvas."""
    return min(SCREEN_COORD_MAX, max(SCREEN_COORD_MIN, value))


def _transform_coordinates(x: int, y: int, orientation: TouchOrientation) -> tuple[int, int]:
    """Apply the configured square-screen orientation transform."""
    if orientation == "rotate_90":
        return SCREEN_COORD_MAX - y, x
    if orientation == "rotate_180":
        return SCREEN_COORD_MAX - x, SCREEN_COORD_MAX - y
    if orientation == "rotate_270":
        return y, SCREEN_COORD_MAX - x
    if orientation == "flip_x":
        return SCREEN_COORD_MAX - x, y
    if orientation == "flip_y":
        return x, SCREEN_COORD_MAX - y
    if orientation == "swap_xy":
        return y, x
    return x, y
