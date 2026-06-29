"""Tests for touchscreen gesture recognition."""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hardware import touch_input
from hardware.touch_input import (
    ABS_MT_POSITION_X,
    ABS_MT_POSITION_Y,
    ABS_MT_TRACKING_ID,
    ABS_X,
    ABS_Y,
    BTN_TOUCH,
    CHIME_THRESHOLD_MS,
    DEBOUNCE_FLOOR_MS,
    EV_ABS,
    EV_KEY,
    INPUT_PROP_DIRECT,
    LONG_PRESS_THRESHOLD_MS,
    READER_POLL_S,
    TAP_MAX_MS,
    TouchEvent,
    TouchInputListener,
    _capability_codes,
    _device_fd,
    _device_has_required_capabilities,
)


@dataclass
class FakeInputEvent:
    """Small evdev event fake."""

    type: int
    code: int
    value: int


class FakeDevice:
    """Minimal device fake for cleanup tests."""

    def __init__(self, path: str = "touch0") -> None:
        self.path = path
        self.closed = False
        self._read_events: list[FakeInputEvent | None] = []

    def close(self) -> None:
        """Record close."""
        self.closed = True

    def fileno(self) -> int:
        """Return a harmless fake file descriptor."""
        return 0

    def read_one(self) -> FakeInputEvent | None:
        """Return the next queued fake input event."""
        if not self._read_events:
            return None
        return self._read_events.pop(0)

    def capabilities(self, absinfo: bool = False) -> dict[int, list[int]]:
        """Return default touch capabilities."""
        del absinfo
        return {EV_ABS: [ABS_X, ABS_Y], EV_KEY: [BTN_TOUCH]}

    def input_props(self) -> list[int]:
        """Return direct-input property by default."""
        return [INPUT_PROP_DIRECT]


class FakeThread:
    """Thread-like object for stop() cleanup tests."""

    def __init__(self) -> None:
        self.joined = False

    def is_alive(self) -> bool:
        """Pretend the worker is alive."""
        return True

    def join(self, timeout: float | None = None) -> None:
        """Record join."""
        del timeout
        self.joined = True


class FakeStartThread(FakeThread):
    """Thread fake that records start() calls."""

    created: list[FakeStartThread] = []

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.started = False
        FakeStartThread.created.append(self)

    def start(self) -> None:
        """Record thread start."""
        self.started = True

    def is_alive(self) -> bool:
        """Return whether start() has been called."""
        return self.started


def key_touch(value: int) -> FakeInputEvent:
    """Build a BTN_TOUCH event."""
    return FakeInputEvent(type=EV_KEY, code=BTN_TOUCH, value=value)


def tracking(value: int) -> FakeInputEvent:
    """Build a multitouch tracking-id event."""
    return FakeInputEvent(type=EV_ABS, code=ABS_MT_TRACKING_ID, value=value)


def abs_pos(code: int, value: int) -> FakeInputEvent:
    """Build an absolute-position event."""
    return FakeInputEvent(type=EV_ABS, code=code, value=value)


def pop_event(listener: TouchInputListener) -> TouchEvent | None:
    """Read one queued high-level touch event."""
    return listener.wait_for_event(timeout=0.0)


def install_fake_evdev(monkeypatch: Any, devices: dict[str, FakeDevice]) -> None:
    """Patch evdev loading to return path-addressed fake devices."""

    def input_device(path: str) -> FakeDevice:
        if path not in devices:
            raise OSError(path)
        return devices[path]

    monkeypatch.setattr(
        touch_input,
        "_load_evdev_module",
        lambda: SimpleNamespace(InputDevice=input_device),
    )


def test_tap_boundaries() -> None:
    """<50 ms is ignored; 50 ms taps; 500 ms enters the ignored release band."""
    assert DEBOUNCE_FLOOR_MS == 50
    assert TAP_MAX_MS == 500
    listener = TouchInputListener(clock=lambda: 0.0)

    listener._handle_input_event(key_touch(1), now=0.0)
    listener._handle_input_event(key_touch(0), now=0.049)
    assert pop_event(listener) is None


def test_start_is_idempotent(monkeypatch: Any) -> None:
    """start() launches one daemon reader thread."""
    FakeStartThread.created.clear()
    monkeypatch.setattr(touch_input.threading, "Thread", FakeStartThread)
    listener = TouchInputListener(clock=lambda: 0.0)

    listener.start()
    listener.start()

    assert len(FakeStartThread.created) == 1
    assert FakeStartThread.created[0].started is True
    assert FakeStartThread.created[0].kwargs["name"] == "mungi-touch-input"

    listener._handle_input_event(key_touch(1), now=1.0)
    listener._handle_input_event(key_touch(0), now=1.05)
    event = pop_event(listener)
    assert event == TouchEvent(type="tap", press_duration_ms=50, timestamp=1.05)

    listener._handle_input_event(key_touch(1), now=2.0)
    listener._handle_input_event(key_touch(0), now=2.5)
    assert pop_event(listener) is None


def test_long_press_chime_and_parent_event() -> None:
    """1.5 s fires chime, and 3.0 s emits long_press."""
    chime_fired = threading.Event()
    listener = TouchInputListener(chime_callback=chime_fired.set, clock=lambda: 0.0)

    listener._handle_input_event(key_touch(1), now=0.0)
    listener._check_deadlines(CHIME_THRESHOLD_MS / 1000.0)
    assert chime_fired.wait(timeout=1.0)
    assert pop_event(listener) is None

    listener._check_deadlines(LONG_PRESS_THRESHOLD_MS / 1000.0)
    event = pop_event(listener)
    assert event is not None
    assert event.type == "long_press"
    assert event.press_duration_ms == 3000
    assert event.timestamp == 3.0


def test_release_after_chime_before_long_press_cancels_parent_event() -> None:
    """Releasing between 1.5 and 3.0 s does not emit long_press."""
    chime_fired = threading.Event()
    listener = TouchInputListener(chime_callback=chime_fired.set, clock=lambda: 0.0)

    listener._handle_input_event(key_touch(1), now=0.0)
    listener._check_deadlines(1.5)
    assert chime_fired.wait(timeout=1.0)
    listener._handle_input_event(key_touch(0), now=2.0)

    assert pop_event(listener) is None


def test_multitouch_primary_contact_wins_until_all_released() -> None:
    """Secondary contacts do not replace the primary touch timing."""
    listener = TouchInputListener(clock=lambda: 0.0)

    listener._handle_input_event(tracking(10), now=0.0)
    listener._handle_input_event(tracking(11), now=0.1)
    listener._handle_input_event(tracking(-1), now=0.2)
    assert pop_event(listener) is None

    listener._handle_input_event(tracking(-1), now=0.25)
    event = pop_event(listener)

    assert event == TouchEvent(type="tap", press_duration_ms=250, timestamp=0.25)


def test_tap_includes_latest_multitouch_coordinates() -> None:
    """ABS_MT_POSITION_X/Y events are emitted on the high-level tap."""
    listener = TouchInputListener(clock=lambda: 0.0)

    listener._handle_input_event(key_touch(1), now=1.0)
    listener._handle_input_event(abs_pos(ABS_MT_POSITION_X, 123), now=1.01)
    listener._handle_input_event(abs_pos(ABS_MT_POSITION_Y, 456), now=1.02)
    listener._handle_input_event(key_touch(0), now=1.12)

    assert pop_event(listener) == TouchEvent(
        type="tap",
        press_duration_ms=120,
        timestamp=1.12,
        x=123,
        y=456,
    )


def test_tap_uses_single_touch_coordinate_fallback_and_latest_value() -> None:
    """ABS_X/Y fallback coordinates are accepted and the latest value wins."""
    listener = TouchInputListener(clock=lambda: 0.0)

    listener._handle_input_event(key_touch(1), now=1.0)
    listener._handle_input_event(abs_pos(ABS_X, 10), now=1.01)
    listener._handle_input_event(abs_pos(ABS_Y, 20), now=1.02)
    listener._handle_input_event(abs_pos(ABS_X, 30), now=1.03)
    listener._handle_input_event(abs_pos(ABS_Y, 40), now=1.04)
    listener._handle_input_event(key_touch(0), now=1.1)

    assert pop_event(listener) == TouchEvent(
        type="tap",
        press_duration_ms=100,
        timestamp=1.1,
        x=30,
        y=40,
    )


def test_touch_coordinates_clamp_transform_and_reset_between_taps() -> None:
    """Coordinates clamp to the panel and do not leak after release."""
    listener = TouchInputListener(clock=lambda: 0.0, orientation="rotate_180")

    listener._handle_input_event(key_touch(1), now=1.0)
    listener._handle_input_event(abs_pos(ABS_MT_POSITION_X, -5), now=1.01)
    listener._handle_input_event(abs_pos(ABS_MT_POSITION_Y, 900), now=1.02)
    listener._handle_input_event(key_touch(0), now=1.1)

    assert pop_event(listener) == TouchEvent(
        type="tap",
        press_duration_ms=100,
        timestamp=1.1,
        x=719,
        y=0,
    )

    listener._handle_input_event(key_touch(1), now=2.0)
    listener._handle_input_event(key_touch(0), now=2.1)

    assert pop_event(listener) == TouchEvent(
        type="tap",
        press_duration_ms=100,
        timestamp=2.1,
    )


def test_stop_joins_thread_closes_device_and_drains_events() -> None:
    """stop() joins, closes fd, and clears queued gestures."""
    listener = TouchInputListener(clock=lambda: 0.0)
    fake_thread = FakeThread()
    fake_device = FakeDevice()
    listener._thread = fake_thread  # type: ignore[assignment]
    listener._device = fake_device
    listener._put_event(TouchEvent(type="tap", press_duration_ms=100, timestamp=1.0))

    listener.stop()

    assert fake_thread.joined is True
    assert fake_device.closed is True
    assert pop_event(listener) is None


def test_wait_for_event_times_out() -> None:
    """An empty event queue returns None after timeout."""
    assert TouchInputListener(clock=lambda: 0.0).wait_for_event(timeout=0.0) is None


def test_drain_taps_removes_taps_and_preserves_long_press() -> None:
    """drain_taps() removes tap events without losing parent-mode requests."""
    listener = TouchInputListener(clock=lambda: 0.0)
    listener._put_event(TouchEvent(type="tap", press_duration_ms=100, timestamp=1.0))
    listener._put_event(TouchEvent(type="long_press", press_duration_ms=3000, timestamp=2.0))
    listener._put_event(TouchEvent(type="tap", press_duration_ms=120, timestamp=3.0))

    listener.drain_taps()

    assert pop_event(listener) == TouchEvent(
        type="long_press",
        press_duration_ms=3000,
        timestamp=2.0,
    )
    assert pop_event(listener) is None


def test_discover_device_path_env_override(monkeypatch: Any) -> None:
    """MUNGI_TOUCH_DEVICE wins when it has required capabilities."""
    device = FakeDevice("/dev/input/event7")
    install_fake_evdev(monkeypatch, {"/dev/input/event7": device})
    monkeypatch.setenv("MUNGI_TOUCH_DEVICE", "/dev/input/event7")

    assert TouchInputListener(clock=lambda: 0.0).discover_device_path() == "/dev/input/event7"
    assert device.closed is True


def test_discover_device_path_rejects_invalid_env(monkeypatch: Any) -> None:
    """An invalid configured touch device returns None."""
    monkeypatch.setenv("MUNGI_TOUCH_DEVICE", "/dev/input/missing")
    install_fake_evdev(monkeypatch, {})

    assert TouchInputListener(clock=lambda: 0.0).discover_device_path() is None


def test_discover_device_path_prefers_direct_candidate(monkeypatch: Any) -> None:
    """Capability-first discovery prefers INPUT_PROP_DIRECT candidates."""
    direct_path = str(Path("/dev/input/by-id/direct-event"))
    indirect_path = str(Path("/dev/input/event0"))
    direct = FakeDevice(direct_path)
    indirect = FakeDevice(indirect_path)
    indirect.input_props = lambda: []  # type: ignore[method-assign]
    install_fake_evdev(
        monkeypatch,
        {
            direct_path: direct,
            indirect_path: indirect,
        },
    )
    monkeypatch.delenv("MUNGI_TOUCH_DEVICE", raising=False)

    def fake_glob(self: Path, pattern: str) -> list[Path]:
        root = str(self).replace("\\", "/")
        if root.endswith("/dev/input/by-id") and pattern == "*event*":
            return [Path("/dev/input/by-id/direct-event")]
        if root.endswith("/dev/input") and pattern == "event*":
            return [Path("/dev/input/event0")]
        return []

    monkeypatch.setattr(touch_input.Path, "glob", fake_glob)

    assert TouchInputListener(clock=lambda: 0.0).discover_device_path() == direct_path


def test_open_device_uses_configured_path(monkeypatch: Any) -> None:
    """_open_device returns an evdev InputDevice for explicit device_path."""
    device = FakeDevice("touch0")
    install_fake_evdev(monkeypatch, {"touch0": device})

    assert TouchInputListener("touch0", clock=lambda: 0.0)._open_device() is device


def test_open_device_raises_when_discovery_fails(monkeypatch: Any) -> None:
    """_open_device raises when no touchscreen can be found."""
    install_fake_evdev(monkeypatch, {})
    monkeypatch.delenv("MUNGI_TOUCH_DEVICE", raising=False)
    monkeypatch.setattr(touch_input.Path, "glob", lambda self, pattern: [])

    with pytest.raises(RuntimeError, match="No touchscreen"):
        TouchInputListener(clock=lambda: 0.0)._open_device()


def test_poll_once_reads_available_events(monkeypatch: Any) -> None:
    """_poll_once drains read_one() while select reports readiness."""
    listener = TouchInputListener(clock=lambda: 0.0)
    device = FakeDevice()
    device._read_events = [key_touch(1), None]
    listener._device = device
    monkeypatch.setattr(touch_input.select, "select", lambda r, w, x, timeout: (r, w, x))

    listener._poll_once()

    assert listener._primary_active is True


def test_poll_once_with_no_device_returns() -> None:
    """_poll_once is safe before a device is opened."""
    TouchInputListener(clock=lambda: 0.0)._poll_once()


def test_next_poll_timeout_contract() -> None:
    """Polling wakes at a bounded cadence and near long-press deadlines."""
    listener = TouchInputListener(clock=lambda: 0.0)
    assert listener._next_poll_timeout() == READER_POLL_S
    listener._primary_down(0.0)
    assert listener._next_poll_timeout() == READER_POLL_S
    listener._chime_sent = True
    listener._long_press_sent = True
    assert listener._next_poll_timeout() == READER_POLL_S


def test_chime_without_callback_is_noop() -> None:
    """Missing chime callback returns without spawning work."""
    TouchInputListener(clock=lambda: 0.0)._fire_chime()


def test_chime_callback_failure_is_logged(caplog: Any) -> None:
    """Chime callback failures are logged and swallowed."""
    listener = TouchInputListener(
        chime_callback=lambda: (_ for _ in ()).throw(RuntimeError("chime")),
        clock=lambda: 0.0,
    )

    with caplog.at_level("WARNING", logger="hardware.touch_input"):
        listener._run_chime_callback()

    assert "chime callback failed" in caplog.text


def test_event_queue_drops_oldest_when_full() -> None:
    """High-level event queue keeps newest events when full."""
    listener = TouchInputListener(clock=lambda: 0.0, event_queue_size=1)
    listener._put_event(TouchEvent(type="tap", press_duration_ms=100, timestamp=1.0))
    listener._put_event(TouchEvent(type="long_press", press_duration_ms=3000, timestamp=2.0))

    assert pop_event(listener) == TouchEvent(
        type="long_press",
        press_duration_ms=3000,
        timestamp=2.0,
    )


def test_primary_release_without_active_touch_resets_state() -> None:
    """Release without a matching down event is harmless."""
    listener = TouchInputListener(clock=lambda: 0.0)
    listener._primary_release(1.0)
    assert listener._primary_active is False


def test_device_disconnect_resets_state(monkeypatch: Any) -> None:
    """A reader-loop OSError closes the device and resets active touch state."""
    listener = TouchInputListener(clock=lambda: 0.0)
    listener._device = FakeDevice()
    listener._primary_down(0.0)

    def raise_disconnect() -> None:
        listener._stop_event.set()
        raise OSError("disconnect")

    monkeypatch.setattr(listener, "_poll_once", raise_disconnect)
    monkeypatch.setattr(listener, "_open_device", lambda: listener._device)

    listener._reader_loop()

    assert listener._primary_active is False


def test_reader_loop_open_failure_retries_then_stops(monkeypatch: Any) -> None:
    """Open failures are logged and retried until stop is requested."""
    listener = TouchInputListener(clock=lambda: 0.0)

    def wait(_timeout: float) -> bool:
        listener._stop_event.set()
        return True

    monkeypatch.setattr(listener._stop_event, "wait", wait)
    monkeypatch.setattr(listener, "_open_device", MagicOpenFailure())

    listener._reader_loop()

    assert listener._device is None


class MagicOpenFailure:
    """Callable that raises one RuntimeError for open-failure coverage."""

    def __call__(self) -> FakeDevice:
        raise RuntimeError("missing")


def test_capability_helpers() -> None:
    """Capability helpers accept single-touch and multitouch coordinate layouts."""
    single = FakeDevice()
    multi = FakeDevice()
    multi.capabilities = lambda absinfo=False: {  # type: ignore[method-assign]
        EV_ABS: [(ABS_MT_POSITION_X, object()), (ABS_MT_POSITION_Y, object())],
        EV_KEY: [BTN_TOUCH],
    }
    missing_button = FakeDevice()
    missing_button.capabilities = lambda absinfo=False: {EV_ABS: [ABS_X, ABS_Y], EV_KEY: []}  # type: ignore[method-assign]

    assert _capability_codes([(ABS_X, object()), ABS_Y]) == {ABS_X, ABS_Y}
    assert _device_has_required_capabilities(single) is True
    assert _device_has_required_capabilities(multi) is True
    assert _device_has_required_capabilities(missing_button) is False


def test_device_fd_accepts_fd_attribute() -> None:
    """_device_fd falls back to an fd attribute when fileno() is absent."""
    assert _device_fd(SimpleNamespace(fd=7)) == 7


def test_lazy_evdev_import(monkeypatch: Any) -> None:
    """_load_evdev_module imports the evdev module lazily."""
    fake_evdev = SimpleNamespace(InputDevice=object)
    monkeypatch.setitem(sys.modules, "evdev", fake_evdev)
    assert touch_input._load_evdev_module() is fake_evdev
