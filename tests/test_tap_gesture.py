"""Tests for generic coordinate-free tap gesture helpers."""

from __future__ import annotations

from core._tap_gesture import DoubleTapGesture
from hardware.touch_input import TouchEvent


def _tap(timestamp: float) -> TouchEvent:
    return TouchEvent(type="tap", press_duration_ms=100, timestamp=timestamp)


def test_double_tap_gesture_classifies_second_tap_inside_window() -> None:
    """Two taps inside the configured window become a double tap."""
    gesture = DoubleTapGesture(double_tap_window_s=0.6)

    assert gesture.classify(_tap(10.0)) == "single"
    assert gesture.classify(_tap(10.4)) == "double"
    assert gesture.classify(_tap(11.0)) == "single"


def test_double_tap_gesture_treats_late_tap_as_single() -> None:
    """Late taps start a new single-tap sequence."""
    gesture = DoubleTapGesture(double_tap_window_s=0.6)

    assert gesture.classify(_tap(10.0)) == "single"
    assert gesture.classify(_tap(10.7)) == "single"


def test_double_tap_gesture_reset_forgets_pending_tap() -> None:
    """Reset clears the pending first tap."""
    gesture = DoubleTapGesture(double_tap_window_s=0.6)

    assert gesture.classify(_tap(10.0)) == "single"
    gesture.reset()

    assert gesture.classify(_tap(10.2)) == "single"
