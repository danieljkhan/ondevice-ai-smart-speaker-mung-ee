"""Generic tap gesture helpers for coordinate-free touchscreen menus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from hardware.touch_input import TouchEvent

TapGestureAction = Literal["single", "double"]

DEFAULT_DOUBLE_TAP_WINDOW_S = 0.6


@dataclass
class DoubleTapGesture:
    """Classify fresh tap events as single taps or double taps."""

    double_tap_window_s: float = DEFAULT_DOUBLE_TAP_WINDOW_S
    _last_tap_timestamp: float | None = None

    def classify(self, event: TouchEvent) -> TapGestureAction:
        """Return ``double`` when two taps arrive inside the configured window."""
        previous = self._last_tap_timestamp
        self._last_tap_timestamp = event.timestamp
        if previous is None:
            return "single"
        if event.timestamp - previous <= self.double_tap_window_s:
            self._last_tap_timestamp = None
            return "double"
        return "single"

    def reset(self) -> None:
        """Forget any pending first tap."""
        self._last_tap_timestamp = None


__all__ = ["DEFAULT_DOUBLE_TAP_WINDOW_S", "DoubleTapGesture", "TapGestureAction"]
