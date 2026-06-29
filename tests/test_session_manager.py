"""Tests for the touchscreen SessionManager state machine."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from core import session_manager
from core.audio_capture import AudioCapture
from core.character_expression import CharacterExpression
from core.event_log import EventLog
from core.funny_english_match import FunnyEnglishMatchResult
from core.pipeline import Utterance
from core.session_manager import (
    AWAITING_TAP_IDLE_TIMEOUT,
    FUNNY_ENGLISH_LISTEN_POLL_SLICE,
    FUNNY_ENGLISH_LISTEN_TIMEOUT,
    HISTORY_CONSENT_REPROMPT_TIMEOUT,
    HISTORY_CONSENT_SLEEP_TIMEOUT,
    HISTORY_SELECT_IDLE_TIMEOUT,
    LISTEN_TURN_TIMEOUT,
    STT_LOAD_EXCEPTIONS,
    WAKE_SCREEN_LEAD_DEFAULT_S,
    WAKE_SCREEN_LEAD_ENV,
    WAKE_SCREEN_LEAD_MAX_S,
    FunnyEnglishListenInterrupt,
    SessionManager,
    SessionState,
)
from hardware.touch_input import TouchEvent

_PLAYBACK_GUARD_CALLS = ["mute", "pause", "drain", "resume", "unmute"]
_DEFAULT_ACK_SENTINEL = object()


class FakeMonotonic:
    """Mutable monotonic clock fake."""

    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        """Return the current fake monotonic value."""
        return self.value

    def advance(self, seconds: float) -> None:
        """Advance fake time."""
        self.value += seconds


class FakeTouch:
    """Touch listener fake with tap-only drain semantics."""

    def __init__(self, events: list[TouchEvent] | None = None) -> None:
        self.events = events if events is not None else []
        self.drain_calls = 0
        self.wait_timeouts: list[float | None] = []
        self.on_empty: Callable[[], None] | None = None
        self.on_wait: Callable[[], TouchEvent | None] | None = None

    def wait_for_event(self, timeout: float | None = None) -> TouchEvent | None:
        """Return the next scripted event."""
        self.wait_timeouts.append(timeout)
        if self.events:
            return self.events.pop(0)
        if self.on_wait is not None:
            event = self.on_wait()
            if event is not None:
                return event
        if self.on_empty is not None:
            self.on_empty()
        return None

    def drain_taps(self) -> None:
        """Drop queued taps and preserve long-press events."""
        self.drain_calls += 1
        self.events = [event for event in self.events if event.type != "tap"]

    def push(self, event: TouchEvent) -> None:
        """Append a scripted event."""
        self.events.append(event)


class FakePipeline:
    """Pipeline fake implementing the step-4 session API."""

    def __init__(
        self,
        utterances: list[Utterance | None],
        error: BaseException | None = None,
        on_run: Callable[[], None] | None = None,
        wait_error: BaseException | None = None,
        on_wait_for_utterance: Callable[[], None] | None = None,
    ) -> None:
        self.utterances = utterances
        self.error = error
        self.on_run = on_run
        self.wait_error = wait_error
        self.on_wait_for_utterance = on_wait_for_utterance
        self.session_id = "session-1"
        self.timeouts: list[float] = []
        self.reset_session = MagicMock()
        self.run_calls = 0
        self.playback_gate_calls: list[
            tuple[Callable[[], None] | None, Callable[[], None] | None]
        ] = []
        self.playback_gate_on_start: Callable[[], None] | None = None
        self.playback_gate_on_end: Callable[[], None] | None = None
        self.expression_sink_calls: list[Callable[[CharacterExpression], None] | None] = []
        self.expression_sink: Callable[[CharacterExpression], None] | None = None
        self.language_sink_calls: list[Callable[[str], None] | None] = []
        self.language_sink: Callable[[str], None] | None = None
        self.history_mode_sink_calls: list[Callable[[], None] | None] = []
        self.history_mode_sink: Callable[[], None] | None = None
        self.funny_english_sink_calls: list[Callable[[], None] | None] = []
        self.funny_english_sink: Callable[[], None] | None = None
        self.session_language = "ko"
        self.set_session_language = MagicMock(side_effect=self._set_session_language)
        self.switch_session_language_with_confirmation = MagicMock(
            side_effect=self._switch_session_language_with_confirmation
        )

    def _set_session_language(self, lang: str) -> None:
        """Update the fake session language and emit the registered sink."""
        self.session_language = lang
        if self.language_sink is not None:
            self.language_sink(lang)

    def _switch_session_language_with_confirmation(self, target_language: str) -> None:
        """Update the fake language through the touch-confirmation path."""
        self._set_session_language(target_language)

    def set_playback_gate(
        self,
        on_start: Callable[[], None] | None,
        on_end: Callable[[], None] | None,
    ) -> None:
        """Record the playback gate callbacks registered by SessionManager."""
        self.playback_gate_on_start = on_start
        self.playback_gate_on_end = on_end
        self.playback_gate_calls.append((on_start, on_end))

    def set_expression_sink(
        self,
        cb: Callable[[CharacterExpression], None] | None,
    ) -> None:
        """Record the expression sink registered by SessionManager."""
        self.expression_sink = cb
        self.expression_sink_calls.append(cb)

    def set_language_sink(
        self,
        cb: Callable[[str], None] | None,
    ) -> None:
        """Record the language sink registered by SessionManager."""
        self.language_sink = cb
        self.language_sink_calls.append(cb)

    def set_history_mode_sink(self, cb: Callable[[], None] | None) -> None:
        """Record the history-mode sink registered by SessionManager."""
        self.history_mode_sink = cb
        self.history_mode_sink_calls.append(cb)

    def set_funny_english_sink(self, cb: Callable[[], None] | None) -> None:
        """Record the Funny English sink registered by SessionManager."""
        self.funny_english_sink = cb
        self.funny_english_sink_calls.append(cb)

    def wait_for_utterance(
        self,
        audio_capture: AudioCapture,
        *,
        timeout: float,
    ) -> Utterance | None:
        """Return the next scripted utterance."""
        del audio_capture
        self.timeouts.append(timeout)
        if self.wait_error is not None:
            wait_error = self.wait_error
            self.wait_error = None
            raise wait_error
        if self.on_wait_for_utterance is not None:
            self.on_wait_for_utterance()
        if not self.utterances:
            return None
        return self.utterances.pop(0)

    def run_turn_with_audio(self, utterance: Utterance) -> None:
        """Record a turn or raise a scripted error."""
        del utterance
        self.run_calls += 1
        if self.on_run is not None:
            self.on_run()
        if self.error is not None:
            raise self.error
        if self.playback_gate_on_start is not None:
            self.playback_gate_on_start()
        if self.playback_gate_on_end is not None:
            self.playback_gate_on_end()

    def run_funny_english_attempt(
        self,
        utterance: Utterance,
        card: Any,
        stt_hotwords_csv: str | None = None,
    ) -> Any:
        """Record a Funny English attempt."""
        del utterance, card, stt_hotwords_csv
        return MagicMock(band="pass")


class FakeAudioCapture:
    """AudioCapture-like fake for lifecycle assertions."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._muted = False

    def start(self) -> None:
        """Record start."""
        self.calls.append("start")

    def stop(self) -> None:
        """Record stop."""
        self.calls.append("stop")

    def mute(self) -> None:
        """Record mute."""
        self._muted = True
        self.calls.append("mute")

    def unmute(self) -> None:
        """Record unmute."""
        self._muted = False
        self.calls.append("unmute")

    def is_muted(self) -> bool:
        """Return whether the fake capture is muted."""
        return self._muted

    def drain(self) -> None:
        """Record drain."""
        self.calls.append("drain")

    def pause(self) -> None:
        """Record pause."""
        self.calls.append("pause")

    def resume(self) -> None:
        """Record resume."""
        self.calls.append("resume")


class RecordingRenderer:
    """Character renderer fake that records state, expression, and close calls."""

    def __init__(
        self,
        *,
        calls: list[tuple[str, Any]] | None = None,
        raise_on_state: bool = False,
        raise_on_expression: bool = False,
    ) -> None:
        self.calls = calls if calls is not None else []
        self.raise_on_state = raise_on_state
        self.raise_on_expression = raise_on_expression
        self.hit_target: Any | None = None

    def on_state_change(self, state: SessionState) -> None:
        """Record a state change or raise a scripted exception."""
        self.calls.append(("state", state))
        if self.raise_on_state:
            raise RuntimeError("state failure")

    def on_expression(self, expression: CharacterExpression) -> None:
        """Record an expression change or raise a scripted exception."""
        self.calls.append(("expression", expression))
        if self.raise_on_expression:
            raise RuntimeError("expression failure")

    def on_language_change(self, lang: str) -> None:
        """Record a session language change."""
        self.calls.append(("language", lang))

    def show_image(self, path: str | Path, *, letterbox: bool = True) -> str:
        """Record an image display request."""
        self.calls.append(("image", (path, letterbox)))
        return "image-token"

    def clear_image(self) -> None:
        """Record an image clear request."""
        self.calls.append(("clear_image", None))

    def show_text(
        self,
        lines: list[str],
        *,
        size: int,
        highlight_index: int | None = None,
        title: str | None = None,
        layout: str = "center",
        show_exit_button: bool = False,
        has_back: bool = False,
        page_index: int = 0,
        page_count: int = 1,
    ) -> None:
        """Record a text display request."""
        self.calls.append(("text", (lines, size, highlight_index, title, layout, show_exit_button)))
        del has_back, page_index, page_count

    def clear_text(self) -> None:
        """Record a text clear request."""
        self.calls.append(("clear_text", None))

    def show_card(
        self,
        *,
        image_path: str | Path | None = None,
        lines: list[str] | None = None,
        highlight_index: int | None = None,
        title: str | None = None,
        sublabel: str | None = None,
        show_exit_button: bool = False,
        show_back: bool = False,
        show_prev_card: bool = False,
        show_next_card: bool = False,
    ) -> str:
        """Record a card display request."""
        self.calls.append(
            (
                "card",
                (
                    image_path,
                    lines,
                    highlight_index,
                    title,
                    sublabel,
                    show_exit_button,
                    show_back,
                    show_prev_card,
                    show_next_card,
                ),
            )
        )
        return "card-token"

    def wait_until_rendered(self, token: str | None, timeout: float) -> bool:
        """Record a token wait."""
        self.calls.append(("wait_until_rendered", (token, timeout)))
        return True

    def show_history_image(self, path: str | Path) -> str:
        """Record a history image display request."""
        self.calls.append(("history_image", path))
        return "history-image-token"

    def show_history_menu(
        self,
        items: list[str],
        highlight: int,
        title: str,
        *,
        has_back: bool = False,
        page_index: int = 0,
        page_count: int = 1,
    ) -> str:
        """Record a history menu display request."""
        self.calls.append(("history_menu", (items, highlight, title)))
        del has_back, page_index, page_count
        return "history-menu-token"

    def hit_test(self, x: int, y: int) -> Any | None:
        """Return a scripted hit target."""
        self.calls.append(("hit_test", (x, y)))
        return self.hit_target

    def show_press_feedback(self, target: Any) -> str:
        """Record press feedback."""
        self.calls.append(("press_feedback", target))
        return "press-token"

    def flash_press_feedback(self, target: Any) -> None:
        """Record synchronous press feedback."""
        self.calls.append(("flash_press_feedback", target))

    def show_portal_status(self, text: str, *, duration: float) -> None:
        """Record transient portal status feedback."""
        self.calls.append(("portal_status", (text, duration)))

    def close(self) -> None:
        """Record renderer close."""
        self.calls.append(("close", None))


class FakeSoundBank:
    """SoundBank fake returning stable audio cues."""

    def __init__(
        self,
        *,
        ack_clip: Any = _DEFAULT_ACK_SENTINEL,
        language_switch_clip: tuple[np.ndarray, int] | None = None,
    ) -> None:
        self.kind_requests: list[str] = []
        self.wake_last_day: Any = None
        self.wake_requests = 0
        self.ack_requests = 0
        self.language_switch_requests = 0
        if ack_clip is _DEFAULT_ACK_SENTINEL:
            ack_clip = (np.full(2, 0.25, dtype=np.float32), 16_000)
        self.ack_clip: tuple[np.ndarray, int] | None = ack_clip
        self.language_switch_clip = language_switch_clip

    def pick_wake(self, now: datetime, last_wake_date: Any) -> tuple[np.ndarray, int]:
        """Return wake audio."""
        del now
        self.wake_requests += 1
        self.wake_last_day = last_wake_date
        return np.zeros(4, dtype=np.float32), 16_000

    def pick_ack(self) -> tuple[np.ndarray, int] | None:
        """Return acknowledgement audio if configured."""
        self.ack_requests += 1
        return self.ack_clip

    def pick_language_switch(self) -> tuple[np.ndarray, int] | None:
        """Return language-switch audio if configured."""
        self.language_switch_requests += 1
        return self.language_switch_clip

    def pick_error(self, kind: str = "stt_load_fail") -> tuple[np.ndarray, int]:
        """Return error audio."""
        self.kind_requests.append(kind)
        return np.ones(4, dtype=np.float32), 16_000

    def pick_sleep(self) -> tuple[np.ndarray, int]:
        """Return sleep audio."""
        return np.full(4, 0.5, dtype=np.float32), 16_000


class FakeHistoryController:
    """History controller fake for session-manager routing tests."""

    def __init__(self, manager: SessionManager) -> None:
        self.manager = manager
        self.enter_calls = 0
        self.exit_calls = 0
        self.close_calls = 0
        self.exit_restore_stt: list[bool] = []
        self.next_exit_future: Future[None] | None = None
        self.select_taps: list[TouchEvent] = []
        self.select_targets: list[int] = []
        self.back_calls = 0
        self.prev_page_calls = 0
        self.next_page_calls = 0
        self.scene_taps: list[TouchEvent] = []
        self.previous_step_calls = 0
        self.consent_taps: list[TouchEvent] = []
        self.consent_timeout_results: list[bool] = []
        self.consent_timeout_calls = 0
        self.poll_calls = 0

    def enter(self) -> None:
        """Enter fake history mode."""
        self.enter_calls += 1
        self.manager.transition_history_state(SessionState.HISTORY_SELECT)

    def exit(self, *, restore_stt: bool) -> Future[None]:
        """Record history exit."""
        self.exit_calls += 1
        self.exit_restore_stt.append(restore_stt)
        if self.next_exit_future is not None:
            return self.next_exit_future
        future: Future[None] = Future()
        future.set_result(None)
        return future

    def close(self) -> None:
        """Record history controller close."""
        self.close_calls += 1

    def handle_select_tap(self, event: TouchEvent) -> None:
        """Record a select tap."""
        self.select_taps.append(event)

    def handle_select_target(self, index: int) -> None:
        """Record a direct selected menu target."""
        self.select_targets.append(index)

    def handle_back(self) -> None:
        """Record a direct back target."""
        self.back_calls += 1

    def handle_prev_page(self) -> None:
        """Record a direct previous-page target."""
        self.prev_page_calls += 1

    def handle_next_page(self) -> None:
        """Record a direct next-page target."""
        self.next_page_calls += 1

    def handle_scene_tap(self, event: TouchEvent) -> None:
        """Record a scene tap."""
        self.scene_taps.append(event)

    def handle_previous_step(self) -> None:
        """Record a direct previous-step back target."""
        self.previous_step_calls += 1

    def handle_consent_tap(self, event: TouchEvent) -> None:
        """Record a consent tap."""
        self.consent_taps.append(event)

    def handle_consent_timeout(self) -> bool:
        """Return scripted consent timeout behavior."""
        self.consent_timeout_calls += 1
        if self.consent_timeout_results:
            return self.consent_timeout_results.pop(0)
        return False

    def handle_paused_tap(self, event: TouchEvent) -> None:
        """Ignore paused taps."""
        del event

    def handle_done_tap(self, event: TouchEvent) -> None:
        """Ignore done taps."""
        del event

    def poll(self) -> None:
        """Record a poll."""
        self.poll_calls += 1


class FakeFunnyEnglishController:
    """Funny English controller fake for session-manager routing tests."""

    def __init__(self, manager: SessionManager) -> None:
        self.manager = manager
        self.enter_calls = 0
        self.exit_calls = 0
        self.select_taps: list[TouchEvent] = []
        self.intro_taps = 0
        self.select_targets: list[int] = []
        self.back_calls = 0
        self.prev_page_calls = 0
        self.next_page_calls = 0
        self.prev_card_calls = 0
        self.next_card_calls = 0
        self.prompt_taps: list[TouchEvent] = []
        self.feedback_taps: list[TouchEvent] = []
        self.done_taps: list[TouchEvent] = []
        self.poll_calls = 0

    def enter(self) -> None:
        """Enter fake Funny English mode."""
        self.enter_calls += 1
        self.manager.transition_funny_english_state(SessionState.FE_INTRO)

    def exit(self) -> None:
        """Record Funny English exit."""
        self.exit_calls += 1

    def handle_select_tap(self, event: TouchEvent) -> None:
        """Record a select tap."""
        self.select_taps.append(event)

    def handle_intro_tap(self) -> None:
        """Record an intro tap."""
        self.intro_taps += 1

    def handle_select_target(self, index: int) -> None:
        """Record a direct selected stage target."""
        self.select_targets.append(index)

    def handle_back(self) -> None:
        """Record a direct back target."""
        self.back_calls += 1

    def handle_prev_page(self) -> None:
        """Record a direct previous-page target."""
        self.prev_page_calls += 1

    def handle_next_page(self) -> None:
        """Record a direct next-page target."""
        self.next_page_calls += 1

    def handle_prev_card(self) -> None:
        """Record a direct previous-card target."""
        self.prev_card_calls += 1

    def handle_next_card(self) -> None:
        """Record a direct next-card target."""
        self.next_card_calls += 1

    def handle_prompt_tap(self, event: TouchEvent) -> None:
        """Record a prompt tap."""
        self.prompt_taps.append(event)

    def handle_feedback_tap(self, event: TouchEvent) -> None:
        """Record a feedback tap."""
        self.feedback_taps.append(event)

    def handle_done_tap(self, event: TouchEvent) -> None:
        """Record a done tap."""
        self.done_taps.append(event)

    def poll(self) -> None:
        """Record a poll."""
        self.poll_calls += 1


def make_utterance() -> Utterance:
    """Return one valid Utterance."""
    return Utterance(audio=np.zeros(160, dtype=np.float32), sample_rate=16_000)


def tap_event(
    timestamp: float,
    duration_ms: int = 100,
    *,
    x: int | None = None,
    y: int | None = None,
) -> TouchEvent:
    """Build a tap event."""
    return TouchEvent(type="tap", press_duration_ms=duration_ms, timestamp=timestamp, x=x, y=y)


def long_press_event(timestamp: float, duration_ms: int = 3000) -> TouchEvent:
    """Build a long-press event."""
    return TouchEvent(type="long_press", press_duration_ms=duration_ms, timestamp=timestamp)


def state_calls(renderer: RecordingRenderer) -> list[SessionState]:
    """Return recorded state transitions."""
    return [call[1] for call in renderer.calls if call[0] == "state"]


def expression_calls(renderer: RecordingRenderer) -> list[CharacterExpression]:
    """Return recorded expression transitions."""
    return [call[1] for call in renderer.calls if call[0] == "expression"]


def flash_feedback_count(renderer: RecordingRenderer) -> int:
    """Return the number of direct press feedback flashes."""
    return sum(1 for call in renderer.calls if call[0] == "flash_press_feedback")


def portal_status_calls(renderer: RecordingRenderer) -> list[tuple[str, float]]:
    """Return recorded portal status messages."""
    return [call[1] for call in renderer.calls if call[0] == "portal_status"]


def assert_contains_subsequence(values: list[str], subsequence: list[str]) -> None:
    """Assert that ``values`` contains ``subsequence`` contiguously."""
    for index in range(len(values) - len(subsequence) + 1):
        if values[index : index + len(subsequence)] == subsequence:
            return
    raise AssertionError(f"{subsequence!r} not found in {values!r}")


def install_playback_recorder(monkeypatch: Any, calls: list[str] | None = None) -> MagicMock:
    """Patch blocking audio playback with an optional call marker."""

    def record_playback(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        if calls is not None:
            calls.append("play_audio")

    play_audio = MagicMock(side_effect=record_playback)
    monkeypatch.setattr(session_manager.audio_player, "play_audio", play_audio)
    return play_audio


def build_manager(
    pipeline: FakePipeline,
    tmp_path: Path,
    monkeypatch: Any,
    *,
    renderer: RecordingRenderer | None = None,
    touch: FakeTouch | None = None,
    sound_bank: FakeSoundBank | None = None,
    monotonic: FakeMonotonic | None = None,
) -> tuple[SessionManager, MagicMock, FakeAudioCapture, FakeSoundBank, EventLog, FakeTouch]:
    """Build a SessionManager with fakes and isolated mutable root."""
    monkeypatch.setenv("MUNGI_MUTABLE_ROOT", str(tmp_path))
    monkeypatch.setenv(WAKE_SCREEN_LEAD_ENV, "0")
    mm = MagicMock()
    audio_capture = FakeAudioCapture()
    fake_sound_bank = sound_bank or FakeSoundBank()
    fake_touch = touch or FakeTouch()
    fake_monotonic = monotonic or FakeMonotonic()
    event_log = EventLog(tmp_path / "events" / "parent_mode.jsonl")
    manager = SessionManager(
        mm=mm,
        pipeline_factory=lambda received_mm: pipeline,
        touch=fake_touch,  # type: ignore[arg-type]
        sound_bank=fake_sound_bank,  # type: ignore[arg-type]
        audio_capture=audio_capture,  # type: ignore[arg-type]
        event_log=event_log,
        renderer=renderer,
        clock=lambda: datetime(2026, 5, 26, 6, 0),
        monotonic_clock=fake_monotonic,
    )
    return manager, mm, audio_capture, fake_sound_bank, event_log, fake_touch


@pytest.mark.parametrize(
    ("state", "expression"),
    [
        (SessionState.IDLE, CharacterExpression.IDLE),
        (SessionState.WAKING, CharacterExpression.GREETING),
        (SessionState.AWAITING_TAP, CharacterExpression.NEUTRAL),
        (SessionState.LISTENING, CharacterExpression.LISTENING),
        (SessionState.RESPONDING, CharacterExpression.THINKING),
        (SessionState.SLEEPING, CharacterExpression.SLEEPY),
        (SessionState.HISTORY_SELECT, CharacterExpression.NEUTRAL),
        (SessionState.HISTORY_SCENE, CharacterExpression.SPEAKING),
        (SessionState.HISTORY_CONSENT, CharacterExpression.NEUTRAL),
        (SessionState.HISTORY_PAUSED, CharacterExpression.NEUTRAL),
        (SessionState.HISTORY_DONE, CharacterExpression.EXCITED),
        (SessionState.FE_INTRO, CharacterExpression.EXCITED),
        (SessionState.FE_SELECT, CharacterExpression.EXCITED),
        (SessionState.FE_PROMPT, CharacterExpression.SPEAKING),
        (SessionState.FE_LISTEN, CharacterExpression.LISTENING),
        (SessionState.FE_SCORE, CharacterExpression.THINKING),
        (SessionState.FE_FEEDBACK, CharacterExpression.HAPPY),
        (SessionState.FE_DONE, CharacterExpression.EXCITED),
    ],
)
def test_state_transitions_emit_mapped_expressions(
    state: SessionState,
    expression: CharacterExpression,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Each state emits the expected expression."""
    renderer = RecordingRenderer()
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )

    manager._transition(state)

    assert renderer.calls[-2:] == [("state", state), ("expression", expression)]


def test_every_session_state_has_expression_mapping() -> None:
    """Every SessionState member must have an expression mapping."""
    assert set(session_manager._STATE_TO_EXPRESSION) == set(SessionState)


def test_run_emits_initial_idle_expression_before_waiting(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """run() draws the startup IDLE frame before blocking for touch input."""
    renderer = RecordingRenderer()
    touch = FakeTouch()
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
        touch=touch,
    )

    def stop_after_first_wait() -> None:
        assert renderer.calls == [("expression", CharacterExpression.IDLE)]
        manager.stop()

    touch.on_empty = stop_after_first_wait

    manager.run()

    assert renderer.calls == [("expression", CharacterExpression.IDLE)]
    assert touch.wait_timeouts == [1.0]
    assert touch.drain_calls == 1


def test_emit_expression_suppresses_renderer_exception_and_transitions_continue(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Expression emit failures are suppressed and do not break later transitions."""
    renderer = RecordingRenderer(raise_on_expression=True)
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )

    manager._emit_expression(CharacterExpression.HAPPY)
    manager._transition(SessionState.IDLE)

    assert manager.state is SessionState.IDLE
    assert renderer.calls == [
        ("expression", CharacterExpression.HAPPY),
        ("state", SessionState.IDLE),
        ("expression", CharacterExpression.IDLE),
    ]


def test_transition_suppresses_state_exception_but_still_emits_expression(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """State callback failures do not prevent the mapped expression emit."""
    renderer = RecordingRenderer(raise_on_state=True)
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )

    manager._transition(SessionState.WAKING)

    assert manager.state is SessionState.WAKING
    assert renderer.calls == [
        ("state", SessionState.WAKING),
        ("expression", CharacterExpression.GREETING),
    ]


def test_funny_english_attempt_alias_rescore_upgrades_pipeline_non_pass(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Session scoring accepts card aliases without changing the pipeline implementation."""
    pipeline = FakePipeline([None])
    pipeline.run_funny_english_attempt = MagicMock(
        return_value=FunnyEnglishMatchResult(
            transcript="비",
            normalized_transcript_tokens=(),
            target_tokens=("b",),
            matched_tokens=(),
            missed_tokens=("b",),
            matched_pct=0.0,
            similarity=0.0,
            band="silent_junk",
        )
    )
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
    )
    utterance = Utterance(audio=np.zeros(160, dtype=np.float32), sample_rate=16_000)
    card = SimpleNamespace(tokens=("b",), accept_aliases=("b", "bee", "비"))

    result = manager.run_funny_english_attempt(utterance, card)

    assert result.band == "pass"
    assert result.matched_pct == 1.0
    pipeline.run_funny_english_attempt.assert_called_once_with(
        utterance,
        card,
        stt_hotwords_csv=None,
    )


def test_funny_english_attempt_forwards_stt_hotwords_csv(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Session scoring forwards an explicit stage-level STT hotword CSV."""
    pipeline = FakePipeline([None])
    pipeline_result = FunnyEnglishMatchResult(
        transcript="cat",
        normalized_transcript_tokens=("cat",),
        target_tokens=("cat",),
        matched_tokens=("cat",),
        missed_tokens=(),
        matched_pct=1.0,
        similarity=1.0,
        band="pass",
    )
    pipeline.run_funny_english_attempt = MagicMock(return_value=pipeline_result)
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
    )
    utterance = Utterance(audio=np.zeros(160, dtype=np.float32), sample_rate=16_000)
    card = SimpleNamespace(tokens=("cat",), accept_aliases=())
    hotwords_csv = "cat,dog,뭉이,뭉이야"

    result = manager.run_funny_english_attempt(
        utterance,
        card,
        stt_hotwords_csv=hotwords_csv,
    )

    assert result is pipeline_result
    pipeline.run_funny_english_attempt.assert_called_once_with(
        utterance,
        card,
        stt_hotwords_csv=hotwords_csv,
    )


def test_set_expression_public_api_emits_expression(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """set_expression emits a caller-requested expression through the renderer."""
    renderer = RecordingRenderer()
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )

    manager.set_expression(CharacterExpression.HAPPY)

    assert renderer.calls[-1] == ("expression", CharacterExpression.HAPPY)


def test_session_manager_registers_pipeline_expression_sink(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """SessionManager should route pipeline expression emits through the renderer."""
    renderer = RecordingRenderer()
    pipeline = FakePipeline([None])
    build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )

    assert len(pipeline.expression_sink_calls) == 1
    assert pipeline.expression_sink is not None
    pipeline.expression_sink(CharacterExpression.HAPPY)

    assert renderer.calls[-1] == ("expression", CharacterExpression.HAPPY)


def test_session_manager_registers_pipeline_language_sink(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """SessionManager should route pipeline language changes through the renderer."""
    renderer = RecordingRenderer()
    pipeline = FakePipeline([None])
    build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )

    assert len(pipeline.language_sink_calls) == 1
    assert pipeline.language_sink is not None
    pipeline.language_sink("en")

    assert renderer.calls[-1] == ("language", "en")


def test_session_manager_registers_pipeline_history_mode_sink(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """SessionManager should route history-mode entry through the controller."""
    pipeline = FakePipeline([None])
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
    )
    controller = FakeHistoryController(manager)
    manager.set_history_controller(controller)

    assert len(pipeline.history_mode_sink_calls) == 1
    assert pipeline.history_mode_sink is not None
    pipeline.history_mode_sink()

    assert controller.enter_calls == 1
    assert manager.state is SessionState.HISTORY_SELECT


def test_language_indicator_cue_plays_only_on_real_language_change(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Language-switch cue is suppressed for boot and repeat emits."""
    cue_clip = (np.array([1, 2, 3], dtype=np.int16), 22_050)
    renderer = RecordingRenderer()
    sound_bank = FakeSoundBank(language_switch_clip=cue_clip)
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
        sound_bank=sound_bank,
    )
    play_guard = MagicMock()
    manager._play_audio_with_capture_guard = play_guard  # type: ignore[method-assign]

    manager.set_language_indicator("ko")
    manager.set_language_indicator("ko")
    manager.set_language_indicator("en")

    assert [call for call in renderer.calls if call[0] == "language"] == [
        ("language", "ko"),
        ("language", "ko"),
        ("language", "en"),
    ]
    assert sound_bank.language_switch_requests == 1
    play_guard.assert_called_once_with(cue_clip[0], cue_clip[1])


def test_language_indicator_cue_missing_clip_is_noop(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A real language change tolerates a missing optional cue pool."""
    renderer = RecordingRenderer()
    sound_bank = FakeSoundBank(language_switch_clip=None)
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
        sound_bank=sound_bank,
    )
    play_guard = MagicMock()
    manager._play_audio_with_capture_guard = play_guard  # type: ignore[method-assign]

    manager.set_language_indicator("ko")
    manager.set_language_indicator("en")

    assert sound_bank.language_switch_requests == 1
    play_guard.assert_not_called()


def test_capture_guard_resumes_when_capture_was_active(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Out-of-turn cues resume and unmute capture after playback."""
    manager, _mm, audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
    )
    install_playback_recorder(monkeypatch, audio_capture.calls)

    manager._play_audio_with_capture_guard(np.zeros(2, dtype=np.float32), 16_000)

    assert audio_capture.calls == ["mute", "pause", "play_audio", "drain", "resume", "unmute"]
    assert manager._capture_paused is False
    assert audio_capture.is_muted() is False


def test_capture_guard_preserves_prepaused_capture(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """In-turn language-switch cues must not reopen capture during RESPONDING."""
    manager, _mm, audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
    )
    manager._pause_audio_capture()
    audio_capture.calls.clear()
    install_playback_recorder(monkeypatch, audio_capture.calls)

    manager._play_audio_with_capture_guard(np.zeros(2, dtype=np.float32), 16_000)

    assert audio_capture.calls == ["mute", "pause", "play_audio"]
    assert manager._capture_paused is True
    assert audio_capture.is_muted() is True


def test_guarded_playback_api_pauses_capture_once_and_uses_blocking_playback(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """History narration uses the explicit guarded playback API."""
    manager, _mm, audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
    )
    install_playback_recorder(monkeypatch, audio_capture.calls)

    manager.begin_guarded_playback()
    manager.play_guarded(np.zeros(2, dtype=np.float32), 16_000)
    manager.play_guarded(np.zeros(2, dtype=np.float32), 16_000)
    manager.end_guarded_playback()

    assert audio_capture.calls == [
        "mute",
        "pause",
        "play_audio",
        "play_audio",
        "drain",
        "resume",
        "unmute",
    ]


def test_history_capture_pause_survives_segment_guard_gaps(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """History capture stays paused when per-segment playback guards end."""
    manager, _mm, audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
    )
    install_playback_recorder(monkeypatch, audio_capture.calls)

    manager.begin_history_capture_pause()
    audio_capture.calls.clear()
    manager.begin_guarded_playback()
    manager.play_guarded(np.zeros(2, dtype=np.float32), 16_000)
    manager.end_guarded_playback()

    assert audio_capture.calls == ["mute", "pause", "play_audio"]
    assert manager._capture_paused is True
    assert audio_capture.is_muted() is True

    manager.release_history_capture_pause(restore_stt=True)
    assert manager._capture_paused is True
    assert audio_capture.is_muted() is True

    manager.release_history_capture_pause(restore_stt=False)
    assert audio_capture.calls[-1] == "stop"


def test_interrupt_playback_delegates_to_audio_player_stop(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Session-owned interrupt releases blocking local playback."""
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
    )
    stop_playback = MagicMock()
    monkeypatch.setattr(session_manager.audio_player, "stop_playback", stop_playback)

    manager.interrupt_playback()

    stop_playback.assert_called_once_with()


def test_shutdown_from_idle_runs_ordered_cleanup_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """shutdown performs ordered cleanup and tolerates repeated calls."""
    call_log: list[tuple[str, Any]] = []
    renderer = RecordingRenderer(calls=call_log)
    manager, mm, audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )

    audio_capture.stop = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda: call_log.append(("audio_stop", None)),
    )
    mm.cancel_preload.side_effect = lambda: call_log.append(("cancel_preload", None))
    mm.unload_stt.side_effect = lambda *, force=False: call_log.append(("unload_stt", force))

    manager.shutdown()
    manager.shutdown()

    assert call_log == [
        ("audio_stop", None),
        ("close", None),
        ("cancel_preload", None),
        ("unload_stt", False),
        ("audio_stop", None),
        ("close", None),
        ("cancel_preload", None),
        ("unload_stt", False),
    ]


def test_shutdown_rejects_active_state_after_run_but_allows_idle(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """shutdown is terminal and rejects active post-run states."""
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
    )
    manager._has_run = True
    manager._state = SessionState.RESPONDING

    with pytest.raises(RuntimeError, match="shutdown\\(\\) must be called"):
        manager.shutdown()

    manager._state = SessionState.IDLE
    manager.shutdown()


def test_add_state_listener_is_prerun_idle_only_and_closes_old_renderer(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """add_state_listener closes the old renderer and rejects unsafe replacement."""
    call_log: list[tuple[str, Any]] = []
    old_renderer = RecordingRenderer(calls=call_log)
    new_renderer = RecordingRenderer(calls=call_log)
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=old_renderer,
    )

    manager.add_state_listener(new_renderer)
    manager._transition(SessionState.IDLE)

    assert call_log == [
        ("close", None),
        ("state", SessionState.IDLE),
        ("expression", CharacterExpression.IDLE),
    ]

    manager._has_run = True
    with pytest.raises(RuntimeError, match="pre-run only"):
        manager.add_state_listener(RecordingRenderer())

    manager._has_run = False
    manager._state = SessionState.WAKING
    with pytest.raises(RuntimeError, match="requires state=IDLE"):
        manager.add_state_listener(RecordingRenderer())


def test_cold_wake_tap_returns_to_awaiting_tap(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """IDLE tap runs one wake sequence and returns to the dispatch loop."""
    renderer = RecordingRenderer()
    monotonic = FakeMonotonic()
    touch = FakeTouch()
    pipeline = FakePipeline([make_utterance()])
    manager, mm, audio_capture, sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
        renderer=renderer,
        touch=touch,
        monotonic=monotonic,
    )
    play_audio = install_playback_recorder(monkeypatch)
    emitted_tap = False

    def emit_tap_once() -> TouchEvent | None:
        nonlocal emitted_tap
        if not emitted_tap:
            emitted_tap = True
            return tap_event(manager._armed_at + 0.1)
        if manager.state is SessionState.AWAITING_TAP:
            manager.stop()
        return None

    touch.on_wait = emit_tap_once

    manager.run()

    assert state_calls(renderer) == [SessionState.WAKING, SessionState.AWAITING_TAP]
    assert manager.state is SessionState.AWAITING_TAP
    assert sound_bank.wake_requests == 1
    assert sound_bank.ack_requests == 0
    assert play_audio.call_count == 1
    pipeline.reset_session.assert_called_once_with()
    assert pipeline.run_calls == 0
    assert audio_capture.calls == ["start", *_PLAYBACK_GUARD_CALLS, "pause", "stop"]
    assert mm.reset_preload_state.call_count == 1
    mm.preload_stt.assert_called_once_with()
    mm.unload_stt.assert_not_called()
    assert touch.drain_calls >= 2


def test_cold_wake_emits_waking_expression_before_wake_audio(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Cold wake draws WAKING before the wake greeting is played."""
    call_log: list[tuple[str, Any]] = []
    renderer = RecordingRenderer(calls=call_log)
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([make_utterance()]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )

    def record_playback(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        call_log.append(("play_audio", None))

    monkeypatch.setattr(
        session_manager.audio_player,
        "play_audio",
        MagicMock(side_effect=record_playback),
    )

    manager._handle_cold_wake()

    waking_expression = ("expression", CharacterExpression.GREETING)
    wake_audio = ("play_audio", None)
    assert waking_expression in call_log
    assert wake_audio in call_log
    assert call_log.index(waking_expression) < call_log.index(wake_audio)


def test_cold_wake_screen_lead_sleeps_remaining_deadline(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Cold wake sleeps only the unspent screen-lead budget."""
    monotonic = FakeMonotonic()
    manager, mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([make_utterance()]),
        tmp_path,
        monkeypatch,
        monotonic=monotonic,
    )
    monkeypatch.setenv(WAKE_SCREEN_LEAD_ENV, "0.4")
    sleeps: list[float] = []
    monkeypatch.setattr(session_manager.time, "sleep", sleeps.append)
    mm.preload_stt.side_effect = lambda: monotonic.advance(0.125)
    install_playback_recorder(monkeypatch)

    manager._handle_cold_wake()

    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(0.275)


def test_cold_wake_screen_lead_skips_sleep_when_preload_covers_budget(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Cold wake adds no sleep when preload already covers the lead budget."""
    monotonic = FakeMonotonic()
    manager, mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([make_utterance()]),
        tmp_path,
        monkeypatch,
        monotonic=monotonic,
    )
    monkeypatch.setenv(WAKE_SCREEN_LEAD_ENV, "0.4")
    sleep = MagicMock()
    monkeypatch.setattr(session_manager.time, "sleep", sleep)
    mm.preload_stt.side_effect = lambda: monotonic.advance(0.5)
    install_playback_recorder(monkeypatch)

    manager._handle_cold_wake()

    sleep.assert_not_called()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, WAKE_SCREEN_LEAD_DEFAULT_S),
        ("0", 0.0),
        ("invalid", WAKE_SCREEN_LEAD_DEFAULT_S),
        ("-0.25", 0.0),
        ("1.25", 1.25),
        ("5.0", WAKE_SCREEN_LEAD_MAX_S),
    ],
)
def test_screen_lead_seconds_env_override(
    tmp_path: Path,
    monkeypatch: Any,
    raw: str | None,
    expected: float,
) -> None:
    """Screen-lead env override is defaulted, disabled, zero-clamped, or max-clamped."""
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([make_utterance()]),
        tmp_path,
        monkeypatch,
    )
    if raw is None:
        monkeypatch.delenv(WAKE_SCREEN_LEAD_ENV, raising=False)
    else:
        monkeypatch.setenv(WAKE_SCREEN_LEAD_ENV, raw)

    assert manager._screen_lead_seconds() == pytest.approx(expected)


def test_await_screen_lead_zero_env_skips_sleep(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A zero screen-lead override disables the wake delay."""
    monotonic = FakeMonotonic()
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([make_utterance()]),
        tmp_path,
        monkeypatch,
        monotonic=monotonic,
    )
    monkeypatch.setenv(WAKE_SCREEN_LEAD_ENV, "0")
    sleep = MagicMock()
    monkeypatch.setattr(session_manager.time, "sleep", sleep)

    manager._await_screen_lead(since=monotonic())

    sleep.assert_not_called()


def test_listen_turn_ack_utterance_response_and_rearm(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """AWAITING_TAP tap runs one listen/respond turn and re-arms."""
    renderer = RecordingRenderer()
    pipeline = FakePipeline([make_utterance()])
    manager, _mm, audio_capture, sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )
    playback_calls: list[str] = audio_capture.calls
    play_audio = install_playback_recorder(monkeypatch, playback_calls)
    manager._enter_awaiting_tap()
    renderer.calls.clear()
    audio_capture.calls.clear()

    manager._handle_listen_turn()

    assert state_calls(renderer) == [
        SessionState.LISTENING,
        SessionState.RESPONDING,
        SessionState.AWAITING_TAP,
    ]
    assert expression_calls(renderer) == [
        CharacterExpression.LISTENING,
        CharacterExpression.THINKING,
        CharacterExpression.NEUTRAL,
    ]
    assert pipeline.timeouts == [LISTEN_TURN_TIMEOUT]
    assert pipeline.run_calls == 1
    assert sound_bank.ack_requests == 1
    assert play_audio.call_count == 1
    assert_contains_subsequence(audio_capture.calls, _PLAYBACK_GUARD_CALLS)
    assert manager.state is SessionState.AWAITING_TAP
    pipeline.set_session_language.assert_not_called()


def test_funny_english_capture_uses_shorter_listen_timeout(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Funny English read-aloud capture polls in bounded slices."""
    pipeline = FakePipeline([make_utterance()])
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
    )
    install_playback_recorder(monkeypatch)

    utterance = manager.capture_funny_english_audio()

    assert utterance is not None
    assert manager.state is SessionState.FE_LISTEN
    assert pipeline.timeouts == [FUNNY_ENGLISH_LISTEN_POLL_SLICE]


def test_funny_english_capture_timeout_uses_slices_without_touch_interrupt(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A silent FE capture exhausts the FE timeout through poll slices."""
    pipeline = FakePipeline([])
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
    )
    install_playback_recorder(monkeypatch)

    result = manager.capture_funny_english_audio()

    assert result is None
    assert manager.state is SessionState.FE_LISTEN
    assert sum(pipeline.timeouts) == pytest.approx(FUNNY_ENGLISH_LISTEN_TIMEOUT)
    # Every poll slice is the poll-slice size (the final slice is the remaining
    # time, equal to the slice within float rounding).
    assert all(t == pytest.approx(FUNNY_ENGLISH_LISTEN_POLL_SLICE) for t in pipeline.timeouts)


@pytest.mark.parametrize("target_kind", ["exit", "back"])
def test_funny_english_capture_returns_direct_nav_interrupt(
    target_kind: str,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Exit/back taps queued during FE capture return typed interrupts."""
    renderer = RecordingRenderer()
    renderer.hit_target = SimpleNamespace(kind=target_kind, index=None, generation=9)
    touch = FakeTouch([tap_event(101.0, x=540, y=40)])
    pipeline = FakePipeline([])
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
        renderer=renderer,
        touch=touch,
    )
    install_playback_recorder(monkeypatch)

    result = manager.capture_funny_english_audio()

    assert isinstance(result, FunnyEnglishListenInterrupt)
    assert result.kind == target_kind
    assert pipeline.timeouts == []
    assert touch.events == []
    assert ("hit_test", (540, 40)) in renderer.calls
    assert any(call[0] == "flash_press_feedback" for call in renderer.calls)


@pytest.mark.parametrize("target_kind", ["exit", "back"])
def test_funny_english_capture_prefers_same_slice_nav_interrupt_over_utterance(
    target_kind: str,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A queued FE nav tap wins over a VAD utterance returned in the same slice."""
    renderer = RecordingRenderer()
    renderer.hit_target = SimpleNamespace(kind=target_kind, index=None, generation=9)
    touch = FakeTouch([])
    pipeline = FakePipeline(
        [make_utterance()],
        on_wait_for_utterance=lambda: touch.push(tap_event(101.0, x=540, y=40)),
    )
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
        renderer=renderer,
        touch=touch,
    )
    install_playback_recorder(monkeypatch)

    result = manager.capture_funny_english_audio()

    assert isinstance(result, FunnyEnglishListenInterrupt)
    assert result.kind == target_kind
    assert pipeline.timeouts == [FUNNY_ENGLISH_LISTEN_POLL_SLICE]
    assert touch.events == []
    assert ("hit_test", (540, 40)) in renderer.calls
    assert any(call[0] == "flash_press_feedback" for call in renderer.calls)


def test_funny_english_capture_discards_coordinate_taps_but_preserves_other_events(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Non-interrupt coordinate taps are consumed while other events stay queued."""
    event = tap_event(101.0, x=80, y=650)
    coordinate_less_tap = tap_event(101.05)
    long_press = long_press_event(101.1)
    renderer = RecordingRenderer()
    renderer.hit_target = SimpleNamespace(kind="prev_card", index=None, generation=9)
    touch = FakeTouch([event, coordinate_less_tap, long_press])
    pipeline = FakePipeline([None, make_utterance()])
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
        renderer=renderer,
        touch=touch,
    )
    install_playback_recorder(monkeypatch)

    result = manager.capture_funny_english_audio()

    assert result is not None
    assert not isinstance(result, FunnyEnglishListenInterrupt)
    assert touch.events == [coordinate_less_tap, long_press]
    assert pipeline.timeouts == [
        FUNNY_ENGLISH_LISTEN_POLL_SLICE,
        FUNNY_ENGLISH_LISTEN_POLL_SLICE,
    ]


def test_language_toggle_tap_in_awaiting_tap_speaks_confirmation_and_rearms(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A coordinate tap on the idle badge toggles language instead of listening."""
    renderer = RecordingRenderer()
    renderer.hit_target = SimpleNamespace(kind="language_toggle", index=None, generation=4)
    pipeline = FakePipeline([make_utterance()])
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )
    manager._enter_awaiting_tap()
    renderer.calls.clear()

    manager._dispatch_tap(tap_event(101.0, x=675, y=20))

    assert ("hit_test", (675, 20)) in renderer.calls
    assert any(call[0] == "flash_press_feedback" for call in renderer.calls)
    pipeline.switch_session_language_with_confirmation.assert_called_once_with("en")
    assert pipeline.session_language == "en"
    assert pipeline.run_calls == 0
    assert manager.state is SessionState.AWAITING_TAP
    assert state_calls(renderer) == [SessionState.RESPONDING, SessionState.AWAITING_TAP]


@pytest.mark.parametrize("state", [SessionState.LISTENING, SessionState.RESPONDING])
def test_language_toggle_tap_is_ignored_outside_awaiting_tap(
    state: SessionState,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """The badge target is inert while listening or responding."""
    renderer = RecordingRenderer()
    renderer.hit_target = SimpleNamespace(kind="language_toggle", index=None, generation=4)
    pipeline = FakePipeline([make_utterance()])
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )
    manager._transition(state)
    renderer.calls.clear()

    manager._dispatch_tap(tap_event(101.0, x=675, y=20))

    assert ("hit_test", (675, 20)) not in renderer.calls
    assert not any(call[0] == "flash_press_feedback" for call in renderer.calls)
    pipeline.switch_session_language_with_confirmation.assert_not_called()
    assert pipeline.run_calls == 0
    assert manager.state is state


def test_history_mode_entry_during_turn_is_not_clobbered_by_finalizer(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A history sink emitted during a turn must survive listen-turn cleanup."""
    pipeline = FakePipeline([make_utterance()])

    def enter_history() -> None:
        assert pipeline.history_mode_sink is not None
        pipeline.history_mode_sink()

    pipeline.on_run = enter_history
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
    )
    controller = FakeHistoryController(manager)
    manager.set_history_controller(controller)
    install_playback_recorder(monkeypatch)

    manager._handle_listen_turn()

    assert controller.enter_calls == 1
    assert manager.state is SessionState.HISTORY_SELECT


def test_funny_english_entry_during_turn_is_not_clobbered_by_finalizer(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A Funny English sink emitted during a turn must survive listen-turn cleanup."""
    pipeline = FakePipeline([make_utterance()])

    def enter_funny_english() -> None:
        assert pipeline.funny_english_sink is not None
        pipeline.funny_english_sink()

    pipeline.on_run = enter_funny_english
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
    )
    controller = FakeFunnyEnglishController(manager)
    manager.set_funny_english_controller(controller)
    install_playback_recorder(monkeypatch)

    manager._handle_listen_turn()

    assert controller.enter_calls == 1
    assert manager.state is SessionState.FE_INTRO


def test_funny_english_long_press_exits_before_english_language_revert(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Long-press in FE exits the mode instead of only flipping EN back to KO."""
    pipeline = FakePipeline([None])
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
    )
    controller = FakeFunnyEnglishController(manager)
    manager.set_funny_english_controller(controller)
    pipeline.session_language = "en"
    manager.transition_funny_english_state(SessionState.FE_SELECT)

    manager._handle_parent_request(long_press_event(101.0))

    assert controller.exit_calls == 1
    pipeline.set_session_language.assert_called_once_with("ko")
    assert manager.state is SessionState.AWAITING_TAP


def test_coordinate_funny_english_select_tap_dispatches_hit_target(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Coordinate taps on FE stage menu items select directly."""
    renderer = RecordingRenderer()
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )
    controller = FakeFunnyEnglishController(manager)
    manager.set_funny_english_controller(controller)
    manager.transition_funny_english_state(SessionState.FE_SELECT)
    renderer.hit_target = SimpleNamespace(kind="menu_item", index=1, generation=9)

    manager._dispatch_tap(tap_event(101.0, x=120, y=220))

    assert controller.select_targets == [1]
    assert controller.select_taps == []
    assert any(call[0] == "flash_press_feedback" for call in renderer.calls)


@pytest.mark.parametrize(
    ("target_kind", "expected_attr"),
    [
        ("prev_page", "prev_page_calls"),
        ("next_page", "next_page_calls"),
    ],
)
def test_coordinate_funny_english_select_tap_dispatches_page_targets(
    target_kind: str,
    expected_attr: str,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Coordinate taps on FE page targets route to the matching controller method."""
    renderer = RecordingRenderer()
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )
    controller = FakeFunnyEnglishController(manager)
    manager.set_funny_english_controller(controller)
    manager.transition_funny_english_state(SessionState.FE_SELECT)
    renderer.hit_target = SimpleNamespace(kind=target_kind, index=None, generation=9)

    manager._dispatch_tap(tap_event(101.0, x=420, y=650))

    assert getattr(controller, expected_attr) == 1
    assert controller.select_taps == []
    assert any(call[0] == "flash_press_feedback" for call in renderer.calls)


def test_funny_english_select_nav_taps_act_immediately_and_menu_items_still_work(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """FE selection nav taps handle repeated taps without a shared cooldown."""
    monotonic = FakeMonotonic()
    renderer = RecordingRenderer()
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
        monotonic=monotonic,
    )
    controller = FakeFunnyEnglishController(manager)
    manager.set_funny_english_controller(controller)
    manager.transition_funny_english_state(SessionState.FE_SELECT)
    renderer.hit_target = SimpleNamespace(kind="next_page", index=None, generation=9)
    renderer.calls.clear()

    manager._dispatch_tap(tap_event(monotonic.value, x=420, y=650))

    assert controller.next_page_calls == 1
    assert flash_feedback_count(renderer) == 1

    renderer.calls.clear()
    manager._dispatch_tap(tap_event(monotonic.value, x=420, y=650))

    assert controller.next_page_calls == 2
    assert flash_feedback_count(renderer) == 1

    renderer.calls.clear()
    renderer.hit_target = SimpleNamespace(kind="menu_item", index=2, generation=9)
    manager._dispatch_tap(tap_event(monotonic.value, x=120, y=220))

    assert controller.select_targets == [2]
    assert flash_feedback_count(renderer) == 1


def test_coordinate_funny_english_exit_tap_exits_mode(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Coordinate taps on the FE exit target leave Funny English mode."""
    renderer = RecordingRenderer()
    pipeline = FakePipeline([None])
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )
    controller = FakeFunnyEnglishController(manager)
    manager.set_funny_english_controller(controller)
    manager.transition_funny_english_state(SessionState.FE_SELECT)
    renderer.hit_target = SimpleNamespace(kind="exit", index=None, generation=9)

    manager._dispatch_tap(tap_event(101.0, x=540, y=40))

    assert controller.exit_calls == 1
    pipeline.set_session_language.assert_called_once_with("ko")
    assert manager.state is SessionState.AWAITING_TAP


def test_funny_english_select_coordinate_miss_does_nothing(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Coordinate-bearing missed FE select taps no longer advance the menu."""
    renderer = RecordingRenderer()
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )
    controller = FakeFunnyEnglishController(manager)
    manager.set_funny_english_controller(controller)
    manager.transition_funny_english_state(SessionState.FE_SELECT)
    renderer.hit_target = None

    manager._dispatch_tap(tap_event(101.0, x=700, y=700))

    assert controller.select_taps == []
    assert controller.select_targets == []


def test_funny_english_select_coordinate_less_tap_keeps_fallback(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Coordinate-less FE select taps keep the fallback select handler."""
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
    )
    controller = FakeFunnyEnglishController(manager)
    manager.set_funny_english_controller(controller)
    manager.transition_funny_english_state(SessionState.FE_SELECT)
    event = tap_event(101.0)

    manager._dispatch_tap(event)

    assert controller.select_taps == [event]
    assert controller.select_targets == []


def test_funny_english_intro_tap_calls_intro_handler(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Any non-exit tap on the FE intro screen advances through the controller."""
    renderer = RecordingRenderer()
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )
    controller = FakeFunnyEnglishController(manager)
    manager.set_funny_english_controller(controller)
    manager.transition_funny_english_state(SessionState.FE_INTRO)
    renderer.hit_target = None

    manager._dispatch_tap(tap_event(101.0, x=120, y=220))

    assert controller.intro_taps == 1


def test_coordinate_funny_english_intro_exit_tap_exits_mode(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """The FE intro screen's exit target leaves the mode instead of starting."""
    renderer = RecordingRenderer()
    pipeline = FakePipeline([None])
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )
    controller = FakeFunnyEnglishController(manager)
    manager.set_funny_english_controller(controller)
    manager.transition_funny_english_state(SessionState.FE_INTRO)
    renderer.hit_target = SimpleNamespace(kind="exit", index=None, generation=9)

    manager._dispatch_tap(tap_event(101.0, x=540, y=40))

    assert controller.exit_calls == 1
    assert controller.intro_taps == 0
    pipeline.set_session_language.assert_called_once_with("ko")
    assert manager.state is SessionState.AWAITING_TAP


def test_funny_english_intro_exit_tap_exits_without_nav_cooldown(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """FE intro exit taps no longer wait for a shared nav cooldown."""
    monotonic = FakeMonotonic()
    renderer = RecordingRenderer()
    pipeline = FakePipeline([None])
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
        renderer=renderer,
        monotonic=monotonic,
    )
    controller = FakeFunnyEnglishController(manager)
    manager.set_funny_english_controller(controller)
    manager.transition_funny_english_state(SessionState.FE_INTRO)
    renderer.hit_target = SimpleNamespace(kind="exit", index=None, generation=9)
    renderer.calls.clear()

    manager._dispatch_tap(tap_event(monotonic.value, x=540, y=40))

    assert controller.exit_calls == 1
    assert controller.intro_taps == 0
    pipeline.set_session_language.assert_called_once_with("ko")
    assert flash_feedback_count(renderer) == 1


@pytest.mark.parametrize(
    ("state", "target_kind"),
    [
        (SessionState.FE_PROMPT, "back"),
        (SessionState.FE_PROMPT, "exit"),
        (SessionState.FE_FEEDBACK, "back"),
        (SessionState.FE_FEEDBACK, "exit"),
        (SessionState.FE_DONE, "back"),
        (SessionState.FE_DONE, "exit"),
    ],
)
def test_coordinate_funny_english_card_nav_tap_dispatches_back_or_exit(
    state: SessionState,
    target_kind: str,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """FE card nav targets flash first, then run back or exit actions."""
    renderer = RecordingRenderer()
    pipeline = FakePipeline([None])
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )
    controller = FakeFunnyEnglishController(manager)
    events: list[str] = []

    def flash_press_feedback(target: Any) -> None:
        del target
        events.append("flash")

    def handle_back() -> None:
        controller.back_calls += 1
        events.append("back")

    def exit_funny_english() -> None:
        controller.exit_calls += 1
        events.append("exit")

    renderer.flash_press_feedback = flash_press_feedback  # type: ignore[method-assign]
    controller.handle_back = handle_back  # type: ignore[method-assign]
    controller.exit = exit_funny_english  # type: ignore[method-assign]
    manager.set_funny_english_controller(controller)
    manager.transition_funny_english_state(state)
    renderer.hit_target = SimpleNamespace(kind=target_kind, index=None, generation=9)

    manager._dispatch_tap(tap_event(101.0, x=40, y=40))

    if target_kind == "back":
        assert events == ["flash", "back"]
        assert controller.back_calls == 1
        assert controller.exit_calls == 0
    else:
        assert events == ["flash", "exit"]
        assert controller.exit_calls == 1
        pipeline.set_session_language.assert_called_once_with("ko")


@pytest.mark.parametrize(
    ("state", "target_kind", "expected_attr", "fallback_attr"),
    [
        (SessionState.FE_PROMPT, "prev_card", "prev_card_calls", "prompt_taps"),
        (SessionState.FE_PROMPT, "next_card", "next_card_calls", "prompt_taps"),
        (SessionState.FE_FEEDBACK, "prev_card", "prev_card_calls", "feedback_taps"),
        (SessionState.FE_FEEDBACK, "next_card", "next_card_calls", "feedback_taps"),
    ],
)
def test_coordinate_funny_english_card_nav_tap_dispatches_prev_or_next_card(
    state: SessionState,
    target_kind: str,
    expected_attr: str,
    fallback_attr: str,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """FE word-nav targets flash first, then route without content fallback."""
    renderer = RecordingRenderer()
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )
    controller = FakeFunnyEnglishController(manager)
    manager.set_funny_english_controller(controller)
    manager.transition_funny_english_state(state)
    renderer.hit_target = SimpleNamespace(kind=target_kind, index=None, generation=9)

    manager._dispatch_tap(tap_event(101.0, x=80, y=650))

    assert getattr(controller, expected_attr) == 1
    assert getattr(controller, fallback_attr) == []
    assert flash_feedback_count(renderer) == 1


def test_funny_english_done_ignores_prev_next_card_targets(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Done state keeps only back/exit card navigation active."""
    renderer = RecordingRenderer()
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )
    controller = FakeFunnyEnglishController(manager)
    manager.set_funny_english_controller(controller)
    manager.transition_funny_english_state(SessionState.FE_DONE)
    renderer.hit_target = SimpleNamespace(kind="next_card", index=None, generation=9)
    event = tap_event(101.0, x=620, y=650)

    manager._dispatch_tap(event)

    assert controller.next_card_calls == 0
    assert controller.done_taps == [event]
    assert flash_feedback_count(renderer) == 0


def test_funny_english_card_nav_repeats_without_content_fallback(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Repeated FE card nav taps run immediately without content tap fallback."""
    monotonic = FakeMonotonic()
    renderer = RecordingRenderer()
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
        monotonic=monotonic,
    )
    controller = FakeFunnyEnglishController(manager)
    manager.set_funny_english_controller(controller)
    manager.transition_funny_english_state(SessionState.FE_PROMPT)
    renderer.hit_target = SimpleNamespace(kind="back", index=None, generation=9)
    renderer.calls.clear()

    manager._dispatch_tap(tap_event(monotonic.value, x=40, y=40))

    assert controller.back_calls == 1
    assert controller.prompt_taps == []
    assert flash_feedback_count(renderer) == 1

    renderer.calls.clear()
    manager._dispatch_tap(tap_event(monotonic.value, x=40, y=40))

    assert controller.back_calls == 2
    assert controller.prompt_taps == []
    assert flash_feedback_count(renderer) == 1


@pytest.mark.parametrize(
    ("state", "expected_attr"),
    [
        (SessionState.FE_PROMPT, "prompt_taps"),
        (SessionState.FE_FEEDBACK, "feedback_taps"),
        (SessionState.FE_DONE, "done_taps"),
    ],
)
def test_coordinate_funny_english_card_non_nav_tap_keeps_state_handler(
    state: SessionState,
    expected_attr: str,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Non-nav card taps still run the current FE card-state handler."""
    renderer = RecordingRenderer()
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )
    controller = FakeFunnyEnglishController(manager)
    manager.set_funny_english_controller(controller)
    manager.transition_funny_english_state(state)
    renderer.hit_target = SimpleNamespace(kind="menu_item", index=0, generation=9)
    event = tap_event(101.0, x=320, y=360)

    manager._dispatch_tap(event)

    assert getattr(controller, expected_attr) == [event]
    assert not any(call[0] == "flash_press_feedback" for call in renderer.calls)


@pytest.mark.parametrize(
    ("state", "expected_attr"),
    [
        (SessionState.FE_PROMPT, "prompt_taps"),
        (SessionState.FE_FEEDBACK, "feedback_taps"),
        (SessionState.FE_DONE, "done_taps"),
    ],
)
def test_funny_english_card_coordinate_less_tap_keeps_fallback(
    state: SessionState,
    expected_attr: str,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Coordinate-less card taps still run the existing FE fallback handler."""
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
    )
    controller = FakeFunnyEnglishController(manager)
    manager.set_funny_english_controller(controller)
    manager.transition_funny_english_state(state)
    event = tap_event(101.0)

    manager._dispatch_tap(event)

    assert getattr(controller, expected_attr) == [event]


class RaisingFunnyEnglishController(FakeFunnyEnglishController):
    """FE controller fake whose prompt tap raises to simulate STT/audio failure.

    Mirrors the real failure surface: ``_run_listen_attempt`` calls
    ``self._mm.load(ModelType.STT, ...)`` (raises FileNotFoundError/OSError/
    RuntimeError on init failure or OOM) and the model/cached-WAV playback path
    calls ``play_audio`` (raises on ALSA/USB-audio device errors).
    """

    def __init__(self, manager: SessionManager, error: BaseException) -> None:
        super().__init__(manager)
        self._error = error

    def handle_prompt_tap(self, event: TouchEvent) -> None:
        """Record the tap, then raise the configured failure."""
        super().handle_prompt_tap(event)
        raise self._error


@pytest.mark.parametrize("exception_type", STT_LOAD_EXCEPTIONS)
def test_funny_english_stt_load_exception_does_not_kill_event_loop(
    exception_type: type[BaseException],
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """An STT-load failure during an FE attempt is contained, not propagated.

    The dispatch firewall must swallow the exception, play the safe error cue,
    and return the device to a stable awake tap state instead of letting the
    exception unwind through ``run()`` and freeze the touchscreen.
    """
    renderer = RecordingRenderer()
    pipeline = FakePipeline([None])
    manager, _mm, _audio_capture, sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )
    play_audio = install_playback_recorder(monkeypatch)
    controller = RaisingFunnyEnglishController(manager, exception_type("stt boom"))
    manager.set_funny_english_controller(controller)
    manager.transition_funny_english_state(SessionState.FE_PROMPT)
    renderer.hit_target = None
    event = tap_event(101.0)

    # Must not raise: the firewall contains the failure.
    manager._dispatch_tap(event)

    assert controller.prompt_taps == [event]
    assert sound_bank.kind_requests == ["stt_load_fail"]
    assert play_audio.call_count == 1
    assert controller.exit_calls == 1
    assert manager.state is SessionState.AWAITING_TAP


def test_funny_english_audio_playback_exception_does_not_kill_event_loop(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """An audio-playback failure during an FE attempt is contained, not propagated."""
    renderer = RecordingRenderer()
    pipeline = FakePipeline([None])
    manager, _mm, _audio_capture, sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )
    play_audio = install_playback_recorder(monkeypatch)
    controller = RaisingFunnyEnglishController(manager, OSError("alsa device busy"))
    manager.set_funny_english_controller(controller)
    manager.transition_funny_english_state(SessionState.FE_PROMPT)
    renderer.hit_target = None
    event = tap_event(101.0)

    # Must not raise: the firewall contains the playback failure.
    manager._dispatch_tap(event)

    assert controller.prompt_taps == [event]
    assert sound_bank.kind_requests == ["stt_load_fail"]
    assert play_audio.call_count == 1
    assert controller.exit_calls == 1
    assert manager.state is SessionState.AWAITING_TAP


def test_funny_english_unexpected_exception_does_not_kill_event_loop(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A non-STT-family exception during an FE attempt is still contained."""
    renderer = RecordingRenderer()
    pipeline = FakePipeline([None])
    manager, _mm, _audio_capture, sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )
    play_audio = install_playback_recorder(monkeypatch)
    controller = RaisingFunnyEnglishController(manager, KeyError("unexpected"))
    manager.set_funny_english_controller(controller)
    manager.transition_funny_english_state(SessionState.FE_PROMPT)
    renderer.hit_target = None
    event = tap_event(101.0)

    # Must not raise: the broad Exception arm of the firewall contains it.
    manager._dispatch_tap(event)

    assert controller.prompt_taps == [event]
    assert sound_bank.kind_requests == ["stt_load_fail"]
    assert play_audio.call_count == 1
    assert controller.exit_calls == 1
    assert manager.state is SessionState.AWAITING_TAP


def test_funny_english_run_loop_survives_fe_dispatch_failure(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """The event loop keeps running after an FE dispatch failure is contained.

    Drives ``run()`` end to end: a failing FE tap must not unwind through the
    loop. After recovery the device is in AWAITING_TAP and the loop stops only
    via the normal ``stop()`` latch, confirming the touchscreen stays responsive.
    """
    renderer = RecordingRenderer()
    touch = FakeTouch()
    pipeline = FakePipeline([None])
    manager, _mm, _audio_capture, sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
        renderer=renderer,
        touch=touch,
    )
    install_playback_recorder(monkeypatch)
    controller = RaisingFunnyEnglishController(manager, RuntimeError("stt init failed"))
    manager.set_funny_english_controller(controller)
    manager.transition_funny_english_state(SessionState.FE_PROMPT)
    renderer.hit_target = None
    touch.events = [tap_event(101.0)]

    polls_after_recovery = {"count": 0}

    def stop_after_recovery() -> None:
        # Called on each empty poll after the queued tap drains.
        polls_after_recovery["count"] += 1
        if polls_after_recovery["count"] >= 1:
            manager.stop()

    touch.on_empty = stop_after_recovery

    # Must complete without the FE failure escaping run().
    manager.run()

    assert controller.prompt_taps  # the failing tap was dispatched
    assert sound_bank.kind_requests == ["stt_load_fail"]
    assert manager.state is SessionState.AWAITING_TAP


def test_no_speech_timeout_rearms_without_sleeping(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A no-speech listen timeout returns to AWAITING_TAP, not SLEEPING."""
    renderer = RecordingRenderer()
    pipeline = FakePipeline([None])
    manager, mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )
    install_playback_recorder(monkeypatch)
    manager._enter_awaiting_tap()
    renderer.calls.clear()

    manager._handle_listen_turn()

    assert state_calls(renderer) == [SessionState.LISTENING, SessionState.AWAITING_TAP]
    assert SessionState.SLEEPING not in state_calls(renderer)
    assert pipeline.timeouts == [LISTEN_TURN_TIMEOUT]
    assert pipeline.run_calls == 0
    mm.unload_stt.assert_not_called()
    assert manager.state is SessionState.AWAITING_TAP


def test_awaiting_tap_idle_timeout_enters_sleeping_then_idle(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """AWAITING_TAP idles out after 45 seconds and releases resources."""
    renderer = RecordingRenderer()
    monotonic = FakeMonotonic()
    touch = FakeTouch()
    pipeline = FakePipeline([None])
    manager, mm, audio_capture, _sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
        renderer=renderer,
        touch=touch,
        monotonic=monotonic,
    )
    install_playback_recorder(monkeypatch)
    manager._enter_awaiting_tap()
    renderer.calls.clear()
    monotonic.advance(AWAITING_TAP_IDLE_TIMEOUT)

    def stop_after_sleep() -> None:
        if manager.state is SessionState.IDLE:
            manager.stop()

    touch.on_empty = stop_after_sleep
    manager.run()

    assert state_calls(renderer) == [SessionState.SLEEPING, SessionState.IDLE]
    mm.cancel_preload.assert_called_once_with()
    mm.unload_stt.assert_called_once_with(force=False)
    assert audio_capture.calls[-1] == "stop"
    assert manager.state is SessionState.IDLE


def test_fresh_tap_during_responding_is_dropped(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Barge-in taps are ignored while RESPONDING."""
    pipeline = FakePipeline([make_utterance()])
    touch = FakeTouch([tap_event(101.0)])
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
        touch=touch,
    )
    manager._transition(SessionState.RESPONDING)
    touch.on_empty = manager.stop

    manager.run()

    assert pipeline.run_calls == 0
    assert manager.state is SessionState.RESPONDING


def test_stale_tap_before_arm_is_discarded(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A tap stamped before _armed_at cannot trigger a turn."""
    pipeline = FakePipeline([make_utterance()])
    touch = FakeTouch([tap_event(99.0)])
    manager, _mm, _audio_capture, sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
        touch=touch,
    )
    manager._state = SessionState.AWAITING_TAP
    manager._armed_at = 100.0
    touch.on_empty = manager.stop

    manager.run()

    assert pipeline.run_calls == 0
    assert sound_bank.ack_requests == 0
    assert manager.state is SessionState.AWAITING_TAP


def test_ack_none_still_listens_and_preserves_turn_playback_guard(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Missing ack audio falls back to a visual-only LISTENING transition."""
    renderer = RecordingRenderer()
    sound_bank = FakeSoundBank(ack_clip=None)
    pipeline = FakePipeline([make_utterance()])
    manager, _mm, audio_capture, _sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
        renderer=renderer,
        sound_bank=sound_bank,
    )
    play_audio = install_playback_recorder(monkeypatch)
    manager._enter_awaiting_tap()
    renderer.calls.clear()
    audio_capture.calls.clear()

    manager._handle_listen_turn()

    assert state_calls(renderer) == [
        SessionState.LISTENING,
        SessionState.RESPONDING,
        SessionState.AWAITING_TAP,
    ]
    assert pipeline.run_calls == 1
    assert sound_bank.ack_requests == 1
    assert play_audio.call_count == 0
    assert_contains_subsequence(audio_capture.calls, _PLAYBACK_GUARD_CALLS)
    assert manager.state is SessionState.AWAITING_TAP


@pytest.mark.parametrize("exception_type", STT_LOAD_EXCEPTIONS)
def test_stt_exception_family_plays_error_and_rearms(
    exception_type: type[BaseException],
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """The configured STT-load exception family routes to error cue then re-arms."""
    pipeline = FakePipeline([make_utterance()], error=exception_type("boom"))
    manager, mm, _audio_capture, sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
    )
    play_audio = install_playback_recorder(monkeypatch)
    manager._enter_awaiting_tap()

    manager._handle_listen_turn()

    assert sound_bank.kind_requests == ["stt_load_fail"]
    assert play_audio.call_count == 2
    assert manager.state is SessionState.AWAITING_TAP
    mm.unload_stt.assert_not_called()


def test_wait_for_utterance_exception_plays_error_rearms_and_next_turn_runs(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A VAD/wait failure is non-fatal and the next valid listen turn runs."""
    pipeline = FakePipeline(
        [make_utterance()],
        wait_error=RuntimeError("vad unavailable"),
    )
    manager, _mm, _audio_capture, sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
    )
    play_audio = install_playback_recorder(monkeypatch)
    manager._enter_awaiting_tap()

    manager._handle_listen_turn()

    assert manager.state is SessionState.AWAITING_TAP
    assert pipeline.run_calls == 0
    assert sound_bank.kind_requests == ["stt_load_fail"]
    assert play_audio.call_count == 2

    manager._handle_listen_turn()

    assert manager.state is SessionState.AWAITING_TAP
    assert pipeline.run_calls == 1
    assert pipeline.timeouts == [LISTEN_TURN_TIMEOUT, LISTEN_TURN_TIMEOUT]
    assert sound_bank.kind_requests == ["stt_load_fail"]
    assert play_audio.call_count == 3


def test_wake_playback_failure_routes_to_error_and_sleep(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Wake audio playback failure is caught by the WAKING outer handler."""
    pipeline = FakePipeline([make_utterance()])
    manager, _mm, audio_capture, sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
    )
    play_audio = MagicMock(side_effect=[RuntimeError("speaker"), None, None])
    monkeypatch.setattr(session_manager.audio_player, "play_audio", play_audio)

    manager._handle_cold_wake()

    assert sound_bank.kind_requests == ["stt_load_fail"]
    assert audio_capture.calls == [
        "start",
        *_PLAYBACK_GUARD_CALLS,
        *_PLAYBACK_GUARD_CALLS,
        *_PLAYBACK_GUARD_CALLS,
        "stop",
    ]
    assert manager.state is SessionState.IDLE


def test_sleep_playback_failure_still_cleans_up(tmp_path: Path, monkeypatch: Any) -> None:
    """Sleep cue failure cannot prevent cleanup and IDLE transition."""
    manager, mm, audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
    )
    play_audio = MagicMock(side_effect=RuntimeError("sleep speaker"))
    monkeypatch.setattr(session_manager.audio_player, "play_audio", play_audio)

    manager._enter_sleeping()

    assert audio_capture.calls == [*_PLAYBACK_GUARD_CALLS, "stop"]
    mm.unload_stt.assert_called_once_with(force=False)
    assert manager.state is SessionState.IDLE


@pytest.mark.parametrize("state", [SessionState.AWAITING_TAP, SessionState.LISTENING])
def test_parent_mode_event_log_schema_accepts_new_states(
    state: SessionState,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Long-press parent requests log the current push-to-talk session state."""
    pipeline = FakePipeline([None])
    manager, _mm, _audio_capture, _sound_bank, event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
    )
    manager._transition(state)

    manager._handle_parent_request(long_press_event(101.0))

    entries = event_log.read_entries()
    assert entries[0]["schema_version"] == 1
    assert entries[0]["event"] == "parent_mode_requested"
    assert entries[0]["session_state_at_request"] in {"awaiting_tap", "listening"}
    assert entries[0]["session_state_at_request"] == state.value
    assert entries[0]["press_duration_ms"] == 3000
    assert entries[0]["session_id"] == "session-1"


def test_long_press_in_english_session_reverts_to_korean(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Long-press provides a non-voice recovery path from English back to Korean."""
    cue_clip = (np.array([1, 2, 3], dtype=np.int16), 22_050)
    pipeline = FakePipeline([None])
    pipeline.session_language = "en"
    renderer = RecordingRenderer()
    sound_bank = FakeSoundBank(language_switch_clip=cue_clip)
    manager, _mm, _audio_capture, _sound_bank, event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
        renderer=renderer,
        sound_bank=sound_bank,
    )
    manager.set_language_indicator("en")
    renderer.calls.clear()
    sound_bank.language_switch_requests = 0
    play_guard = MagicMock()
    manager._play_audio_with_capture_guard = play_guard  # type: ignore[method-assign]

    manager._handle_parent_request(long_press_event(101.0))

    assert pipeline.session_language == "ko"
    pipeline.set_session_language.assert_called_once_with("ko")
    assert renderer.calls == [("language", "ko")]
    assert sound_bank.language_switch_requests == 1
    play_guard.assert_called_once_with(cue_clip[0], cue_clip[1])
    assert event_log.read_entries() == []


def test_long_press_exits_history_before_parent_mode_or_language_recovery(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Long-press exits history mode before parent-mode or EN-to-KO handling."""
    pipeline = FakePipeline([None])
    pipeline.session_language = "en"
    manager, _mm, _audio_capture, _sound_bank, event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
    )
    monkeypatch.setattr(session_manager.audio_player, "stop_playback", MagicMock())
    controller = FakeHistoryController(manager)
    manager.set_history_controller(controller)
    manager.transition_history_state(SessionState.HISTORY_SELECT)

    manager._handle_parent_request(long_press_event(101.0))

    assert controller.exit_calls == 1
    pipeline.set_session_language.assert_not_called()
    assert event_log.read_entries() == []
    assert manager.state is SessionState.AWAITING_TAP


def test_tap_in_history_consent_routes_to_consent_handler(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Tap at the consent gate continues through the history controller."""
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
    )
    controller = FakeHistoryController(manager)
    manager.set_history_controller(controller)
    manager.transition_history_state(SessionState.HISTORY_CONSENT)

    event = tap_event(101.0)
    manager._dispatch_tap(event)

    assert controller.consent_taps == [event]


def test_coordinate_history_select_tap_dispatches_hit_target(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Coordinate taps on menu items select directly without fallback cycling."""
    renderer = RecordingRenderer()
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )
    controller = FakeHistoryController(manager)
    manager.set_history_controller(controller)
    manager.transition_history_state(SessionState.HISTORY_SELECT)
    renderer.hit_target = SimpleNamespace(kind="menu_item", index=2, generation=7)

    manager._dispatch_tap(tap_event(101.0, x=100, y=200))

    assert controller.select_targets == [2]
    assert controller.select_taps == []
    assert ("hit_test", (100, 200)) in renderer.calls
    assert any(call[0] == "flash_press_feedback" for call in renderer.calls)


@pytest.mark.parametrize(
    ("target_kind", "expected_attr"),
    [
        ("back", "back_calls"),
        ("prev_page", "prev_page_calls"),
        ("next_page", "next_page_calls"),
    ],
)
def test_coordinate_history_select_tap_dispatches_nav_targets(
    target_kind: str,
    expected_attr: str,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Coordinate taps on history nav targets route to the matching controller method."""
    renderer = RecordingRenderer()
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )
    controller = FakeHistoryController(manager)
    manager.set_history_controller(controller)
    manager.transition_history_state(SessionState.HISTORY_SELECT)
    renderer.hit_target = SimpleNamespace(kind=target_kind, index=None, generation=7)

    manager._dispatch_tap(tap_event(101.0, x=100, y=200))

    assert getattr(controller, expected_attr) == 1
    assert controller.select_taps == []
    assert any(call[0] == "flash_press_feedback" for call in renderer.calls)


def test_history_select_nav_taps_act_immediately_and_menu_items_still_work(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """History selection nav taps handle repeated taps without a shared cooldown."""
    monotonic = FakeMonotonic()
    renderer = RecordingRenderer()
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
        monotonic=monotonic,
    )
    controller = FakeHistoryController(manager)
    manager.set_history_controller(controller)
    manager.transition_history_state(SessionState.HISTORY_SELECT)
    renderer.hit_target = SimpleNamespace(kind="back", index=None, generation=7)
    renderer.calls.clear()

    manager._dispatch_tap(tap_event(monotonic.value, x=100, y=200))

    assert controller.back_calls == 1
    assert flash_feedback_count(renderer) == 1

    renderer.calls.clear()
    manager._dispatch_tap(tap_event(monotonic.value, x=100, y=200))

    assert controller.back_calls == 2
    assert flash_feedback_count(renderer) == 1

    renderer.calls.clear()
    renderer.hit_target = SimpleNamespace(kind="menu_item", index=1, generation=7)
    manager._dispatch_tap(tap_event(monotonic.value, x=100, y=200))

    assert controller.select_targets == [1]
    assert flash_feedback_count(renderer) == 1


def test_coordinate_history_exit_tap_exits_mode(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Coordinate taps on the renderer exit target leave history mode."""
    renderer = RecordingRenderer()
    manager, _mm, _audio_capture, _sound_bank, event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )
    monkeypatch.setattr(session_manager.audio_player, "stop_playback", MagicMock())
    controller = FakeHistoryController(manager)
    manager.set_history_controller(controller)
    manager.transition_history_state(SessionState.HISTORY_SCENE)
    renderer.hit_target = SimpleNamespace(kind="exit", index=None, generation=8)

    manager._dispatch_tap(tap_event(101.0, x=530, y=40))

    assert controller.exit_calls == 1
    assert controller.scene_taps == []
    assert event_log.read_entries() == []
    assert manager.state is SessionState.AWAITING_TAP


def test_history_exit_target_exits_without_content_fallback_or_nav_cooldown(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """History exit target runs immediately and does not fall back to scene taps."""
    monotonic = FakeMonotonic()
    renderer = RecordingRenderer()
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
        monotonic=monotonic,
    )
    monkeypatch.setattr(session_manager.audio_player, "stop_playback", MagicMock())
    controller = FakeHistoryController(manager)
    manager.set_history_controller(controller)
    manager.transition_history_state(SessionState.HISTORY_SCENE)
    renderer.hit_target = SimpleNamespace(kind="exit", index=None, generation=8)
    renderer.calls.clear()

    manager._dispatch_tap(tap_event(monotonic.value, x=530, y=40))

    assert controller.exit_calls == 1
    assert controller.scene_taps == []
    assert flash_feedback_count(renderer) == 1
    assert manager.state is SessionState.AWAITING_TAP


def test_history_scene_back_target_routes_to_previous_step(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A scene-view back target rewinds to the previous step, not a scene tap."""
    renderer = RecordingRenderer()
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )
    controller = FakeHistoryController(manager)
    manager.set_history_controller(controller)
    manager.transition_history_state(SessionState.HISTORY_SCENE)
    renderer.hit_target = SimpleNamespace(kind="back", index=None, generation=8)

    manager._dispatch_tap(tap_event(101.0, x=60, y=40))

    assert controller.previous_step_calls == 1
    assert controller.scene_taps == []
    assert controller.exit_calls == 0
    assert flash_feedback_count(renderer) == 1
    assert manager.state is SessionState.HISTORY_SCENE


def test_history_scene_non_nav_target_falls_back_to_scene_tap(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A scene tap that misses the back/exit targets advances the slideshow."""
    renderer = RecordingRenderer()
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )
    controller = FakeHistoryController(manager)
    manager.set_history_controller(controller)
    manager.transition_history_state(SessionState.HISTORY_SCENE)
    renderer.hit_target = None

    event = tap_event(101.0, x=360, y=300)
    manager._dispatch_tap(event)

    assert controller.previous_step_calls == 0
    assert controller.exit_calls == 0
    assert controller.scene_taps == [event]


def test_history_pending_teardown_drops_next_listen_tap(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A listen tap immediately after history exit is dropped while teardown is pending."""
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([make_utterance()]),
        tmp_path,
        monkeypatch,
    )
    controller = FakeHistoryController(manager)
    controller.next_exit_future = Future()
    manager.set_history_controller(controller)
    manager.transition_history_state(SessionState.HISTORY_SCENE)
    listen_turn = MagicMock()
    manager._handle_listen_turn = listen_turn  # type: ignore[method-assign]

    manager._exit_history_mode()
    manager._dispatch_tap(tap_event(manager._armed_at + 0.1))

    assert controller.exit_restore_stt == [True]
    assert manager.state is SessionState.AWAITING_TAP
    listen_turn.assert_not_called()

    controller.next_exit_future.set_result(None)


def test_history_pending_teardown_defers_awaiting_tap_idle_sleep(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """AWAITING_TAP idle timeout is deferred until history teardown completes."""
    monotonic = FakeMonotonic()
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        monotonic=monotonic,
    )
    controller = FakeHistoryController(manager)
    controller.next_exit_future = Future()
    manager.set_history_controller(controller)
    manager.transition_history_state(SessionState.HISTORY_SCENE)
    enter_sleeping = MagicMock()
    manager._enter_sleeping = enter_sleeping  # type: ignore[method-assign]

    manager._exit_history_mode()
    monotonic.advance(AWAITING_TAP_IDLE_TIMEOUT)
    manager._handle_idle_poll()

    enter_sleeping.assert_not_called()
    assert manager.state is SessionState.AWAITING_TAP

    controller.next_exit_future.set_result(None)


def test_history_select_coordinate_miss_does_nothing(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Coordinate-bearing missed select taps no longer advance the menu."""
    renderer = RecordingRenderer()
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )
    controller = FakeHistoryController(manager)
    manager.set_history_controller(controller)
    manager.transition_history_state(SessionState.HISTORY_SELECT)
    renderer.hit_target = None
    event = tap_event(101.0, x=700, y=700)

    manager._dispatch_tap(event)

    assert controller.select_taps == []
    assert controller.select_targets == []


def test_history_select_coordinate_less_tap_keeps_fallback(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Coordinate-less select taps preserve the accessibility cycle fallback."""
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
    )
    controller = FakeHistoryController(manager)
    manager.set_history_controller(controller)
    manager.transition_history_state(SessionState.HISTORY_SELECT)
    event = tap_event(101.0)

    manager._dispatch_tap(event)

    assert controller.select_taps == [event]
    assert controller.select_targets == []


def test_history_sleep_path_runs_cleanup_after_teardown_future(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """History sleep exit returns immediately and finalizes sleep after teardown."""
    renderer = RecordingRenderer()
    manager, mm, audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
    )
    install_playback_recorder(monkeypatch)
    controller = FakeHistoryController(manager)
    controller.next_exit_future = Future()
    manager.set_history_controller(controller)
    manager.transition_history_state(SessionState.HISTORY_CONSENT)
    renderer.calls.clear()

    manager._sleep_from_history_mode()

    assert controller.exit_restore_stt == [False]
    assert state_calls(renderer) == [SessionState.SLEEPING]
    assert mm.unload_stt.call_count == 0
    assert audio_capture.calls == []
    assert manager.state is SessionState.SLEEPING

    controller.next_exit_future.set_result(None)
    manager._join_history_sleep_finalizer(timeout=1.0)

    mm.cancel_preload.assert_called_once_with()
    mm.unload_stt.assert_called_once_with(force=False)
    assert audio_capture.calls[-1] == "stop"
    assert manager.state is SessionState.IDLE


def test_stop_closes_controller_and_cancels_pending_history_sleep_finalizer(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """stop() drains controller teardown before audio teardown and skips finalizer cleanup."""
    call_log: list[str] = []
    manager, mm, audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
    )
    controller = FakeHistoryController(manager)
    pending = Future()
    controller.next_exit_future = pending
    manager.set_history_controller(controller)
    manager.transition_history_state(SessionState.HISTORY_CONSENT)
    manager._sleep_from_history_mode()
    sleep_cleanup = MagicMock(side_effect=lambda: call_log.append("sleep_cleanup"))
    manager._run_sleep_cleanup = sleep_cleanup  # type: ignore[method-assign]

    def close_controller() -> None:
        call_log.append("controller_close")
        pending.set_result(None)

    controller.close = close_controller  # type: ignore[method-assign]
    audio_capture.stop = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda: call_log.append("audio_stop")
    )

    manager.stop()

    assert call_log == ["controller_close", "audio_stop"]
    sleep_cleanup.assert_not_called()
    mm.unload_stt.assert_not_called()


def test_history_consent_idle_reprompts_once_then_sleeps(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Consent idle timeout asks once, then exits history and enters sleep."""
    monotonic = FakeMonotonic()
    renderer = RecordingRenderer()
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
        monotonic=monotonic,
    )
    install_playback_recorder(monkeypatch)
    monkeypatch.setattr(session_manager.audio_player, "stop_playback", MagicMock())
    controller = FakeHistoryController(manager)
    controller.consent_timeout_results = [True, False]
    manager.set_history_controller(controller)
    manager.transition_history_state(SessionState.HISTORY_CONSENT)

    monotonic.advance(HISTORY_CONSENT_REPROMPT_TIMEOUT - 0.1)
    manager._handle_idle_poll()

    assert controller.consent_timeout_calls == 0

    monotonic.advance(0.1)
    manager._handle_idle_poll()

    assert controller.poll_calls == 2
    assert controller.consent_timeout_calls == 1
    assert controller.exit_calls == 0
    assert manager.state is SessionState.HISTORY_CONSENT

    monotonic.advance(HISTORY_CONSENT_SLEEP_TIMEOUT)
    manager._handle_idle_poll()
    manager._join_history_sleep_finalizer(timeout=1.0)

    assert controller.consent_timeout_calls == 2
    assert controller.exit_calls == 1
    assert controller.exit_restore_stt == [False]
    assert manager.state is SessionState.IDLE


def test_history_select_idle_uses_dedicated_timeout(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """History selection sleeps on its own child-friendly timeout."""
    monotonic = FakeMonotonic()
    renderer = RecordingRenderer()
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
        renderer=renderer,
        monotonic=monotonic,
    )
    install_playback_recorder(monkeypatch)
    monkeypatch.setattr(session_manager.audio_player, "stop_playback", MagicMock())
    controller = FakeHistoryController(manager)
    manager.set_history_controller(controller)
    manager.transition_history_state(SessionState.HISTORY_SELECT)

    monotonic.advance(AWAITING_TAP_IDLE_TIMEOUT)
    manager._handle_idle_poll()

    assert controller.exit_calls == 0
    assert manager.state is SessionState.HISTORY_SELECT

    monotonic.advance(HISTORY_SELECT_IDLE_TIMEOUT - AWAITING_TAP_IDLE_TIMEOUT)
    manager._handle_idle_poll()
    manager._join_history_sleep_finalizer(timeout=1.0)

    assert controller.exit_calls == 1
    assert controller.exit_restore_stt == [False]
    assert manager.state is SessionState.IDLE
    assert SessionState.SLEEPING in state_calls(renderer)


def test_long_press_in_korean_session_keeps_parent_stub(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Korean-session long-press keeps the existing parent-request stub behavior."""
    pipeline = FakePipeline([None])
    manager, _mm, _audio_capture, _sound_bank, event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
    )

    manager._handle_parent_request(long_press_event(101.0))

    pipeline.set_session_language.assert_not_called()
    entries = event_log.read_entries()
    assert entries[0]["event"] == "parent_mode_requested"
    assert entries[0]["handled"] is False


def test_tap_in_english_session_does_not_trigger_language_recovery(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Only long-press, not a normal tap, triggers the English-to-Korean recovery path."""
    pipeline = FakePipeline([None])
    pipeline.session_language = "en"
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
    )
    install_playback_recorder(monkeypatch)
    manager._enter_awaiting_tap()

    manager._dispatch_tap(tap_event(manager._armed_at + 0.1))

    assert pipeline.session_language == "en"
    pipeline.set_session_language.assert_not_called()


def test_wake_day_state_file_and_memory_fallback(tmp_path: Path, monkeypatch: Any) -> None:
    """Wake-day state uses runtime mutable root and falls back to memory on write errors."""
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        FakePipeline([None]),
        tmp_path,
        monkeypatch,
    )
    day = manager._compute_wake_day(datetime(2026, 5, 26, 4, 30))
    assert day.isoformat() == "2026-05-25"

    manager._write_last_wake_day(day)
    assert manager._read_last_wake_day() == day

    state_path = tmp_path / "state" / "last_wake_date.json"
    state_path.write_text("{not-json", encoding="utf-8")
    manager._last_wake_day_fallback = None
    assert manager._read_last_wake_day() is None

    monkeypatch.setattr(session_manager.Path, "open", MagicMock(side_effect=OSError("ro")))
    fallback_day = manager._compute_wake_day(datetime(2026, 5, 27, 6, 0))
    manager._write_last_wake_day(fallback_day)
    assert manager._read_last_wake_day() == fallback_day


def test_run_logs_long_press_without_dropping_parent_request(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Long-press requests are preserved and logged from the run loop."""
    pipeline = FakePipeline([None])
    touch = FakeTouch([long_press_event(101.0)])
    manager, _mm, _audio_capture, _sound_bank, event_log, _touch = build_manager(
        pipeline,
        tmp_path,
        monkeypatch,
        touch=touch,
    )
    manager._transition(SessionState.AWAITING_TAP)
    touch.on_empty = manager.stop

    manager.run()

    assert event_log.read_entries()[0]["event"] == "parent_mode_requested"
    assert manager.state is SessionState.AWAITING_TAP
    pipeline.set_session_language.assert_not_called()


def test_pipeline_factory_receives_model_manager(tmp_path: Path) -> None:
    """SessionManager constructs its pipeline through pipeline_factory(mm)."""
    pipeline = FakePipeline([None])
    mm = MagicMock()
    factory = MagicMock(return_value=pipeline)

    SessionManager(
        mm=mm,
        pipeline_factory=factory,
        touch=FakeTouch(),  # type: ignore[arg-type]
        sound_bank=FakeSoundBank(),  # type: ignore[arg-type]
        audio_capture=FakeAudioCapture(),  # type: ignore[arg-type]
        event_log=EventLog(tmp_path / "events.jsonl"),
        monotonic_clock=FakeMonotonic(),
    )

    factory.assert_called_once_with(mm)


def test_portal_activate_double_tap_starts_portal_and_confirms(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A deliberate second portal tap starts and confirms the download portal."""
    from parental.download_portal import control as portal_control

    renderer = RecordingRenderer()
    renderer.hit_target = SimpleNamespace(kind="portal_activate", index=None, generation=4)
    pipeline = FakePipeline([make_utterance()])
    manager, _mm, _audio_capture, _sound_bank, _event_log, _touch = build_manager(
        pipeline, tmp_path, monkeypatch, renderer=renderer
    )
    starts: list[str] = []
    statuses_seen_at_start: list[tuple[str, float]] = []

    def start_service(**_: Any) -> Any:
        statuses_seen_at_start.extend(portal_status_calls(renderer))
        starts.append("start")
        return portal_control.ControlResult(True, "start", "ok")

    monkeypatch.setattr(portal_control, "start_service", start_service)
    play_audio = install_playback_recorder(monkeypatch)
    manager._enter_awaiting_tap()

    manager._dispatch_tap(tap_event(101.0, x=20, y=20))  # first tap: single -> wait
    assert starts == []
    assert portal_status_calls(renderer) == [("한 번 더", session_manager._PORTAL_CONFIRM_WINDOW_S)]
    assert flash_feedback_count(renderer) == 1
    manager._dispatch_tap(tap_event(102.5, x=20, y=20))  # second tap in window: double

    assert starts == ["start"]
    assert statuses_seen_at_start == [
        ("한 번 더", session_manager._PORTAL_CONFIRM_WINDOW_S),
        ("다운로드 모드 켜짐", session_manager._PORTAL_ACTIVATED_STATUS_S),
    ]
    assert portal_status_calls(renderer) == [
        ("한 번 더", session_manager._PORTAL_CONFIRM_WINDOW_S),
        ("다운로드 모드 켜짐", session_manager._PORTAL_ACTIVATED_STATUS_S),
    ]
    assert flash_feedback_count(renderer) == 2
    play_audio.assert_called_once()
    assert pipeline.run_calls == 0
    assert manager.state is SessionState.AWAITING_TAP


def test_portal_activate_double_tap_clears_confirmation_when_start_fails(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A failed portal start clears the optimistic confirmation and plays error feedback."""
    from parental.download_portal import control as portal_control

    renderer = RecordingRenderer()
    renderer.hit_target = SimpleNamespace(kind="portal_activate", index=None, generation=4)
    pipeline = FakePipeline([make_utterance()])
    manager, _mm, _audio_capture, sound_bank, _event_log, _touch = build_manager(
        pipeline, tmp_path, monkeypatch, renderer=renderer
    )
    starts: list[str] = []

    def start_service(**_: Any) -> Any:
        starts.append("start")
        return portal_control.ControlResult(False, "start", "failed")

    monkeypatch.setattr(portal_control, "start_service", start_service)
    play_audio = install_playback_recorder(monkeypatch)
    manager._enter_awaiting_tap()

    manager._dispatch_tap(tap_event(101.0, x=20, y=20))
    manager._dispatch_tap(tap_event(102.5, x=20, y=20))

    assert starts == ["start"]
    assert portal_status_calls(renderer) == [
        ("한 번 더", session_manager._PORTAL_CONFIRM_WINDOW_S),
        ("다운로드 모드 켜짐", session_manager._PORTAL_ACTIVATED_STATUS_S),
        ("", 0.0),
    ]
    assert sound_bank.ack_requests == 0
    assert sound_bank.kind_requests == ["stt_load_fail"]
    play_audio.assert_called_once()
    assert manager.state is SessionState.AWAITING_TAP


def test_portal_activate_first_tap_shows_progress_hint_and_does_not_start(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A lone portal tap shows the progress hint and waits for the second tap."""
    from parental.download_portal import control as portal_control

    renderer = RecordingRenderer()
    renderer.hit_target = SimpleNamespace(kind="portal_activate", index=None, generation=4)
    pipeline = FakePipeline([make_utterance()])
    manager, *_rest = build_manager(pipeline, tmp_path, monkeypatch, renderer=renderer)
    starts: list[str] = []
    monkeypatch.setattr(portal_control, "start_service", lambda **_: starts.append("start"))
    manager._enter_awaiting_tap()

    manager._dispatch_tap(tap_event(101.0, x=20, y=20))

    assert starts == []
    assert portal_status_calls(renderer) == [("한 번 더", session_manager._PORTAL_CONFIRM_WINDOW_S)]
    assert flash_feedback_count(renderer) == 1
    assert pipeline.run_calls == 0


def test_portal_activate_tap_outside_awaiting_tap_is_ignored_without_feedback(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Portal taps outside AWAITING_TAP do not activate or flash the disabled button."""
    from parental.download_portal import control as portal_control

    renderer = RecordingRenderer()
    pipeline = FakePipeline([make_utterance()])
    manager, *_rest = build_manager(pipeline, tmp_path, monkeypatch, renderer=renderer)
    starts: list[str] = []
    monkeypatch.setattr(
        portal_control,
        "start_service",
        lambda **_: (starts.append("start"), portal_control.ControlResult(True, "start", "ok"))[1],
    )
    manager._transition(SessionState.RESPONDING)

    manager._handle_portal_activate_tap(tap_event(101.0, x=20, y=20))
    manager._handle_portal_activate_tap(tap_event(101.3, x=20, y=20))

    assert starts == []
    assert portal_status_calls(renderer) == []
    assert flash_feedback_count(renderer) == 0
