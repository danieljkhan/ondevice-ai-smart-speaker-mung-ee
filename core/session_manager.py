"""Touchscreen-driven conversation session state machine."""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import suppress
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from core._tap_gesture import DoubleTapGesture
from core.audio_capture import AudioCapture
from core.character_expression import CharacterExpression
from core.event_log import EventLog
from core.funny_english_match import (
    DEFAULT_FUNNY_ENGLISH_PASS_PCT,
    DEFAULT_FUNNY_ENGLISH_PASS_SIMILARITY,
    match_funny_english_attempt,
)
from core.model_manager import ModelManager, ModelType
from core.pipeline import ConversationPipeline, Utterance
from core.runtime import detect_runtime_paths
from core.sound_bank import WAKE_DAY_HOUR_OFFSET, SoundBank
from hardware import audio_player
from hardware.touch_input import TouchEvent, TouchInputListener

logger = logging.getLogger(__name__)

LISTEN_TURN_TIMEOUT = 17.0
FUNNY_ENGLISH_LISTEN_TIMEOUT = 10.0
FUNNY_ENGLISH_LISTEN_POLL_SLICE = 0.4
AWAITING_TAP_IDLE_TIMEOUT = 45.0
HISTORY_SELECT_IDLE_TIMEOUT = 180.0
HISTORY_CONSENT_REPROMPT_TIMEOUT = 75.0
HISTORY_CONSENT_SLEEP_TIMEOUT = 150.0
PRELOAD_JOIN_TIMEOUT = 2.0
WAKE_SCREEN_LEAD_ENV = "MUNGI_WAKE_SCREEN_LEAD_S"
WAKE_SCREEN_LEAD_DEFAULT_S = 0.4
WAKE_SCREEN_LEAD_MAX_S = 3.0
_PORTAL_CONFIRM_WINDOW_S = 3.0
_PORTAL_ACTIVATED_STATUS_S = 3.0

STT_LOAD_EXCEPTIONS = (ImportError, ValueError, FileNotFoundError, RuntimeError, OSError)


class SessionState(Enum):
    """Top-level touchscreen session states."""

    IDLE = "idle"
    WAKING = "waking"
    AWAITING_TAP = "awaiting_tap"
    LISTENING = "listening"
    RESPONDING = "responding"
    SLEEPING = "sleeping"
    HISTORY_SELECT = "history_select"
    HISTORY_SCENE = "history_scene"
    HISTORY_CONSENT = "history_consent"
    HISTORY_PAUSED = "history_paused"
    HISTORY_DONE = "history_done"
    FE_INTRO = "fe_intro"
    FE_SELECT = "fe_select"
    FE_PROMPT = "fe_prompt"
    FE_LISTEN = "fe_listen"
    FE_SCORE = "fe_score"
    FE_FEEDBACK = "fe_feedback"
    FE_DONE = "fe_done"


FunnyEnglishListenInterruptKind = Literal["exit", "back"]


@dataclass(frozen=True)
class FunnyEnglishListenInterrupt:
    """A direct-touch navigation request captured during FE listening."""

    kind: FunnyEnglishListenInterruptKind


class RenderHitTarget(Protocol):
    """Renderer-owned direct-touch hit-test target.

    The attributes are read-only so a frozen dataclass satisfies the protocol.
    """

    @property
    def kind(self) -> str:
        """Target kind, such as "menu_item", "exit", or "language_toggle"."""

    @property
    def index(self) -> int | None:
        """Menu item index (None for the exit target)."""

    @property
    def generation(self) -> int:
        """Layout generation this target was computed for."""


class CharacterRenderer(Protocol):
    """Renderer protocol for Phase 2 character output."""

    def on_state_change(self, state: SessionState) -> None:
        """Handle a session state change."""

    def on_expression(self, expression: CharacterExpression) -> None:
        """Handle a character expression change."""

    def on_language_change(self, lang: str) -> None:
        """Handle a session language change."""

    def show_image(self, path: str | Path, *, letterbox: bool = True) -> str | None:
        """Display a full-screen image."""

    def clear_image(self) -> None:
        """Clear the full-screen image layer."""

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
        """Display generic text or menu content."""

    def clear_text(self) -> None:
        """Clear the text layer."""

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
    ) -> str | None:
        """Display a generic image+text card."""

    def wait_until_rendered(self, token: str | None, timeout: float) -> bool:
        """Wait until a tokenized render request is displayed."""

    def show_history_image(self, path: str | Path) -> str | None:
        """Display a history-mode image."""

    def show_history_menu(
        self,
        items: list[str],
        highlight: int,
        title: str,
        *,
        has_back: bool = False,
        page_index: int = 0,
        page_count: int = 1,
    ) -> str | None:
        """Display a history-mode menu."""

    def hit_test(self, x: int, y: int) -> RenderHitTarget | None:
        """Return the current direct-touch target at screen coordinate ``x,y``."""

    def show_press_feedback(self, target: RenderHitTarget) -> str | None:
        """Render short pressed-state feedback for a direct-touch target."""

    def flash_press_feedback(self, target: RenderHitTarget) -> None:
        """Render and briefly hold direct-touch press feedback."""

    def show_portal_status(self, text: str, *, duration: float) -> None:
        """Render a transient status message next to the portal button."""

    def close(self) -> None:
        """Release renderer resources."""


class NullCharacterRenderer:
    """No-op renderer used until Phase 2 display work lands."""

    def on_state_change(self, state: SessionState) -> None:
        """Ignore a session state change."""
        del state

    def on_expression(self, expression: CharacterExpression) -> None:
        """Ignore a character expression change."""
        del expression

    def on_language_change(self, lang: str) -> None:
        """Ignore a session language change."""
        del lang

    def show_image(self, path: str | Path, *, letterbox: bool = True) -> str | None:
        """Ignore an image display request."""
        del path, letterbox
        return None

    def clear_image(self) -> None:
        """Ignore an image clear request."""

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
        """Ignore a text display request."""
        del lines, size, highlight_index, title, layout, show_exit_button
        del has_back, page_index, page_count

    def clear_text(self) -> None:
        """Ignore a text clear request."""

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
    ) -> str | None:
        """Ignore a card display request."""
        del image_path, lines, highlight_index, title, sublabel, show_exit_button, show_back
        del show_prev_card, show_next_card
        return None

    def wait_until_rendered(self, token: str | None, timeout: float) -> bool:
        """Treat no-op rendering as complete."""
        del token, timeout
        return True

    def show_history_image(self, path: str | Path) -> str | None:
        """Ignore a history image display request."""
        del path
        return None

    def show_history_menu(
        self,
        items: list[str],
        highlight: int,
        title: str,
        *,
        has_back: bool = False,
        page_index: int = 0,
        page_count: int = 1,
    ) -> str | None:
        """Ignore a history menu display request."""
        del items, highlight, title, has_back, page_index, page_count
        return None

    def hit_test(self, x: int, y: int) -> RenderHitTarget | None:
        """No-op renderers have no direct-touch targets."""
        del x, y
        return None

    def show_press_feedback(self, target: RenderHitTarget) -> str | None:
        """Ignore press feedback."""
        del target
        return None

    def flash_press_feedback(self, target: RenderHitTarget) -> None:
        """Ignore synchronous press feedback."""
        del target

    def show_portal_status(self, text: str, *, duration: float) -> None:
        """Ignore portal status feedback."""
        del text, duration

    def close(self) -> None:
        """Release no resources."""


class SessionPipeline(Protocol):
    """Step-4 pipeline API consumed by the session manager."""

    @property
    def session_id(self) -> str | None:
        """Return the active session identifier if one exists."""

    @property
    def session_language(self) -> str:
        """Return the current explicit session response language."""

    def set_session_language(self, lang: str) -> None:
        """Set the explicit session response language."""

    def switch_session_language_with_confirmation(self, target_language: str) -> Any:
        """Set a session language and play its deterministic confirmation."""

    def reset_session(self) -> None:
        """Reset the pipeline to a fresh conversation session."""

    def set_playback_gate(
        self,
        on_start: Callable[[], None] | None,
        on_end: Callable[[], None] | None,
    ) -> None:
        """Register callbacks invoked around local TTS playback."""

    def set_expression_sink(
        self,
        cb: Callable[[CharacterExpression], None] | None,
    ) -> None:
        """Register a callback for content-driven character expressions."""

    def set_language_sink(
        self,
        cb: Callable[[str], None] | None,
    ) -> None:
        """Register a callback for session-language changes."""

    def set_history_mode_sink(
        self,
        cb: Callable[[], None] | None,
    ) -> None:
        """Register a callback for history-mode entry."""

    def set_funny_english_sink(
        self,
        cb: Callable[[], None] | None,
    ) -> None:
        """Register a callback for Funny English entry."""

    def wait_for_utterance(
        self,
        audio_capture: AudioCapture,
        *,
        timeout: float,
    ) -> Utterance | None:
        """Wait for one streaming VAD utterance."""

    def run_turn_with_audio(self, utterance: Utterance) -> None:
        """Run one turn from an already-detected utterance."""

    def run_funny_english_attempt(
        self,
        utterance: Utterance,
        card: Any,
        *,
        stt_hotwords_csv: str | None = None,
    ) -> Any:
        """Run one Funny English STT-only attempt."""


class SessionModelManager(Protocol):
    """Model-manager API consumed by the session manager."""

    def load(self, model_type: ModelType, *, stt_hotwords_csv: str | None = None) -> None:
        """Load one model."""

    def reset_preload_state(self) -> None:
        """Reset any in-flight STT preload before a new session."""

    def preload_stt(self) -> None:
        """Start STT preload best-effort."""

    def cancel_preload(self) -> None:
        """Cancel pending STT preload work."""

    def cancel_preload_and_join(self) -> None:
        """Cancel pending STT preload work and wait briefly for it."""

    def unload_stt(self, *, force: bool = False) -> None:
        """Unload the STT model."""

    def unload_tts(self, *, force: bool = False) -> None:
        """Unload the TTS model."""


class HistoryController(Protocol):
    """History-mode controller API consumed by the session manager."""

    def enter(self) -> None:
        """Enter history mode."""

    def exit(self, *, restore_stt: bool) -> Future[None]:
        """Exit history mode."""

    def close(self) -> None:
        """Drain and stop history-mode background resources."""

    def handle_select_tap(self, event: TouchEvent) -> None:
        """Handle a tap in catalogue selection."""

    def handle_select_target(self, index: int) -> None:
        """Handle a direct menu-item target in catalogue selection."""

    def handle_back(self) -> None:
        """Handle a direct back target in catalogue selection."""

    def handle_prev_page(self) -> None:
        """Handle a direct previous-page target in catalogue selection."""

    def handle_next_page(self) -> None:
        """Handle a direct next-page target in catalogue selection."""

    def handle_scene_tap(self, event: TouchEvent) -> None:
        """Handle a tap in slideshow scene playback."""

    def handle_previous_step(self) -> None:
        """Handle a direct previous-step back target in slideshow playback."""

    def handle_consent_tap(self, event: TouchEvent) -> None:
        """Handle a tap at a history section consent gate."""

    def handle_consent_timeout(self) -> bool:
        """Handle a consent idle timeout and return True when re-prompted."""

    def handle_paused_tap(self, event: TouchEvent) -> None:
        """Handle a tap in the reserved paused state."""

    def handle_done_tap(self, event: TouchEvent) -> None:
        """Handle a tap in the done state."""

    def poll(self) -> None:
        """Run non-blocking timer maintenance."""


class FunnyEnglishController(Protocol):
    """Funny English controller API consumed by the session manager."""

    def enter(self) -> None:
        """Enter Funny English mode."""

    def exit(self) -> None:
        """Exit Funny English mode."""

    def handle_select_tap(self, event: TouchEvent) -> None:
        """Handle a tap in stage/card selection."""

    def handle_intro_tap(self) -> None:
        """Handle a tap on the Funny English instruction screen."""

    def handle_select_target(self, index: int) -> None:
        """Handle a direct stage menu-item target."""

    def handle_prev_page(self) -> None:
        """Handle a direct previous-page target in stage selection."""

    def handle_next_page(self) -> None:
        """Handle a direct next-page target in stage selection."""

    def handle_prev_card(self) -> None:
        """Handle a direct previous-card target in a stage."""

    def handle_next_card(self) -> None:
        """Handle a direct next-card target in a stage."""

    def handle_back(self) -> None:
        """Handle a direct back target from an in-stage screen."""

    def handle_prompt_tap(self, event: TouchEvent) -> None:
        """Handle a tap on the current prompt card."""

    def handle_feedback_tap(self, event: TouchEvent) -> None:
        """Handle a tap on feedback/retry state."""

    def handle_done_tap(self, event: TouchEvent) -> None:
        """Handle a tap on the done state."""

    def poll(self) -> None:
        """Run non-blocking timer maintenance."""


PipelineFactory = Callable[[SessionModelManager], SessionPipeline]


_STATE_TO_EXPRESSION: dict[SessionState, CharacterExpression] = {
    SessionState.IDLE: CharacterExpression.IDLE,
    SessionState.WAKING: CharacterExpression.GREETING,
    SessionState.AWAITING_TAP: CharacterExpression.NEUTRAL,
    SessionState.LISTENING: CharacterExpression.LISTENING,
    SessionState.RESPONDING: CharacterExpression.THINKING,
    SessionState.SLEEPING: CharacterExpression.SLEEPY,
    SessionState.HISTORY_SELECT: CharacterExpression.NEUTRAL,
    SessionState.HISTORY_SCENE: CharacterExpression.SPEAKING,
    SessionState.HISTORY_CONSENT: CharacterExpression.NEUTRAL,
    SessionState.HISTORY_PAUSED: CharacterExpression.NEUTRAL,
    SessionState.HISTORY_DONE: CharacterExpression.EXCITED,
    SessionState.FE_INTRO: CharacterExpression.EXCITED,
    SessionState.FE_SELECT: CharacterExpression.EXCITED,
    SessionState.FE_PROMPT: CharacterExpression.SPEAKING,
    SessionState.FE_LISTEN: CharacterExpression.LISTENING,
    SessionState.FE_SCORE: CharacterExpression.THINKING,
    SessionState.FE_FEEDBACK: CharacterExpression.HAPPY,
    SessionState.FE_DONE: CharacterExpression.EXCITED,
}

_HISTORY_STATES: frozenset[SessionState] = frozenset(
    {
        SessionState.HISTORY_SELECT,
        SessionState.HISTORY_SCENE,
        SessionState.HISTORY_CONSENT,
        SessionState.HISTORY_PAUSED,
        SessionState.HISTORY_DONE,
    }
)

_FE_STATES: frozenset[SessionState] = frozenset(
    {
        SessionState.FE_INTRO,
        SessionState.FE_SELECT,
        SessionState.FE_PROMPT,
        SessionState.FE_LISTEN,
        SessionState.FE_SCORE,
        SessionState.FE_FEEDBACK,
        SessionState.FE_DONE,
    }
)

# Funny English states that accept and route a fresh tap. FE_LISTEN and
# FE_SCORE intentionally drop taps (they are mid-attempt transient states), so
# they are excluded here and fall through to the drop branch in _dispatch_tap.
_FE_TAP_STATES: frozenset[SessionState] = frozenset(
    {
        SessionState.FE_INTRO,
        SessionState.FE_SELECT,
        SessionState.FE_PROMPT,
        SessionState.FE_FEEDBACK,
        SessionState.FE_DONE,
    }
)


class SessionManager:
    """Coordinate touch events, audio cues, capture, and conversation turns."""

    def __init__(
        self,
        mm: SessionModelManager,
        pipeline_factory: PipelineFactory | None,
        touch: TouchInputListener,
        sound_bank: SoundBank,
        audio_capture: AudioCapture,
        event_log: EventLog,
        renderer: CharacterRenderer | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now().astimezone(),
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a touchscreen session manager."""
        self._mm = mm
        self._pipeline_factory = pipeline_factory or _default_pipeline_factory
        self._pipeline = self._pipeline_factory(mm)
        self._touch = touch
        self._sound_bank = sound_bank
        self._audio_capture = audio_capture
        self._pipeline.set_playback_gate(
            self._mute_and_pause_audio_capture,
            self._drain_resume_and_unmute_audio_capture,
        )
        self._event_log = event_log
        self._renderer = renderer or NullCharacterRenderer()
        self._pipeline.set_expression_sink(self.set_expression)
        self._pipeline.set_language_sink(self.set_language_indicator)
        self._pipeline.set_history_mode_sink(self._enter_history_mode)
        self._pipeline.set_funny_english_sink(self._enter_funny_english_mode)
        self._language_indicator_lock = threading.Lock()
        self._last_indicator_language: str | None = None
        self._clock = clock
        self._monotonic_clock = monotonic_clock
        self._state = SessionState.IDLE
        self._stop = False
        self._capture_paused = False
        self._playback_lock = threading.Lock()
        self._playback_owner_thread_id: int | None = None
        self._playback_should_resume_capture = False
        self._history_controller: HistoryController | None = None
        self._funny_english_controller: FunnyEnglishController | None = None
        self._history_consent_reprompted = False
        self._history_teardown_lock = threading.Lock()
        self._history_teardown_pending = False
        self._history_teardown_future: Future[None] | None = None
        self._history_teardown_started_at: float | None = None
        self._history_sleep_finalizer: threading.Thread | None = None
        self._language_toggle_in_progress = False
        self._portal_activate_in_progress = False
        self._portal_tap_gesture = DoubleTapGesture(double_tap_window_s=_PORTAL_CONFIRM_WINDOW_S)
        self._current_tap_dispatch_started_at: float | None = None
        self._pending_history_exit_dispatch_started_at: float | None = None
        self._last_wake_day_fallback: date | None = None
        self._has_run: bool = False
        self._armed_at: float = self._monotonic_clock()

    @property
    def state(self) -> SessionState:
        """Return the current session state."""
        return self._state

    def add_state_listener(self, renderer: CharacterRenderer) -> None:
        """Replace the active state renderer. PRE-RUN ONLY.

        Calling after run() has been invoked raises RuntimeError to prevent
        runtime swap races. The old renderer is closed before assignment to
        keep renderer ownership singular.
        """
        if self._has_run:
            raise RuntimeError(
                "add_state_listener() is pre-run only; cannot replace renderer after run() "
                "has started"
            )
        if self._state != SessionState.IDLE:
            raise RuntimeError(f"add_state_listener() requires state=IDLE; current={self._state}")
        old = self._renderer
        with suppress(Exception):
            old.close()
        self._renderer = renderer

    def set_history_controller(self, controller: HistoryController | None) -> None:
        """Attach the controller that owns history-mode runtime behavior."""
        self._history_controller = controller

    def set_funny_english_controller(self, controller: FunnyEnglishController | None) -> None:
        """Attach the controller that owns Funny English runtime behavior."""
        self._funny_english_controller = controller

    def is_history_mode_active(self) -> bool:
        """Return whether the session is currently in a history mode state."""
        return self._state in _HISTORY_STATES

    def is_funny_english_mode_active(self) -> bool:
        """Return whether the session is currently in a Funny English state."""
        return self._state in _FE_STATES

    def exit_funny_english_mode(self) -> None:
        """Exit Funny English mode from controller-owned interrupt handling."""
        self._exit_funny_english_mode()

    def stop(self) -> None:
        """Request shutdown of the run loop."""
        self._stop = True
        self._close_history_controller()
        self._join_history_sleep_finalizer(timeout=PRELOAD_JOIN_TIMEOUT)
        with suppress(Exception):
            self._audio_capture.stop()

    def set_expression(self, expression: CharacterExpression) -> None:
        """Emit an expression change from an external caller.

        A later state transition may overwrite this transient expression.
        """
        self._emit_expression(expression)

    def set_language_indicator(self, lang: str) -> None:
        """Emit a session language change from an external caller."""
        with self._language_indicator_lock:
            previous = self._last_indicator_language
            with suppress(Exception):
                self._renderer.on_language_change(lang)
            if previous is not None and lang != previous:
                with suppress(Exception):
                    cue_clip = self._sound_bank.pick_language_switch()
                    if cue_clip is not None:
                        audio_samples, sample_rate = cue_clip
                        self._play_audio_with_capture_guard(audio_samples, sample_rate)
            self._last_indicator_language = lang

    def shutdown(self) -> None:
        """Release all session-owned resources after or before the run loop.

        Raises RuntimeError if called during an active session state. Cleanup
        order is audio capture, renderer, then model-manager resources.
        """
        if self._has_run and self._state not in (SessionState.IDLE, SessionState.SLEEPING):
            raise RuntimeError(
                "shutdown() must be called after run() returns or before run() starts; "
                f"current state={self._state}"
            )
        self._stop = True
        self._close_history_controller()
        self._join_history_sleep_finalizer(timeout=PRELOAD_JOIN_TIMEOUT)
        with suppress(Exception):
            self._audio_capture.stop()
        with suppress(Exception):
            self._renderer.close()
        with suppress(Exception):
            self._mm.cancel_preload()
        with suppress(Exception):
            self._mm.unload_stt(force=False)

    def run(self) -> None:
        """Run the event loop until ``stop()`` is called.

        Sets the run latch on entry; the latch is sticky across stop/start
        cycles so renderers cannot be swapped after runtime begins.
        """
        self._has_run = True
        if self._state == SessionState.IDLE:
            self._rearm_taps()
            self._emit_expression(_STATE_TO_EXPRESSION[SessionState.IDLE])
        while not self._stop:
            event = self._touch.wait_for_event(timeout=1.0)
            if event is None:
                self._handle_idle_poll()
                continue
            if event.type == "tap":
                self._current_tap_dispatch_started_at = self._monotonic_clock()
                self._dispatch_tap(event)
                self._log_history_exit_dispatch_return()
                self._current_tap_dispatch_started_at = None
            elif event.type == "long_press":
                self._handle_parent_request(event)

    def _transition(self, new_state: SessionState) -> None:
        """Update state, notify the renderer, and emit the mapped expression."""
        self._state = new_state
        with suppress(Exception):
            self._renderer.on_state_change(new_state)
        self._emit_expression(_STATE_TO_EXPRESSION[new_state])

    def _is_history_teardown_pending(self) -> bool:
        with self._history_teardown_lock:
            return self._history_teardown_pending

    def _track_history_teardown(self, future: Future[None] | None) -> None:
        if future is None:
            return
        started_at = self._monotonic_clock()
        with self._history_teardown_lock:
            self._history_teardown_pending = True
            self._history_teardown_future = future
            self._history_teardown_started_at = started_at
        future.add_done_callback(self._on_history_teardown_done)

    def _on_history_teardown_done(self, future: Future[None]) -> None:
        completed_at = self._monotonic_clock()
        with self._history_teardown_lock:
            if self._history_teardown_future is not future:
                return
            started_at = self._history_teardown_started_at
            self._history_teardown_pending = False
            self._history_teardown_future = None
            self._history_teardown_started_at = None
        self._armed_at = completed_at
        try:
            exc = future.exception()
        except Exception as exc:
            logger.warning("History teardown future inspection failed: %s", exc)
        else:
            if exc is not None:
                logger.warning("History teardown completed with an error: %s", exc)
        if started_at is not None:
            logger.info(
                "History teardown completed in %.3fs",
                max(0.0, completed_at - started_at),
            )

    @staticmethod
    def _completed_future() -> Future[None]:
        future: Future[None] = Future()
        future.set_result(None)
        return future

    def _mark_history_exit_tap_dispatch(self) -> None:
        started_at = self._current_tap_dispatch_started_at or self._monotonic_clock()
        self._pending_history_exit_dispatch_started_at = started_at
        logger.info("History exit tap dispatch entry at %.6f", started_at)

    def _log_history_exit_dispatch_return(self) -> None:
        started_at = self._pending_history_exit_dispatch_started_at
        if started_at is None:
            return
        returned_at = self._monotonic_clock()
        self._pending_history_exit_dispatch_started_at = None
        logger.info(
            "History exit tap dispatch returned in %.3fs",
            max(0.0, returned_at - started_at),
        )

    def _close_history_controller(self) -> None:
        controller = self._history_controller
        if controller is None:
            return
        with suppress(Exception):
            controller.close()

    def _join_history_sleep_finalizer(self, *, timeout: float) -> None:
        thread = self._history_sleep_finalizer
        if thread is None or thread is threading.current_thread():
            return
        thread.join(timeout=timeout)
        if not thread.is_alive():
            self._history_sleep_finalizer = None

    def _start_history_sleep_finalizer(self, future: Future[None] | None) -> None:
        if future is None:
            return
        thread = self._history_sleep_finalizer
        if thread is not None and thread.is_alive():
            logger.warning("History sleep finalizer already running")
            return
        thread = threading.Thread(
            target=self._run_history_sleep_finalizer,
            args=(future,),
            name="HistorySleepFinalizer",
            daemon=True,
        )
        self._history_sleep_finalizer = thread
        thread.start()

    def _run_history_sleep_finalizer(self, future: Future[None]) -> None:
        try:
            with suppress(Exception):
                future.result()
            if not self._stop:
                self._run_sleep_cleanup()
        finally:
            if self._history_sleep_finalizer is threading.current_thread():
                self._history_sleep_finalizer = None

    def _emit_expression(self, expression: CharacterExpression) -> None:
        """Emit an expression change best-effort."""
        with suppress(Exception):
            self._renderer.on_expression(expression)

    def _mute_and_pause_audio_capture(self) -> None:
        """Mute callback frames and pause the input stream for playback."""
        self._audio_capture.mute()
        self._audio_capture.pause()
        self._capture_paused = True

    def _drain_resume_and_unmute_audio_capture(self) -> None:
        """Discard residual frames, restart capture, and unmute callbacks."""
        try:
            try:
                self._audio_capture.drain()
            finally:
                self._audio_capture.resume()
        finally:
            self._audio_capture.unmute()
            self._capture_paused = False

    def _play_audio_with_capture_guard(self, audio_samples: Any, sample_rate: int) -> None:
        """Play blocking local audio while the microphone stream is paused."""
        with self._playback_lock:
            was_paused = self._capture_paused
            self._audio_capture.mute()
            try:
                self._audio_capture.pause()
                self._capture_paused = True
                audio_player.play_audio(audio_samples, sample_rate=sample_rate, blocking=True)
            finally:
                if not was_paused:
                    self._drain_resume_and_unmute_audio_capture()

    def begin_guarded_playback(self) -> None:
        """Begin a serialized playback section with capture paused once."""
        self._playback_lock.acquire()
        self._playback_owner_thread_id = threading.get_ident()
        self._playback_should_resume_capture = not self._capture_paused
        try:
            self._mute_and_pause_audio_capture()
        except Exception:
            self._playback_should_resume_capture = False
            self._playback_owner_thread_id = None
            self._playback_lock.release()
            raise

    def play_guarded(self, audio_samples: Any, sample_rate: int) -> None:
        """Play blocking audio inside an active guarded playback section."""
        if self._playback_owner_thread_id != threading.get_ident():
            raise RuntimeError("play_guarded() requires begin_guarded_playback() on this thread")
        audio_player.play_audio(audio_samples, sample_rate=sample_rate, blocking=True)

    def interrupt_playback(self) -> None:
        """Stop any currently blocking local playback best-effort."""
        with suppress(Exception):
            audio_player.stop_playback()

    def end_guarded_playback(self) -> None:
        """End a serialized playback section and resume capture once."""
        if self._playback_owner_thread_id != threading.get_ident():
            raise RuntimeError("end_guarded_playback() requires ownership by this thread")
        try:
            if self._playback_should_resume_capture:
                self._drain_resume_and_unmute_audio_capture()
        finally:
            self._playback_should_resume_capture = False
            self._playback_owner_thread_id = None
            self._playback_lock.release()

    def begin_history_capture_pause(self) -> None:
        """Hold microphone capture muted and paused across history narration."""
        with suppress(Exception):
            self._mute_and_pause_audio_capture()

    def release_history_capture_pause(self, *, restore_stt: bool) -> None:
        """Release history capture policy without reopening capture at exit."""
        if restore_stt:
            with suppress(Exception):
                self._mute_and_pause_audio_capture()
            return
        with suppress(Exception):
            self._audio_capture.stop()

    def play_history_ack(self) -> None:
        """Play the history-navigation acknowledgement cue best-effort."""
        with suppress(Exception):
            ack_clip = self._sound_bank.pick_ack()
            if ack_clip is None:
                return
            ack_audio, sample_rate = ack_clip
            self._play_audio_with_capture_guard(ack_audio, sample_rate)

    def _dispatch_tap(self, event: TouchEvent) -> None:
        """Route a fresh tap according to the current state."""
        if not self._is_fresh_tap(event):
            return
        if self._state in _FE_TAP_STATES:
            self._dispatch_funny_english_tap(event)
        elif self._state == SessionState.HISTORY_SELECT:
            self._handle_history_select_tap(event)
        elif self._state == SessionState.HISTORY_SCENE:
            self._handle_history_scene_tap(event)
        elif self._state == SessionState.HISTORY_CONSENT:
            self._handle_history_consent_tap(event)
        elif self._state == SessionState.HISTORY_PAUSED:
            self._handle_history_paused_tap(event)
        elif self._state == SessionState.HISTORY_DONE:
            self._handle_history_done_tap(event)
        elif self._state == SessionState.IDLE:
            self._handle_cold_wake()
        elif self._state == SessionState.AWAITING_TAP:
            if self._is_history_teardown_pending():
                logger.info("Dropped AWAITING_TAP listen tap while history teardown is pending")
                self._rearm_taps()
                return
            target = None
            if event.x is not None and event.y is not None:
                target = self._renderer.hit_test(event.x, event.y)
            if target is not None and target.kind == "language_toggle":
                self._flash_press_feedback(target)
                self._handle_language_toggle_tap()
                return
            if target is not None and target.kind == "portal_activate":
                self._flash_press_feedback(target)
                self._handle_portal_activate_tap(event)
                return
            self._handle_listen_turn()
        elif self._state in (
            SessionState.WAKING,
            SessionState.LISTENING,
            SessionState.RESPONDING,
            SessionState.FE_LISTEN,
            SessionState.FE_SCORE,
        ):
            logger.debug("Dropped tap during %s", self._state.value)

    def _dispatch_funny_english_tap(self, event: TouchEvent) -> None:
        """Route a Funny English tap behind an exception firewall.

        Funny English listen/playback (STT load, attempt scoring, model/cached
        WAV playback) can raise on STT init failure, OOM, or ALSA/USB-audio
        device errors. Without a firewall here, such an exception would unwind
        through ``_dispatch_tap`` and ``run()``, killing the event loop and
        freezing the touchscreen mid-mode. Mirror ``_handle_listen_turn``: log
        the failure, play the safe error cue, and exit Funny English to the
        stable awake tap state instead of letting the exception escape.
        """
        try:
            if self._state == SessionState.FE_INTRO:
                self._handle_funny_english_intro_tap(event)
            elif self._state == SessionState.FE_SELECT:
                self._handle_funny_english_select_tap(event)
            elif self._state == SessionState.FE_PROMPT:
                self._handle_funny_english_prompt_tap(event)
            elif self._state == SessionState.FE_FEEDBACK:
                self._handle_funny_english_feedback_tap(event)
            elif self._state == SessionState.FE_DONE:
                self._handle_funny_english_done_tap(event)
        except STT_LOAD_EXCEPTIONS as exc:
            logger.warning("Funny English tap failed: %s", exc, exc_info=True)
            self._recover_funny_english_from_failure()
        except Exception as exc:
            logger.error("Funny English tap failed unexpectedly: %s", exc, exc_info=True)
            self._recover_funny_english_from_failure()

    def _recover_funny_english_from_failure(self) -> None:
        """Play the error cue and return to a safe state after an FE failure."""
        self._play_error_safe()
        with suppress(Exception):
            self._exit_funny_english_mode()
        if self._state in _FE_STATES:
            self._enter_awaiting_tap()

    def _is_fresh_tap(self, event: TouchEvent) -> bool:
        """Return whether ``event`` was stamped after the current arm point."""
        if event.timestamp < self._armed_at:
            logger.debug(
                "Dropped stale tap stamped %.6f before arm %.6f",
                event.timestamp,
                self._armed_at,
            )
            return False
        return True

    def _handle_idle_poll(self) -> None:
        """Apply the AWAITING_TAP idle timeout on a polling tick."""
        if self._state in (SessionState.HISTORY_SCENE, SessionState.HISTORY_DONE):
            self._poll_history_mode()
            return
        if self._state == SessionState.HISTORY_CONSENT:
            self._poll_history_mode()
            timeout = (
                HISTORY_CONSENT_SLEEP_TIMEOUT
                if self._history_consent_reprompted
                else HISTORY_CONSENT_REPROMPT_TIMEOUT
            )
            if self._monotonic_clock() - self._armed_at >= timeout:
                reprompted = False
                if self._history_controller is not None:
                    with suppress(Exception):
                        reprompted = self._history_controller.handle_consent_timeout()
                if reprompted:
                    self._history_consent_reprompted = True
                    self._rearm_taps()
                else:
                    self._sleep_from_history_mode()
            return
        if self._state in _FE_STATES:
            self._poll_funny_english_mode()
            if self._state in (SessionState.FE_INTRO, SessionState.FE_SELECT) and (
                self._monotonic_clock() - self._armed_at >= AWAITING_TAP_IDLE_TIMEOUT
            ):
                self._sleep_from_funny_english_mode()
            return
        if self._state == SessionState.HISTORY_SELECT:
            self._poll_history_mode()
            if self._monotonic_clock() - self._armed_at >= HISTORY_SELECT_IDLE_TIMEOUT:
                self._sleep_from_history_mode()
            return
        if self._state == SessionState.AWAITING_TAP and self._is_history_teardown_pending():
            self._rearm_taps()
            return
        if self._state == SessionState.AWAITING_TAP and (
            self._monotonic_clock() - self._armed_at >= AWAITING_TAP_IDLE_TIMEOUT
        ):
            self._enter_sleeping()

    def _rearm_taps(self) -> None:
        """Discard queued taps and advance the freshness boundary."""
        drain_taps = getattr(self._touch, "drain_taps", None)
        if callable(drain_taps):
            with suppress(Exception):
                drain_taps()
        self._armed_at = self._monotonic_clock()

    def _pause_audio_capture(self) -> None:
        """Pause microphone capture best-effort."""
        with suppress(Exception):
            self._audio_capture.pause()
            self._capture_paused = True

    def _enter_awaiting_tap(self) -> None:
        """Pause capture, arm for the next tap, and enter AWAITING_TAP."""
        self._pause_audio_capture()
        self._rearm_taps()
        self._transition(SessionState.AWAITING_TAP)

    def _enter_idle_ready(self) -> None:
        """Arm IDLE so only future taps can cold-wake."""
        self._rearm_taps()
        self._transition(SessionState.IDLE)

    def _screen_lead_seconds(self) -> float:
        """Return the screen-lead settle time before wake audio (env-overridable, clamped)."""
        raw = os.getenv(WAKE_SCREEN_LEAD_ENV)
        if raw is None:
            return WAKE_SCREEN_LEAD_DEFAULT_S
        try:
            value = float(raw)
        except ValueError:
            return WAKE_SCREEN_LEAD_DEFAULT_S
        if value < 0:
            return 0.0
        return min(value, WAKE_SCREEN_LEAD_MAX_S)

    def _await_screen_lead(self, *, since: float) -> None:
        """Hold the remaining screen-lead budget so the panel/character lead the wake audio."""
        lead = self._screen_lead_seconds()
        if lead <= 0:
            return
        remaining = lead - (self._monotonic_clock() - since)
        if remaining > 0:
            time.sleep(remaining)

    def _handle_cold_wake(self) -> None:
        """Run the WAKING sequence and return to the dispatch loop."""
        self._transition(SessionState.WAKING)
        wake_emit_at = self._monotonic_clock()
        try:
            with suppress(Exception):
                self._mm.reset_preload_state()

            with suppress(Exception):
                self._pipeline.reset_session()

            self._audio_capture.start()
            # start() resumes the stream, so keep the capture-pause shadow flag
            # consistent with real stream state (prevents a stale True from a prior
            # paused→sleep path suppressing a later cue's post-playback resume).
            self._capture_paused = False

            with suppress(Exception):
                self._mm.preload_stt()

            self._await_screen_lead(since=wake_emit_at)

            now = self._clock()
            wake_audio, sample_rate = self._sound_bank.pick_wake(
                now=now,
                last_wake_date=self._read_last_wake_day(),
            )
            self._play_audio_with_capture_guard(wake_audio, sample_rate)

            self._write_last_wake_day(self._compute_wake_day(now))
            self._enter_awaiting_tap()
        except Exception as exc:
            logger.warning("WAKING failed: %s", exc, exc_info=True)
            self._play_error_safe()
            self._enter_sleeping()

    def _capture_listen_audio(
        self,
        *,
        listen_state: SessionState = SessionState.LISTENING,
    ) -> Utterance | None:
        """Capture one utterance without running any pipeline response stages."""
        if listen_state not in (SessionState.LISTENING, SessionState.FE_LISTEN):
            raise ValueError(f"unsupported listen state: {listen_state}")
        ack_clip = self._sound_bank.pick_ack()
        if ack_clip is not None:
            ack_audio, sample_rate = ack_clip
            self._play_audio_with_capture_guard(ack_audio, sample_rate)
            self._drain_resume_and_unmute_audio_capture()
        else:
            self._drain_resume_and_unmute_audio_capture()
        self._transition(listen_state)
        listen_timeout = (
            FUNNY_ENGLISH_LISTEN_TIMEOUT
            if listen_state is SessionState.FE_LISTEN
            else LISTEN_TURN_TIMEOUT
        )
        return self._pipeline.wait_for_utterance(
            self._audio_capture,
            timeout=listen_timeout,
        )

    def capture_funny_english_audio(self) -> Utterance | FunnyEnglishListenInterrupt | None:
        """Capture one Funny English read-aloud attempt, allowing FE nav interrupts."""
        ack_clip = self._sound_bank.pick_ack()
        if ack_clip is not None:
            ack_audio, sample_rate = ack_clip
            self._play_audio_with_capture_guard(ack_audio, sample_rate)
            self._drain_resume_and_unmute_audio_capture()
        else:
            self._drain_resume_and_unmute_audio_capture()
        self._transition(SessionState.FE_LISTEN)

        elapsed = 0.0
        while elapsed + 1.0e-9 < FUNNY_ENGLISH_LISTEN_TIMEOUT:
            interrupt = self._poll_funny_english_listen_interrupt()
            if interrupt is not None:
                return interrupt
            slice_timeout = min(
                FUNNY_ENGLISH_LISTEN_POLL_SLICE,
                FUNNY_ENGLISH_LISTEN_TIMEOUT - elapsed,
            )
            utterance = self._pipeline.wait_for_utterance(
                self._audio_capture,
                timeout=slice_timeout,
            )
            if utterance is not None:
                interrupt = self._poll_funny_english_listen_interrupt()
                if interrupt is not None:
                    return interrupt
                return utterance
            interrupt = self._poll_funny_english_listen_interrupt()
            if interrupt is not None:
                return interrupt
            elapsed += slice_timeout
        return None

    def _poll_funny_english_listen_interrupt(self) -> FunnyEnglishListenInterrupt | None:
        """Return an FE listen interrupt from queued touch events, preserving others."""
        events = self._pop_pending_touch_events()
        if not events:
            return None

        preserved: list[TouchEvent] = []
        for index, event in enumerate(events):
            if event.type == "tap" and event.x is not None and event.y is not None:
                if self._is_fresh_tap(event):
                    target = self._history_hit_target(event)
                    if target is not None and target.kind in ("exit", "back"):
                        self._flash_press_feedback(target)
                        preserved.extend(
                            self._preservable_funny_english_listen_events(events[index + 1 :])
                        )
                        self._restore_pending_touch_events(preserved)
                        return FunnyEnglishListenInterrupt(
                            kind=cast(FunnyEnglishListenInterruptKind, target.kind)
                        )
                continue
            preserved.append(event)

        self._restore_pending_touch_events(preserved)
        return None

    @staticmethod
    def _preservable_funny_english_listen_events(events: list[TouchEvent]) -> list[TouchEvent]:
        """Return queued FE-listen events that should survive coordinate tap draining."""
        return [
            event for event in events if event.type != "tap" or event.x is None or event.y is None
        ]

    def _pop_pending_touch_events(self) -> list[TouchEvent]:
        """Remove currently queued touch events from the listener without blocking."""
        events: list[TouchEvent] = []
        while True:
            event = self._touch.wait_for_event(timeout=0)
            if event is None:
                return events
            events.append(event)

    def _restore_pending_touch_events(self, events: list[TouchEvent]) -> None:
        """Restore touch events so the run loop can dispatch non-interrupt taps."""
        if not events:
            return

        event_buffer = getattr(self._touch, "events", None)
        if isinstance(event_buffer, list):
            event_buffer[0:0] = events
            return

        queue_obj = getattr(self._touch, "_events", None)
        queue_items = getattr(queue_obj, "queue", None)
        queue_mutex = getattr(queue_obj, "mutex", None)
        append_left = getattr(queue_items, "appendleft", None)
        if queue_mutex is not None and callable(append_left):
            with queue_mutex:
                for event in reversed(events):
                    append_left(event)
                not_empty = getattr(queue_obj, "not_empty", None)
                if not_empty is not None:
                    not_empty.notify_all()
            return

        put_nowait = getattr(queue_obj, "put_nowait", None)
        if callable(put_nowait):
            for event in events:
                try:
                    put_nowait(event)
                except queue.Full:
                    logger.warning("Touch event queue full; dropped preserved event %s", event.type)
            return

        logger.warning("Unable to restore %d pending touch events", len(events))

    def run_funny_english_attempt(
        self,
        utterance: Utterance,
        card: Any,
        *,
        stt_hotwords_csv: str | None = None,
    ) -> Any:
        """Run the pipeline's Funny English STT-only scoring path."""
        result = self._pipeline.run_funny_english_attempt(
            utterance,
            card,
            stt_hotwords_csv=stt_hotwords_csv,
        )
        accept_aliases = tuple(str(token) for token in getattr(card, "accept_aliases", ()))
        if not accept_aliases or getattr(result, "band", None) == "pass":
            return result
        tokens = tuple(str(token) for token in getattr(card, "tokens", ()))
        try:
            alias_result = match_funny_english_attempt(
                str(getattr(result, "transcript", "")),
                tokens,
                accept_aliases=accept_aliases,
                pass_pct=DEFAULT_FUNNY_ENGLISH_PASS_PCT,
                pass_similarity=DEFAULT_FUNNY_ENGLISH_PASS_SIMILARITY,
            )
        except ValueError:
            return result
        if alias_result.band == "pass":
            return alias_result
        return result

    def _handle_listen_turn(self) -> None:
        """Process one push-to-talk listening turn and re-arm for the next tap."""
        try:
            utterance = self._capture_listen_audio()
            if utterance is None:
                self._enter_awaiting_tap()
                return

            self._transition(SessionState.RESPONDING)
            self._pause_audio_capture()
            self._pipeline.run_turn_with_audio(utterance)
        except STT_LOAD_EXCEPTIONS as exc:
            logger.warning("Turn failed: %s", exc, exc_info=True)
            self._play_error_safe()
        except Exception as exc:
            logger.error("Turn failed unexpectedly: %s", exc, exc_info=True)
            self._play_error_safe()
        finally:
            if (
                self._state != SessionState.AWAITING_TAP
                and self._state not in _HISTORY_STATES
                and self._state not in _FE_STATES
            ):
                self._enter_awaiting_tap()

    def _handle_language_toggle_tap(self) -> None:
        """Toggle the idle session language and play its confirmation."""
        if self._language_toggle_in_progress:
            logger.debug("Dropped duplicate language-toggle tap while confirmation is active")
            return
        if self._state != SessionState.AWAITING_TAP:
            logger.debug("Dropped language-toggle tap during %s", self._state.value)
            return

        self._language_toggle_in_progress = True
        try:
            current_language = str(getattr(self._pipeline, "session_language", "ko")).lower()
            target_language = "ko" if current_language == "en" else "en"
            self._transition(SessionState.RESPONDING)
            self._pause_audio_capture()
            self._pipeline.switch_session_language_with_confirmation(target_language)
        except Exception as exc:
            logger.error("Language-toggle tap failed unexpectedly: %s", exc, exc_info=True)
            self._play_error_safe()
        finally:
            self._language_toggle_in_progress = False
            if (
                self._state != SessionState.AWAITING_TAP
                and self._state not in _HISTORY_STATES
                and self._state not in _FE_STATES
            ):
                self._enter_awaiting_tap()

    def _handle_portal_activate_tap(self, event: TouchEvent) -> None:
        """Top-left double-tap → start the download portal on demand.

        ``classify`` runs first on every hotspot tap to track the double-tap
        window; a single tap shows progress text and waits for the second.
        Only a confirmed double tap starts the portal. Re-entrancy is guarded
        and the privileged start runs through the narrow sudo helper.
        """
        action = self._portal_tap_gesture.classify(event)
        if self._state != SessionState.AWAITING_TAP:
            logger.debug("Dropped portal-activate tap during %s", self._state.value)
            return
        if self._portal_activate_in_progress:
            logger.debug("Dropped duplicate portal-activate tap while activation is active")
            return
        if action == "single":
            with suppress(Exception):
                self._renderer.show_portal_status("한 번 더", duration=_PORTAL_CONFIRM_WINDOW_S)
            return
        if action != "double":
            return

        self._portal_activate_in_progress = True
        with suppress(Exception):
            self._renderer.show_portal_status(
                "다운로드 모드 켜짐",
                duration=_PORTAL_ACTIVATED_STATUS_S,
            )
        try:
            from parental.download_portal import control

            self._transition(SessionState.RESPONDING)
            self._pause_audio_capture()
            result = control.start_service()
            if result.ok:
                logger.info("Download portal activated via touchscreen double-tap")
                self._play_portal_activated_feedback()
            else:
                with suppress(Exception):
                    self._renderer.show_portal_status("", duration=0.0)
                logger.error("Portal activation failed: %s", result.detail)
                self._play_error_safe()
        except Exception as exc:
            with suppress(Exception):
                self._renderer.show_portal_status("", duration=0.0)
            logger.error("Portal-activate tap failed unexpectedly: %s", exc, exc_info=True)
            self._play_error_safe()
        finally:
            self._portal_activate_in_progress = False
            if (
                self._state != SessionState.AWAITING_TAP
                and self._state not in _HISTORY_STATES
                and self._state not in _FE_STATES
            ):
                self._enter_awaiting_tap()

    def _play_portal_activated_feedback(self) -> None:
        """Play the tap-ack cue to confirm the portal started, best-effort."""
        with suppress(Exception):
            ack_clip = self._sound_bank.pick_ack()
            if ack_clip is None:
                return
            ack_audio, sample_rate = ack_clip
            self._play_audio_with_capture_guard(ack_audio, sample_rate)

    def _handle_wake(self) -> None:
        """Run the cold-wake sequence.

        Kept as a compatibility wrapper for focused tests and any private callers.
        """
        self._handle_cold_wake()

    def _enter_sleeping(self) -> None:
        """Play the sleep cue and return resources to IDLE."""
        self.interrupt_playback()
        self._transition(SessionState.SLEEPING)
        self._run_sleep_cleanup()

    def _run_sleep_cleanup(self) -> None:
        """Run blocking sleep cleanup after the visible SLEEPING transition."""
        try:
            sleep_audio, sample_rate = self._sound_bank.pick_sleep()
            self._play_audio_with_capture_guard(sleep_audio, sample_rate)
        except Exception as exc:
            logger.warning("Sleep cue playback failed: %s", exc)
        finally:
            with suppress(Exception):
                self._mm.cancel_preload()
                self._mm.unload_stt(force=False)
            with suppress(Exception):
                self._audio_capture.stop()
            self._enter_idle_ready()

    def _play_error_safe(self) -> None:
        """Play the STT-load failure cue best-effort."""
        with suppress(Exception):
            error_audio, sample_rate = self._sound_bank.pick_error("stt_load_fail")
            self._play_audio_with_capture_guard(error_audio, sample_rate)

    def _handle_parent_request(self, event: TouchEvent) -> None:
        """Log a Phase 1 parent-mode request without opening a PIN UI."""
        if self._state in _FE_STATES:
            self._exit_funny_english_mode()
            return
        if self._state in _HISTORY_STATES:
            self._exit_history_mode()
            return
        if getattr(self._pipeline, "session_language", "ko") == "en":
            self._pipeline.set_session_language("ko")
            return
        with suppress(Exception):
            self._event_log.append(
                {
                    "schema_version": 1,
                    "event": "parent_mode_requested",
                    "timestamp": self._clock().isoformat(),
                    "source": "touchscreen_long_press",
                    "session_state_at_request": self._state.value,
                    "press_duration_ms": event.press_duration_ms,
                    "session_id": getattr(self._pipeline, "session_id", None),
                    "handled": False,
                    "reason": None,
                }
            )

    def transition_history_state(self, state: SessionState) -> None:
        """Transition to a history state for the history controller."""
        if state not in _HISTORY_STATES:
            raise ValueError(f"not a history state: {state}")
        self._history_consent_reprompted = False
        if state in (SessionState.HISTORY_SELECT, SessionState.HISTORY_CONSENT):
            self._rearm_taps()
        self._transition(state)

    def transition_funny_english_state(self, state: SessionState) -> None:
        """Transition to a Funny English state for the Funny English controller."""
        if state not in _FE_STATES:
            raise ValueError(f"not a Funny English state: {state}")
        if state in (SessionState.FE_INTRO, SessionState.FE_SELECT):
            self._rearm_taps()
        self._transition(state)

    def _enter_history_mode(self) -> None:
        """Enter the history controller from a deterministic voice trigger."""
        if self._history_controller is None:
            logger.warning("History mode trigger ignored because no controller is registered")
            return
        try:
            self._history_controller.enter()
        except Exception as exc:
            logger.warning("History mode entry failed: %s", exc, exc_info=True)
            if self._state in _HISTORY_STATES:
                self._enter_awaiting_tap()

    def _enter_funny_english_mode(self) -> None:
        """Enter the Funny English controller from a deterministic voice trigger."""
        if self._funny_english_controller is None:
            logger.warning("Funny English trigger ignored because no controller is registered")
            return
        try:
            self._funny_english_controller.enter()
        except Exception as exc:
            logger.warning("Funny English entry failed: %s", exc, exc_info=True)
            if self._state in _FE_STATES:
                self._enter_awaiting_tap()

    def _exit_history_mode(self) -> None:
        """Exit the history controller and return to the awake tap state."""
        self.interrupt_playback()
        future: Future[None] | None = None
        if self._history_controller is not None:
            with suppress(Exception):
                future = self._history_controller.exit(restore_stt=True)
        self._track_history_teardown(future)
        if self._state in _HISTORY_STATES:
            self._enter_awaiting_tap()

    def _exit_funny_english_mode(self) -> None:
        """Exit Funny English and return to the awake tap state."""
        if self._funny_english_controller is not None:
            with suppress(Exception):
                self._funny_english_controller.exit()
        with suppress(Exception):
            self._pipeline.set_session_language("ko")
        if self._state in _FE_STATES:
            self._enter_awaiting_tap()

    def _sleep_from_history_mode(self) -> None:
        """Exit history mode and enter the normal sleeping path."""
        self.interrupt_playback()
        future: Future[None] | None = None
        if self._history_controller is not None:
            with suppress(Exception):
                future = self._history_controller.exit(restore_stt=False)
        if future is None:
            future = self._completed_future()
        else:
            self._track_history_teardown(future)
        self._transition(SessionState.SLEEPING)
        self._start_history_sleep_finalizer(future)

    def _sleep_from_funny_english_mode(self) -> None:
        """Exit Funny English and enter the normal sleeping path."""
        if self._funny_english_controller is not None:
            with suppress(Exception):
                self._funny_english_controller.exit()
        with suppress(Exception):
            self._pipeline.set_session_language("ko")
        self._enter_sleeping()

    def _poll_history_mode(self) -> None:
        """Run non-blocking history controller maintenance."""
        if self._history_controller is not None:
            with suppress(Exception):
                self._history_controller.poll()

    def _poll_funny_english_mode(self) -> None:
        """Run non-blocking Funny English controller maintenance."""
        if self._funny_english_controller is not None:
            with suppress(Exception):
                self._funny_english_controller.poll()

    def _handle_history_select_tap(self, event: TouchEvent) -> None:
        """Route a tap in history catalogue selection."""
        target = self._history_hit_target(event)
        if target is not None:
            if target.kind == "exit":
                self._flash_press_feedback(target)
                self._mark_history_exit_tap_dispatch()
                self._exit_history_mode()
                return
            if target.kind == "back" and self._history_controller is not None:
                self._flash_press_feedback(target)
                self._history_controller.handle_back()
                return
            if target.kind == "prev_page" and self._history_controller is not None:
                self._flash_press_feedback(target)
                self._history_controller.handle_prev_page()
                return
            if target.kind == "next_page" and self._history_controller is not None:
                self._flash_press_feedback(target)
                self._history_controller.handle_next_page()
                return
            if (
                target.kind == "menu_item"
                and target.index is not None
                and self._history_controller is not None
            ):
                self._flash_press_feedback(target)
                self._history_controller.handle_select_target(target.index)
                return
        if event.x is not None and event.y is not None:
            return
        if self._history_controller is not None:
            self._history_controller.handle_select_tap(event)

    def _handle_history_scene_tap(self, event: TouchEvent) -> None:
        """Route a tap in history slideshow playback."""
        target = self._history_hit_target(event)
        if target is not None and target.kind == "exit":
            self._flash_press_feedback(target)
            self._mark_history_exit_tap_dispatch()
            self._exit_history_mode()
            return
        if target is not None and target.kind == "back" and self._history_controller is not None:
            self._flash_press_feedback(target)
            self._history_controller.handle_previous_step()
            return
        if self._history_controller is not None:
            self._history_controller.handle_scene_tap(event)

    def _handle_history_consent_tap(self, event: TouchEvent) -> None:
        """Route a tap at a history section consent gate."""
        if self._handle_history_exit_target(event):
            return
        if self._history_controller is not None:
            self._history_controller.handle_consent_tap(event)

    def _handle_history_paused_tap(self, event: TouchEvent) -> None:
        """Route a tap in the reserved history paused state."""
        if self._handle_history_exit_target(event):
            return
        if self._history_controller is not None:
            self._history_controller.handle_paused_tap(event)

    def _handle_history_done_tap(self, event: TouchEvent) -> None:
        """Route a tap in the transient history done state."""
        if self._handle_history_exit_target(event):
            return
        if self._history_controller is not None:
            self._history_controller.handle_done_tap(event)

    def _history_hit_target(self, event: TouchEvent) -> RenderHitTarget | None:
        """Return a direct-touch history target when a fresh tap carries coordinates."""
        if event.x is None or event.y is None:
            return None
        with suppress(Exception):
            return self._renderer.hit_test(event.x, event.y)
        return None

    def _handle_history_exit_target(self, event: TouchEvent) -> bool:
        """Exit history mode when a coordinate tap hits the renderer exit target."""
        target = self._history_hit_target(event)
        if target is None or target.kind != "exit":
            return False
        self._flash_press_feedback(target)
        self._mark_history_exit_tap_dispatch()
        self._exit_history_mode()
        return True

    def _flash_press_feedback(self, target: RenderHitTarget) -> None:
        """Render direct-touch press feedback before an action runs."""
        with suppress(Exception):
            self._renderer.flash_press_feedback(target)

    def _handle_funny_english_select_tap(self, event: TouchEvent) -> None:
        """Route a tap in Funny English selection."""
        target = self._history_hit_target(event)
        if target is not None:
            if target.kind == "exit":
                self._flash_press_feedback(target)
                self._exit_funny_english_mode()
                return
            if (
                target.kind == "menu_item"
                and target.index is not None
                and self._funny_english_controller is not None
            ):
                self._flash_press_feedback(target)
                self._funny_english_controller.handle_select_target(target.index)
                return
            if target.kind == "prev_page" and self._funny_english_controller is not None:
                self._flash_press_feedback(target)
                self._funny_english_controller.handle_prev_page()
                return
            if target.kind == "next_page" and self._funny_english_controller is not None:
                self._flash_press_feedback(target)
                self._funny_english_controller.handle_next_page()
                return
        if event.x is not None and event.y is not None:
            return
        if self._funny_english_controller is not None:
            self._funny_english_controller.handle_select_tap(event)

    def _handle_funny_english_intro_tap(self, event: TouchEvent) -> None:
        """Route a tap on the Funny English instruction screen."""
        target = self._history_hit_target(event)
        if target is not None and target.kind == "exit":
            self._flash_press_feedback(target)
            self._exit_funny_english_mode()
            return
        if self._funny_english_controller is not None:
            self._funny_english_controller.handle_intro_tap()

    def _handle_funny_english_card_nav_target(
        self,
        event: TouchEvent,
        *,
        allow_card_nav: bool = True,
    ) -> bool:
        """Handle direct navigation targets on Funny English card-like screens."""
        target = self._history_hit_target(event)
        if target is None:
            return False
        if target.kind == "exit":
            self._flash_press_feedback(target)
            self._exit_funny_english_mode()
            return True
        if target.kind == "back" and self._funny_english_controller is not None:
            self._flash_press_feedback(target)
            self._funny_english_controller.handle_back()
            return True
        if (
            allow_card_nav
            and target.kind == "prev_card"
            and self._funny_english_controller is not None
        ):
            self._flash_press_feedback(target)
            self._funny_english_controller.handle_prev_card()
            return True
        if (
            allow_card_nav
            and target.kind == "next_card"
            and self._funny_english_controller is not None
        ):
            self._flash_press_feedback(target)
            self._funny_english_controller.handle_next_card()
            return True
        return False

    def _handle_funny_english_prompt_tap(self, event: TouchEvent) -> None:
        """Route a tap on a Funny English prompt."""
        if self._handle_funny_english_card_nav_target(event):
            return
        if self._funny_english_controller is not None:
            self._funny_english_controller.handle_prompt_tap(event)

    def _handle_funny_english_feedback_tap(self, event: TouchEvent) -> None:
        """Route a tap on Funny English feedback."""
        if self._handle_funny_english_card_nav_target(event):
            return
        if self._funny_english_controller is not None:
            self._funny_english_controller.handle_feedback_tap(event)

    def _handle_funny_english_done_tap(self, event: TouchEvent) -> None:
        """Route a tap on the Funny English done state."""
        if self._handle_funny_english_card_nav_target(event, allow_card_nav=False):
            return
        if self._funny_english_controller is not None:
            self._funny_english_controller.handle_done_tap(event)

    def _read_last_wake_day(self) -> date | None:
        """Read the last wake-day state, treating missing or damaged data as absent."""
        path = _last_wake_day_path()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            wake_day = payload.get("wake_day")
            if not isinstance(wake_day, str):
                return self._last_wake_day_fallback
            return date.fromisoformat(wake_day)
        except (OSError, ValueError, json.JSONDecodeError):
            return self._last_wake_day_fallback

    def _write_last_wake_day(self, day: date) -> None:
        """Write the last wake-day state atomically, falling back to memory on OSError."""
        self._last_wake_day_fallback = day
        path = _last_wake_day_path()
        temp_path = path.with_name(f"{path.name}.tmp")
        payload = json.dumps({"wake_day": day.isoformat()}, separators=(",", ":"))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with temp_path.open("w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temp_path.replace(path)
            _fsync_directory(path.parent)
        except OSError as exc:
            logger.warning("Failed to write last wake-day state; using memory fallback: %s", exc)
            with suppress(OSError):
                temp_path.unlink()

    def _compute_wake_day(self, now: datetime) -> date:
        """Return the 05:00-boundary wake day for ``now``."""
        return (now - timedelta(hours=WAKE_DAY_HOUR_OFFSET)).date()


def _default_pipeline_factory(mm: SessionModelManager) -> SessionPipeline:
    """Build the live-demo tuned conversation pipeline lazily."""
    from safety.content_filter import ContentFilter
    from scripts.demo_live import _build_pipeline_config

    config = _build_pipeline_config()
    content_filter: ContentFilter | None = None
    if config.enable_content_filter:
        content_filter = ContentFilter.from_default()
        logger.info(
            "Content filter active for default live pipeline: %d categories, %d patterns",
            content_filter.category_count,
            content_filter.pattern_count,
        )
    return cast(
        SessionPipeline,
        ConversationPipeline(
            cast(ModelManager, mm),
            config,
            content_filter=content_filter,
        ),
    )


def _last_wake_day_path() -> Path:
    """Return the mutable runtime path for wake-day state."""
    return Path(detect_runtime_paths().mutable_root) / "state" / "last_wake_date.json"


def _fsync_directory(path: Path) -> None:
    """Best-effort fsync for a directory after atomic replace."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
