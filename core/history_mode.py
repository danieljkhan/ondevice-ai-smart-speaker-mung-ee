"""Runtime controller for the curated Korean-history picture-storytelling mode."""

from __future__ import annotations

import json
import logging
import math
import threading
import weakref
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Any, Literal, Protocol

from core._tap_gesture import DoubleTapGesture
from core.character_expression import CharacterExpression
from core.model_manager import ModelType
from core.session_manager import SessionState
from hardware.touch_input import TouchEvent
from models import tts_cache
from models.tts_runner import _split_text_into_sentences, normalize_tts_text
from scripts.generate_history_font_subset import displayable_history_text

logger = logging.getLogger(__name__)

HISTORY_TITLE = "재미있는 우리역사"
BACK_LABEL = "뒤로"
DONE_LABEL = "전체 끝!"
MISSING_IMAGE_LABEL = "이미지를 준비 중이에요."
CONSENT_PROMPT = "다음 이야기 들려줄까? 화면을 톡 누르면 들려줄게."
CONSENT_REPROMPT = "다음 이야기가 듣고 싶으면 화면을 톡 눌러줘."
MENU_PAGE_SIZE = 4
RENDER_ACK_TIMEOUT_S = 1.0
TITLE_BOUNDARY_PAUSE_S = 0.35
MISSING_ORDER_SENTINEL = 1_000_000_000

MenuLevel = Literal["era", "doc"]
_TeardownWorkItem = tuple[Future[None], Callable[[], None]] | None


def _history_sentence_segments(text: str) -> list[str]:
    sentences = _split_text_into_sentences(text)
    if sentences:
        return sentences
    stripped = text.strip()
    return [stripped] if stripped else []


def history_narration_segments(narration: str, section_title: str | None) -> Sequence[str | None]:
    """Return runtime TTS segments for one history narration.

    ``None`` entries are timing pause sentinels used by the runtime and must be
    skipped by offline cache bakes.
    """
    title = (section_title or "").strip()
    if not title or title not in narration:
        return _history_sentence_segments(narration)

    before, rest = narration.split(title, 1)
    after = rest
    segments: list[str | None] = []
    segments.extend(_history_sentence_segments(before))
    if segments:
        segments.append(None)
    segments.extend(_history_sentence_segments(title))
    if after.strip():
        segments.append(None)
        segments.extend(_history_sentence_segments(after))
    return segments


class _SerializedTeardownWorker:
    def __init__(self, *, name: str) -> None:
        self._tasks: Queue[_TeardownWorkItem] = Queue()
        self._shutdown = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            args=(self._tasks,),
            name=name,
            daemon=True,
        )
        self._finalizer = weakref.finalize(self, self._tasks.put, None)
        self._thread.start()

    def submit(self, callback: Callable[[], None]) -> Future[None]:
        future: Future[None] = Future()
        with self._lock:
            if self._shutdown:
                future.set_exception(RuntimeError("teardown worker is shut down"))
                return future
            self._tasks.put((future, callback))
        return future

    def shutdown(self, *, wait: bool) -> None:
        with self._lock:
            if not self._shutdown:
                self._shutdown = True
                self._tasks.put(None)
                self._finalizer.detach()
        if wait and self._thread is not threading.current_thread():
            self._thread.join()

    @staticmethod
    def _run(tasks: Queue[_TeardownWorkItem]) -> None:
        while True:
            item = tasks.get()
            try:
                if item is None:
                    return
                future, callback = item
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    callback()
                except Exception as exc:
                    future.set_exception(exc)
                else:
                    future.set_result(None)
            finally:
                tasks.task_done()


class HistoryRenderer(Protocol):
    """Renderer API used by the curated-history mode."""

    def show_history_image(self, path: str | Path) -> str | None:
        """Display a history-mode image."""

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
    ) -> str | None:
        """Display a generic image+text card."""

    def wait_until_rendered(self, token: str | None, timeout: float) -> bool:
        """Wait until a tokenized render request reaches the display."""

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
        """Display generic text."""

    def clear_image(self) -> None:
        """Clear any image layer."""

    def clear_text(self) -> None:
        """Clear any text layer."""


class HistorySession(Protocol):
    """Session-manager API used by the curated-history mode."""

    def transition_history_state(self, state: SessionState) -> None:
        """Transition to one of the history states."""

    def set_expression(self, expression: CharacterExpression) -> None:
        """Emit a character expression."""

    def play_history_ack(self) -> None:
        """Play the history-navigation acknowledgement cue."""

    def begin_guarded_playback(self) -> None:
        """Begin one guarded playback section."""

    def play_guarded(self, audio_samples: Any, sample_rate: int) -> None:
        """Play audio while the guarded playback section is active."""

    def end_guarded_playback(self) -> None:
        """End one guarded playback section."""

    def begin_history_capture_pause(self) -> None:
        """Hold capture muted/paused for the whole history section."""

    def release_history_capture_pause(self, *, restore_stt: bool) -> None:
        """Release the history capture pause according to the exit target state."""

    def interrupt_playback(self) -> None:
        """Stop any active blocking audio playback."""


class HistoryModelManager(Protocol):
    """Model-manager API required by the curated-history mode."""

    @property
    def tts(self) -> Any:
        """Return the loaded TTS engine."""

    def reset_preload_state(self) -> None:
        """Reset any pending STT preload state."""

    def preload_stt(self) -> None:
        """Start STT preload best-effort."""

    def cancel_preload(self) -> None:
        """Cancel pending preload work."""

    def unload_stt(self, *, force: bool = False) -> None:
        """Unload the STT model."""

    def unload_llm(self) -> None:
        """Unload the LLM model."""

    def load(self, model_type: ModelType) -> None:
        """Load a model."""

    def unload_tts(self, *, force: bool = False) -> None:
        """Unload the TTS model."""


@dataclass(frozen=True)
class HistoryImage:
    """One optional image attached to a history scene."""

    path: Path
    caption: str | None
    letterboxed: bool
    clean: bool
    is_infographic: bool
    anchor_ratio: float | None = None


@dataclass(frozen=True)
class HistoryScene:
    """One narrated history scene."""

    seq: int
    section_index: int
    section_title: str | None
    narration: str
    est_speech_ms: int
    tail_silence_ms: int
    image_captions: tuple[str, ...]
    images: tuple[HistoryImage, ...]


@dataclass(frozen=True)
class HistorySection:
    """One consent-paced section containing one or more scenes."""

    section_index: int
    section_title: str | None
    scenes: tuple[HistoryScene, ...]
    image_captions: tuple[str, ...]


@dataclass(frozen=True)
class HistoryDocument:
    """One loaded history document and its scenes."""

    doc_hash: str
    title: str
    kind: str
    era: str
    scenes: tuple[HistoryScene, ...]
    sections: tuple[HistorySection, ...]


@dataclass(frozen=True)
class HistoryCatalogEntry:
    """One manifest entry for the history catalogue."""

    doc_hash: str
    title: str
    kind: str
    era: str
    doc_path: Path
    scene_count: int
    image_count: int
    order: int


class HistoryModeController:
    """Coordinate history catalogue navigation, rendering, and TTS narration."""

    def __init__(
        self,
        *,
        session: HistorySession,
        model_manager: HistoryModelManager,
        renderer: HistoryRenderer,
        manifest_path: Path = Path("assets/history/manifest.json"),
        repo_root: Path = Path("."),
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        """Create the history mode controller."""
        self._session = session
        self._mm = model_manager
        self._renderer = renderer
        self._repo_root = repo_root
        self._manifest_path = self._resolve_path(manifest_path)
        self._monotonic_clock = monotonic_clock
        self._catalog, self._era_order = self._load_catalog(self._manifest_path)
        self._entries_by_era = self._index_catalog(self._catalog)
        self._active = False
        self._menu_level: MenuLevel = "era"
        self._selected_era: str | None = None
        self._highlight = 0
        self._page_start = 0
        self._current_doc: HistoryDocument | None = None
        self._section_index = 0
        self._scene_index = 0
        self._section_scene_offset = 0
        self._last_shown_history_image: HistoryImage | None = None
        self._scene_gesture = DoubleTapGesture()
        self._advance_event = threading.Event()
        self._exit_event = threading.Event()
        self._narration_done_event = threading.Event()
        self._narration_done_at: float | None = None
        self._done_until: float | None = None
        self._pending_previous = False
        self._awaiting_consent = False
        self._consent_reprompted = False
        self._narration_thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._teardown_executor = _SerializedTeardownWorker(name="HistoryTeardown")
        self._teardown_lock = threading.Lock()
        self._pending_teardown: Future[None] | None = None
        self._closed = False

    @property
    def active(self) -> bool:
        """Return whether the history mode is currently active."""
        with self._lock:
            return self._active

    def enter(self) -> None:
        """Enter catalogue selection mode and prepare TTS residency."""
        self._await_pending_teardown()
        self._mm.cancel_preload()
        try:
            # Pause capture FIRST so the mic queue does not flood the main loop
            # while the (blocking) model unload/load steps run during entry.
            self._session.begin_history_capture_pause()
            self._mm.reset_preload_state()
            self._mm.unload_stt(force=True)
            self._mm.unload_llm()
            self._mm.load(ModelType.TTS)
        except Exception:
            self._restore_residency_after_entry_failure()
            raise

        with self._lock:
            self._active = True
            self._menu_level = "era"
            self._selected_era = None
            self._highlight = 0
            self._page_start = 0
            self._current_doc = None
            self._section_index = 0
            self._scene_index = 0
            self._section_scene_offset = 0
            self._last_shown_history_image = None
            self._scene_gesture.reset()
            self._advance_event.clear()
            self._exit_event.clear()
            self._narration_done_event.clear()
            self._narration_done_at = None
            self._done_until = None
            self._pending_previous = False
            self._awaiting_consent = False
            self._consent_reprompted = False

        self._session.transition_history_state(SessionState.HISTORY_SELECT)
        self._render_menu()

    def exit(self, *, restore_stt: bool) -> Future[None]:
        """Exit history mode promptly and restore residency off the caller thread."""
        thread = self._narration_thread
        self._exit_event.set()
        self._advance_event.set()
        self._session.interrupt_playback()
        self._renderer.clear_image()
        self._renderer.clear_text()

        with self._lock:
            self._active = False
            self._menu_level = "era"
            self._selected_era = None
            self._highlight = 0
            self._page_start = 0
            self._current_doc = None
            self._section_index = 0
            self._scene_index = 0
            self._section_scene_offset = 0
            self._last_shown_history_image = None
            self._narration_done_at = None
            self._done_until = None
            self._pending_previous = False
            self._awaiting_consent = False
            self._consent_reprompted = False
            self._scene_gesture.reset()

        return self._enqueue_teardown(thread, restore_stt=restore_stt)

    def close(self) -> None:
        """Drain pending history teardown work and stop the controller executor."""
        self._closed = True
        self._await_pending_teardown()
        self._teardown_executor.shutdown(wait=True)

    def _enqueue_teardown(
        self,
        thread: threading.Thread | None,
        *,
        restore_stt: bool,
    ) -> Future[None]:
        with self._teardown_lock:
            if self._pending_teardown is not None and not self._pending_teardown.done():
                return self._pending_teardown
            if self._closed:
                return self._completed_future()
            future = self._teardown_executor.submit(
                lambda: self._run_exit_teardown(thread, restore_stt),
            )
            self._pending_teardown = future
        future.add_done_callback(self._clear_pending_teardown)
        return future

    @staticmethod
    def _completed_future() -> Future[None]:
        future: Future[None] = Future()
        future.set_result(None)
        return future

    def _clear_pending_teardown(self, future: Future[None]) -> None:
        with self._teardown_lock:
            if self._pending_teardown is future:
                self._pending_teardown = None

    def _await_pending_teardown(self) -> None:
        with self._teardown_lock:
            future = self._pending_teardown
        if future is not None:
            future.result()

    def _run_exit_teardown(
        self,
        thread: threading.Thread | None,
        restore_stt: bool,
    ) -> None:
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        with self._lock:
            if self._narration_thread is thread:
                self._narration_thread = None
        self._run_teardown_step("unload history TTS", lambda: self._mm.unload_tts(force=False))
        self._run_teardown_step("reset STT preload", self._mm.reset_preload_state)
        if restore_stt:
            self._run_teardown_step("preload STT after history exit", self._mm.preload_stt)
        self._run_teardown_step(
            "release history capture pause",
            lambda: self._session.release_history_capture_pause(restore_stt=restore_stt),
        )

    @staticmethod
    def _run_teardown_step(label: str, callback: Callable[[], None]) -> None:
        try:
            callback()
        except Exception:
            logger.warning("History teardown step failed: %s", label, exc_info=True)

    def handle_select_tap(self, event: TouchEvent) -> None:
        """Handle a coordinate-less or missed catalogue-navigation fallback tap."""
        if not self.active:
            return
        del event
        self._session.play_history_ack()
        self._advance_highlight()
        self._render_menu()

    def handle_select_target(self, index: int) -> None:
        """Select the visible menu item at ``index`` from a direct-touch hit."""
        if not self.active:
            return
        options = self._current_menu_options()
        selected_index = self._page_start + index
        if selected_index < 0 or selected_index >= len(options):
            return
        self._session.play_history_ack()
        self._highlight = selected_index
        self._page_start = (self._highlight // MENU_PAGE_SIZE) * MENU_PAGE_SIZE
        self._select_highlighted_item()

    def handle_back(self) -> None:
        """Return from the document menu to the era menu."""
        if not self.active or self._menu_level != "doc":
            return
        self._session.play_history_ack()
        self._go_back()

    def handle_prev_page(self) -> None:
        """Move to the previous visible menu page and re-render."""
        if not self.active:
            return
        options = self._current_menu_options()
        if not options:
            return
        previous_page_start = max(0, self._page_start - MENU_PAGE_SIZE)
        self._page_start = previous_page_start
        self._highlight = min(previous_page_start, len(options) - 1)
        self._session.play_history_ack()
        self._render_menu()

    def handle_next_page(self) -> None:
        """Move to the next visible menu page and re-render."""
        if not self.active:
            return
        options = self._current_menu_options()
        if not options:
            return
        last_page_start = ((len(options) - 1) // MENU_PAGE_SIZE) * MENU_PAGE_SIZE
        next_page_start = min(last_page_start, self._page_start + MENU_PAGE_SIZE)
        self._page_start = next_page_start
        self._highlight = min(next_page_start, len(options) - 1)
        self._session.play_history_ack()
        self._render_menu()

    def handle_scene_tap(self, event: TouchEvent) -> None:
        """Handle one slideshow tap."""
        if not self.active:
            return
        action = self._scene_gesture.classify(event)
        if action == "double":
            self._pending_previous = True
            self._advance_event.set()
            self._session.interrupt_playback()
            if self._narration_done_event.is_set():
                self._go_previous_scene()
            return
        self._advance_event.set()
        self._session.interrupt_playback()
        if self._narration_done_event.is_set():
            if self._is_last_section():
                self._finish_document()
            else:
                self._enter_consent_state()

    def handle_previous_step(self) -> None:
        """Go back one scene from a direct-touch scene back button.

        A step is one scene, so this reuses the same machinery as a scene
        double-tap: interrupt the current narration and rewind one scene
        (the previous scene in the section, or the previous section's last
        scene when crossing a boundary). When narration is still in flight the
        rewind is deferred via ``_pending_previous`` and applied by ``poll``.
        The button is hidden on the document's first scene, so reaching this
        method always has a previous scene to land on.
        """
        if not self.active or not self._can_go_previous_step():
            return
        self._session.play_history_ack()
        self._pending_previous = True
        self._advance_event.set()
        self._session.interrupt_playback()
        if self._narration_done_event.is_set():
            self._go_previous_scene()

    def handle_consent_tap(self, event: TouchEvent) -> None:
        """Continue from a section consent gate to the next section."""
        del event
        if not self.active or not self._awaiting_consent:
            return
        self._session.interrupt_playback()
        self._go_next_section()

    def handle_consent_timeout(self) -> bool:
        """Re-prompt once at the consent gate, then ask the session to sleep."""
        if not self.active or not self._awaiting_consent:
            return False
        if self._consent_reprompted:
            self._session.interrupt_playback()
            return False
        self._consent_reprompted = True
        self._render_consent_prompt(CONSENT_REPROMPT)
        self._play_system_prompt(CONSENT_REPROMPT)
        return True

    def handle_paused_tap(self, event: TouchEvent) -> None:
        """Resume from the reserved paused state."""
        del event
        if self._current_doc is None:
            self._session.transition_history_state(SessionState.HISTORY_SELECT)
            self._render_menu()
            return
        self._start_section()

    def handle_done_tap(self, event: TouchEvent) -> None:
        """Return from the done cue to the catalogue."""
        del event
        self._return_to_catalogue()

    def poll(self) -> None:
        """Advance delayed narration and done-state timers."""
        if not self.active:
            return
        if self._done_until is not None:
            if self._now() >= self._done_until:
                self._return_to_catalogue()
            return
        if self._exit_event.is_set():
            return
        current_doc = self._current_doc
        if current_doc is None:
            return
        if self._narration_done_event.is_set():
            if self._pending_previous:
                self._go_previous_scene()
                return
            if self._is_last_section():
                self._finish_document()
                return
            if not self._awaiting_consent:
                self._enter_consent_state()

    def _advance_highlight(self) -> None:
        options = self._current_menu_options()
        if not options:
            self._highlight = 0
            self._page_start = 0
            return
        self._highlight = (self._highlight + 1) % len(options)
        self._page_start = (self._highlight // MENU_PAGE_SIZE) * MENU_PAGE_SIZE

    def _select_highlighted_item(self) -> None:
        options = self._current_menu_options()
        if not options:
            return
        item = options[self._highlight]
        if self._menu_level == "era":
            self._selected_era = item
            self._menu_level = "doc"
            self._reset_menu_position()
            self._render_menu()
            return
        entry = self._current_doc_entries()[self._highlight]
        self._open_document(entry)

    def _go_back(self) -> None:
        if self._menu_level == "doc":
            self._menu_level = "era"
            self._selected_era = None
        self._reset_menu_position()
        self._render_menu()

    def _reset_menu_position(self) -> None:
        self._highlight = 0
        self._page_start = 0

    def _current_menu_options(self) -> list[str]:
        if self._menu_level == "era":
            return [era for era in self._era_order if self._era_has_docs(era)]
        return [entry.title for entry in self._current_doc_entries()]

    def _current_doc_entries(self) -> list[HistoryCatalogEntry]:
        if self._selected_era is None:
            return []
        return self._entries_by_era.get(self._selected_era, [])

    def _era_has_docs(self, era: str) -> bool:
        return bool(self._entries_by_era.get(era))

    def _render_menu(self) -> None:
        options = self._current_menu_options()
        has_back = self._menu_level == "doc"
        if not options:
            self._renderer.show_text(
                ["목록이 비어 있어요.", BACK_LABEL],
                size=34,
                highlight_index=1,
                title=HISTORY_TITLE,
                layout="menu",
                show_exit_button=True,
                has_back=has_back,
                page_index=0,
                page_count=1,
            )
            return

        self._highlight = min(self._highlight, len(options) - 1)
        self._page_start = (self._highlight // MENU_PAGE_SIZE) * MENU_PAGE_SIZE
        page_count = max(1, math.ceil(len(options) / MENU_PAGE_SIZE))
        page_index = self._page_start // MENU_PAGE_SIZE
        visible = options[self._page_start : self._page_start + MENU_PAGE_SIZE]
        visible_display = [displayable_history_text(item) for item in visible]
        local_highlight = self._highlight - self._page_start
        self._renderer.show_history_menu(
            visible_display,
            local_highlight,
            displayable_history_text(self._menu_title()) or HISTORY_TITLE,
            has_back=has_back,
            page_index=page_index,
            page_count=page_count,
        )

    def _menu_title(self) -> str:
        if self._menu_level == "era":
            return HISTORY_TITLE
        return self._selected_era or HISTORY_TITLE

    def _open_document(self, entry: HistoryCatalogEntry) -> None:
        try:
            document = self._load_document(entry)
        except (OSError, ValueError, json.JSONDecodeError):
            logger.warning("Failed to load history document: %s", entry.doc_hash, exc_info=True)
            self._render_menu()
            return
        if not document.scenes:
            logger.warning("History document has no scenes: %s", entry.doc_hash)
            self._render_menu()
            return
        if not document.sections:
            logger.warning("History document has no sections: %s", entry.doc_hash)
            self._render_menu()
            return
        self._current_doc = document
        self._section_index = 0
        self._scene_index = 0
        self._section_scene_offset = 0
        self._last_shown_history_image = None
        self._scene_gesture.reset()
        self._start_section()

    def _start_section(self) -> None:
        document = self._current_doc
        if document is None:
            self._return_to_catalogue()
            return
        if self._section_index >= len(document.sections):
            self._finish_document()
            return
        section = document.sections[self._section_index]
        if not section.scenes:
            self._go_next_section()
            return
        scene_offset = min(max(self._section_scene_offset, 0), len(section.scenes) - 1)
        self._section_scene_offset = scene_offset
        first_scene = section.scenes[scene_offset]
        self._scene_index = document.scenes.index(first_scene)
        self._advance_event.clear()
        self._exit_event.clear()
        self._narration_done_event.clear()
        self._narration_done_at = None
        self._done_until = None
        self._pending_previous = False
        self._awaiting_consent = False
        self._consent_reprompted = False
        self._session.transition_history_state(SessionState.HISTORY_SCENE)
        render_token = self._render_scene_visual(
            first_scene,
            fallback_image=self._resolve_imageless_scene_image(document, first_scene),
        )
        self._session.set_expression(CharacterExpression.SPEAKING)
        # The lead-in only plays for the document's very first scene, never when a
        # mid-document scene rewind re-enters section 0 at a later scene.
        include_lead_in = self._section_index == 0 and scene_offset == 0
        self._narration_thread = threading.Thread(
            target=self._run_narration_worker,
            args=(document, section, render_token, include_lead_in, scene_offset),
            name="HistoryNarrationWorker",
            daemon=True,
        )
        self._narration_thread.start()

    def _render_scene_visual(
        self,
        scene: HistoryScene,
        progress: float = 0.0,
        *,
        fallback_image: HistoryImage | None = None,
    ) -> str | None:
        image = self._select_scene_image_by_progress(scene, progress)
        uses_own_image = image is not None
        if image is None:
            image = fallback_image
        if image is not None and image.path.exists():
            image_caption = self._display_text(image.caption) if uses_own_image else None
            caption = image_caption or self._first_caption(scene.image_captions)
            lines = [caption] if caption else []
            token = self._renderer.show_card(
                image_path=image.path,
                lines=lines,
                title=None,
                sublabel=None,
                show_exit_button=True,
                show_back=self._can_go_previous_step(),
            )
            self._last_shown_history_image = image
            return token
        if image is not None:
            logger.warning("History image missing, rendering text fallback: %s", image.path)
        self._renderer.show_text(
            [MISSING_IMAGE_LABEL],
            size=34,
            title=None,
            layout="center",
            show_exit_button=True,
        )
        return None

    def _resolve_imageless_scene_image(
        self,
        document: HistoryDocument,
        scene: HistoryScene,
    ) -> HistoryImage | None:
        """Return the look-ahead or carried image for a scene without images."""
        if self._select_focal_image(scene) is not None:
            return None
        return self._next_focal_image(document, scene) or self._last_shown_history_image

    @staticmethod
    def _next_focal_image(
        document: HistoryDocument,
        scene: HistoryScene,
    ) -> HistoryImage | None:
        """Return the focal image from the next document-order scene that has one."""
        try:
            scene_index = document.scenes.index(scene)
        except ValueError:
            return None
        for next_scene in document.scenes[scene_index + 1 :]:
            image = HistoryModeController._select_focal_image(next_scene)
            if image is not None:
                return image
        return None

    @staticmethod
    def _select_focal_image(scene: HistoryScene) -> HistoryImage | None:
        """Return the first display image using the scene display ordering."""
        return HistoryModeController._select_scene_image(scene, 0)

    @staticmethod
    def _select_scene_image(scene: HistoryScene, index: int) -> HistoryImage | None:
        """Return the image at ``index`` from the preferred display order."""
        ordered = HistoryModeController._ordered_scene_images(scene)
        if not ordered:
            return None
        return ordered[index % len(ordered)]

    @staticmethod
    def _anchored_display_sequence(
        scene: HistoryScene,
    ) -> tuple[tuple[HistoryImage, float], ...]:
        """Return scene images paired with anchors, ordered by playback position.

        When every image carries a build-time ``anchor_ratio`` the sequence is
        sorted by that ratio so each image appears as the narration reaches its
        position. When anchors are absent (older data) the display order falls
        back to the infographic-first ordering with even spacing, preserving the
        previous reading-order behavior. The result is always non-decreasing in
        anchor value so the runtime never flips an image backward.
        """
        ordered = HistoryModeController._ordered_scene_images(scene)
        count = len(ordered)
        if count == 0:
            return ()
        if all(image.anchor_ratio is not None for image in ordered):
            paired = sorted(
                ordered,
                key=lambda image: float(image.anchor_ratio or 0.0),
            )
            return tuple((image, float(image.anchor_ratio or 0.0)) for image in paired)
        return tuple((image, index / count) for index, image in enumerate(ordered))

    @staticmethod
    def _select_scene_image_by_progress(
        scene: HistoryScene, progress: float
    ) -> HistoryImage | None:
        """Return the last image whose anchor has been reached at ``progress``.

        ``progress`` is the playback position within the scene in ``[0, 1]``.
        Defaults to the first display image at the scene start.
        """
        sequence = HistoryModeController._anchored_display_sequence(scene)
        if not sequence:
            return None
        selected = sequence[0][0]
        for image, anchor in sequence:
            if anchor <= progress:
                selected = image
            else:
                break
        return selected

    @staticmethod
    def _ordered_scene_images(scene: HistoryScene) -> tuple[HistoryImage, ...]:
        """Return scene images with infographics first, preserving stable order."""
        if not scene.images:
            return ()
        return tuple(
            image
            for _index, image in sorted(
                enumerate(scene.images),
                key=lambda item: (
                    not item[1].is_infographic,
                    not item[1].clean,
                    item[0],
                ),
            )
        )

    @staticmethod
    def _first_caption(captions: tuple[str, ...]) -> str | None:
        for caption in captions:
            stripped = displayable_history_text(caption)
            if stripped:
                return stripped
        return None

    @staticmethod
    def _display_text(text: str | None) -> str | None:
        if text is None:
            return None
        filtered = displayable_history_text(text)
        return filtered or None

    def _narration_segments(self, scene: HistoryScene) -> Sequence[str | None]:
        return history_narration_segments(scene.narration, scene.section_title)

    def _should_stop_narration(self) -> bool:
        return self._advance_event.is_set() or self._exit_event.is_set()

    def _play_tts_text(self, tts: Any, text: str) -> bool:
        cache_path = tts_cache.lookup(text, "ko")
        if cache_path is not None:
            try:
                from core.funny_english_mode import _load_wav

                audio_samples, sample_rate = _load_wav(cache_path)
            except Exception:
                logger.warning("History cached TTS WAV load failed: %s", cache_path, exc_info=True)
            else:
                return self._play_tts_audio_samples(audio_samples, sample_rate)

        # Sanitize before live synthesis so Korean-history Hanja, middle dots,
        # and fullwidth punctuation never reach (and crash) the TTS engine.
        # The cache lookup above uses the original text on purpose: its key is
        # normalized internally and idempotently, so this does not affect hits.
        synth_text = normalize_tts_text(text)
        if not synth_text:
            logger.warning("History narration text empty after sanitization; skipping segment")
            return True
        try:
            audio_samples, sample_rate = tts.synthesize(synth_text, language="ko")
        except Exception:
            logger.warning(
                "History narration synthesis failed; skipping segment",
                exc_info=True,
            )
            return True
        return self._play_tts_audio_samples(audio_samples, sample_rate)

    def _play_tts_audio_samples(self, audio_samples: Any, sample_rate: int) -> bool:
        if self._should_stop_narration():
            return False
        self._session.begin_guarded_playback()
        try:
            if self._should_stop_narration():
                return False
            self._session.play_guarded(audio_samples, sample_rate)
        finally:
            self._session.end_guarded_playback()
        return True

    def _play_system_prompt(self, text: str) -> None:
        try:
            tts = self._mm.tts
            if tts is None:
                raise RuntimeError("History system prompt requires a loaded TTS engine")
            self._play_tts_text(tts, text)
        except Exception:
            logger.warning("History system prompt playback failed", exc_info=True)

    def _run_narration_worker(
        self,
        document: HistoryDocument,
        section: HistorySection,
        initial_render_token: str | None,
        include_lead_in: bool,
        scene_offset: int = 0,
    ) -> None:
        try:
            render_token = initial_render_token
            if not self._renderer.wait_until_rendered(
                render_token,
                timeout=RENDER_ACK_TIMEOUT_S,
            ):
                logger.debug("History render token timed out before narration")
            tts = self._mm.tts
            if tts is None:
                raise RuntimeError("History narration requires a loaded TTS engine")
            if include_lead_in and not self._play_tts_text(
                tts,
                f"지금부터 '{document.title}' 이야기를 들려줄게.",
            ):
                return
            start_offset = min(max(scene_offset, 0), len(section.scenes) - 1)
            for relative_index, scene in enumerate(section.scenes[start_offset:]):
                if self._should_stop_narration():
                    break
                # Track the document-absolute index of the scene now on screen so the
                # scene back button reflects the current scene, not the section start.
                self._scene_index = document.scenes.index(scene)
                if relative_index > 0:
                    if self._should_stop_narration():
                        break
                    render_token = self._render_scene_visual(
                        scene,
                        fallback_image=self._resolve_imageless_scene_image(document, scene),
                    )
                    if not self._renderer.wait_until_rendered(
                        render_token,
                        timeout=RENDER_ACK_TIMEOUT_S,
                    ):
                        logger.debug("History render token timed out before narration")
                if self._should_stop_narration():
                    break
                segments = list(self._narration_segments(scene))
                spoken_total = sum(1 for segment in segments if segment is not None)
                image_count = len(self._ordered_scene_images(scene))
                spoken_index = 0
                current_image = self._select_scene_image_by_progress(scene, 0.0)
                for segment in segments:
                    if segment is None:
                        if self._exit_event.wait(TITLE_BOUNDARY_PAUSE_S):
                            break
                        continue
                    if self._should_stop_narration():
                        break
                    progress = spoken_index / (spoken_total - 1) if spoken_total > 1 else 0.0
                    if image_count > 1:
                        next_image = self._select_scene_image_by_progress(scene, progress)
                        if spoken_index > 0 and next_image is not current_image:
                            if self._should_stop_narration():
                                break
                            self._render_scene_visual(scene, progress)
                        current_image = next_image
                    if not self._play_tts_text(tts, segment):
                        break
                    spoken_index += 1
                if self._should_stop_narration():
                    break
        except Exception:
            logger.warning("History narration worker failed", exc_info=True)
        finally:
            self._narration_done_at = self._now()
            self._narration_done_event.set()

    def _enter_consent_state(self) -> None:
        if self._awaiting_consent:
            return
        self._advance_event.clear()
        self._pending_previous = False
        self._awaiting_consent = True
        self._consent_reprompted = False
        self._session.transition_history_state(SessionState.HISTORY_CONSENT)
        self._session.set_expression(CharacterExpression.NEUTRAL)
        self._render_consent_prompt(CONSENT_PROMPT)
        self._play_system_prompt(CONSENT_PROMPT)

    def _render_consent_prompt(self, prompt: str) -> None:
        document = self._current_doc
        title = HISTORY_TITLE
        if document is not None and self._section_index < len(document.sections):
            section = document.sections[self._section_index]
            title = (
                self._display_text(section.section_title)
                or self._display_text(document.title)
                or HISTORY_TITLE
            )
        self._renderer.show_text(
            [prompt],
            size=30,
            title=title,
            layout="center",
            show_exit_button=True,
        )

    def _is_last_section(self) -> bool:
        document = self._current_doc
        return document is None or self._section_index + 1 >= len(document.sections)

    def _can_go_previous_step(self) -> bool:
        """Return whether a previous scene exists to rewind to.

        Granularity is one scene (not one section): the back button is shown
        whenever the current scene is not the document's very first scene, so it
        appears partway through section 0 as well as on every later section.
        """
        return self._current_doc is not None and self._scene_index > 0

    def _go_previous_section(self) -> None:
        document = self._current_doc
        if document is None:
            self._return_to_catalogue()
            return
        self._section_index = max(0, self._section_index - 1)
        self._section_scene_offset = 0
        self._start_section()

    def _go_previous_scene(self) -> None:
        """Rewind one scene, crossing into the previous section when needed."""
        document = self._current_doc
        if document is None:
            self._return_to_catalogue()
            return
        target_scene_index = self._scene_index - 1
        if target_scene_index < 0:
            self._section_scene_offset = 0
            self._start_section()
            return
        target_scene = document.scenes[target_scene_index]
        section_index, scene_offset = self._locate_scene_in_sections(document, target_scene)
        self._section_index = section_index
        self._section_scene_offset = scene_offset
        self._start_section()

    @staticmethod
    def _locate_scene_in_sections(
        document: HistoryDocument,
        scene: HistoryScene,
    ) -> tuple[int, int]:
        """Return the ``(section_index, within_section_offset)`` owning ``scene``."""
        for section_index, section in enumerate(document.sections):
            for scene_offset, section_scene in enumerate(section.scenes):
                if section_scene is scene:
                    return section_index, scene_offset
        return 0, 0

    def _go_next_section(self) -> None:
        document = self._current_doc
        if document is None:
            self._return_to_catalogue()
            return
        self._awaiting_consent = False
        self._consent_reprompted = False
        if self._section_index + 1 >= len(document.sections):
            self._finish_document()
            return
        self._section_index += 1
        self._section_scene_offset = 0
        self._start_section()

    def _finish_document(self) -> None:
        self._current_doc = None
        self._section_index = 0
        self._scene_index = 0
        self._section_scene_offset = 0
        self._last_shown_history_image = None
        self._advance_event.clear()
        self._pending_previous = False
        self._awaiting_consent = False
        self._consent_reprompted = False
        self._session.transition_history_state(SessionState.HISTORY_DONE)
        self._session.set_expression(CharacterExpression.EXCITED)
        self._renderer.show_text(
            [DONE_LABEL],
            size=46,
            title=HISTORY_TITLE,
            layout="center",
            show_exit_button=True,
        )
        self._done_until = self._now() + 1.0

    def _return_to_catalogue(self) -> None:
        self._current_doc = None
        self._section_index = 0
        self._scene_index = 0
        self._section_scene_offset = 0
        self._last_shown_history_image = None
        self._narration_done_event.clear()
        self._narration_done_at = None
        self._done_until = None
        self._awaiting_consent = False
        self._consent_reprompted = False
        self._scene_gesture.reset()
        self._session.transition_history_state(SessionState.HISTORY_SELECT)
        self._render_menu()

    def _load_catalog(self, manifest_path: Path) -> tuple[list[HistoryCatalogEntry], list[str]]:
        with manifest_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") not in (1, 2):
            raise ValueError("history manifest schema_version must be 1 or 2")
        raw_era_order = payload.get("era_order")
        raw_docs = payload.get("docs")
        if not isinstance(raw_era_order, list) or not all(
            isinstance(item, str) for item in raw_era_order
        ):
            raise ValueError("history manifest era_order must be list[str]")
        if not isinstance(raw_docs, list):
            raise ValueError("history manifest docs must be a list")
        entries = [self._parse_catalog_entry(raw) for raw in raw_docs]
        return entries, raw_era_order

    def _parse_catalog_entry(self, raw: Any) -> HistoryCatalogEntry:
        if not isinstance(raw, dict):
            raise ValueError("history manifest doc entries must be objects")
        doc_path = self._resolve_path(Path(self._require_str(raw, "doc_path")))
        return HistoryCatalogEntry(
            doc_hash=self._require_str(raw, "doc_hash"),
            title=self._require_str(raw, "title"),
            kind=self._require_str(raw, "kind"),
            era=self._require_str(raw, "era"),
            doc_path=doc_path,
            scene_count=self._require_int(raw, "scene_count"),
            image_count=self._require_int(raw, "image_count"),
            order=self._parse_catalog_order(raw),
        )

    @staticmethod
    def _parse_catalog_order(raw: dict[str, Any]) -> int:
        """Parse manifest ``order``, defaulting old manifests to legacy sorting."""
        value = raw.get("order", MISSING_ORDER_SENTINEL)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("history manifest doc order must be int")
        return int(value)

    @staticmethod
    def _index_catalog(
        entries: list[HistoryCatalogEntry],
    ) -> dict[str, list[HistoryCatalogEntry]]:
        indexed: dict[str, list[HistoryCatalogEntry]] = {}
        for entry in entries:
            indexed.setdefault(entry.era, []).append(entry)
        for group in indexed.values():
            if any(entry.order == MISSING_ORDER_SENTINEL for entry in group):
                group.sort(key=lambda item: item.title)
            else:
                group.sort(key=lambda item: item.order)
        return indexed

    def _load_document(self, entry: HistoryCatalogEntry) -> HistoryDocument:
        with entry.doc_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        raw_scenes = payload.get("scenes")
        if not isinstance(raw_scenes, list):
            raise ValueError(f"history document {entry.doc_hash} scenes must be a list")
        scenes = tuple(self._parse_scene(raw, index) for index, raw in enumerate(raw_scenes))
        return HistoryDocument(
            doc_hash=entry.doc_hash,
            title=self._require_str(payload, "title"),
            kind=self._require_str(payload, "kind"),
            era=self._require_str(payload, "era"),
            scenes=scenes,
            sections=self._group_scenes_into_sections(scenes),
        )

    def _parse_scene(self, raw: Any, fallback_section_index: int) -> HistoryScene:
        if not isinstance(raw, dict):
            raise ValueError("history scenes must be objects")
        narration = self._require_str(raw, "narration")
        raw_images = raw.get("images", [])
        if not isinstance(raw_images, list):
            raise ValueError("history scene images must be a list")
        section_index_raw = raw.get("section_index", fallback_section_index)
        if not isinstance(section_index_raw, int):
            raise ValueError("history scene section_index must be int")
        section_title = raw.get("section_title")
        if section_title is not None and not isinstance(section_title, str):
            raise ValueError("history scene section_title must be text or null")
        raw_captions = raw.get("image_captions", [])
        if not isinstance(raw_captions, list) or not all(
            isinstance(item, str) for item in raw_captions
        ):
            raise ValueError("history scene image_captions must be list[str]")
        return HistoryScene(
            seq=self._require_int(raw, "seq"),
            section_index=section_index_raw,
            section_title=section_title,
            narration=narration,
            est_speech_ms=self._require_int(raw, "est_speech_ms"),
            tail_silence_ms=self._require_int(raw, "tail_silence_ms"),
            image_captions=tuple(raw_captions),
            images=tuple(self._parse_image(image) for image in raw_images),
        )

    def _parse_image(self, raw: Any) -> HistoryImage:
        if not isinstance(raw, dict):
            raise ValueError("history scene images must be objects")
        path = self._resolve_path(Path(self._require_str(raw, "path")))
        caption = raw.get("caption")
        if caption is not None and not isinstance(caption, str):
            raise ValueError("history image caption must be text or null")
        anchor_ratio = self._parse_anchor_ratio(raw.get("anchor_ratio"))
        return HistoryImage(
            path=path,
            caption=caption,
            letterboxed=bool(raw.get("letterboxed", False)),
            clean=bool(raw.get("clean", False)),
            is_infographic=bool(raw.get("is_infographic", False)),
            anchor_ratio=anchor_ratio,
        )

    @staticmethod
    def _parse_anchor_ratio(value: Any) -> float | None:
        """Parse the optional ``anchor_ratio`` clamped to ``[0, 1]``."""
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("history image anchor_ratio must be a number or null")
        numeric = float(value)
        if math.isnan(numeric):
            raise ValueError("history image anchor_ratio must be a number or null")
        return min(max(numeric, 0.0), 1.0)

    @staticmethod
    def _group_scenes_into_sections(scenes: tuple[HistoryScene, ...]) -> tuple[HistorySection, ...]:
        grouped: dict[int, list[HistoryScene]] = {}
        for scene in scenes:
            grouped.setdefault(scene.section_index, []).append(scene)
        sections: list[HistorySection] = []
        for section_index in sorted(grouped):
            section_scenes = tuple(grouped[section_index])
            title = next(
                (scene.section_title for scene in section_scenes if scene.section_title),
                None,
            )
            captions: list[str] = []
            for scene in section_scenes:
                captions.extend(scene.image_captions)
            sections.append(
                HistorySection(
                    section_index=section_index,
                    section_title=title,
                    scenes=section_scenes,
                    image_captions=tuple(captions),
                )
            )
        return tuple(sections)

    def _resolve_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return self._repo_root / path

    def _restore_residency_after_entry_failure(self) -> None:
        """Best-effort restore when entering history mode fails before activation."""
        with self._lock:
            self._active = False
        with suppress(Exception):
            self._mm.unload_tts(force=False)
        with suppress(Exception):
            self._mm.reset_preload_state()
        with suppress(Exception):
            self._mm.preload_stt()

    @staticmethod
    def _require_str(payload: dict[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"history payload field {key!r} must be non-empty text")
        return value

    @staticmethod
    def _require_int(payload: dict[str, Any], key: str) -> int:
        value = payload.get(key)
        if not isinstance(value, int):
            raise ValueError(f"history payload field {key!r} must be int")
        return value

    def _now(self) -> float:
        if self._monotonic_clock is not None:
            return self._monotonic_clock()
        import time

        return time.monotonic()


__all__ = [
    "BACK_LABEL",
    "CONSENT_PROMPT",
    "CONSENT_REPROMPT",
    "DONE_LABEL",
    "HISTORY_TITLE",
    "MISSING_IMAGE_LABEL",
    "HistoryCatalogEntry",
    "HistoryDocument",
    "HistoryImage",
    "HistoryModeController",
    "HistoryScene",
    "HistorySection",
    "history_narration_segments",
]
