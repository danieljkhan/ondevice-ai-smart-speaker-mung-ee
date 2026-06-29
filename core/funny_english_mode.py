"""Runtime controller for the Funny English read-along mode."""

from __future__ import annotations

import json
import logging
import math
import os
import wave
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from core.character_expression import CharacterExpression
from core.funny_english_match import FunnyEnglishMatchResult, normalize_hotword_csv
from core.model_manager import ModelType
from core.pipeline import Utterance
from core.session_manager import FunnyEnglishListenInterrupt, SessionState
from hardware.touch_input import TouchEvent
from models import tts_cache

logger = logging.getLogger(__name__)

FUNNY_ENGLISH_TITLE = "Funny English"
BACK_LABEL = "Back"
DONE_LABEL = "All done!"
MISSING_CONTENT_LABEL = "Card is getting ready."
FE_INTRO_TITLE = "퍼니 잉글리시"
FE_INTRO_LINES = ["그림을 보고", "들리는 영어를 따라 말해요"]
FE_INTRO_SUBLABEL = "화면을 누르면 시작해요"
FE_INTRO_SPEECH = (
    "그림을 보고, 뭉이가 들려주는 영어를 잘 듣고 따라 말해보세요. 준비됐으면 화면을 눌러요!"
)
FE_REPEAT_CUE_SUBLABEL = "눌러서 띠링 소리가 나면 따라 말해요"
FE_REPEAT_CUE_SPEECH = "눌러서 따라 말해봐"
FE_PASS_FEEDBACK_SPEECH = "좋아! 아주 잘 읽었어!"
FE_LAST_ATTEMPT_ADVANCE_SPEECH = "괜찮아! 뭉이가 한 번 더 들려주고 다음으로 갈게!"
FE_SILENT_JUNK_RETRY_SPEECH = "한 번 더 듣고 같이 해보자!"
FE_CLOSE_RETRY_SPEECH = "거의 다 왔어! 한 번 더 해보자!"
FE_OTHER_RETRY_SPEECH = "좋아, 첫 소리부터 다시 해보자!"
FE_DONE_SPEECH = "오늘 영어 읽기 끝! 멋지게 해냈어!"
FE_KOREAN_SPEECH_LINES = (
    FE_INTRO_SPEECH,
    FE_REPEAT_CUE_SPEECH,
    FE_PASS_FEEDBACK_SPEECH,
    FE_LAST_ATTEMPT_ADVANCE_SPEECH,
    FE_SILENT_JUNK_RETRY_SPEECH,
    FE_CLOSE_RETRY_SPEECH,
    FE_OTHER_RETRY_SPEECH,
    FE_DONE_SPEECH,
)
FE_LISTENING_CUE_SUBLABEL = "지금 따라 말해요"
FE_CAPTURED_CUE_SUBLABEL = "들었어요!"
FE_RENDER_ACK_TIMEOUT_S = 1.0
DEFAULT_MAX_ATTEMPTS = 3
MENU_PAGE_SIZE = 4
_FE_TARGET_RMS = 0.14
_FE_MAX_GAIN = 8.0
_FE_PEAK_CEILING = 0.97
_FE_RMS_EPSILON = 1.0e-5
FeedbackAction = Literal["retry", "advance", "select"]
_FILLED_STAR = "★"
_EMPTY_STAR = "☆"


def funny_english_score_line(result: FunnyEnglishMatchResult | None) -> str | None:
    """Return a pronunciation score line for one attempt, or ``None`` for no speech."""
    score = funny_english_score(result)
    if score is None:
        return None
    star_count = _star_count(score)
    stars = _FILLED_STAR * star_count + _EMPTY_STAR * (5 - star_count)
    return f"{stars} ({score})"


def funny_english_score(result: FunnyEnglishMatchResult | None) -> int | None:
    """Return the 0-100 pronunciation score for a captured attempt."""
    if result is None or result.band == "silent_junk":
        return None
    raw_score = round(max(result.matched_pct, result.similarity) * 100)
    return min(100, max(0, raw_score))


def _star_count(score: int) -> int:
    """Return the 0-5 star bucket for a pronunciation score."""
    if score >= 90:
        return 5
    if score >= 75:
        return 4
    if score >= 60:
        return 3
    if score >= 40:
        return 2
    if score >= 1:
        return 1
    return 0


class FunnyEnglishRenderer(Protocol):
    """Renderer API used by Funny English."""

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
        """Display an image+text card."""

    def wait_until_rendered(self, token: str | None, timeout: float) -> bool:
        """Wait until a tokenized render request is displayed."""

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


class FunnyEnglishSession(Protocol):
    """Session-manager API used by Funny English."""

    def transition_funny_english_state(self, state: SessionState) -> None:
        """Transition to one of the Funny English states."""

    def set_expression(self, expression: CharacterExpression) -> None:
        """Emit a character expression."""

    def begin_guarded_playback(self) -> None:
        """Begin one guarded playback section."""

    def play_guarded(self, audio_samples: Any, sample_rate: int) -> None:
        """Play audio while the guarded playback section is active."""

    def end_guarded_playback(self) -> None:
        """End one guarded playback section."""

    def capture_funny_english_audio(self) -> Utterance | FunnyEnglishListenInterrupt | None:
        """Capture one read-aloud utterance."""

    def exit_funny_english_mode(self) -> None:
        """Exit Funny English mode at the session level."""

    def run_funny_english_attempt(
        self,
        utterance: Utterance,
        card: Any,
        *,
        stt_hotwords_csv: str | None = None,
    ) -> FunnyEnglishMatchResult:
        """Score one Funny English attempt through STT only."""


class FunnyEnglishModelManager(Protocol):
    """Model-manager API required by Funny English."""

    @property
    def tts(self) -> Any:
        """Return the loaded TTS engine."""

    def cancel_preload_and_join(self) -> None:
        """Cancel and join any pending STT preload."""

    def reset_preload_state(self) -> None:
        """Reset pending preload state."""

    def preload_stt(self, *, stt_hotwords_csv: str | None = None) -> None:
        """Start STT preload best-effort."""

    def unload_stt(self, *, force: bool = False) -> None:
        """Unload the STT model."""

    def load(self, model_type: ModelType, *, stt_hotwords_csv: str | None = None) -> None:
        """Load a model."""

    def unload_tts(self, *, force: bool = False) -> None:
        """Unload the TTS model."""


class FunnyEnglishBgm(Protocol):
    """BGM channel API used by Funny English."""

    def start_loop(self, audio_samples: Any, sample_rate: int, *, volume: float = 0.16) -> None:
        """Start BGM looping."""

    def duck(self, *, volume: float = 0.0) -> None:
        """Duck BGM."""

    def unduck(self) -> None:
        """Restore BGM volume."""

    def stop(self) -> None:
        """Stop BGM."""


@dataclass(frozen=True)
class FunnyEnglishCard:
    """One read-along card from the committed curriculum."""

    card_id: str
    stage: int
    card_type: str
    text: str
    tokens: tuple[str, ...]
    ko_gloss: str
    syllables: tuple[str, ...]
    first_sound: str
    image_path: Path | None
    model_audio_path: Path | None
    accept_threshold: float
    hotwords: tuple[str, ...]
    accept_aliases: tuple[str, ...] = ()

    @property
    def hotword_tokens(self) -> tuple[str, ...]:
        """Return the pre-normalized STT hotword tokens for this card."""
        return self.hotwords or (*self.tokens, *self.accept_aliases)

    @property
    def hotwords_csv(self) -> str:
        """Return the STT hotword CSV for this card."""
        return normalize_hotword_csv(self.hotword_tokens)


@dataclass(frozen=True)
class FunnyEnglishStage:
    """One curriculum stage."""

    stage: int
    title: str
    cards: tuple[FunnyEnglishCard, ...]


def _normalize_fe_playback_audio(audio_samples: Any) -> Any:
    """Return FE playback samples normalized to the shared local loudness target."""
    if not isinstance(audio_samples, np.ndarray):
        return audio_samples
    if audio_samples.size == 0:
        return audio_samples
    if not np.issubdtype(audio_samples.dtype, np.number):
        return audio_samples

    original_dtype = audio_samples.dtype
    integer_info: np.iinfo[Any] | None = None
    integer_scale = 1.0
    if np.issubdtype(original_dtype, np.integer):
        integer_info = np.iinfo(original_dtype)
        if integer_info.min >= 0:
            return audio_samples
        integer_scale = float(max(abs(integer_info.min), integer_info.max))
        working = audio_samples.astype(np.float64) / integer_scale
    else:
        working = audio_samples.astype(np.float64, copy=True)

    np.nan_to_num(working, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    rms = float(np.sqrt(np.mean(np.square(working))))
    if not math.isfinite(rms) or rms < _FE_RMS_EPSILON:
        return audio_samples

    gain = min(_FE_TARGET_RMS / rms, _FE_MAX_GAIN)
    adjusted = working * gain
    peak = float(np.max(np.abs(adjusted)))
    if math.isfinite(peak) and peak > _FE_PEAK_CEILING:
        adjusted *= _FE_PEAK_CEILING / peak

    if integer_info is not None:
        restored = np.rint(adjusted * integer_scale)
        restored = np.clip(restored, integer_info.min, integer_info.max)
        return restored.astype(original_dtype)
    if np.issubdtype(original_dtype, np.floating):
        dtype_ceiling = float(
            np.nextafter(
                np.array(_FE_PEAK_CEILING, dtype=original_dtype),
                np.array(0.0, dtype=original_dtype),
            )
        )
        adjusted = np.clip(adjusted, -dtype_ceiling, dtype_ceiling)
        return adjusted.astype(original_dtype, copy=False)
    return audio_samples


class FunnyEnglishModeController:
    """Coordinate Funny English selection, rendering, audio modeling, and scoring."""

    def __init__(
        self,
        *,
        session: FunnyEnglishSession,
        model_manager: FunnyEnglishModelManager,
        renderer: FunnyEnglishRenderer,
        manifest_path: Path = Path("assets/funny_english/manifest.json"),
        repo_root: Path = Path("."),
        bgm_player: FunnyEnglishBgm | None = None,
        bgm_clip: tuple[Any, int] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        """Create the Funny English controller."""
        self._session = session
        self._mm = model_manager
        self._renderer = renderer
        self._repo_root = repo_root
        self._manifest_path = self._resolve_path(manifest_path)
        self._bgm_player = bgm_player
        self._bgm_clip = bgm_clip
        self._monotonic_clock = monotonic_clock
        self._stages = self._load_manifest(self._manifest_path)
        self._active = False
        self._stage_index = 0
        self._card_index = 0
        self._highlight = 0
        self._page_start = 0
        self._feedback_action: FeedbackAction = "select"
        self._attempts_by_card: dict[str, int] = {}
        self._successes_by_card: dict[str, int] = {}

    @property
    def active(self) -> bool:
        """Return whether Funny English is currently active."""
        return self._active

    def enter(self) -> None:
        """Enter stage selection mode and clear any unprompted STT preload."""
        self._mm.cancel_preload_and_join()
        try:
            self._mm.unload_stt(force=True)
            self._mm.load(ModelType.TTS)
            self._mm.reset_preload_state()
        except Exception:
            self._restore_residency_after_entry_failure()
            raise
        self._active = True
        self._stage_index = 0
        self._card_index = 0
        self._highlight = 0
        self._page_start = 0
        self._feedback_action = "select"
        self._attempts_by_card.clear()
        self._successes_by_card.clear()
        self._start_bgm()
        self._session.transition_funny_english_state(SessionState.FE_INTRO)
        self._render_intro()
        self._speak_feedback(FE_INTRO_SPEECH, language="ko")

    def exit(self) -> None:
        """Exit Funny English and restore normal STT preload behavior."""
        self._active = False
        self._stop_bgm()
        self._renderer.clear_image()
        self._renderer.clear_text()
        self._mm.unload_tts(force=False)
        self._mm.reset_preload_state()
        self._mm.preload_stt()

    def handle_select_tap(self, event: TouchEvent) -> None:
        """Handle a coordinate-less stage-selection fallback tap."""
        if not self.active:
            return
        del event
        self._advance_highlight()
        self._render_stage_menu()

    def handle_intro_tap(self) -> None:
        """Advance from the instruction screen into stage selection."""
        if not self.active:
            return
        self._session.transition_funny_english_state(SessionState.FE_SELECT)
        self._render_stage_menu()

    def handle_select_target(self, index: int) -> None:
        """Select the visible stage at ``index`` from a direct-touch hit."""
        if not self.active:
            return
        options = self._stage_options()
        selected_index = self._page_start + index
        if selected_index < 0 or selected_index >= len(options):
            return
        self._stage_index = selected_index
        self._card_index = 0
        self._highlight = selected_index
        self._page_start = (self._highlight // MENU_PAGE_SIZE) * MENU_PAGE_SIZE
        self._start_prompt()

    def handle_prev_page(self) -> None:
        """Move the stage menu to the previous page."""
        if not self.active:
            return
        options = self._stage_options()
        if not options:
            return
        previous_page_start = max(0, self._page_start - MENU_PAGE_SIZE)
        self._page_start = previous_page_start
        self._highlight = min(previous_page_start, len(options) - 1)
        self._render_stage_menu()

    def handle_next_page(self) -> None:
        """Move the stage menu to the next page."""
        if not self.active:
            return
        options = self._stage_options()
        if not options:
            return
        last_page_start = ((len(options) - 1) // MENU_PAGE_SIZE) * MENU_PAGE_SIZE
        next_page_start = min(last_page_start, self._page_start + MENU_PAGE_SIZE)
        self._page_start = next_page_start
        self._highlight = min(next_page_start, len(options) - 1)
        self._render_stage_menu()

    def handle_prev_card(self) -> None:
        """Move to the previous card in the current stage, clamped at the first card."""
        self._move_card(-1)

    def handle_next_card(self) -> None:
        """Move to the next card in the current stage, clamped at the last card."""
        self._move_card(1)

    def handle_prompt_tap(self, event: TouchEvent) -> None:
        """Start listening for the current card."""
        del event
        if not self.active:
            return
        self._run_listen_attempt()

    def handle_feedback_tap(self, event: TouchEvent) -> None:
        """Advance from feedback according to the last scoring band."""
        del event
        if not self.active:
            return
        if self._feedback_action == "retry":
            self._start_prompt()
            return
        if self._feedback_action == "advance":
            self._advance_card()
            return
        self._return_to_select()

    def handle_done_tap(self, event: TouchEvent) -> None:
        """Return to stage selection after a done card."""
        del event
        if not self.active:
            return
        self._return_to_select()

    def handle_back(self) -> None:
        """Return to the stage-selection menu from an in-stage screen."""
        if not self.active:
            return
        self._return_to_select()

    def poll(self) -> None:
        """Run non-blocking timer maintenance."""

    def _start_prompt(self) -> None:
        card = self._current_card()
        if card is None:
            self._finish_stage()
            return
        self._bgm_unduck()
        self._session.transition_funny_english_state(SessionState.FE_PROMPT)
        attempt_index = self._attempts_by_card.get(card.card_id, 0)
        self._show_current_card(
            card,
            lines=[card.text],
            sublabel=self._prompt_sublabel(card, attempt_index),
        )
        self._session.set_expression(CharacterExpression.SPEAKING)
        self._play_model_for_attempt(card, attempt_index)
        self._show_current_card(card, lines=[card.text], sublabel=FE_REPEAT_CUE_SUBLABEL)
        self._session.set_expression(CharacterExpression.LISTENING)
        if attempt_index == 0:
            self._speak_feedback(FE_REPEAT_CUE_SPEECH, language="ko")
        with suppress(Exception):
            self._mm.preload_stt(stt_hotwords_csv=self._stage_hotwords_csv())

    def _run_listen_attempt(self) -> None:
        card = self._current_card()
        if card is None:
            self._finish_stage()
            return
        stage_hotwords_csv = self._stage_hotwords_csv()
        self._session.transition_funny_english_state(SessionState.FE_LISTEN)
        self._bgm_duck()
        self._mm.load(ModelType.STT, stt_hotwords_csv=stage_hotwords_csv)
        token = self._show_current_card(card, lines=[card.text], sublabel=FE_LISTENING_CUE_SUBLABEL)
        self._session.set_expression(CharacterExpression.LISTENING)
        self._renderer.wait_until_rendered(token, timeout=FE_RENDER_ACK_TIMEOUT_S)
        utterance = self._session.capture_funny_english_audio()
        if isinstance(utterance, FunnyEnglishListenInterrupt):
            if utterance.kind == "exit":
                self._bgm_unduck()
                self._session.exit_funny_english_mode()
            else:
                self._bgm_unduck()
                self.handle_back()
            return
        if utterance is not None:
            token = self._show_current_card(
                card,
                lines=[card.text],
                sublabel=FE_CAPTURED_CUE_SUBLABEL,
            )
            self._session.set_expression(CharacterExpression.HAPPY)
            self._renderer.wait_until_rendered(token, timeout=FE_RENDER_ACK_TIMEOUT_S)
        self._session.transition_funny_english_state(SessionState.FE_SCORE)
        if utterance is None:
            self._handle_score(card, None)
            return
        result = self._session.run_funny_english_attempt(
            utterance,
            card,
            stt_hotwords_csv=stage_hotwords_csv,
        )
        self._handle_score(card, result)

    def _handle_score(
        self,
        card: FunnyEnglishCard,
        result: FunnyEnglishMatchResult | None,
    ) -> None:
        self._bgm_unduck()
        self._session.transition_funny_english_state(SessionState.FE_FEEDBACK)
        attempt_index = self._attempts_by_card.get(card.card_id, 0)
        score_line = funny_english_score_line(result)
        if result is not None and result.band == "pass":
            self._successes_by_card[card.card_id] = self._successes_by_card.get(card.card_id, 0) + 1
            self._attempts_by_card[card.card_id] = 0
            self._feedback_action = "advance"
            self._render_feedback(card, "Great reading!", "advance", score_line=score_line)
            self._speak_feedback(FE_PASS_FEEDBACK_SPEECH, language="ko")
            return

        max_attempts = _max_attempts()
        if attempt_index + 1 >= max_attempts:
            self._attempts_by_card[card.card_id] = 0
            self._feedback_action = "advance"
            self._render_feedback(card, "Let's keep going!", "advance", score_line=score_line)
            self._play_model_for_attempt(card, 0)
            self._speak_feedback(FE_LAST_ATTEMPT_ADVANCE_SPEECH, language="ko")
            return

        self._attempts_by_card[card.card_id] = attempt_index + 1
        self._feedback_action = "retry"
        if result is None or result.band == "silent_junk":
            message = "Listen again, then try."
            spoken = FE_SILENT_JUNK_RETRY_SPEECH
        elif result.band == "close":
            message = "Almost there!"
            spoken = FE_CLOSE_RETRY_SPEECH
        else:
            missed = ", ".join(result.missed_tokens[:1]) if result.missed_tokens else card.text
            message = f"Try: {missed}"
            spoken = FE_OTHER_RETRY_SPEECH
        self._render_feedback(card, message, "retry", score_line=score_line)
        self._speak_feedback(spoken, language="ko")

    def _render_feedback(
        self,
        card: FunnyEnglishCard,
        message: str,
        action: FeedbackAction,
        *,
        score_line: str | None = None,
    ) -> None:
        lines = [message]
        if score_line is not None:
            lines.insert(0, score_line)
        self._show_current_card(
            card,
            lines=lines,
            sublabel="Tap to try again"
            if action == "retry"
            else "다음으로 가려면 화면을 톡 눌러봐",
        )

    def _advance_card(self) -> None:
        stage = self._current_stage()
        if stage is None:
            self._return_to_select()
            return
        self._card_index += 1
        if self._card_index >= len(stage.cards):
            self._finish_stage()
            return
        self._start_prompt()

    def _move_card(self, delta: int) -> None:
        """Move within the current stage by ``delta`` cards without wrapping."""
        if not self.active:
            return
        stage = self._current_stage()
        if stage is None or not stage.cards:
            return
        last_index = len(stage.cards) - 1
        current_index = min(max(self._card_index, 0), last_index)
        next_index = min(max(current_index + delta, 0), last_index)
        if next_index == current_index:
            self._card_index = current_index
            return
        self._card_index = next_index
        self._attempts_by_card[stage.cards[next_index].card_id] = 0
        self._start_prompt()

    def _finish_stage(self) -> None:
        self._session.transition_funny_english_state(SessionState.FE_DONE)
        self._session.set_expression(CharacterExpression.EXCITED)
        self._feedback_action = "select"
        self._renderer.show_text(
            [DONE_LABEL],
            size=46,
            title=FUNNY_ENGLISH_TITLE,
            layout="center",
            show_exit_button=True,
            has_back=True,
        )
        self._speak_feedback(FE_DONE_SPEECH, language="ko")

    def _return_to_select(self) -> None:
        self._session.transition_funny_english_state(SessionState.FE_SELECT)
        self._highlight = min(self._stage_index, max(0, len(self._stages) - 1))
        self._page_start = (self._highlight // MENU_PAGE_SIZE) * MENU_PAGE_SIZE
        self._render_stage_menu()

    def _render_intro(self) -> None:
        self._renderer.show_text(
            [*FE_INTRO_LINES, FE_INTRO_SUBLABEL],
            size=40,
            title=FE_INTRO_TITLE,
            layout="center",
            show_exit_button=True,
            has_back=False,
        )

    def _render_stage_menu(self) -> None:
        options = self._stage_options()
        if not options:
            self._renderer.show_text(
                [MISSING_CONTENT_LABEL],
                size=34,
                title=FUNNY_ENGLISH_TITLE,
                layout="center",
            )
            return
        self._highlight = min(self._highlight, len(options) - 1)
        self._page_start = (self._highlight // MENU_PAGE_SIZE) * MENU_PAGE_SIZE
        page_count = max(1, math.ceil(len(options) / MENU_PAGE_SIZE))
        page_index = self._page_start // MENU_PAGE_SIZE
        visible = options[self._page_start : self._page_start + MENU_PAGE_SIZE]
        local_highlight = self._highlight - self._page_start
        self._renderer.show_text(
            visible,
            size=34,
            highlight_index=local_highlight,
            title=FUNNY_ENGLISH_TITLE,
            layout="menu",
            show_exit_button=True,
            has_back=False,
            page_index=page_index,
            page_count=page_count,
        )

    def _stage_options(self) -> list[str]:
        return [stage.title for stage in self._stages]

    def _advance_highlight(self) -> None:
        options = self._stage_options()
        if not options:
            self._highlight = 0
            self._page_start = 0
            return
        self._highlight = (self._highlight + 1) % len(options)
        self._page_start = (self._highlight // MENU_PAGE_SIZE) * MENU_PAGE_SIZE

    def _current_stage(self) -> FunnyEnglishStage | None:
        if not self._stages:
            return None
        return self._stages[min(self._stage_index, len(self._stages) - 1)]

    def _stage_hotwords_csv(self) -> str:
        stage = self._current_stage()
        if stage is None or not stage.cards:
            return ""
        return normalize_hotword_csv(
            tuple(token for card in stage.cards for token in card.hotword_tokens)
        )

    def _current_card(self) -> FunnyEnglishCard | None:
        stage = self._current_stage()
        if stage is None or not stage.cards:
            return None
        return stage.cards[min(self._card_index, len(stage.cards) - 1)]

    def _play_model_for_attempt(self, card: FunnyEnglishCard, attempt_index: int) -> None:
        del attempt_index
        cache_path = tts_cache.lookup(card.text, "en")
        if cache_path is not None and self._play_cached_wav(cache_path):
            return

        if card.model_audio_path is not None and card.model_audio_path.exists():
            try:
                audio, sample_rate = _load_wav(card.model_audio_path)
                self._play_guarded(audio, sample_rate)
                return
            except OSError:
                logger.warning("Failed to play Funny English model WAV: %s", card.model_audio_path)

        self._speak_feedback(card.text, language="en")

    def _show_current_card(
        self,
        card: FunnyEnglishCard,
        *,
        lines: list[str],
        sublabel: str | None,
    ) -> str | None:
        """Render a card with current in-stage navigation targets."""
        show_prev_card, show_next_card = self._card_nav_flags()
        return self._renderer.show_card(
            image_path=card.image_path,
            lines=lines,
            highlight_index=0,
            title=FUNNY_ENGLISH_TITLE,
            sublabel=sublabel,
            show_exit_button=True,
            show_back=True,
            show_prev_card=show_prev_card,
            show_next_card=show_next_card,
        )

    def _card_nav_flags(self) -> tuple[bool, bool]:
        """Return whether previous/next card buttons should be visible."""
        stage = self._current_stage()
        if stage is None or not stage.cards:
            return False, False
        current_index = min(max(self._card_index, 0), len(stage.cards) - 1)
        return current_index > 0, current_index + 1 < len(stage.cards)

    def _speak_feedback(self, text: str, *, language: str) -> None:
        try:
            cache_path = tts_cache.lookup(text, language)
            if cache_path is not None and self._play_cached_wav(cache_path):
                return

            self._mm.load(ModelType.TTS)
            tts = self._mm.tts
            if tts is None:
                raise RuntimeError("Funny English feedback requires a loaded TTS engine")
            audio, sample_rate = tts.synthesize(text, language=language)
            self._play_guarded(audio, sample_rate)
        except Exception:
            logger.warning("Funny English feedback synthesis failed", exc_info=True)

    def _play_cached_wav(self, path: Path) -> bool:
        try:
            audio, sample_rate = _load_wav(path)
        except Exception:
            logger.warning("Failed to load Funny English cached TTS WAV: %s", path, exc_info=True)
            return False
        self._play_guarded(audio, sample_rate)
        return True

    def _play_guarded(self, audio_samples: Any, sample_rate: int) -> None:
        normalized_samples = _normalize_fe_playback_audio(audio_samples)
        self._session.begin_guarded_playback()
        try:
            self._session.play_guarded(normalized_samples, sample_rate)
        finally:
            self._session.end_guarded_playback()

    def _start_bgm(self) -> None:
        if self._bgm_player is None or self._bgm_clip is None:
            return
        audio, sample_rate = self._bgm_clip
        with suppress(Exception):
            self._bgm_player.start_loop(audio, sample_rate, volume=0.16)

    def _bgm_duck(self) -> None:
        if self._bgm_player is not None:
            with suppress(Exception):
                self._bgm_player.duck(volume=0.0)

    def _bgm_unduck(self) -> None:
        if self._bgm_player is not None:
            with suppress(Exception):
                self._bgm_player.unduck()

    def _stop_bgm(self) -> None:
        if self._bgm_player is not None:
            with suppress(Exception):
                self._bgm_player.stop()

    def _prompt_sublabel(self, card: FunnyEnglishCard, attempt_index: int) -> str:
        if attempt_index == 1:
            return "Say it with me"
        if attempt_index >= 2:
            return f"First sound: {card.first_sound}"
        if card.ko_gloss:
            return card.ko_gloss
        return f"Stage {card.stage}"

    def _load_manifest(self, manifest_path: Path) -> tuple[FunnyEnglishStage, ...]:
        with manifest_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != 1:
            raise ValueError("Funny English manifest schema_version must be 1")
        raw_stages = payload.get("stages")
        if not isinstance(raw_stages, list):
            raise ValueError("Funny English manifest stages must be a list")
        stages: list[FunnyEnglishStage] = []
        for raw_stage in raw_stages:
            stage_entry = self._require_object(raw_stage, "manifest stage")
            stage_path = self._resolve_path(Path(self._require_str(stage_entry, "stage_path")))
            stages.append(self._load_stage(stage_path))
        return tuple(stages)

    def _load_stage(self, stage_path: Path) -> FunnyEnglishStage:
        with stage_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != 1:
            raise ValueError(f"{stage_path} schema_version must be 1")
        raw_cards = payload.get("cards")
        if not isinstance(raw_cards, list):
            raise ValueError(f"{stage_path} cards must be a list")
        return FunnyEnglishStage(
            stage=self._require_int(payload, "stage"),
            title=self._require_str(payload, "title"),
            cards=tuple(self._parse_card(raw) for raw in raw_cards),
        )

    def _parse_card(self, raw: Any) -> FunnyEnglishCard:
        card = self._require_object(raw, "stage card")
        image = card.get("image")
        model_audio = card.get("model_audio")
        return FunnyEnglishCard(
            card_id=self._require_str(card, "card_id"),
            stage=self._require_int(card, "stage"),
            card_type=self._require_str(card, "type"),
            text=self._require_str(card, "text"),
            tokens=tuple(self._require_str_list(card, "tokens")),
            ko_gloss=str(card.get("ko_gloss") or ""),
            syllables=tuple(self._optional_str_list(card, "syllables")),
            first_sound=str(card.get("first_sound") or ""),
            image_path=self._resolve_optional_path(image),
            model_audio_path=self._resolve_optional_path(model_audio),
            accept_threshold=float(card.get("accept_threshold", 0.6)),
            hotwords=tuple(self._optional_str_list(card, "hotwords")),
            accept_aliases=tuple(self._optional_str_list(card, "accept_aliases")),
        )

    def _resolve_optional_path(self, value: Any) -> Path | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            return None
        return self._resolve_path(Path(value))

    def _resolve_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return self._repo_root / path

    def _restore_residency_after_entry_failure(self) -> None:
        self._active = False
        with suppress(Exception):
            self._mm.unload_tts(force=False)
        with suppress(Exception):
            self._mm.reset_preload_state()
        with suppress(Exception):
            self._mm.preload_stt()

    @staticmethod
    def _require_object(value: Any, label: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be an object")
        return value

    @staticmethod
    def _require_str(payload: dict[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Funny English payload field {key!r} must be non-empty text")
        return value

    @staticmethod
    def _require_int(payload: dict[str, Any], key: str) -> int:
        value = payload.get(key)
        if not isinstance(value, int):
            raise ValueError(f"Funny English payload field {key!r} must be int")
        return value

    @staticmethod
    def _require_str_list(payload: dict[str, Any], key: str) -> list[str]:
        value = payload.get(key)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"Funny English payload field {key!r} must be list[str]")
        if not value:
            raise ValueError(f"Funny English payload field {key!r} must not be empty")
        return list(value)

    @staticmethod
    def _optional_str_list(payload: dict[str, Any], key: str) -> list[str]:
        value = payload.get(key, [])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"Funny English payload field {key!r} must be list[str]")
        return list(value)


def _max_attempts() -> int:
    raw = os.getenv("MUNGI_FE_MAX_ATTEMPTS", "").strip()
    if not raw:
        return DEFAULT_MAX_ATTEMPTS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_ATTEMPTS
    return max(1, value)


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load a mono WAV file with the standard library."""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise OSError(f"unsupported WAV sample width: {sample_width}")
    samples = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return samples, sample_rate


__all__ = [
    "BACK_LABEL",
    "DONE_LABEL",
    "FE_CAPTURED_CUE_SUBLABEL",
    "FE_CLOSE_RETRY_SPEECH",
    "FE_DONE_SPEECH",
    "FE_LISTENING_CUE_SUBLABEL",
    "FE_KOREAN_SPEECH_LINES",
    "FE_LAST_ATTEMPT_ADVANCE_SPEECH",
    "FE_OTHER_RETRY_SPEECH",
    "FE_PASS_FEEDBACK_SPEECH",
    "FE_RENDER_ACK_TIMEOUT_S",
    "FE_REPEAT_CUE_SPEECH",
    "FE_REPEAT_CUE_SUBLABEL",
    "FE_SILENT_JUNK_RETRY_SPEECH",
    "FUNNY_ENGLISH_TITLE",
    "FunnyEnglishCard",
    "FunnyEnglishModeController",
    "FunnyEnglishStage",
    "MISSING_CONTENT_LABEL",
    "funny_english_score",
    "funny_english_score_line",
]
