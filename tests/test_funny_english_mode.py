"""Tests for the Funny English runtime controller."""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, call

import numpy as np

from core.character_expression import CharacterExpression
from core.funny_english_match import (
    FunnyEnglishMatchResult,
    match_funny_english_attempt,
    normalize_hotword_csv,
)
from core.funny_english_mode import (
    _FE_MAX_GAIN,
    _FE_PEAK_CEILING,
    _FE_TARGET_RMS,
    FE_CAPTURED_CUE_SUBLABEL,
    FE_INTRO_LINES,
    FE_INTRO_SPEECH,
    FE_INTRO_SUBLABEL,
    FE_INTRO_TITLE,
    FE_LISTENING_CUE_SUBLABEL,
    FE_RENDER_ACK_TIMEOUT_S,
    FE_REPEAT_CUE_SPEECH,
    FE_REPEAT_CUE_SUBLABEL,
    FUNNY_ENGLISH_TITLE,
    FunnyEnglishCard,
    FunnyEnglishModeController,
    funny_english_score_line,
)
from core.model_manager import ModelType
from core.pipeline import Utterance
from core.session_manager import FunnyEnglishListenInterrupt, SessionState
from hardware.touch_input import TouchEvent


class FakeSession:
    """Session fake for Funny English controller tests."""

    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.states: list[SessionState] = []
        self.played_audio: list[tuple[Any, int]] = []
        self.next_utterance: Utterance | FunnyEnglishListenInterrupt | None = Utterance(
            audio=np.zeros(160, dtype=np.float32),
            sample_rate=16_000,
        )

    def transition_funny_english_state(self, state: SessionState) -> None:
        """Record state transition."""
        self.states.append(state)
        self.log.append(f"state:{state.value}")

    def set_expression(self, expression: Any) -> None:
        """Record expression."""
        self.log.append(f"expression:{expression.value}")

    def begin_guarded_playback(self) -> None:
        """Record guarded playback start."""
        self.log.append("begin_playback")

    def play_guarded(self, audio_samples: Any, sample_rate: int) -> None:
        """Record guarded playback."""
        self.played_audio.append((audio_samples, sample_rate))
        self.log.append("play_guarded")

    def end_guarded_playback(self) -> None:
        """Record guarded playback end."""
        self.log.append("end_playback")

    def capture_funny_english_audio(self) -> Utterance | FunnyEnglishListenInterrupt | None:
        """Record capture resume and return one utterance."""
        self.log.append("capture_resume")
        return self.next_utterance

    def exit_funny_english_mode(self) -> None:
        """Record session-level Funny English exit."""
        self.log.append("exit_funny_english_mode")

    def run_funny_english_attempt(
        self,
        utterance: Utterance,
        card: Any,
        *,
        stt_hotwords_csv: str | None = None,
    ) -> Any:
        """Return a passing match result."""
        del utterance
        self.log.append(f"run_fe_attempt:{stt_hotwords_csv}")
        return match_funny_english_attempt(
            card.text,
            card.tokens,
            accept_aliases=getattr(card, "accept_aliases", ()),
        )


class FakeModelManager:
    """Model-manager fake for Funny English controller tests."""

    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.tts = MagicMock()
        self.tts.synthesize.return_value = (np.zeros(10, dtype=np.float32), 16_000)

    def cancel_preload_and_join(self) -> None:
        """Record preload cancellation."""
        self.log.append("cancel_preload_and_join")

    def reset_preload_state(self) -> None:
        """Record preload reset."""
        self.log.append("reset_preload_state")

    def preload_stt(self, *, stt_hotwords_csv: str | None = None) -> None:
        """Record preload restore."""
        if stt_hotwords_csv is None:
            self.log.append("preload_stt")
            return
        self.log.append(f"preload_stt:{stt_hotwords_csv}")

    def unload_stt(self, *, force: bool = False) -> None:
        """Record STT unload."""
        self.log.append(f"unload_stt:{force}")

    def load(self, model_type: ModelType, *, stt_hotwords_csv: str | None = None) -> None:
        """Record model load."""
        if stt_hotwords_csv is None:
            self.log.append(f"load:{model_type.value}")
            return
        self.log.append(f"load:{model_type.value}:{stt_hotwords_csv}")

    def unload_tts(self, *, force: bool = False) -> None:
        """Record TTS unload."""
        self.log.append(f"unload_tts:{force}")


class FakeRenderer:
    """Renderer fake for Funny English controller tests."""

    def __init__(self, log: list[str] | None = None) -> None:
        self.log = log
        self.card_calls: list[dict[str, Any]] = []
        self.text_calls: list[dict[str, Any]] = []
        self.wait_calls: list[tuple[str | None, float]] = []

    def show_card(self, **kwargs: Any) -> str:
        """Record card render."""
        self.card_calls.append(kwargs)
        token = f"card-{len(self.card_calls)}"
        if self.log is not None:
            self.log.append(f"show_card:{kwargs.get('sublabel')}")
        return token

    def wait_until_rendered(self, token: str | None, timeout: float) -> bool:
        """Record render-token waits as immediately complete."""
        self.wait_calls.append((token, timeout))
        if self.log is not None:
            self.log.append(f"wait:{token}")
        return True

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
        """Record text render."""
        self.text_calls.append(
            {
                "lines": lines,
                "size": size,
                "highlight_index": highlight_index,
                "title": title,
                "layout": layout,
                "show_exit_button": show_exit_button,
                "has_back": has_back,
                "page_index": page_index,
                "page_count": page_count,
            }
        )

    def clear_image(self) -> None:
        """Ignore clear image."""

    def clear_text(self) -> None:
        """Ignore clear text."""


class FakeBgm:
    """BGM fake recording duck/start/stop ordering."""

    def __init__(self, log: list[str]) -> None:
        self.log = log

    def start_loop(self, audio_samples: Any, sample_rate: int, *, volume: float = 0.16) -> None:
        """Record BGM start."""
        del audio_samples, sample_rate, volume
        self.log.append("bgm_start")

    def duck(self, *, volume: float = 0.0) -> None:
        """Record BGM duck."""
        del volume
        self.log.append("bgm_duck")

    def unduck(self) -> None:
        """Record BGM unduck."""
        self.log.append("bgm_unduck")

    def stop(self) -> None:
        """Record BGM stop."""
        self.log.append("bgm_stop")


def _write_manifest(
    tmp_path: Path,
    *,
    stage_count: int = 1,
    cards_per_stage: int = 1,
) -> Path:
    root = tmp_path / "assets" / "funny_english"
    stage_dir = root / "stages"
    stage_dir.mkdir(parents=True, exist_ok=True)
    stages: list[dict[str, Any]] = []
    words = ("cat", "dog", "sun", "pig")
    for index in range(stage_count):
        stage_path = stage_dir / f"stage_{index}.json"
        cards: list[dict[str, Any]] = []
        for card_index in range(cards_per_stage):
            word = words[card_index % len(words)]
            cards.append(
                {
                    "schema_version": 1,
                    "card_id": (
                        f"card-cat-{index}"
                        if cards_per_stage == 1
                        else f"card-{index}-{card_index}"
                    ),
                    "stage": index,
                    "type": "word",
                    "text": word,
                    "tokens": [word],
                    "ko_gloss": "고양이",
                    "syllables": list(word),
                    "first_sound": word[0],
                    "image": None,
                    "model_audio": None,
                    "accept_threshold": 0.6,
                    "hotwords": [word],
                }
            )
        stage_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "stage": index,
                    "title": f"Stage {index}",
                    "cards": cards,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        stages.append(
            {
                "stage": index,
                "title": f"Stage {index}",
                "stage_path": stage_path.relative_to(tmp_path).as_posix(),
                "card_count": 1,
            }
        )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "title": "Funny English",
                "stages": stages,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return manifest_path


def _make_controller(tmp_path: Path, log: list[str]) -> FunnyEnglishModeController:
    manifest_path = _write_manifest(tmp_path)
    renderer = FakeRenderer()
    return _make_controller_with_renderer(tmp_path, log, manifest_path, renderer)


def _make_controller_with_renderer(
    tmp_path: Path,
    log: list[str],
    manifest_path: Path,
    renderer: FakeRenderer,
) -> FunnyEnglishModeController:
    return FunnyEnglishModeController(
        session=FakeSession(log),
        model_manager=FakeModelManager(log),
        renderer=renderer,
        manifest_path=manifest_path.relative_to(tmp_path),
        repo_root=tmp_path,
        bgm_player=FakeBgm(log),
        bgm_clip=(np.zeros(10, dtype=np.float32), 16_000),
    )


def _stt_preload_csvs(log: list[str]) -> list[str]:
    prefix = "preload_stt:"
    return [entry.removeprefix(prefix) for entry in log if entry.startswith(prefix)]


def _stt_load_csvs(log: list[str]) -> list[str]:
    prefix = f"load:{ModelType.STT.value}:"
    return [entry.removeprefix(prefix) for entry in log if entry.startswith(prefix)]


def _run_attempt_csvs(log: list[str]) -> list[str]:
    prefix = "run_fe_attempt:"
    return [entry.removeprefix(prefix) for entry in log if entry.startswith(prefix)]


def _write_mono_wav(path: Path) -> None:
    """Write a tiny mono 16-bit WAV fixture."""
    samples = np.array([0, 4000, -4000, 0], dtype=np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(samples.tobytes())


def _write_mono_wav_values(path: Path, values: list[int]) -> None:
    """Write a mono 16-bit WAV fixture with caller-selected samples."""
    samples = np.array(values, dtype=np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(samples.tobytes())


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


def test_card_loader_accepts_aliases_and_uses_them_as_hotword_fallback(tmp_path: Path) -> None:
    """Stage cards load optional accept aliases while older hotword-less cards still work."""
    root = tmp_path / "assets" / "funny_english"
    stage_dir = root / "stages"
    stage_dir.mkdir(parents=True, exist_ok=True)
    stage_path = stage_dir / "stage_0.json"
    stage_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": 0,
                "title": "Alphabet sounds",
                "cards": [
                    {
                        "schema_version": 1,
                        "card_id": "letter-b",
                        "stage": 0,
                        "type": "word",
                        "text": "B",
                        "tokens": ["b"],
                        "ko_gloss": "공의 첫 소리",
                        "syllables": ["B"],
                        "first_sound": "b",
                        "image": None,
                        "model_audio": None,
                        "accept_threshold": 0.6,
                        "accept_aliases": ["b", "bee", "비"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "title": "Funny English",
                "stages": [
                    {
                        "stage": 0,
                        "title": "Alphabet sounds",
                        "stage_path": stage_path.relative_to(tmp_path).as_posix(),
                        "card_count": 1,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    controller = _make_controller_with_renderer(tmp_path, [], manifest_path, FakeRenderer())
    card = controller._stages[0].cards[0]

    assert card.accept_aliases == ("b", "bee", "비")
    assert card.hotwords == ()
    assert card.hotwords_csv == "b,bee,비,뭉이,뭉이야"


def test_score_line_maps_scores_to_stars_and_omits_silent_results() -> None:
    """Pronunciation score lines use stars plus a number for captured attempts only."""
    high = match_funny_english_attempt("cat", ("cat",))
    medium_pass = FunnyEnglishMatchResult(
        transcript="I see",
        normalized_transcript_tokens=("i", "see"),
        target_tokens=("i", "see", "a", "cat"),
        matched_tokens=("i", "see"),
        missed_tokens=("a", "cat"),
        matched_pct=0.62,
        similarity=0.61,
        band="pass",
    )
    silent = match_funny_english_attempt("", ("cat",))

    assert funny_english_score_line(high) == "★★★★★ (100)"
    assert funny_english_score_line(medium_pass) == "★★★☆☆ (62)"
    assert funny_english_score_line(silent) is None
    assert funny_english_score_line(None) is None


def test_feedback_render_includes_score_line_and_omits_silent_score(tmp_path: Path) -> None:
    """Feedback cards show attempt score lines but avoid misleading no-speech scores."""
    log: list[str] = []
    renderer = FakeRenderer()
    controller = _make_controller_with_renderer(tmp_path, log, _write_manifest(tmp_path), renderer)
    card = controller._stages[0].cards[0]
    high = match_funny_english_attempt("cat", ("cat",))

    controller._handle_score(card, high)

    assert renderer.card_calls[-1]["lines"] == ["★★★★★ (100)", "Great reading!"]

    silent = match_funny_english_attempt("", ("cat",))
    controller._handle_score(card, silent)

    assert renderer.card_calls[-1]["lines"] == ["Listen again, then try."]


def test_entry_cancels_preload_and_force_unloads_before_tts(tmp_path: Path) -> None:
    """FE entry clears any unprompted STT preload before mode activation."""
    log: list[str] = []
    controller = _make_controller(tmp_path, log)

    controller.enter()

    assert log[:3] == ["cancel_preload_and_join", "unload_stt:True", "load:tts"]


def test_enter_renders_intro_and_speaks_korean_instruction(tmp_path: Path) -> None:
    """FE entry starts with a spoken repeat-after-me instruction screen."""
    log: list[str] = []
    renderer = FakeRenderer()
    manifest_path = _write_manifest(tmp_path)
    controller = _make_controller_with_renderer(tmp_path, log, manifest_path, renderer)

    controller.enter()

    assert controller._session.states == [SessionState.FE_INTRO]
    assert renderer.text_calls[-1] == {
        "lines": [*FE_INTRO_LINES, FE_INTRO_SUBLABEL],
        "size": 40,
        "highlight_index": None,
        "title": FE_INTRO_TITLE,
        "layout": "center",
        "show_exit_button": True,
        "has_back": False,
        "page_index": 0,
        "page_count": 1,
    }
    controller._mm.tts.synthesize.assert_called_once_with(FE_INTRO_SPEECH, language="ko")
    assert log[-3:] == ["begin_playback", "play_guarded", "end_playback"]


def test_intro_tap_advances_to_stage_select(tmp_path: Path) -> None:
    """A tap on the intro screen reveals the stage-selection menu."""
    log: list[str] = []
    manifest_path = _write_manifest(tmp_path, stage_count=2)
    renderer = FakeRenderer()
    controller = _make_controller_with_renderer(tmp_path, log, manifest_path, renderer)
    controller.enter()

    controller.handle_intro_tap()

    assert controller._session.states[-1] is SessionState.FE_SELECT
    assert renderer.text_calls[-1]["lines"] == ["Stage 0", "Stage 1"]
    assert renderer.text_calls[-1]["layout"] == "menu"


def test_stage_menu_paging_over_six_stages(tmp_path: Path) -> None:
    """The FE stage menu pages explicitly over four visible rows."""
    log: list[str] = []
    manifest_path = _write_manifest(tmp_path, stage_count=6)
    renderer = FakeRenderer()
    controller = _make_controller_with_renderer(tmp_path, log, manifest_path, renderer)

    controller.enter()
    controller.handle_intro_tap()

    assert renderer.text_calls[-1] == {
        "lines": ["Stage 0", "Stage 1", "Stage 2", "Stage 3"],
        "size": 34,
        "highlight_index": 0,
        "title": "Funny English",
        "layout": "menu",
        "show_exit_button": True,
        "has_back": False,
        "page_index": 0,
        "page_count": 2,
    }

    controller.handle_next_page()

    assert renderer.text_calls[-1] == {
        "lines": ["Stage 4", "Stage 5"],
        "size": 34,
        "highlight_index": 0,
        "title": "Funny English",
        "layout": "menu",
        "show_exit_button": True,
        "has_back": False,
        "page_index": 1,
        "page_count": 2,
    }

    controller.handle_prev_page()

    assert renderer.text_calls[-1]["lines"] == ["Stage 0", "Stage 1", "Stage 2", "Stage 3"]
    assert renderer.text_calls[-1]["page_index"] == 0


def test_direct_touch_stage_select_starts_prompt_from_paged_index(tmp_path: Path) -> None:
    """Direct menu targets select page_start + local index without double-tap."""
    log: list[str] = []
    manifest_path = _write_manifest(tmp_path, stage_count=6)
    renderer = FakeRenderer()
    controller = _make_controller_with_renderer(tmp_path, log, manifest_path, renderer)
    controller.enter()
    controller.handle_intro_tap()
    controller.handle_next_page()

    controller.handle_select_target(1)

    assert controller._stage_index == 5
    assert controller._card_index == 0
    assert "state:fe_prompt" in log
    assert renderer.card_calls[-1]["lines"] == ["cat"]
    assert renderer.card_calls[-1]["show_back"] is True
    assert renderer.card_calls[-1]["show_exit_button"] is True


def test_prompt_first_attempt_renders_and_speaks_repeat_cue(tmp_path: Path) -> None:
    """First card prompt ends with a cue and hotword-aware STT preload."""
    log: list[str] = []
    renderer = FakeRenderer()
    manifest_path = _write_manifest(tmp_path)
    controller = _make_controller_with_renderer(tmp_path, log, manifest_path, renderer)
    controller._active = True

    controller._start_prompt()

    assert renderer.card_calls[-2]["sublabel"] == "고양이"
    cue_card = renderer.card_calls[-1]
    assert cue_card["lines"] == ["cat"]
    assert cue_card["sublabel"] == FE_REPEAT_CUE_SUBLABEL
    assert cue_card["show_back"] is True
    assert cue_card["show_exit_button"] is True
    assert cue_card["show_prev_card"] is False
    assert cue_card["show_next_card"] is False
    assert f"expression:{CharacterExpression.LISTENING.value}" in log
    assert log.index(f"expression:{CharacterExpression.LISTENING.value}") < log.index(
        f"preload_stt:{normalize_hotword_csv(('cat',))}"
    )
    assert controller._mm.tts.synthesize.call_args_list == [
        call("cat", language="en"),
        call(FE_REPEAT_CUE_SPEECH, language="ko"),
    ]


def test_prompt_retry_renders_repeat_cue_without_speaking_it(tmp_path: Path) -> None:
    """Retry prompts keep the visual cue but do not repeat the Korean cue speech."""
    log: list[str] = []
    renderer = FakeRenderer()
    manifest_path = _write_manifest(tmp_path)
    controller = _make_controller_with_renderer(tmp_path, log, manifest_path, renderer)
    controller._active = True
    controller._attempts_by_card["card-cat-0"] = 1

    controller._start_prompt()

    assert renderer.card_calls[-1]["sublabel"] == FE_REPEAT_CUE_SUBLABEL
    assert f"expression:{CharacterExpression.LISTENING.value}" in log
    assert f"preload_stt:{normalize_hotword_csv(('cat',))}" in log
    assert controller._mm.tts.synthesize.call_args_list == [call("cat", language="en")]


def test_retry_attempt_replays_word_wav_instead_of_letter_tts(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Retry modeling uses the whole-word WAV, never syllable or first-sound TTS."""
    log: list[str] = []
    controller = _make_controller(tmp_path, log)
    wav_path = tmp_path / "dog.wav"
    _write_mono_wav(wav_path)
    monkeypatch.setattr(
        "core.funny_english_mode.tts_cache.lookup",
        lambda text, lang: None,
    )
    card = FunnyEnglishCard(
        card_id="dog",
        stage=0,
        card_type="word",
        text="dog",
        tokens=("dog",),
        ko_gloss="강아지",
        syllables=("d", "o", "g"),
        first_sound="d",
        image_path=None,
        model_audio_path=wav_path,
        accept_threshold=0.6,
        hotwords=("dog",),
    )

    controller._play_model_for_attempt(card, attempt_index=2)

    assert controller._session.played_audio
    controller._mm.tts.synthesize.assert_not_called()


def test_retry_attempt_tts_fallback_uses_whole_word(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Missing model WAV falls back to whole-word TTS instead of spelling letters."""
    log: list[str] = []
    controller = _make_controller(tmp_path, log)
    monkeypatch.setattr(
        "core.funny_english_mode.tts_cache.lookup",
        lambda text, lang: None,
    )
    card = FunnyEnglishCard(
        card_id="dog",
        stage=0,
        card_type="word",
        text="dog",
        tokens=("dog",),
        ko_gloss="강아지",
        syllables=("d", "o", "g"),
        first_sound="d",
        image_path=None,
        model_audio_path=tmp_path / "missing.wav",
        accept_threshold=0.6,
        hotwords=("dog",),
    )

    controller._play_model_for_attempt(card, attempt_index=2)

    controller._mm.tts.synthesize.assert_called_once_with("dog", language="en")


def test_speak_feedback_uses_valid_cache_hit_without_loading_tts(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Validated feedback cache hits play directly and avoid live TTS loading."""
    log: list[str] = []
    controller = _make_controller(tmp_path, log)
    wav_path = tmp_path / "feedback-cache.wav"
    _write_mono_wav(wav_path)
    monkeypatch.setattr(
        "core.funny_english_mode.tts_cache.lookup",
        lambda text, lang: wav_path,
    )

    controller._speak_feedback("Great!", language="en")

    controller._mm.tts.synthesize.assert_not_called()
    assert "load:tts" not in log
    assert controller._session.played_audio[-1][1] == 16_000


def test_speak_feedback_live_synths_on_cache_miss(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Feedback cache misses preserve the existing live synthesis path."""
    log: list[str] = []
    controller = _make_controller(tmp_path, log)
    monkeypatch.setattr(
        "core.funny_english_mode.tts_cache.lookup",
        lambda text, lang: None,
    )

    controller._speak_feedback("Great!", language="en")

    assert "load:tts" in log
    controller._mm.tts.synthesize.assert_called_once_with("Great!", language="en")
    assert controller._session.played_audio[-1][1] == 16_000


def test_speak_feedback_live_synths_on_bad_cache_hit(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A bad feedback cache path falls back to live synthesis."""
    log: list[str] = []
    controller = _make_controller(tmp_path, log)
    bad_wav_path = tmp_path / "bad-feedback-cache.wav"
    bad_wav_path.write_bytes(b"not a wav")
    monkeypatch.setattr(
        "core.funny_english_mode.tts_cache.lookup",
        lambda text, lang: bad_wav_path,
    )

    controller._speak_feedback("Great!", language="en")

    assert "load:tts" in log
    controller._mm.tts.synthesize.assert_called_once_with("Great!", language="en")


def test_play_model_for_attempt_prefers_valid_cache_over_committed_wav(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """English modeling uses the validated cache before the committed WAV fallback."""
    log: list[str] = []
    controller = _make_controller(tmp_path, log)
    cache_wav = tmp_path / "cache-dog.wav"
    model_wav = tmp_path / "model-dog.wav"
    _write_mono_wav_values(cache_wav, [0, 100, 0])
    _write_mono_wav_values(model_wav, [0, 100, 200, 100, 0])
    monkeypatch.setattr(
        "core.funny_english_mode.tts_cache.lookup",
        lambda text, lang: cache_wav,
    )
    card = FunnyEnglishCard(
        card_id="dog",
        stage=0,
        card_type="word",
        text="dog",
        tokens=("dog",),
        ko_gloss="강아지",
        syllables=("d", "o", "g"),
        first_sound="d",
        image_path=None,
        model_audio_path=model_wav,
        accept_threshold=0.6,
        hotwords=("dog",),
    )

    controller._play_model_for_attempt(card, attempt_index=0)

    played_samples = controller._session.played_audio[-1][0]
    assert len(played_samples) == 3
    controller._mm.tts.synthesize.assert_not_called()


def test_play_model_for_attempt_bad_cache_falls_back_to_live_when_model_missing(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Bad cache plus missing committed WAV falls through to English live TTS."""
    log: list[str] = []
    controller = _make_controller(tmp_path, log)
    bad_wav_path = tmp_path / "bad-model-cache.wav"
    bad_wav_path.write_bytes(b"not a wav")
    monkeypatch.setattr(
        "core.funny_english_mode.tts_cache.lookup",
        lambda text, lang: bad_wav_path,
    )
    card = FunnyEnglishCard(
        card_id="dog",
        stage=0,
        card_type="word",
        text="dog",
        tokens=("dog",),
        ko_gloss="강아지",
        syllables=("d", "o", "g"),
        first_sound="d",
        image_path=None,
        model_audio_path=tmp_path / "missing.wav",
        accept_threshold=0.6,
        hotwords=("dog",),
    )

    controller._play_model_for_attempt(card, attempt_index=0)

    controller._mm.tts.synthesize.assert_called_once_with("dog", language="en")


def test_stage_select_taps_no_longer_double_tap_select(tmp_path: Path) -> None:
    """Coordinate-less FE select taps only advance highlight as a fallback."""
    log: list[str] = []
    manifest_path = _write_manifest(tmp_path, stage_count=6)
    renderer = FakeRenderer()
    controller = _make_controller_with_renderer(tmp_path, log, manifest_path, renderer)
    controller.enter()
    controller.handle_intro_tap()

    controller.handle_select_tap(TouchEvent(type="tap", press_duration_ms=100, timestamp=1.0))
    controller.handle_select_tap(TouchEvent(type="tap", press_duration_ms=100, timestamp=1.2))

    assert controller._stage_index == 0
    assert controller._highlight == 2
    assert "state:fe_prompt" not in log


def test_bgm_ducks_before_capture_resume(tmp_path: Path) -> None:
    """BGM ducking happens before capture resumes for FE_LISTEN."""
    log: list[str] = []
    controller = _make_controller(tmp_path, log)
    controller._active = True
    expected_csv = normalize_hotword_csv(("cat",))

    controller.handle_prompt_tap(TouchEvent(type="tap", press_duration_ms=100, timestamp=1.0))

    assert "state:fe_listen" in log
    assert log.index("bgm_duck") < log.index("capture_resume")
    assert _stt_load_csvs(log) == [expected_csv]
    assert _run_attempt_csvs(log) == [expected_csv]


def test_listen_attempt_shows_listening_and_captured_cues(tmp_path: Path) -> None:
    """A captured FE attempt shows listening and heard-you cues before scoring."""
    log: list[str] = []
    renderer = FakeRenderer(log)
    manifest_path = _write_manifest(tmp_path)
    controller = _make_controller_with_renderer(tmp_path, log, manifest_path, renderer)
    controller._active = True
    attempt_entry = f"run_fe_attempt:{normalize_hotword_csv(('cat',))}"

    controller.handle_prompt_tap(TouchEvent(type="tap", press_duration_ms=100, timestamp=1.0))

    assert f"show_card:{FE_LISTENING_CUE_SUBLABEL}" in log
    assert f"show_card:{FE_CAPTURED_CUE_SUBLABEL}" in log
    assert log.index(f"show_card:{FE_LISTENING_CUE_SUBLABEL}") < log.index("capture_resume")
    assert log.index(f"show_card:{FE_CAPTURED_CUE_SUBLABEL}") < log.index(attempt_entry)
    assert (f"card-{len(renderer.card_calls) - 2}", FE_RENDER_ACK_TIMEOUT_S) in renderer.wait_calls
    assert (f"card-{len(renderer.card_calls) - 1}", FE_RENDER_ACK_TIMEOUT_S) in renderer.wait_calls


def test_listen_attempt_without_capture_skips_captured_cue_and_scoring(tmp_path: Path) -> None:
    """A silent FE attempt keeps the retry path without showing heard-you feedback."""
    log: list[str] = []
    renderer = FakeRenderer(log)
    manifest_path = _write_manifest(tmp_path)
    controller = _make_controller_with_renderer(tmp_path, log, manifest_path, renderer)
    controller._active = True
    cast(FakeSession, controller._session).next_utterance = None

    controller.handle_prompt_tap(TouchEvent(type="tap", press_duration_ms=100, timestamp=1.0))

    assert f"show_card:{FE_LISTENING_CUE_SUBLABEL}" in log
    assert f"show_card:{FE_CAPTURED_CUE_SUBLABEL}" not in log
    assert _run_attempt_csvs(log) == []
    assert renderer.card_calls[-1]["sublabel"] == "Tap to try again"


def test_listen_attempt_exit_interrupt_exits_without_scoring(tmp_path: Path) -> None:
    """An exit interrupt during FE capture exits before FE_SCORE/scoring."""
    log: list[str] = []
    renderer = FakeRenderer(log)
    manifest_path = _write_manifest(tmp_path)
    controller = _make_controller_with_renderer(tmp_path, log, manifest_path, renderer)
    controller._active = True
    cast(FakeSession, controller._session).next_utterance = FunnyEnglishListenInterrupt("exit")

    controller.handle_prompt_tap(TouchEvent(type="tap", press_duration_ms=100, timestamp=1.0))

    assert "exit_funny_english_mode" in log
    assert "bgm_unduck" in log
    assert log.index("bgm_duck") < log.index("bgm_unduck") < log.index("exit_funny_english_mode")
    assert "state:fe_score" not in log
    assert f"show_card:{FE_CAPTURED_CUE_SUBLABEL}" not in log
    assert _run_attempt_csvs(log) == []


def test_listen_attempt_back_interrupt_returns_to_select_without_scoring(tmp_path: Path) -> None:
    """A back interrupt during FE capture returns to selection before scoring."""
    log: list[str] = []
    renderer = FakeRenderer(log)
    manifest_path = _write_manifest(tmp_path)
    controller = _make_controller_with_renderer(tmp_path, log, manifest_path, renderer)
    controller._active = True
    cast(FakeSession, controller._session).next_utterance = FunnyEnglishListenInterrupt("back")

    controller.handle_prompt_tap(TouchEvent(type="tap", press_duration_ms=100, timestamp=1.0))

    assert "state:fe_select" in log
    assert "bgm_unduck" in log
    assert log.index("bgm_duck") < log.index("bgm_unduck") < log.index("state:fe_select")
    assert "state:fe_score" not in log
    assert f"show_card:{FE_CAPTURED_CUE_SUBLABEL}" not in log
    assert _run_attempt_csvs(log) == []
    assert renderer.text_calls[-1]["title"] == FUNNY_ENGLISH_TITLE


def test_multi_card_stage_threads_stage_hotwords_for_prompt_and_attempts(
    tmp_path: Path,
) -> None:
    """Every STT load site uses the same stage hotword CSV within a multi-card stage."""
    log: list[str] = []
    renderer = FakeRenderer()
    manifest_path = _write_manifest(tmp_path, cards_per_stage=3)
    controller = _make_controller_with_renderer(tmp_path, log, manifest_path, renderer)
    controller._active = True
    expected_csv = normalize_hotword_csv(("cat", "dog", "sun"))

    for card_index, word in ((0, "cat"), (2, "sun")):
        log.clear()
        controller._card_index = card_index
        card = controller._current_card()
        assert card is not None
        assert card.text == word

        controller._start_prompt()
        controller._run_listen_attempt()

        assert _stt_preload_csvs(log) == [expected_csv]
        assert _stt_load_csvs(log) == [expected_csv]
        assert _run_attempt_csvs(log) == [expected_csv]


def test_multi_card_stage_reentry_keeps_stage_hotword_preload_csv(
    tmp_path: Path,
) -> None:
    """Card re-entry paths keep the stage-level hotword CSV stable."""
    log: list[str] = []
    renderer = FakeRenderer()
    manifest_path = _write_manifest(tmp_path, cards_per_stage=3)
    controller = _make_controller_with_renderer(tmp_path, log, manifest_path, renderer)
    controller._active = True
    expected_csv = normalize_hotword_csv(("cat", "dog", "sun"))

    controller._start_prompt()

    assert controller._card_index == 0
    assert renderer.card_calls[-1]["lines"] == ["cat"]
    assert _stt_preload_csvs(log) == [expected_csv]

    log.clear()
    controller._advance_card()

    assert controller._card_index == 1
    assert renderer.card_calls[-1]["lines"] == ["dog"]
    assert _stt_preload_csvs(log) == [expected_csv]

    log.clear()
    controller._move_card(1)

    assert controller._card_index == 2
    assert renderer.card_calls[-1]["lines"] == ["sun"]
    assert _stt_preload_csvs(log) == [expected_csv]

    log.clear()
    controller._move_card(-1)

    assert controller._card_index == 1
    assert renderer.card_calls[-1]["lines"] == ["dog"]
    assert _stt_preload_csvs(log) == [expected_csv]


def test_prev_next_card_clamp_and_render_nav_targets(tmp_path: Path) -> None:
    """Word navigation moves within stage bounds and exposes matching card targets."""
    log: list[str] = []
    renderer = FakeRenderer()
    manifest_path = _write_manifest(tmp_path, cards_per_stage=3)
    controller = _make_controller_with_renderer(tmp_path, log, manifest_path, renderer)
    controller._active = True

    controller._start_prompt()

    assert controller._card_index == 0
    assert renderer.card_calls[-1]["lines"] == ["cat"]
    assert renderer.card_calls[-1]["show_prev_card"] is False
    assert renderer.card_calls[-1]["show_next_card"] is True

    controller._attempts_by_card["card-0-1"] = 2
    controller.handle_next_card()

    assert controller._card_index == 1
    assert controller._attempts_by_card["card-0-1"] == 0
    assert renderer.card_calls[-1]["lines"] == ["dog"]
    assert renderer.card_calls[-1]["show_prev_card"] is True
    assert renderer.card_calls[-1]["show_next_card"] is True

    controller.handle_next_card()
    render_count = len(renderer.card_calls)
    controller.handle_next_card()

    assert controller._card_index == 2
    assert len(renderer.card_calls) == render_count
    assert renderer.card_calls[-1]["lines"] == ["sun"]
    assert renderer.card_calls[-1]["show_prev_card"] is True
    assert renderer.card_calls[-1]["show_next_card"] is False

    controller.handle_prev_card()

    assert controller._card_index == 1
    assert renderer.card_calls[-1]["lines"] == ["dog"]


def test_feedback_card_renders_back_and_exit_buttons(tmp_path: Path) -> None:
    """Feedback cards expose stage back and mode exit buttons."""
    log: list[str] = []
    renderer = FakeRenderer()
    manifest_path = _write_manifest(tmp_path)
    controller = _make_controller_with_renderer(tmp_path, log, manifest_path, renderer)
    controller._active = True
    card = controller._current_card()
    assert card is not None

    controller._render_feedback(card, "Almost there!", "retry")

    assert renderer.card_calls[-1]["show_back"] is True
    assert renderer.card_calls[-1]["show_exit_button"] is True


def test_done_screen_renders_back_and_exit_buttons(tmp_path: Path) -> None:
    """The done screen has nav controls even though it uses center text layout."""
    log: list[str] = []
    renderer = FakeRenderer()
    manifest_path = _write_manifest(tmp_path)
    controller = _make_controller_with_renderer(tmp_path, log, manifest_path, renderer)
    controller._active = True

    controller._finish_stage()

    assert renderer.text_calls[-1]["show_exit_button"] is True
    assert renderer.text_calls[-1]["has_back"] is True


def test_handle_back_returns_to_stage_select(tmp_path: Path) -> None:
    """Card-level back returns to the stage menu without leaving FE."""
    log: list[str] = []
    renderer = FakeRenderer()
    manifest_path = _write_manifest(tmp_path, stage_count=2)
    controller = _make_controller_with_renderer(tmp_path, log, manifest_path, renderer)
    controller._active = True

    controller.handle_back()

    assert controller._session.states[-1] is SessionState.FE_SELECT
    assert renderer.text_calls[-1]["layout"] == "menu"


def test_play_guarded_boosts_quiet_clip_to_target_rms(tmp_path: Path) -> None:
    """Quiet FE clips are boosted toward the shared target RMS."""
    log: list[str] = []
    controller = _make_controller(tmp_path, log)
    samples = np.full(2048, 0.02, dtype=np.float32)

    controller._play_guarded(samples, 16_000)

    played = controller._session.played_audio[-1][0]
    assert played.dtype == np.float32
    assert np.isclose(_rms(played), _FE_TARGET_RMS, rtol=0.03)


def test_play_guarded_equalizes_different_input_rms_clips(tmp_path: Path) -> None:
    """Different FE input levels land at the same output RMS."""
    log: list[str] = []
    controller = _make_controller(tmp_path, log)
    quiet = np.full(2048, 0.04, dtype=np.float32)
    loud = np.full(2048, 0.20, dtype=np.float32)

    controller._play_guarded(quiet, 16_000)
    controller._play_guarded(loud, 16_000)

    quiet_out = controller._session.played_audio[-2][0]
    loud_out = controller._session.played_audio[-1][0]
    assert np.isclose(_rms(quiet_out), _rms(loud_out), rtol=0.03)
    assert np.isclose(_rms(loud_out), _FE_TARGET_RMS, rtol=0.03)


def test_play_guarded_peak_limits_loud_transient(tmp_path: Path) -> None:
    """Peak limiting prevents boosted transients from clipping."""
    log: list[str] = []
    controller = _make_controller(tmp_path, log)
    samples = np.zeros(2048, dtype=np.float32)
    samples[0] = 1.0

    controller._play_guarded(samples, 16_000)

    played = controller._session.played_audio[-1][0]
    assert float(np.max(np.abs(played))) <= _FE_PEAK_CEILING


def test_play_guarded_leaves_silent_clip_unchanged_without_nan(tmp_path: Path) -> None:
    """Silent FE clips remain unchanged and finite."""
    log: list[str] = []
    controller = _make_controller(tmp_path, log)
    samples = np.zeros(2048, dtype=np.float32)

    controller._play_guarded(samples, 16_000)

    played = controller._session.played_audio[-1][0]
    assert np.array_equal(played, samples)
    assert np.all(np.isfinite(played))


def test_play_guarded_caps_large_gain(tmp_path: Path) -> None:
    """Very quiet non-silent clips respect the maximum FE gain cap."""
    log: list[str] = []
    controller = _make_controller(tmp_path, log)
    samples = np.full(2048, 0.001, dtype=np.float32)

    controller._play_guarded(samples, 16_000)

    played = controller._session.played_audio[-1][0]
    assert np.isclose(_rms(played), _rms(samples) * _FE_MAX_GAIN, rtol=0.03)
    assert _rms(played) < _FE_TARGET_RMS


def test_play_guarded_preserves_int16_dtype_shape_and_avoids_wrap(tmp_path: Path) -> None:
    """Int16 model WAV clips keep shape/dtype and stay inside the safe peak ceiling."""
    log: list[str] = []
    controller = _make_controller(tmp_path, log)
    samples = np.full((2, 512), 1000, dtype=np.int16)

    controller._play_guarded(samples, 16_000)

    played = controller._session.played_audio[-1][0]
    assert played.dtype == np.int16
    assert played.shape == samples.shape
    assert int(np.max(np.abs(played))) <= int(round(_FE_PEAK_CEILING * 32768))
