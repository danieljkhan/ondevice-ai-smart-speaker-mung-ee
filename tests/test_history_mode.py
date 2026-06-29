"""Tests for the Korean-history runtime controller."""

from __future__ import annotations

import inspect
import json
import threading
import wave
from pathlib import Path
from typing import Any

import pytest

from core.character_expression import CharacterExpression
from core.history_mode import (
    CONSENT_PROMPT,
    MISSING_IMAGE_LABEL,
    HistoryDocument,
    HistoryImage,
    HistoryModeController,
    HistoryScene,
    HistorySection,
)
from core.model_manager import ModelType
from core.session_manager import SessionState
from hardware.touch_input import TouchEvent


class FakeSession:
    """Session fake recording history-mode calls."""

    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events if events is not None else []
        self.states: list[SessionState] = []
        self.expressions: list[CharacterExpression] = []
        self.ack_calls = 0
        self.interrupt_calls = 0
        self.guard_calls: list[str] = []
        self.played: list[tuple[Any, int]] = []
        self.history_capture_calls: list[tuple[str, bool | None]] = []

    def transition_history_state(self, state: SessionState) -> None:
        """Record a history state transition."""
        self.events.append(f"state:{state.value}")
        self.states.append(state)

    def set_expression(self, expression: CharacterExpression) -> None:
        """Record an expression."""
        self.expressions.append(expression)

    def play_history_ack(self) -> None:
        """Record an acknowledgement cue."""
        self.ack_calls += 1

    def begin_guarded_playback(self) -> None:
        """Record guarded playback begin."""
        self.events.append("guard:begin")
        self.guard_calls.append("begin")

    def play_guarded(self, audio_samples: Any, sample_rate: int) -> None:
        """Record guarded playback."""
        self.events.append("guard:play")
        self.guard_calls.append("play")
        self.played.append((audio_samples, sample_rate))

    def end_guarded_playback(self) -> None:
        """Record guarded playback end."""
        self.events.append("guard:end")
        self.guard_calls.append("end")

    def begin_history_capture_pause(self) -> None:
        """Record history capture pause acquisition."""
        self.events.append("history_capture:begin")
        self.history_capture_calls.append(("begin", None))

    def release_history_capture_pause(self, *, restore_stt: bool) -> None:
        """Record history capture pause release."""
        self.events.append(f"history_capture:release:{restore_stt}")
        self.history_capture_calls.append(("release", restore_stt))

    def interrupt_playback(self) -> None:
        """Record a playback interrupt."""
        self.events.append("interrupt")
        self.interrupt_calls += 1


class FakeRenderer:
    """Renderer fake recording history-mode display calls."""

    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events if events is not None else []
        self.calls: list[tuple[str, Any]] = []
        self.history_menus: list[dict[str, Any]] = []
        self._token_index = 0

    def show_history_image(self, path: str | Path) -> str:
        """Record a history image."""
        self.calls.append(("history_image", Path(path)))
        self._token_index += 1
        token = f"image-{self._token_index}"
        self.events.append(f"show_history_image:{token}")
        return token

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
        """Record a card render and return a token."""
        self._token_index += 1
        token = f"card-{self._token_index}"
        self.calls.append(
            (
                "card",
                {
                    "image_path": Path(image_path) if image_path is not None else None,
                    "lines": lines or [],
                    "highlight_index": highlight_index,
                    "title": title,
                    "sublabel": sublabel,
                    "show_exit_button": show_exit_button,
                    "show_back": show_back,
                    "show_prev_card": show_prev_card,
                    "show_next_card": show_next_card,
                    "token": token,
                },
            )
        )
        self.events.append(f"show_card:{token}")
        return token

    def wait_until_rendered(self, token: str | None, timeout: float) -> bool:
        """Record a token wait."""
        self.calls.append(("wait_until_rendered", (token, timeout)))
        self.events.append(f"wait:{token}")
        return True

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
        """Record a history menu."""
        self.calls.append(("history_menu", (items, highlight, title)))
        self.history_menus.append(
            {
                "items": items,
                "highlight": highlight,
                "title": title,
                "has_back": has_back,
                "page_index": page_index,
                "page_count": page_count,
            }
        )
        return "menu-token"

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
        """Record generic text."""
        self.calls.append(("text", (lines, size, highlight_index, title, layout, show_exit_button)))
        del has_back, page_index, page_count

    def clear_image(self) -> None:
        """Record image clear."""
        self.calls.append(("clear_image", None))

    def clear_text(self) -> None:
        """Record text clear."""
        self.calls.append(("clear_text", None))


class FakeTTS:
    """TTS fake returning stable audio."""

    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events if events is not None else []
        self.sentences: list[str] = []

    def synthesize(self, text: str, language: str = "ko") -> tuple[list[float], int]:
        """Record one synthesized sentence."""
        assert language == "ko"
        self.events.append(f"tts:{text}")
        self.sentences.append(text)
        return [0.1, 0.2], 22_050


class BlockingTTS:
    """TTS fake that blocks inside synthesize until a test releases it."""

    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events if events is not None else []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.sentences: list[str] = []

    def synthesize(self, text: str, language: str = "ko") -> tuple[list[float], int]:
        """Block one synthesis call until released by the test."""
        assert language == "ko"
        self.events.append(f"tts_enter:{text}")
        self.sentences.append(text)
        self.entered.set()
        if not self.release.wait(timeout=1.0):
            raise TimeoutError("blocking TTS was not released")
        self.events.append(f"tts_exit:{text}")
        return [0.3, 0.4], 22_050


class FakeModelManager:
    """Model-manager fake recording residency calls."""

    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events if events is not None else []
        self.tts = FakeTTS(self.events)
        self.calls: list[tuple[str, Any]] = []

    def reset_preload_state(self) -> None:
        """Record preload reset."""
        self.calls.append(("reset_preload_state", None))

    def preload_stt(self) -> None:
        """Record STT preload."""
        self.calls.append(("preload_stt", None))

    def cancel_preload(self) -> None:
        """Record preload cancellation."""
        self.calls.append(("cancel_preload", None))

    def unload_stt(self, *, force: bool = False) -> None:
        """Record STT unload."""
        self.calls.append(("unload_stt", force))

    def unload_llm(self) -> None:
        """Record LLM unload."""
        self.calls.append(("unload_llm", None))

    def load(self, model_type: ModelType) -> None:
        """Record model load."""
        self.calls.append(("load", model_type))

    def unload_tts(self, *, force: bool = False) -> None:
        """Record TTS unload."""
        self.calls.append(("unload_tts", force))


def _tap(timestamp: float) -> TouchEvent:
    return TouchEvent(type="tap", press_duration_ms=100, timestamp=timestamp)


def _write_history_fixture(tmp_path: Path) -> Path:
    doc_path = tmp_path / "assets" / "history" / "docs" / "doc1.json"
    doc_path.parent.mkdir(parents=True)
    image_path = tmp_path / "assets" / "history" / "images" / "doc1" / "fig_001.jpg"
    second_image_path = tmp_path / "assets" / "history" / "images" / "doc1" / "fig_002.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake image")
    second_image_path.write_bytes(b"fake image 2")
    doc_payload = {
        "schema_version": 2,
        "doc_hash": "doc1",
        "source_file": "eh_n0010_0010.pdf",
        "title": "단군왕검, 아사달에 나라를 세우다",
        "kind": "people",
        "era": "고조선",
        "era_source": "keyword",
        "scene_count": 2,
        "section_count": 2,
        "image_count": 2,
        "est_total_ms": 2000,
        "sections": [
            {
                "section_index": 0,
                "section_title": "첫 이야기",
                "scene_indices": [0],
                "scene_seq": [1],
                "image_captions": ["첫 그림", "둘째 그림"],
                "is_infographic": True,
            },
            {
                "section_index": 1,
                "section_title": "둘째 이야기",
                "scene_indices": [1],
                "scene_seq": [2],
                "image_captions": [],
                "is_infographic": False,
            },
        ],
        "scenes": [
            {
                "seq": 1,
                "section_index": 0,
                "section_title": "첫 이야기",
                "narration": "첫 문장입니다. 둘째 문장입니다.",
                "est_speech_ms": 1000,
                "tail_silence_ms": 0,
                "image_captions": ["첫 그림", "둘째 그림"],
                "images": [
                    {
                        "path": "assets/history/images/doc1/fig_001.jpg",
                        "caption": "첫 그림",
                        "letterboxed": True,
                        "clean": True,
                        "is_infographic": True,
                        "anchor_ratio": 0.0,
                    },
                    {
                        "path": "assets/history/images/doc1/fig_002.jpg",
                        "caption": "둘째 그림",
                        "letterboxed": True,
                        "clean": True,
                        "is_infographic": False,
                        "anchor_ratio": 0.5,
                    },
                ],
            },
            {
                "seq": 2,
                "section_index": 1,
                "section_title": "둘째 이야기",
                "narration": "셋째 문장입니다.",
                "est_speech_ms": 1000,
                "tail_silence_ms": 0,
                "image_captions": [],
                "images": [],
            },
        ],
    }
    doc_path.write_text(json.dumps(doc_payload, ensure_ascii=False), encoding="utf-8")
    manifest_path = tmp_path / "assets" / "history" / "manifest.json"
    manifest_payload = {
        "schema_version": 2,
        "title": "재미있는 우리역사",
        "era_order": ["고조선"],
        "docs": [
            {
                "doc_hash": "doc1",
                "source_file": "eh_n0010_0010.pdf",
                "title": "단군왕검, 아사달에 나라를 세우다",
                "kind": "people",
                "era": "고조선",
                "scene_count": 2,
                "section_count": 2,
                "image_count": 2,
                "est_total_ms": 2000,
                "doc_path": "assets/history/docs/doc1.json",
                "title_curated": True,
                "era_source": "keyword",
                "order": 0,
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def _write_history_tts_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00\x01\x00")


def _build_controller(
    tmp_path: Path,
) -> tuple[HistoryModeController, FakeSession, FakeModelManager, FakeRenderer]:
    manifest_path = _write_history_fixture(tmp_path)
    events: list[str] = []
    session = FakeSession(events)
    model_manager = FakeModelManager(events)
    renderer = FakeRenderer(events)
    controller = HistoryModeController(
        session=session,
        model_manager=model_manager,
        renderer=renderer,
        manifest_path=manifest_path,
        repo_root=tmp_path,
        monotonic_clock=lambda: 100.0,
    )
    return controller, session, model_manager, renderer


def _renderable_history_image(
    tmp_path: Path,
    name: str,
    caption: str,
    *,
    anchor_ratio: float | None = 0.0,
) -> HistoryImage:
    image_path = tmp_path / "assets" / "history" / "images" / "doc1" / name
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"fake image")
    return HistoryImage(
        path=image_path,
        caption=caption,
        letterboxed=True,
        clean=True,
        is_infographic=False,
        anchor_ratio=anchor_ratio,
    )


def _history_scene(
    seq: int,
    images: tuple[HistoryImage, ...] = (),
    *,
    narration: str | None = None,
) -> HistoryScene:
    return HistoryScene(
        seq=seq,
        section_index=0,
        section_title=None,
        narration=narration or f"{seq}번 장면입니다.",
        est_speech_ms=1000,
        tail_silence_ms=0,
        image_captions=tuple(image.caption or "" for image in images),
        images=images,
    )


def _single_section_document(scenes: tuple[HistoryScene, ...]) -> HistoryDocument:
    section = HistorySection(
        section_index=0,
        section_title=None,
        scenes=scenes,
        image_captions=tuple(caption for scene in scenes for caption in scene.image_captions),
    )
    return HistoryDocument(
        doc_hash="doc1",
        title="테스트 문서",
        kind="people",
        era="고조선",
        scenes=scenes,
        sections=(section,),
    )


def _two_section_document() -> HistoryDocument:
    first_scene = HistoryScene(
        seq=1,
        section_index=0,
        section_title=None,
        narration="첫째 절 문장입니다.",
        est_speech_ms=1000,
        tail_silence_ms=0,
        image_captions=(),
        images=(),
    )
    second_scene = HistoryScene(
        seq=2,
        section_index=1,
        section_title=None,
        narration="둘째 절 문장입니다.",
        est_speech_ms=1000,
        tail_silence_ms=0,
        image_captions=(),
        images=(),
    )
    scenes = (first_scene, second_scene)
    sections = (
        HistorySection(
            section_index=0,
            section_title=None,
            scenes=(first_scene,),
            image_captions=(),
        ),
        HistorySection(
            section_index=1,
            section_title=None,
            scenes=(second_scene,),
            image_captions=(),
        ),
    )
    return HistoryDocument(
        doc_hash="doc1",
        title="테스트 문서",
        kind="people",
        era="고조선",
        scenes=scenes,
        sections=sections,
    )


def _play_single_section_document(
    controller: HistoryModeController,
    document: HistoryDocument,
) -> None:
    controller._current_doc = document
    controller._section_index = 0
    controller._scene_index = 0
    controller._start_section()
    assert controller._narration_thread is not None
    controller._narration_thread.join(timeout=1.0)
    assert controller._narration_thread.is_alive() is False


def _missing_image_text_calls(renderer: FakeRenderer) -> list[tuple[str, Any]]:
    return [
        call for call in renderer.calls if call[0] == "text" and MISSING_IMAGE_LABEL in call[1][0]
    ]


def _write_ordered_menu_fixture(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "assets" / "history" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_payload = {
        "schema_version": 2,
        "title": "재미있는 우리역사",
        "era_order": ["고조선"],
        "docs": [
            {
                "doc_hash": "later_people",
                "source_file": "eh_n0010_0010.pdf",
                "title": "가나다 뒤 문서",
                "kind": "people",
                "era": "고조선",
                "scene_count": 1,
                "section_count": 1,
                "image_count": 0,
                "est_total_ms": 1000,
                "doc_path": "assets/history/docs/later_people.json",
                "title_curated": True,
                "era_source": "keyword",
                "order": 20,
            },
            {
                "doc_hash": "earlier_artifact",
                "source_file": "eh_r0005_0010.pdf",
                "title": "하하 앞 문서",
                "kind": "artifact",
                "era": "고조선",
                "scene_count": 1,
                "section_count": 1,
                "image_count": 0,
                "est_total_ms": 1000,
                "doc_path": "assets/history/docs/earlier_artifact.json",
                "title_curated": True,
                "era_source": "keyword",
                "order": 10,
            },
        ],
    }
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def _write_paged_menu_fixture(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "assets" / "history" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    docs: list[dict[str, Any]] = []
    for index in range(6):
        docs.append(
            {
                "doc_hash": f"doc{index}",
                "source_file": f"eh_n0010_00{index}.pdf",
                "title": f"문서 {index}",
                "kind": "people",
                "era": "고조선",
                "scene_count": 1,
                "section_count": 1,
                "image_count": 0,
                "est_total_ms": 1000,
                "doc_path": f"assets/history/docs/doc{index}.json",
                "title_curated": True,
                "era_source": "keyword",
                "order": index,
            }
        )
    manifest_payload = {
        "schema_version": 2,
        "title": "재미있는 우리역사",
        "era_order": ["고조선"],
        "docs": docs,
    }
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def _start_blocking_narration(
    controller: HistoryModeController,
    model_manager: FakeModelManager,
) -> tuple[BlockingTTS, threading.Thread]:
    document = controller._load_document(controller._catalog[0])
    section = document.sections[0]
    blocking_tts = BlockingTTS(model_manager.events)
    model_manager.tts = blocking_tts  # type: ignore[assignment]
    thread = threading.Thread(
        target=controller._run_narration_worker,
        args=(document, section, None, False),
        name="TestHistoryNarrationWorker",
        daemon=True,
    )
    controller._narration_thread = thread
    thread.start()
    assert blocking_tts.entered.wait(timeout=1.0)
    return blocking_tts, thread


def test_history_mode_enter_loads_tts_unloads_stt_and_renders_menu(tmp_path: Path) -> None:
    """Entry applies the residency swap and renders the first catalogue menu."""
    controller, session, model_manager, renderer = _build_controller(tmp_path)

    controller.enter()

    assert model_manager.calls[:5] == [
        ("cancel_preload", None),
        ("reset_preload_state", None),
        ("unload_stt", True),
        ("unload_llm", None),
        ("load", ModelType.TTS),
    ]
    assert session.history_capture_calls == [("begin", None)]
    assert session.states == [SessionState.HISTORY_SELECT]
    assert renderer.calls[-1] == ("history_menu", (["고조선"], 0, "재미있는 우리역사"))


def test_history_mode_entry_failure_restores_preload(tmp_path: Path) -> None:
    """Failed entry leaves the controller inactive and restores normal preload policy."""
    controller, _session, model_manager, _renderer = _build_controller(tmp_path)

    def fail_load(model_type: ModelType) -> None:
        model_manager.calls.append(("load", model_type))
        raise RuntimeError("tts unavailable")

    model_manager.load = fail_load  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="tts unavailable"):
        controller.enter()

    assert controller.active is False
    assert model_manager.calls == [
        ("cancel_preload", None),
        ("reset_preload_state", None),
        ("unload_stt", True),
        ("unload_llm", None),
        ("load", ModelType.TTS),
        ("unload_tts", False),
        ("reset_preload_state", None),
        ("preload_stt", None),
    ]


def test_history_mode_select_drills_down_and_narrates_with_guarded_playback(
    tmp_path: Path,
) -> None:
    """Selection reaches a document and narration waits for the section card."""
    controller, session, model_manager, renderer = _build_controller(tmp_path)
    controller.enter()

    controller.handle_select_target(0)
    controller.handle_select_target(0)
    assert controller._narration_thread is not None
    controller._narration_thread.join(timeout=1.0)

    assert SessionState.HISTORY_SCENE in session.states
    assert session.guard_calls == [
        "begin",
        "play",
        "end",
        "begin",
        "play",
        "end",
        "begin",
        "play",
        "end",
    ]
    assert model_manager.tts.sentences == [
        "지금부터 '단군왕검, 아사달에 나라를 세우다' 이야기를 들려줄게.",
        "첫 문장입니다.",
        "둘째 문장입니다.",
    ]
    card_calls = [payload for kind, payload in renderer.calls if kind == "card"]
    assert card_calls[0]["lines"] == ["첫 그림"]
    assert card_calls[0]["title"] is None
    assert card_calls[0]["sublabel"] is None
    assert card_calls[1]["lines"] == ["둘째 그림"]
    assert "wait:card-1" in session.events
    assert session.events.index("wait:card-1") < session.events.index(
        "tts:지금부터 '단군왕검, 아사달에 나라를 세우다' 이야기를 들려줄게."
    )
    assert session.events.index("show_card:card-2") < session.events.index("tts:둘째 문장입니다.")
    assert all(call[0] != "history_image" for call in renderer.calls)
    assert all("첫 문장입니다" not in str(call) for call in renderer.calls)


def test_history_mode_select_tap_is_cycle_fallback_only(tmp_path: Path) -> None:
    """Missed or coordinate-less select taps move highlight without selecting."""
    controller, session, _model_manager, renderer = _build_controller(tmp_path)
    controller.enter()
    controller.handle_select_target(0)

    controller.handle_select_tap(_tap(1.0))

    assert session.states == [SessionState.HISTORY_SELECT]
    assert renderer.calls[-1] == (
        "history_menu",
        (["단군왕검, 아사달에 나라를 세우다"], 0, "고조선"),
    )
    assert controller._current_doc is None


def test_history_mode_doc_menu_merges_kinds_and_follows_order(tmp_path: Path) -> None:
    """The doc menu merges people/artifact entries and sorts by manifest order."""
    manifest_path = _write_ordered_menu_fixture(tmp_path)
    events: list[str] = []
    renderer = FakeRenderer(events)
    controller = HistoryModeController(
        session=FakeSession(events),
        model_manager=FakeModelManager(events),
        renderer=renderer,
        manifest_path=manifest_path,
        repo_root=tmp_path,
        monotonic_clock=lambda: 100.0,
    )

    controller.enter()
    controller.handle_select_target(0)

    assert renderer.calls[-1] == (
        "history_menu",
        (["하하 앞 문서", "가나다 뒤 문서"], 0, "고조선"),
    )
    assert renderer.history_menus[-1]["has_back"] is True
    assert "인물" not in controller._current_menu_options()
    assert "유물" not in controller._current_menu_options()


def test_history_mode_back_from_doc_menu_returns_to_era(tmp_path: Path) -> None:
    """The doc menu back target returns directly to the era menu."""
    controller, _session, _model_manager, renderer = _build_controller(tmp_path)
    controller.enter()
    controller.handle_select_target(0)

    controller.handle_back()

    assert renderer.calls[-1] == ("history_menu", (["고조선"], 0, "재미있는 우리역사"))
    assert renderer.history_menus[-1]["has_back"] is False
    assert controller._current_doc is None


def test_history_mode_explicit_paging_changes_page_and_rerenders(tmp_path: Path) -> None:
    """Prev/next page targets page by four rows and reset highlight into the page."""
    manifest_path = _write_paged_menu_fixture(tmp_path)
    events: list[str] = []
    session = FakeSession(events)
    renderer = FakeRenderer(events)
    controller = HistoryModeController(
        session=session,
        model_manager=FakeModelManager(events),
        renderer=renderer,
        manifest_path=manifest_path,
        repo_root=tmp_path,
        monotonic_clock=lambda: 100.0,
    )

    controller.enter()
    assert renderer.history_menus[-1] == {
        "items": ["고조선"],
        "highlight": 0,
        "title": "재미있는 우리역사",
        "has_back": False,
        "page_index": 0,
        "page_count": 1,
    }

    controller.handle_select_target(0)
    assert renderer.history_menus[-1] == {
        "items": ["문서 0", "문서 1", "문서 2", "문서 3"],
        "highlight": 0,
        "title": "고조선",
        "has_back": True,
        "page_index": 0,
        "page_count": 2,
    }

    controller.handle_next_page()
    assert renderer.history_menus[-1] == {
        "items": ["문서 4", "문서 5"],
        "highlight": 0,
        "title": "고조선",
        "has_back": True,
        "page_index": 1,
        "page_count": 2,
    }
    assert session.ack_calls == 2

    controller.handle_prev_page()
    assert renderer.history_menus[-1] == {
        "items": ["문서 0", "문서 1", "문서 2", "문서 3"],
        "highlight": 0,
        "title": "고조선",
        "has_back": True,
        "page_index": 0,
        "page_count": 2,
    }


def test_history_mode_enters_consent_and_tap_starts_next_section(tmp_path: Path) -> None:
    """Section completion pauses at consent until a tap continues."""
    controller, session, model_manager, renderer = _build_controller(tmp_path)
    controller.enter()
    controller.handle_select_target(0)
    controller.handle_select_target(0)
    assert controller._narration_thread is not None
    controller._narration_thread.join(timeout=1.0)

    controller.poll()

    assert session.states[-1] is SessionState.HISTORY_CONSENT
    assert model_manager.tts.sentences[-1] == CONSENT_PROMPT
    assert (
        "text",
        ([CONSENT_PROMPT], 30, None, "첫 이야기", "center", True),
    ) in renderer.calls

    controller.handle_consent_tap(_tap(4.0))
    assert controller._narration_thread is not None
    controller._narration_thread.join(timeout=1.0)

    assert session.states[-1] is SessionState.HISTORY_SCENE
    assert model_manager.tts.sentences[-1] == "셋째 문장입니다."


def test_history_narration_segments_add_timing_boundary_without_title_duplication(
    tmp_path: Path,
) -> None:
    """Section titles already in narration are segmented only by pause markers."""
    controller, _session, _model_manager, _renderer = _build_controller(tmp_path)
    scene = HistoryScene(
        seq=1,
        section_index=0,
        section_title="작은 제목",
        narration="앞말입니다. 작은 제목 뒤말입니다.",
        est_speech_ms=1000,
        tail_silence_ms=0,
        image_captions=(),
        images=(),
    )

    assert controller._narration_segments(scene) == [
        "앞말입니다.",
        None,
        "작은 제목",
        None,
        "뒤말입니다.",
    ]


def test_render_scene_visual_omits_top_title_for_card(tmp_path: Path) -> None:
    """Scene cards do not send narration or document text to the top label."""
    controller, _session, _model_manager, renderer = _build_controller(tmp_path)
    document = controller._load_document(controller._catalog[0])
    section = document.sections[0]
    scene = section.scenes[0]

    token = controller._render_scene_visual(scene)

    assert token == "card-1"
    assert renderer.calls[-1] == (
        "card",
        {
            "image_path": tmp_path / "assets" / "history" / "images" / "doc1" / "fig_001.jpg",
            "lines": ["첫 그림"],
            "highlight_index": None,
            "title": None,
            "sublabel": None,
            "show_exit_button": True,
            "show_back": False,
            "show_prev_card": False,
            "show_next_card": False,
            "token": "card-1",
        },
    )


def test_render_scene_visual_fallback_omits_top_title(tmp_path: Path) -> None:
    """Missing-image fallback does not send narration or document text as a title."""
    controller, _session, _model_manager, renderer = _build_controller(tmp_path)
    document = controller._load_document(controller._catalog[0])
    section = document.sections[0]
    scene = section.scenes[0]
    image_path = tmp_path / "assets" / "history" / "images" / "doc1" / "fig_001.jpg"
    image_path.unlink()

    token = controller._render_scene_visual(scene)

    assert token is None
    assert renderer.calls[-1] == (
        "text",
        (
            ["이미지를 준비 중이에요."],
            34,
            None,
            None,
            "center",
            True,
        ),
    )


def test_imageless_scene_renders_next_focal_image(tmp_path: Path) -> None:
    """An imageless scene shows the next scene's focal image instead of text."""
    controller, _session, _model_manager, renderer = _build_controller(tmp_path)
    next_image = _renderable_history_image(tmp_path, "next.jpg", "다음 그림")
    imageless_scene = _history_scene(1)
    next_scene = _history_scene(2, (next_image,))
    document = _single_section_document((imageless_scene, next_scene))

    _play_single_section_document(controller, document)

    card_calls = [payload for kind, payload in renderer.calls if kind == "card"]
    assert card_calls[0]["image_path"] == next_image.path
    assert card_calls[0]["lines"] == []
    assert _missing_image_text_calls(renderer) == []


def test_trailing_imageless_scene_renders_last_shown_image(tmp_path: Path) -> None:
    """Trailing imageless scenes stick to the last rendered image."""
    controller, _session, _model_manager, renderer = _build_controller(tmp_path)
    image = _renderable_history_image(tmp_path, "shown.jpg", "보이는 그림")
    image_scene = _history_scene(1, (image,))
    imageless_scene = _history_scene(2)
    document = _single_section_document((image_scene, imageless_scene))

    _play_single_section_document(controller, document)

    card_paths = [payload["image_path"] for kind, payload in renderer.calls if kind == "card"]
    assert card_paths[:2] == [image.path, image.path]
    assert _missing_image_text_calls(renderer) == []


def test_zero_image_document_still_renders_missing_image_label(tmp_path: Path) -> None:
    """A document with no images anywhere keeps the placeholder fallback."""
    controller, _session, _model_manager, renderer = _build_controller(tmp_path)
    document = _single_section_document((_history_scene(1),))

    _play_single_section_document(controller, document)

    assert [payload for kind, payload in renderer.calls if kind == "card"] == []
    assert _missing_image_text_calls(renderer) == [
        (
            "text",
            (
                [MISSING_IMAGE_LABEL],
                34,
                None,
                None,
                "center",
                True,
            ),
        )
    ]


def test_scene_with_own_image_ignores_fallback_image(tmp_path: Path) -> None:
    """Scenes with images keep progress-based selection even if fallback exists."""
    controller, _session, _model_manager, renderer = _build_controller(tmp_path)
    document = controller._load_document(controller._catalog[0])
    section = document.sections[0]
    scene = section.scenes[0]
    fallback_image = _renderable_history_image(tmp_path, "fallback.jpg", "대체 그림")

    token = controller._render_scene_visual(
        scene,
        progress=0.5,
        fallback_image=fallback_image,
    )

    assert token == "card-1"
    assert renderer.calls[-1] == (
        "card",
        {
            "image_path": tmp_path / "assets" / "history" / "images" / "doc1" / "fig_002.jpg",
            "lines": ["둘째 그림"],
            "highlight_index": None,
            "title": None,
            "sublabel": None,
            "show_exit_button": True,
            "show_back": False,
            "show_prev_card": False,
            "show_next_card": False,
            "token": "card-1",
        },
    )


def _anchored_image(name: str, anchor: float | None) -> HistoryImage:
    return HistoryImage(
        path=Path(f"assets/history/images/doc1/{name}"),
        caption=name,
        letterboxed=True,
        clean=True,
        is_infographic=False,
        anchor_ratio=anchor,
    )


def _scene_with_images(images: tuple[HistoryImage, ...]) -> HistoryScene:
    return HistoryScene(
        seq=1,
        section_index=0,
        section_title=None,
        narration="문장 하나. 문장 둘. 문장 셋. 문장 넷.",
        est_speech_ms=1000,
        tail_silence_ms=0,
        image_captions=tuple(image.caption or "" for image in images),
        images=images,
    )


def test_history_image_selection_follows_anchor_ratio() -> None:
    """A two-image scene shows image0 before its anchor and image1 after."""
    first = _anchored_image("fig_001.jpg", 0.0)
    second = _anchored_image("fig_002.jpg", 0.5)
    scene = _scene_with_images((first, second))

    select = HistoryModeController._select_scene_image_by_progress
    assert select(scene, 0.0) is first
    assert select(scene, 0.25) is first
    assert select(scene, 0.49) is first
    assert select(scene, 0.5) is second
    assert select(scene, 0.9) is second


def test_history_image_selection_never_flips_backward() -> None:
    """Anchor-driven selection is monotonic across increasing progress."""
    images = (
        _anchored_image("fig_001.jpg", 0.0),
        _anchored_image("fig_002.jpg", 0.33),
        _anchored_image("fig_003.jpg", 0.67),
    )
    scene = _scene_with_images(images)

    select = HistoryModeController._select_scene_image_by_progress
    selected = [
        images.index(select(scene, step / 20))  # type: ignore[arg-type]
        for step in range(21)
    ]
    assert selected == sorted(selected)
    assert selected[0] == 0
    assert selected[-1] == 2


def test_history_image_selection_falls_back_to_even_distribution() -> None:
    """Without anchor_ratio, images distribute evenly over playback progress."""
    images = (
        _anchored_image("fig_001.jpg", None),
        _anchored_image("fig_002.jpg", None),
    )
    scene = _scene_with_images(images)

    select = HistoryModeController._select_scene_image_by_progress
    # Even spacing for two images yields anchors [0.0, 0.5].
    assert select(scene, 0.0) is images[0]
    assert select(scene, 0.4) is images[0]
    assert select(scene, 0.5) is images[1]
    assert select(scene, 0.99) is images[1]


def test_history_narration_worker_renders_image_anchored_at_one(tmp_path: Path) -> None:
    """The narration worker actually shows the LAST image anchored at 1.0 (F1).

    With multiple spoken segments the final segment must map to playback
    progress 1.0, so an image anchored at exactly 1.0 is reached and rendered.
    Before the F1 fix, max progress was ``(n-1)/n < 1.0`` and the final image
    was never displayed.
    """
    controller, _session, _model_manager, renderer = _build_controller(tmp_path)
    image_dir = tmp_path / "assets" / "history" / "images" / "doc1"
    first_path = image_dir / "fig_001.jpg"
    last_path = image_dir / "fig_002.jpg"

    first = HistoryImage(
        path=first_path,
        caption="첫 그림",
        letterboxed=True,
        clean=True,
        is_infographic=False,
        anchor_ratio=0.0,
    )
    last = HistoryImage(
        path=last_path,
        caption="마지막 그림",
        letterboxed=True,
        clean=True,
        is_infographic=False,
        anchor_ratio=1.0,
    )
    scene = HistoryScene(
        seq=1,
        section_index=0,
        section_title=None,
        narration="첫 문장입니다. 둘째 문장입니다. 셋째 문장입니다.",
        est_speech_ms=1000,
        tail_silence_ms=0,
        image_captions=("첫 그림", "마지막 그림"),
        images=(first, last),
    )
    section = HistorySection(
        section_index=0,
        section_title=None,
        scenes=(scene,),
        image_captions=scene.image_captions,
    )
    document = HistoryDocument(
        doc_hash="doc1",
        title="테스트 문서",
        kind="people",
        era="고조선",
        scenes=(scene,),
        sections=(section,),
    )

    controller._session.begin_guarded_playback = lambda: None  # type: ignore[method-assign]
    controller._session.end_guarded_playback = lambda: None  # type: ignore[method-assign]

    controller._run_narration_worker(
        document,
        section,
        initial_render_token=None,
        include_lead_in=False,
    )

    card_image_paths = [payload["image_path"] for kind, payload in renderer.calls if kind == "card"]
    # The worker must reach and render the image anchored at 1.0 (it only does so
    # once playback progress hits 1.0 on the final spoken segment). Before F1 the
    # 1.0-anchored image was unreachable and never rendered.
    assert last_path in card_image_paths
    # The final selected/rendered image is the 1.0-anchored last image.
    assert card_image_paths[-1] == last_path


def test_parse_anchor_ratio_rejects_nan() -> None:
    """A NaN anchor_ratio is rejected rather than silently passed through (F3)."""
    parse = HistoryModeController._parse_anchor_ratio
    with pytest.raises(ValueError, match="must be a number or null"):
        parse(float("nan"))
    # Infinities still clamp into the [0, 1] range (pre-existing behavior kept).
    assert parse(float("inf")) == 1.0
    assert parse(float("-inf")) == 0.0
    assert parse(None) is None
    assert parse(0.5) == 0.5


def test_parse_anchor_ratio_rejects_json_nan_token(tmp_path: Path) -> None:
    """``json.load`` accepts a bare ``NaN`` token; the parser must reject it (F3)."""
    payload = json.loads('{"anchor_ratio": NaN}')
    with pytest.raises(ValueError, match="must be a number or null"):
        HistoryModeController._parse_anchor_ratio(payload["anchor_ratio"])
    del tmp_path


def test_history_mode_bad_document_returns_to_menu(tmp_path: Path) -> None:
    """Corrupt selected documents are logged and return to the current menu."""
    controller, _session, _model_manager, renderer = _build_controller(tmp_path)
    controller.enter()
    controller._catalog[0].doc_path.write_text("{not-json", encoding="utf-8")

    controller.handle_select_target(0)
    controller.handle_select_target(0)

    assert renderer.calls[-1] == (
        "history_menu",
        (["단군왕검, 아사달에 나라를 세우다"], 0, "고조선"),
    )


def test_scene_back_button_hidden_on_first_section_shown_after(tmp_path: Path) -> None:
    """The scene back button is hidden on the first scene and shown once past it."""
    controller, session, _model_manager, renderer = _build_controller(tmp_path)
    session.begin_guarded_playback = lambda: None  # type: ignore[method-assign]
    session.end_guarded_playback = lambda: None  # type: ignore[method-assign]
    image = _renderable_history_image(tmp_path, "fig_001.jpg", "그림")
    document = _two_section_document()
    # Give every scene a renderable focal image so show_card is exercised.
    document = HistoryDocument(
        doc_hash=document.doc_hash,
        title=document.title,
        kind=document.kind,
        era=document.era,
        scenes=tuple(
            HistoryScene(
                seq=scene.seq,
                section_index=scene.section_index,
                section_title=scene.section_title,
                narration=scene.narration,
                est_speech_ms=scene.est_speech_ms,
                tail_silence_ms=scene.tail_silence_ms,
                image_captions=("그림",),
                images=(image,),
            )
            for scene in document.scenes
        ),
        sections=document.sections,
    )

    controller._current_doc = document
    controller._section_index = 0
    controller._scene_index = 0
    controller._render_scene_visual(document.scenes[0], fallback_image=image)
    first_back_flags = [payload["show_back"] for kind, payload in renderer.calls if kind == "card"]
    assert first_back_flags == [False]

    # Scene granularity: the back button appears once the current scene is past
    # the document's first scene, even while still inside section 0.
    controller._section_index = 0
    controller._scene_index = 1
    controller._render_scene_visual(document.scenes[1], fallback_image=image)
    all_back_flags = [payload["show_back"] for kind, payload in renderer.calls if kind == "card"]
    assert all_back_flags == [False, True]


def test_handle_previous_step_rewinds_to_previous_section(tmp_path: Path) -> None:
    """The scene back button rewinds playback to the previous section."""
    controller, session, _model_manager, _renderer = _build_controller(tmp_path)
    session.begin_guarded_playback = lambda: None  # type: ignore[method-assign]
    session.end_guarded_playback = lambda: None  # type: ignore[method-assign]
    document = _two_section_document()

    controller._active = True
    controller._current_doc = document
    controller._section_index = 1
    controller._scene_index = 1
    controller._narration_done_event.set()

    controller.handle_previous_step()

    assert session.ack_calls == 1
    assert session.interrupt_calls == 1
    assert controller._section_index == 0
    if controller._narration_thread is not None:
        controller._narration_thread.join(timeout=1.0)
        assert controller._narration_thread.is_alive() is False


def test_handle_previous_step_noop_on_first_scene(tmp_path: Path) -> None:
    """On the document's first scene there is nothing to rewind to (no-op)."""
    controller, session, _model_manager, _renderer = _build_controller(tmp_path)
    document = _two_section_document()

    controller._active = True
    controller._current_doc = document
    controller._section_index = 0
    controller._scene_index = 0
    controller._narration_done_event.set()

    controller.handle_previous_step()

    assert session.ack_calls == 0
    assert session.interrupt_calls == 0
    assert controller._section_index == 0
    assert controller._narration_thread is None


def _multi_scene_section_document() -> HistoryDocument:
    """Build a document whose sections own multiple contiguous scenes.

    Section 0 owns scenes 0-2 and section 1 owns scenes 3-4, so scene-level back
    differs from section-level back within section 0.
    """
    scenes = tuple(
        HistoryScene(
            seq=index + 1,
            section_index=0 if index < 3 else 1,
            section_title=None,
            narration=f"문장 {index} 입니다.",
            est_speech_ms=1000,
            tail_silence_ms=0,
            image_captions=(),
            images=(),
        )
        for index in range(5)
    )
    sections = (
        HistorySection(
            section_index=0,
            section_title=None,
            scenes=scenes[0:3],
            image_captions=(),
        ),
        HistorySection(
            section_index=1,
            section_title=None,
            scenes=scenes[3:5],
            image_captions=(),
        ),
    )
    return HistoryDocument(
        doc_hash="doc-multi",
        title="여러 장면 문서",
        kind="people",
        era="고조선",
        scenes=scenes,
        sections=sections,
    )


def test_handle_previous_step_rewinds_one_scene_within_section(tmp_path: Path) -> None:
    """Scene back inside a multi-scene section steps back exactly one scene."""
    controller, session, _model_manager, _renderer = _build_controller(tmp_path)
    session.begin_guarded_playback = lambda: None  # type: ignore[method-assign]
    session.end_guarded_playback = lambda: None  # type: ignore[method-assign]
    document = _multi_scene_section_document()

    controller._active = True
    controller._current_doc = document
    controller._section_index = 0
    controller._scene_index = 2  # third scene of section 0
    controller._narration_done_event.set()

    controller.handle_previous_step()
    if controller._narration_thread is not None:
        controller._narration_thread.join(timeout=1.0)

    assert session.ack_calls == 1
    assert session.interrupt_calls == 1
    # Stays in section 0 but starts the worker at the second scene (offset 1).
    assert controller._section_index == 0
    assert controller._section_scene_offset == 1


def test_handle_previous_step_crosses_into_previous_section_last_scene(tmp_path: Path) -> None:
    """Scene back on a section's first scene lands on the prior section's last scene."""
    controller, session, _model_manager, _renderer = _build_controller(tmp_path)
    session.begin_guarded_playback = lambda: None  # type: ignore[method-assign]
    session.end_guarded_playback = lambda: None  # type: ignore[method-assign]
    document = _multi_scene_section_document()

    controller._active = True
    controller._current_doc = document
    controller._section_index = 1
    controller._scene_index = 3  # first scene of section 1
    controller._narration_done_event.set()

    controller.handle_previous_step()
    if controller._narration_thread is not None:
        controller._narration_thread.join(timeout=1.0)

    # Crosses the boundary back into section 0 at its last scene (offset 2).
    assert controller._section_index == 0
    assert controller._section_scene_offset == 2


def test_narration_worker_tracks_scene_index_for_scene_back(tmp_path: Path) -> None:
    """The worker advances ``_scene_index`` so the back button reflects the scene."""
    controller, session, _model_manager, _renderer = _build_controller(tmp_path)
    session.begin_guarded_playback = lambda: None  # type: ignore[method-assign]
    session.end_guarded_playback = lambda: None  # type: ignore[method-assign]
    document = _multi_scene_section_document()
    section = document.sections[0]

    controller._current_doc = document
    controller._section_index = 0
    controller._scene_index = 0

    controller._run_narration_worker(
        document,
        section,
        initial_render_token=None,
        include_lead_in=False,
        scene_offset=0,
    )

    # After narrating all three scenes of section 0 the index lands on scene 2.
    assert controller._scene_index == 2


def test_start_section_honors_scene_offset(tmp_path: Path) -> None:
    """Starting a section at a scene offset renders that scene first."""
    controller, session, _model_manager, _renderer = _build_controller(tmp_path)
    session.begin_guarded_playback = lambda: None  # type: ignore[method-assign]
    session.end_guarded_playback = lambda: None  # type: ignore[method-assign]
    document = _multi_scene_section_document()

    controller._active = True
    controller._current_doc = document
    controller._section_index = 0
    controller._section_scene_offset = 1
    controller._start_section()
    if controller._narration_thread is not None:
        controller._narration_thread.join(timeout=1.0)

    # The absolute scene index for section 0 offset 1 is scene 1.
    assert document.scenes.index(document.sections[0].scenes[1]) == 1
    # Worker walked from offset 1 to the section's last scene (index 2).
    assert controller._scene_index == 2


def test_history_mode_exit_restores_preload_and_unloads_nonresident_tts(tmp_path: Path) -> None:
    """Exit restores STT preload policy and releases nonresident TTS."""
    controller, session, model_manager, renderer = _build_controller(tmp_path)
    controller.enter()

    future = controller.exit(restore_stt=True)
    future.result(timeout=1.0)

    assert session.interrupt_calls == 1
    assert ("unload_tts", False) in model_manager.calls
    assert model_manager.calls[-2:] == [("reset_preload_state", None), ("preload_stt", None)]
    assert session.history_capture_calls[-1] == ("release", True)
    assert renderer.calls[-2:] == [("clear_image", None), ("clear_text", None)]


def test_history_mode_sleep_exit_skips_stt_preload_and_releases_sleep_capture(
    tmp_path: Path,
) -> None:
    """Sleep-target history exit skips STT preload and releases capture as stopped."""
    controller, session, model_manager, _renderer = _build_controller(tmp_path)
    controller.enter()

    future = controller.exit(restore_stt=False)
    future.result(timeout=1.0)

    assert ("unload_tts", False) in model_manager.calls
    assert model_manager.calls[-2:] == [("unload_tts", False), ("reset_preload_state", None)]
    assert ("preload_stt", None) not in model_manager.calls[5:]
    assert session.history_capture_calls[-1] == ("release", False)


def test_history_exit_does_not_play_or_render_after_blocked_synth(
    tmp_path: Path,
) -> None:
    """A worker finishing synth after exit must not play audio or render more scenes."""
    controller, session, model_manager, renderer = _build_controller(tmp_path)
    controller.enter()
    blocking_tts, thread = _start_blocking_narration(controller, model_manager)

    future = controller.exit(restore_stt=True)
    blocking_tts.release.set()
    future.result(timeout=1.0)
    thread.join(timeout=1.0)

    assert thread.is_alive() is False
    assert session.played == []
    assert all(kind != "card" for kind, _payload in renderer.calls)


def test_history_exit_waits_for_blocked_synth_before_unloading_tts(
    tmp_path: Path,
) -> None:
    """Teardown joins the narration worker before unloading TTS."""
    controller, _session, model_manager, _renderer = _build_controller(tmp_path)
    controller.enter()
    blocking_tts, thread = _start_blocking_narration(controller, model_manager)

    future = controller.exit(restore_stt=True)

    assert ("unload_tts", False) not in model_manager.calls
    blocking_tts.release.set()
    future.result(timeout=1.0)
    thread.join(timeout=1.0)

    assert ("unload_tts", False) in model_manager.calls


def test_history_reentry_waits_for_pending_teardown_and_coalesces_second_exit(
    tmp_path: Path,
) -> None:
    """Immediate re-entry waits for teardown and repeated exits share the same future."""
    controller, _session, model_manager, _renderer = _build_controller(tmp_path)
    controller.enter()
    blocking_tts, thread = _start_blocking_narration(controller, model_manager)

    future = controller.exit(restore_stt=True)
    second_future = controller.exit(restore_stt=True)
    entered = threading.Event()

    def enter_again() -> None:
        controller.enter()
        entered.set()

    enter_thread = threading.Thread(target=enter_again, daemon=True)
    enter_thread.start()

    assert second_future is future
    assert entered.wait(timeout=0.05) is False
    assert model_manager.calls.count(("load", ModelType.TTS)) == 1

    blocking_tts.release.set()
    future.result(timeout=1.0)
    thread.join(timeout=1.0)
    enter_thread.join(timeout=1.0)

    assert entered.is_set()
    assert model_manager.calls.count(("load", ModelType.TTS)) == 2
    first_unload_tts = model_manager.calls.index(("unload_tts", False))
    second_load_tts = (
        len(model_manager.calls)
        - 1
        - list(reversed(model_manager.calls)).index(("load", ModelType.TTS))
    )
    assert first_unload_tts < second_load_tts


def test_history_controller_close_drains_pending_teardown_executor(
    tmp_path: Path,
) -> None:
    """Controller close waits for a pending teardown and stops the worker lane."""
    controller, _session, model_manager, _renderer = _build_controller(tmp_path)
    controller.enter()
    blocking_tts, thread = _start_blocking_narration(controller, model_manager)
    future = controller.exit(restore_stt=True)
    close_entered = threading.Event()
    close_done = threading.Event()

    def close_controller() -> None:
        close_entered.set()
        controller.close()
        close_done.set()

    close_thread = threading.Thread(target=close_controller, daemon=True)
    close_thread.start()

    assert close_entered.wait(timeout=1.0)
    assert close_done.wait(timeout=0.05) is False
    blocking_tts.release.set()
    future.result(timeout=1.0)
    thread.join(timeout=1.0)
    close_thread.join(timeout=1.0)

    assert close_done.is_set()
    assert controller._teardown_executor._thread.is_alive() is False
    assert not any(
        live_thread.name.startswith("HistoryTeardown")
        and live_thread.is_alive()
        and not live_thread.daemon
        for live_thread in threading.enumerate()
    )


def test_history_narration_worker_has_no_direct_audio_or_filter_calls() -> None:
    """Narration must not call unguarded playback or conversational filters."""
    source = inspect.getsource(HistoryModeController._run_narration_worker)

    assert "audio_player" not in source
    assert "play_audio" not in source
    assert "_filter_text" not in source


def test_history_play_tts_text_uses_valid_cache_hit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validated cache hits play through the guarded path without live synthesis."""
    controller, session, model_manager, _renderer = _build_controller(tmp_path)
    wav_path = tmp_path / "cache-hit.wav"
    _write_history_tts_wav(wav_path)
    monkeypatch.setattr(
        "core.history_mode.tts_cache.lookup",
        lambda text, lang: wav_path,
    )

    assert controller._play_tts_text(model_manager.tts, "cached text") is True

    assert model_manager.tts.sentences == []
    assert session.guard_calls == ["begin", "play", "end"]
    assert session.played[-1][1] == 16_000


def test_history_play_tts_text_live_synths_on_cache_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache misses preserve the existing live Korean synthesis path."""
    controller, session, model_manager, _renderer = _build_controller(tmp_path)
    monkeypatch.setattr(
        "core.history_mode.tts_cache.lookup",
        lambda text, lang: None,
    )

    assert controller._play_tts_text(model_manager.tts, "live text") is True

    assert model_manager.tts.sentences == ["live text"]
    assert session.guard_calls == ["begin", "play", "end"]
    assert session.played[-1][1] == 22_050


def test_history_play_tts_text_live_synths_on_bad_cache_hit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raced or corrupt cache path falls back to live synthesis."""
    controller, session, model_manager, _renderer = _build_controller(tmp_path)
    bad_wav_path = tmp_path / "bad-cache.wav"
    bad_wav_path.write_bytes(b"not a wav")
    monkeypatch.setattr(
        "core.history_mode.tts_cache.lookup",
        lambda text, lang: bad_wav_path,
    )

    assert controller._play_tts_text(model_manager.tts, "fallback text") is True

    assert model_manager.tts.sentences == ["fallback text"]
    assert session.guard_calls == ["begin", "play", "end"]
    assert session.played[-1][1] == 22_050


def test_history_play_tts_text_sanitizes_before_live_synthesis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hanja, middle dots, and fullwidth punctuation are cleaned before synthesis."""
    controller, _session, model_manager, _renderer = _build_controller(tmp_path)
    monkeypatch.setattr(
        "core.history_mode.tts_cache.lookup",
        lambda text, lang: None,
    )

    # "3·1운동 (古) 이야기，끝。" mixing a middle dot, a Hanja gloss, and fullwidth punctuation.
    dirty = "3·1운동 (古) 이야기，끝。"
    assert controller._play_tts_text(model_manager.tts, dirty) is True

    assert len(model_manager.tts.sentences) == 1
    spoken = model_manager.tts.sentences[0]
    # The engine must never receive a CJK ideograph, middle dot, or fullwidth punctuation.
    for char in spoken:
        codepoint = ord(char)
        assert char not in ("·", "，", "。")
        assert not (0x3400 <= codepoint <= 0x9FFF)
        assert not (0xF900 <= codepoint <= 0xFAFF)


def test_history_play_tts_text_skips_segment_on_synthesis_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing synthesis call skips the segment instead of crashing the narration."""
    controller, session, model_manager, _renderer = _build_controller(tmp_path)
    monkeypatch.setattr(
        "core.history_mode.tts_cache.lookup",
        lambda text, lang: None,
    )

    def boom(_text: str, language: str = "ko") -> tuple[list[float], int]:
        raise RuntimeError("Supertonic synthesis failed: Found 1 unsupported character(s): ['?']")

    monkeypatch.setattr(model_manager.tts, "synthesize", boom)

    # Returns True (continue narration) and does not propagate the failure.
    assert controller._play_tts_text(model_manager.tts, "bad segment") is True
    # No audio was played for the skipped segment.
    assert session.guard_calls == []
