"""Headless tests for the Pygame character renderer."""

from __future__ import annotations

import contextlib
import logging
import struct
import threading
import time
import zlib
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pygame  # type: ignore[import-not-found, import-untyped]
import pytest

from core.character_expression import CharacterExpression
from core.character_renderer import (
    _BACK_BUTTON_RECT,
    _CARD_CAPTION_RECT,
    _CARD_HEADER_RECT,
    _CARD_IMAGE_RECT,
    _CARD_IMAGE_RECT_NO_HEADER,
    _CARD_NEXT_BUTTON_RECT,
    _CARD_PREV_BUTTON_RECT,
    _EXIT_BUTTON_RECT,
    _INDICATOR_ANIM_TOTAL_FRAMES,
    _INDICATOR_POSITION,
    _INDICATOR_SIZE,
    _MENU_LEFT,
    _NEXT_PAGE_BUTTON_RECT,
    _PAGE_BUTTON_SIZE,
    _PAGE_BUTTON_Y,
    _PAGE_INDICATOR_RECT,
    _PAGE_NAV_TOUCH_PAD,
    _PORTAL_HOTSPOT_POSITION,
    _PORTAL_STATUS_GAP,
    _PREV_PAGE_BUTTON_RECT,
    _SAFE_BOTTOM,
    _TEXT_BG,
    _TEXT_SAFE_TOP_RIGHT_MARGIN,
    _WAIT_EXPRESSION_CYCLE,
    _WAIT_EXPRESSION_CYCLE_FRAMES,
    HitTarget,
    PygameCharacterRenderer,
    _CardRequest,
    _centered_title_x,
    _HitRegion,
    _top_title_span,
)
from core.session_manager import SessionState

_CANVAS_SIZE = (720, 720)
_LOGGER_NAME = "core.character_renderer"


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum)


def _write_rgba_png(
    path: Path,
    rgba: tuple[int, int, int, int],
    size: tuple[int, int] = _CANVAS_SIZE,
) -> None:
    width, height = size
    row = b"\x00" + bytes(rgba) * width
    raw = row * height
    payload = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(raw)),
            _png_chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(payload)


def _create_sprites(
    asset_dir: Path,
    expressions: Iterable[CharacterExpression] = CharacterExpression,
) -> Path:
    asset_dir.mkdir(parents=True, exist_ok=True)
    for index, expression in enumerate(expressions):
        channel = (24 + index * 20) % 256
        _write_rgba_png(asset_dir / f"{expression.value}.png", (channel, channel, channel, 255))
    return asset_dir


def _make_surface_renderer() -> PygameCharacterRenderer:
    renderer = PygameCharacterRenderer.__new__(PygameCharacterRenderer)
    renderer._display = pygame.Surface(_CANVAS_SIZE)
    renderer._sprites = {}
    renderer._frame_paths = {}
    renderer._active_frames = None
    renderer._active_frames_expr = None
    renderer._failed_frame_expressions = set()
    renderer._frame_index = 0
    renderer._wait_cycle_armed = False
    renderer._wait_cycle_index = 0
    renderer._wait_cycle_frames_elapsed = 0
    renderer._black_fallback = None
    renderer._lock = threading.Lock()
    renderer._wake = threading.Event()
    renderer._pending_expression = None
    renderer._pending_language = None
    renderer._pending_image = None
    renderer._pending_text = None
    renderer._pending_card = None
    renderer._pending_feedback_token = None
    renderer._render_events = {}
    renderer._render_token_counter = 0
    renderer._layout_generation = 0
    renderer._displayed_layout_generation = 0
    renderer._hit_regions = ()
    renderer._pressed_hit_target = None
    renderer._current_expression_value = None
    renderer._current_language_value = None
    renderer._current_image_request = None
    renderer._current_text_request = None
    renderer._current_card_request = None
    renderer._indicator_anim_frames = 0
    renderer._portal_button_active = False
    renderer._portal_button_icons = {}
    renderer._portal_button_pressed_icon = None
    renderer._portal_status_text = None
    renderer._portal_status_timer = None
    renderer._portal_status_generation = 0
    renderer._language_indicators = {}
    renderer._font_cache = {}
    renderer._close_called = False
    return renderer


def _init_dummy_display(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    pygame.display.set_mode(_CANVAS_SIZE)


def test_has_frame_to_render_tracks_portal_status_only() -> None:
    """Portal status text alone should keep the render loop awake."""
    renderer = _make_surface_renderer()

    assert not renderer._has_frame_to_render()

    renderer._portal_status_text = "다운로드 모드 켜짐"
    assert renderer._has_frame_to_render()

    renderer._portal_status_text = None
    assert not renderer._has_frame_to_render()


@pytest.fixture(autouse=True)
def _quit_pygame_display_after_test() -> Iterator[None]:
    yield
    with contextlib.suppress(Exception):
        pygame.display.quit()


def _wait_for_expression(
    renderer: PygameCharacterRenderer,
    expected: CharacterExpression,
    *,
    timeout: float = 2.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if renderer.current_expression == expected:
            return
        time.sleep(0.01)
    pytest.fail(f"Renderer did not display {expected} within {timeout:.1f}s")


def _wait_for_no_ui_threads(timeout: float = 0.3) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(thread.name == "PygameRendererUI" for thread in threading.enumerate()):
            return
        time.sleep(0.01)
    active = [thread.name for thread in threading.enumerate()]
    pytest.fail(f"PygameRendererUI thread still active: {active}")


def _wait_for_log(caplog: pytest.LogCaptureFixture, message: str, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if message in caplog.text:
            return
        time.sleep(0.01)
    pytest.fail(f"Log message not captured within {timeout:.1f}s: {message}")


def test_init_with_all_sprites_succeeds_without_initial_expression(tmp_path: Path) -> None:
    asset_dir = _create_sprites(tmp_path)
    renderer = PygameCharacterRenderer(asset_dir, sdl_driver="dummy")
    try:
        assert renderer.current_expression is None
        assert renderer._ui_thread.is_alive()
        time.sleep(0.05)
        assert renderer.current_expression is None
    finally:
        renderer.close()


def test_on_expression_renders_requested_sprite(tmp_path: Path) -> None:
    asset_dir = _create_sprites(tmp_path)
    renderer = PygameCharacterRenderer(asset_dir, sdl_driver="dummy")
    try:
        renderer.on_expression(CharacterExpression.LISTENING)
        _wait_for_expression(renderer, CharacterExpression.LISTENING)
    finally:
        renderer.close()


def test_wait_expression_cycle_arms_only_on_thinking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit THINKING emission arms the wait expression cycle."""
    asset_dir = _create_sprites(tmp_path)

    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()
    try:
        renderer._load_sprites(asset_dir)
        renderer._discover_frame_sequences(asset_dir)

        renderer.on_expression(CharacterExpression.LISTENING)

        assert renderer._pending_expression is CharacterExpression.LISTENING
        assert not renderer._wait_cycle_armed

        renderer._render_once()
        assert renderer.current_expression is CharacterExpression.LISTENING
        assert not renderer._has_frame_to_render()

        renderer.on_expression(CharacterExpression.THINKING)

        assert renderer._pending_expression is CharacterExpression.THINKING
        assert renderer._wait_cycle_armed
        assert renderer._wait_cycle_index == 0
        assert renderer._wait_cycle_frames_elapsed == 0
        assert renderer._has_frame_to_render()

        renderer._render_once()

        assert renderer.current_expression is CharacterExpression.THINKING
        assert renderer._wait_cycle_armed
    finally:
        pygame.quit()


def test_wait_expression_cycle_rotates_through_positive_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Armed THINKING wait state advances through the ordered positive cycle."""
    asset_dir = _create_sprites(tmp_path)
    blocked_expressions = {
        CharacterExpression.CONCERNED,
        CharacterExpression.SAD,
        CharacterExpression.ANGRY,
        CharacterExpression.SULKY,
        CharacterExpression.SLEEPY,
        CharacterExpression.TIRED,
    }

    assert blocked_expressions.isdisjoint(_WAIT_EXPRESSION_CYCLE)
    assert len(_WAIT_EXPRESSION_CYCLE) % 3 == 0
    assert all(
        expression is CharacterExpression.THINKING
        for index, expression in enumerate(_WAIT_EXPRESSION_CYCLE)
        if index % 3 == 0
    )
    assert all(
        expression is not CharacterExpression.THINKING
        for index, expression in enumerate(_WAIT_EXPRESSION_CYCLE)
        if index % 3 != 0
    )
    assert {
        CharacterExpression.HAPPY,
        CharacterExpression.JOYFUL,
        CharacterExpression.EXCITED,
        CharacterExpression.SURPRISED,
        CharacterExpression.GREETING,
        CharacterExpression.AFFECTIONATE,
        CharacterExpression.WINKING,
    }.issubset(set(_WAIT_EXPRESSION_CYCLE))

    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()
    try:
        renderer._load_sprites(asset_dir)
        renderer._discover_frame_sequences(asset_dir)

        renderer.on_expression(CharacterExpression.THINKING)
        renderer._render_once()
        assert renderer.current_expression is _WAIT_EXPRESSION_CYCLE[0]

        seen = [renderer.current_expression]
        for expected in _WAIT_EXPRESSION_CYCLE[1:]:
            for _index in range(_WAIT_EXPRESSION_CYCLE_FRAMES):
                assert renderer._has_frame_to_render()
                renderer._render_once()
            seen.append(renderer.current_expression)
            assert renderer.current_expression is expected
            assert renderer._pending_expression is None
            assert renderer._wait_cycle_armed

        assert tuple(seen) == _WAIT_EXPRESSION_CYCLE
    finally:
        pygame.quit()


@pytest.mark.parametrize(
    "expression",
    [CharacterExpression.CONCERNED, CharacterExpression.SPEAKING],
)
def test_wait_expression_cycle_disarms_and_holds_on_non_thinking_expression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    expression: CharacterExpression,
) -> None:
    """Safety and deliberate non-wait expressions stop the wait cycle."""
    asset_dir = _create_sprites(tmp_path)

    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()
    try:
        renderer._load_sprites(asset_dir)
        renderer._discover_frame_sequences(asset_dir)

        renderer.on_expression(CharacterExpression.THINKING)
        renderer._render_once()
        for _index in range(_WAIT_EXPRESSION_CYCLE_FRAMES):
            renderer._render_once()
        assert renderer.current_expression is _WAIT_EXPRESSION_CYCLE[1]
        assert renderer._wait_cycle_armed

        renderer.on_expression(expression)

        assert not renderer._wait_cycle_armed

        renderer._render_once()
        assert renderer.current_expression is expression
        assert not renderer._has_frame_to_render()

        for _index in range(_WAIT_EXPRESSION_CYCLE_FRAMES * 2):
            renderer._render_once()

        assert renderer.current_expression is expression
        assert renderer._wait_cycle_index == 1
        assert renderer._wait_cycle_frames_elapsed == 0
        assert not renderer._wait_cycle_armed
    finally:
        pygame.quit()


def test_show_text_renders_text_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generic text rendering updates the display surface without a sprite."""
    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()

    renderer.show_text(["재미있는 우리역사"], size=34, title="고조선", layout="center")
    renderer._render_once()

    assert renderer._current_text_request is not None
    assert renderer._display.get_at((_CANVAS_SIZE[0] // 2, _CANVAS_SIZE[1] // 2)) != (0, 0, 0, 255)


def test_center_text_layout_renders_back_and_exit_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Center text can expose the same top nav targets as menu/card layouts."""
    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()

    renderer.show_text(
        ["All done!"],
        size=46,
        title="Funny English",
        layout="center",
        show_exit_button=True,
        has_back=True,
    )
    renderer._render_once()
    generation = renderer._displayed_layout_generation

    assert renderer.hit_test(40, 40) == HitTarget("back", None, generation)
    assert renderer.hit_test(540, 40) == HitTarget("exit", None, generation)


def test_show_image_suppresses_language_indicator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Image mode does not draw the top-right language badge."""
    _init_dummy_display(monkeypatch)
    image_path = tmp_path / "green.png"
    _write_rgba_png(image_path, (0, 255, 0, 255))
    renderer = _make_surface_renderer()
    renderer._language_indicators["ko"] = pygame.Surface((48, 48))
    renderer._language_indicators["ko"].fill((255, 0, 0))

    renderer.set_language_indicator("ko")
    renderer.show_image(image_path)
    renderer._render_once()

    assert renderer._current_image_request is not None
    assert renderer._display.get_at(_INDICATOR_POSITION) == (0, 255, 0, 255)
    assert renderer.hit_test(_INDICATOR_POSITION[0] + 1, _INDICATOR_POSITION[1] + 1) is None


def test_show_card_renders_image_and_text_without_language_indicator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Card mode composites image and text while suppressing the language badge."""
    _init_dummy_display(monkeypatch)
    image_path = tmp_path / "blue.png"
    _write_rgba_png(image_path, (0, 0, 255, 255), size=(200, 200))
    renderer = _make_surface_renderer()
    renderer._language_indicators["en"] = pygame.Surface((48, 48))
    renderer._language_indicators["en"].fill((255, 0, 0))

    renderer.set_language_indicator("en")
    renderer.show_card(image_path=image_path, lines=["cat"], title="Funny English")
    renderer._render_once()

    assert renderer._current_card_request is not None
    assert renderer._current_image_request is None
    assert renderer._current_text_request is None
    assert renderer._display.get_at(_INDICATOR_POSITION) != (255, 0, 0, 255)
    assert renderer.hit_test(_INDICATOR_POSITION[0] + 1, _INDICATOR_POSITION[1] + 1) is None


def test_idle_language_indicator_registers_padded_toggle_hit_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The idle language badge is a padded direct-touch toggle target."""
    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()
    indicator = pygame.Surface((_INDICATOR_SIZE, _INDICATOR_SIZE), pygame.SRCALPHA)
    indicator.fill((255, 0, 0, 255))
    renderer._language_indicators["ko"] = indicator

    renderer.set_language_indicator("ko")
    renderer._render_once()
    generation = renderer._displayed_layout_generation
    center = (
        _INDICATOR_POSITION[0] + _INDICATOR_SIZE // 2,
        _INDICATOR_POSITION[1] + _INDICATOR_SIZE // 2,
    )
    padded_margin = (_INDICATOR_POSITION[0] - 1, center[1])
    target = HitTarget("language_toggle", None, generation)

    assert renderer.hit_test(*center) == target
    assert renderer.hit_test(*padded_margin) == target

    before = renderer._display.get_at(center)
    renderer._pressed_hit_target = target
    renderer._render_once()
    after = renderer._display.get_at(center)

    assert after != before


def test_on_state_change_updates_portal_button_active_and_wakes_renderer() -> None:
    """Session state changes control the portal button active flag."""
    renderer = _make_surface_renderer()

    renderer.on_state_change(SessionState.AWAITING_TAP)

    assert renderer._portal_button_active
    assert renderer._wake.is_set()

    renderer._wake.clear()
    renderer.on_state_change(SessionState.RESPONDING)

    assert not renderer._portal_button_active
    assert renderer._wake.is_set()


def test_idle_download_button_renders_active_and_disabled_styles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The top-left download button renders full opacity only while awaiting tap."""
    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()
    indicator = pygame.Surface((_INDICATOR_SIZE, _INDICATOR_SIZE), pygame.SRCALPHA)
    indicator.fill((255, 0, 0, 255))
    renderer._language_indicators["ko"] = indicator
    renderer.set_language_indicator("ko")
    sample = (
        _PORTAL_HOTSPOT_POSITION[0] + (_INDICATOR_SIZE // 2),
        _PORTAL_HOTSPOT_POSITION[1] + (_INDICATOR_SIZE // 2),
    )

    renderer.on_state_change(SessionState.WAKING)
    renderer._render_once()
    disabled = renderer._display.get_at(sample)

    renderer.on_state_change(SessionState.AWAITING_TAP)
    renderer._render_once()
    active = renderer._display.get_at(sample)

    assert sum(active[:3]) > sum(disabled[:3])
    assert False in renderer._portal_button_icons
    assert True in renderer._portal_button_icons


def test_portal_button_pressed_style_applies_only_when_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A portal target renders the pressed style only for the active button."""
    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()
    indicator = pygame.Surface((_INDICATOR_SIZE, _INDICATOR_SIZE), pygame.SRCALPHA)
    indicator.fill((255, 0, 0, 255))
    renderer._language_indicators["ko"] = indicator
    renderer.set_language_indicator("ko")
    renderer.on_state_change(SessionState.AWAITING_TAP)
    renderer._render_once()
    target = HitTarget("portal_activate", None, renderer._displayed_layout_generation)
    sample = (
        _PORTAL_HOTSPOT_POSITION[0] + (_INDICATOR_SIZE // 2),
        _PORTAL_HOTSPOT_POSITION[1] + (_INDICATOR_SIZE // 2),
    )
    active = renderer._display.get_at(sample)

    renderer._pressed_hit_target = target
    renderer._render_once()
    pressed = renderer._display.get_at(sample)

    assert sum(pressed[:3]) < sum(active[:3])

    renderer.on_state_change(SessionState.WAKING)
    assert renderer.show_press_feedback(target, duration=0.5) is None


def test_show_portal_status_draws_text_and_timer_clears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Portal status text is transient and redrawn near the top-left button."""
    _init_dummy_display(monkeypatch)
    timers: list[Any] = []

    class FakeTimer:
        def __init__(self, interval: float, function: Any, args: tuple[Any, ...]) -> None:
            self.interval = interval
            self.function = function
            self.args = args
            self.daemon = False
            self.started = False
            self.cancelled = False
            timers.append(self)

        def start(self) -> None:
            self.started = True

        def cancel(self) -> None:
            self.cancelled = True

        def fire(self) -> None:
            self.function(*self.args)

    monkeypatch.setattr(threading, "Timer", FakeTimer)
    renderer = _make_surface_renderer()
    indicator = pygame.Surface((_INDICATOR_SIZE, _INDICATOR_SIZE), pygame.SRCALPHA)
    indicator.fill((255, 0, 0, 255))
    renderer._language_indicators["ko"] = indicator
    renderer.set_language_indicator("ko")

    renderer.show_portal_status("한 번 더", duration=1.2)

    assert renderer._portal_status_text == "한 번 더"
    assert timers[0].interval == 1.2
    assert timers[0].started

    renderer._render_once()
    status_panel_sample = (
        _PORTAL_HOTSPOT_POSITION[0] + _INDICATOR_SIZE + _PORTAL_STATUS_GAP + 1,
        _PORTAL_HOTSPOT_POSITION[1] + (_INDICATOR_SIZE // 2),
    )
    assert renderer._display.get_at(status_panel_sample) != (0, 0, 0, 255)

    renderer._wake.clear()
    timers[0].fire()

    assert renderer._portal_status_text is None
    assert renderer._portal_status_timer is None
    assert renderer._wake.is_set()


def test_card_back_and_exit_targets_follow_show_card_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Card direct-touch nav targets are opt-in per render request."""
    _init_dummy_display(monkeypatch)
    image_path = tmp_path / "blue.png"
    _write_rgba_png(image_path, (0, 0, 255, 255), size=(200, 200))
    renderer = _make_surface_renderer()

    renderer.show_card(image_path=image_path, lines=["cat"], title="Funny English")
    renderer._render_once()

    assert renderer.hit_test(40, 40) is None
    assert renderer.hit_test(540, 40) is None

    renderer.show_card(
        image_path=image_path,
        lines=["cat"],
        title="Funny English",
        show_exit_button=True,
        show_back=True,
    )
    renderer._render_once()
    generation = renderer._displayed_layout_generation

    assert renderer.hit_test(40, 40) == HitTarget("back", None, generation)
    assert renderer.hit_test(540, 40) == HitTarget("exit", None, generation)


def test_card_prev_next_word_targets_follow_show_card_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Card word-navigation targets are opt-in and stay within the safe bottom."""
    _init_dummy_display(monkeypatch)
    image_path = tmp_path / "blue.png"
    _write_rgba_png(image_path, (0, 0, 255, 255), size=(200, 200))
    renderer = _make_surface_renderer()

    renderer.show_card(
        image_path=image_path,
        lines=["dog"],
        title="Funny English",
        show_prev_card=True,
        show_next_card=True,
    )
    renderer._render_once()
    generation = renderer._displayed_layout_generation

    prev_x = _CARD_PREV_BUTTON_RECT[0] + (_CARD_PREV_BUTTON_RECT[2] // 2)
    next_x = _CARD_NEXT_BUTTON_RECT[0] + (_CARD_NEXT_BUTTON_RECT[2] // 2)
    button_y = _CARD_PREV_BUTTON_RECT[1] + (_CARD_PREV_BUTTON_RECT[3] // 2)

    assert _CARD_PREV_BUTTON_RECT[1] + _CARD_PREV_BUTTON_RECT[3] <= _SAFE_BOTTOM
    assert _CARD_NEXT_BUTTON_RECT[1] + _CARD_NEXT_BUTTON_RECT[3] <= _SAFE_BOTTOM
    assert _CARD_PREV_BUTTON_RECT[1] >= _CARD_CAPTION_RECT[1]
    assert renderer.hit_test(prev_x, button_y) == HitTarget("prev_card", None, generation)
    assert renderer.hit_test(next_x, button_y) == HitTarget("next_card", None, generation)

    renderer.show_card(
        image_path=image_path,
        lines=["dog"],
        title="Funny English",
        show_prev_card=False,
        show_next_card=True,
    )
    renderer._render_once()
    generation = renderer._displayed_layout_generation

    assert renderer.hit_test(prev_x, button_y) is None
    assert renderer.hit_test(next_x, button_y) == HitTarget("next_card", None, generation)


def test_render_token_waits_for_exact_card_flip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A card token is acknowledged only after that request flips."""
    _init_dummy_display(monkeypatch)
    image_path = tmp_path / "blue.png"
    _write_rgba_png(image_path, (0, 0, 255, 255), size=(200, 200))
    renderer = _make_surface_renderer()

    token = renderer.show_card(image_path=image_path, lines=["caption"], title="제목")

    assert token is not None
    assert renderer.wait_until_rendered(token, timeout=0.0) is False
    renderer._render_once()
    assert renderer.wait_until_rendered(token, timeout=0.0) is True


def test_superseded_render_token_times_out_but_latest_token_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overwritten render requests do not acknowledge the old token."""
    _init_dummy_display(monkeypatch)
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    _write_rgba_png(first_path, (255, 0, 0, 255), size=(200, 200))
    _write_rgba_png(second_path, (0, 255, 0, 255), size=(200, 200))
    renderer = _make_surface_renderer()

    old_token = renderer.show_card(image_path=first_path, lines=["old"], title="첫째")
    latest_token = renderer.show_card(image_path=second_path, lines=["new"], title="둘째")
    renderer._render_once()

    assert old_token is not None
    assert latest_token is not None
    assert renderer.wait_until_rendered(old_token, timeout=0.0) is False
    assert renderer.wait_until_rendered(latest_token, timeout=0.0) is True


def test_history_menu_returns_token_and_records_hit_regions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """History menus publish generation-tagged menu and exit hit targets."""
    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()

    token = renderer.show_history_menu(["고조선", "뒤로"], 0, "재미있는 우리역사")

    assert token is not None
    assert renderer.wait_until_rendered(token, timeout=0.0) is False
    renderer._render_once()
    assert renderer.wait_until_rendered(token, timeout=0.0) is True

    menu_target = renderer.hit_test(60, 180)
    exit_target = renderer.hit_test(540, 40)

    assert menu_target == HitTarget("menu_item", 0, renderer._displayed_layout_generation)
    assert exit_target == HitTarget("exit", None, renderer._displayed_layout_generation)


def test_menu_item_hit_regions_are_full_width_and_gapless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Menu item hit regions cover the former right strip and row gaps only."""
    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()

    renderer.show_history_menu(
        ["첫째", "둘째", "셋째", "넷째"], 0, "제목", page_index=0, page_count=2
    )
    renderer._render_once()

    generation = renderer._displayed_layout_generation

    assert renderer.hit_test(700, 180) == HitTarget("menu_item", 0, generation)
    assert renderer.hit_test(700, 273) == HitTarget("menu_item", 1, generation)
    assert renderer.hit_test(360, 120) is None
    arrow_band_target = renderer.hit_test(
        _NEXT_PAGE_BUTTON_RECT[0] - 20,
        _NEXT_PAGE_BUTTON_RECT[1] + 20,
    )
    assert arrow_band_target is None or arrow_band_target.kind != "menu_item"


def test_menu_back_target_presence_follows_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Back hit target is rendered only for menus with a parent level."""
    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()

    renderer.show_history_menu(["문서"], 0, "고조선")
    renderer._render_once()
    assert renderer.hit_test(40, 40) is None

    renderer.show_history_menu(["문서"], 0, "고조선", has_back=True)
    renderer._render_once()

    assert renderer.hit_test(40, 40) == HitTarget(
        "back",
        None,
        renderer._displayed_layout_generation,
    )


def test_menu_page_targets_follow_current_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prev/next page buttons are omitted at their page boundaries."""
    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()
    prev_x = _PREV_PAGE_BUTTON_RECT[0] + 10
    next_x = _NEXT_PAGE_BUTTON_RECT[0] + 10
    button_y = _NEXT_PAGE_BUTTON_RECT[1] + 10

    renderer.show_history_menu(["첫째"], 0, "제목", page_index=0, page_count=2)
    renderer._render_once()
    generation = renderer._displayed_layout_generation

    assert renderer.hit_test(prev_x, button_y) is None
    assert renderer.hit_test(next_x, button_y) == HitTarget("next_page", None, generation)

    renderer.show_history_menu(["다섯째"], 0, "제목", page_index=1, page_count=2)
    renderer._render_once()
    generation = renderer._displayed_layout_generation

    assert renderer.hit_test(prev_x, button_y) == HitTarget("prev_page", None, generation)
    assert renderer.hit_test(next_x, button_y) is None


def test_show_text_menu_renders_page_targets_for_two_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A generic two-page menu exposes page navigation hit targets."""
    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()

    renderer.show_text(
        ["첫째", "둘째", "셋째", "넷째"],
        size=34,
        title="제목",
        layout="menu",
        page_index=0,
        page_count=2,
    )
    renderer._render_once()
    generation = renderer._displayed_layout_generation
    next_x = _NEXT_PAGE_BUTTON_RECT[0] + (_NEXT_PAGE_BUTTON_RECT[2] // 2)
    button_y = _NEXT_PAGE_BUTTON_RECT[1] + (_NEXT_PAGE_BUTTON_RECT[3] // 2)

    assert renderer.hit_test(next_x, button_y) == HitTarget("next_page", None, generation)

    renderer.show_text(
        ["다섯째", "여섯째"],
        size=34,
        title="제목",
        layout="menu",
        page_index=1,
        page_count=2,
    )
    renderer._render_once()
    generation = renderer._displayed_layout_generation
    prev_x = _PREV_PAGE_BUTTON_RECT[0] + (_PREV_PAGE_BUTTON_RECT[2] // 2)

    assert renderer.hit_test(prev_x, button_y) == HitTarget("prev_page", None, generation)


def _inflated_touch_rect(rect: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    """Return ``rect`` inflated by ``pad`` per side, clipped to the canvas."""
    left = max(0, rect[0] - pad)
    top = max(0, rect[1] - pad)
    right = min(_CANVAS_SIZE[0], rect[0] + rect[2] + pad)
    bottom = min(_CANVAS_SIZE[1], rect[1] + rect[3] + pad)
    return (left, top, right - left, bottom - top)


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Return whether two ``(x, y, w, h)`` rects share any interior area."""
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    inter_w = min(ax1, bx1) - max(ax0, bx0)
    inter_h = min(ay1, by1) - max(ay0, by0)
    return inter_w > 0 and inter_h > 0


def test_page_nav_buttons_are_large_side_placed_and_non_overlapping() -> None:
    """Enlarged page buttons sit at the sides with generous, separated touch areas."""
    prev_touch = _inflated_touch_rect(_PREV_PAGE_BUTTON_RECT, _PAGE_NAV_TOUCH_PAD)
    next_touch = _inflated_touch_rect(_NEXT_PAGE_BUTTON_RECT, _PAGE_NAV_TOUCH_PAD)

    # Each touch area is comfortably above the child-friendly first-try target.
    assert prev_touch[2] >= 120 and prev_touch[3] >= 72
    assert next_touch[2] >= 120 and next_touch[3] >= 72

    # No overlap between the two buttons or with the centered indicator.
    assert not _rects_overlap(prev_touch, next_touch)
    assert not _rects_overlap(prev_touch, _PAGE_INDICATOR_RECT)
    assert not _rects_overlap(next_touch, _PAGE_INDICATOR_RECT)

    # All touch areas stay inside the active-area horizontal safe band.
    for touch in (prev_touch, next_touch):
        assert touch[0] >= 8
        assert touch[0] + touch[2] <= 712

    # Buttons are pushed to the screen sides, not clustered in the center.
    assert _PREV_PAGE_BUTTON_RECT[0] < 200
    assert _NEXT_PAGE_BUTTON_RECT[0] > _CANVAS_SIZE[0] - 200

    # The indicator stays centered between the two buttons.
    indicator_center = _PAGE_INDICATOR_RECT[0] + _PAGE_INDICATOR_RECT[2] // 2
    assert abs(indicator_center - _CANVAS_SIZE[0] // 2) <= 1
    prev_right = _PREV_PAGE_BUTTON_RECT[0] + _PREV_PAGE_BUTTON_RECT[2]
    assert prev_right < _PAGE_INDICATOR_RECT[0]
    assert _PAGE_INDICATOR_RECT[0] + _PAGE_INDICATOR_RECT[2] < _NEXT_PAGE_BUTTON_RECT[0]


def test_menu_page_targets_have_safe_visual_rect_and_padded_touch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Page controls sit inside the safe area and accept near-edge touches."""
    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()
    above_visual_y = _PAGE_BUTTON_Y + 4
    prev_x = _PREV_PAGE_BUTTON_RECT[0] + (_PREV_PAGE_BUTTON_RECT[2] // 2)
    next_x = _NEXT_PAGE_BUTTON_RECT[0] + (_NEXT_PAGE_BUTTON_RECT[2] // 2)

    assert _PAGE_BUTTON_SIZE == (132, 76)
    assert _PAGE_BUTTON_Y == 612
    assert _PAGE_NAV_TOUCH_PAD == 16
    assert _PAGE_BUTTON_Y + _PAGE_BUTTON_SIZE[1] <= _SAFE_BOTTOM
    assert _PAGE_INDICATOR_RECT[1] + _PAGE_INDICATOR_RECT[3] <= _SAFE_BOTTOM

    renderer.show_history_menu(["첫째"], 0, "제목", page_index=0, page_count=2)
    renderer._render_once()
    generation = renderer._displayed_layout_generation

    assert renderer.hit_test(next_x, above_visual_y) == HitTarget("next_page", None, generation)

    renderer.show_history_menu(["다섯째"], 0, "제목", page_index=1, page_count=2)
    renderer._render_once()
    generation = renderer._displayed_layout_generation

    assert renderer.hit_test(prev_x, above_visual_y) == HitTarget("prev_page", None, generation)


def test_history_menu_generation_updates_and_clear_removes_stale_hits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hit regions are tied to the displayed layout and cleared by pending renders."""
    _init_dummy_display(monkeypatch)
    image_path = tmp_path / "green.png"
    _write_rgba_png(image_path, (0, 255, 0, 255))
    renderer = _make_surface_renderer()

    renderer.show_history_menu(["첫째"], 0, "제목")
    renderer._render_once()
    first_target = renderer.hit_test(60, 180)
    assert first_target is not None
    first_generation = first_target.generation

    renderer.show_history_menu(["둘째"], 0, "제목")
    renderer._render_once()
    second_target = renderer.hit_test(60, 180)
    assert second_target is not None
    assert second_target.generation != first_generation

    renderer.show_image(image_path)
    renderer._render_once()

    assert renderer.hit_test(60, 180) is None
    assert renderer.hit_test(540, 40) is None


def test_press_feedback_rerenders_pressed_menu_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pressed feedback redraws the current menu item in a darker inset style."""
    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()
    renderer.show_history_menu(["고조선"], 0, "제목")
    renderer._render_once()
    target = renderer.hit_test(60, 180)
    assert target is not None
    before = renderer._display.get_at((56, 168))

    renderer._pressed_hit_target = target
    renderer._render_once()
    after = renderer._display.get_at((56, 168))

    assert after != before


def test_press_feedback_token_renders_before_follow_up_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A feedback token acknowledges the pressed frame before a new layout replaces it."""
    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()
    renderer.show_history_menu(["고조선"], 0, "제목")
    renderer._render_once()
    target = renderer.hit_test(60, 180)
    assert target is not None

    token = renderer.show_press_feedback(target, duration=0.5)

    assert token is not None
    assert renderer.wait_until_rendered(token, timeout=0.0) is False
    renderer._render_once()
    assert renderer.wait_until_rendered(token, timeout=0.0) is True

    renderer.show_history_menu(["삼국"], 0, "다음")
    renderer._render_once()

    assert renderer.hit_test(60, 180) == HitTarget(
        "menu_item",
        0,
        renderer._displayed_layout_generation,
    )


def test_text_wrapping_breaks_long_tokens_to_safe_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Long text is hard-wrapped so rendered parts fit the safe text width."""
    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()
    font = renderer._get_font(34)
    max_width = 240

    parts = renderer._wrap_text("가나다라마바사아자차카타파하" * 4, font, max_width)

    assert parts
    assert all(font.size(part)[0] <= max_width for part in parts)


def test_card_text_renders_long_caption_without_panel_overflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Long card captions render through the shrink/clip path without errors."""
    _init_dummy_display(monkeypatch)
    image_path = tmp_path / "blue.png"
    _write_rgba_png(image_path, (0, 0, 255, 255), size=(200, 200))
    renderer = _make_surface_renderer()

    token = renderer.show_card(
        image_path=image_path,
        lines=["긴 설명 " * 30],
        title="아주 긴 제목 " * 10,
        sublabel="문서 제목",
        show_exit_button=True,
    )
    renderer._render_once()

    assert token is not None
    assert renderer.wait_until_rendered(token, timeout=0.0) is True
    assert renderer.hit_test(540, 40) == HitTarget(
        "exit",
        None,
        renderer._displayed_layout_generation,
    )


def test_history_card_layout_rects_match_ux_spec() -> None:
    """History cards reserve the spec regions for header, image, caption, and exit."""
    assert _BACK_BUTTON_RECT == (20, 20, 190, 64)
    assert _EXIT_BUTTON_RECT == (510, 20, 190, 64)
    assert _BACK_BUTTON_RECT[2:] == _EXIT_BUTTON_RECT[2:]
    assert _BACK_BUTTON_RECT[0] == _CANVAS_SIZE[0] - (_EXIT_BUTTON_RECT[0] + _EXIT_BUTTON_RECT[2])
    assert _CARD_IMAGE_RECT == (20, 88, 680, 508)
    assert _CARD_IMAGE_RECT_NO_HEADER == (20, 20, 680, 584)
    assert _CARD_CAPTION_RECT == (20, 612, 680, 92)
    assert _CARD_HEADER_RECT == (234, 20, 252, 44)
    assert _CARD_CAPTION_RECT[3] == 92

    old_image_area = 620 * 388
    new_image_area = _CARD_IMAGE_RECT[2] * _CARD_IMAGE_RECT[3]
    assert new_image_area == 345_440
    assert new_image_area > old_image_area * 1.4
    assert _CARD_IMAGE_RECT_NO_HEADER[1] == 20
    assert _CARD_IMAGE_RECT_NO_HEADER[1] + _CARD_IMAGE_RECT_NO_HEADER[3] == 604
    assert (
        _CARD_CAPTION_RECT[1] - (_CARD_IMAGE_RECT_NO_HEADER[1] + _CARD_IMAGE_RECT_NO_HEADER[3]) == 8
    )


def test_top_title_span_is_centered_between_equal_nav_buttons() -> None:
    """Top titles stay centered in the symmetric back/exit gap and clamp inside it."""
    span_left, span_right = _top_title_span(
        has_back=True,
        show_exit_button=True,
        base_left=24,
    )
    span_width = span_right - span_left

    assert span_left == _BACK_BUTTON_RECT[0] + _BACK_BUTTON_RECT[2] + 24
    assert span_right == _EXIT_BUTTON_RECT[0] - 24
    assert span_left + span_right == _CANVAS_SIZE[0]
    assert _centered_title_x(120, span_left, span_right) == (_CANVAS_SIZE[0] - 120) // 2
    assert _centered_title_x(span_width, span_left, span_right) == span_left


def test_menu_title_renders_centered_between_nav_buttons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Menu title pixels land at the centered title x when both nav buttons are visible."""
    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()
    title = "Funny English"
    title_color = (23, 45, 67, 255)
    original_fit_font = renderer._fit_font

    class SolidFont:
        """Deterministic title font for menu title placement checks."""

        def size(self, text: str) -> tuple[int, int]:
            """Return deterministic text bounds."""
            return len(text) * 12, 20

        def render(
            self,
            text: str,
            antialias: bool,
            color: tuple[int, int, int],
        ) -> pygame.Surface:
            """Render a solid title block."""
            del antialias, color
            width, height = self.size(text)
            surface = pygame.Surface((width, height), pygame.SRCALPHA)
            surface.fill(title_color)
            return surface

    solid_font = SolidFont()
    span_left, span_right = _top_title_span(
        has_back=True,
        show_exit_button=True,
        base_left=_MENU_LEFT,
    )
    expected_x = _centered_title_x(solid_font.size(title)[0], span_left, span_right)

    def fake_fit_font(text: str, base_size: int, max_width: int) -> Any:
        if text == title and base_size == 38:
            assert max_width == span_right - span_left
            return solid_font
        return original_fit_font(text, base_size, max_width)

    monkeypatch.setattr(renderer, "_fit_font", fake_fit_font)

    renderer.show_text(
        ["Stage 0"],
        size=34,
        title=title,
        layout="menu",
        show_exit_button=True,
        has_back=True,
    )
    renderer._render_once()

    assert renderer._display.get_at((expected_x, 60)) == title_color


def test_center_title_renders_centered_between_nav_buttons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Center-layout title pixels use the centered top-title path."""
    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()
    title = "Funny English"
    title_color = (67, 45, 23, 255)
    original_fit_font = renderer._fit_font

    class SolidFont:
        """Deterministic title font for center title placement checks."""

        def size(self, text: str) -> tuple[int, int]:
            """Return deterministic text bounds."""
            return len(text) * 12, 20

        def render(
            self,
            text: str,
            antialias: bool,
            color: tuple[int, int, int],
        ) -> pygame.Surface:
            """Render a solid title block."""
            del antialias, color
            width, height = self.size(text)
            surface = pygame.Surface((width, height), pygame.SRCALPHA)
            surface.fill(title_color)
            return surface

    solid_font = SolidFont()
    span_left, span_right = _top_title_span(
        has_back=True,
        show_exit_button=True,
        base_left=_TEXT_SAFE_TOP_RIGHT_MARGIN,
    )
    expected_x = _centered_title_x(solid_font.size(title)[0], span_left, span_right)

    def fake_fit_font(text: str, base_size: int, max_width: int) -> Any:
        if text == title and base_size == 34:
            assert max_width == span_right - span_left
            return solid_font
        return original_fit_font(text, base_size, max_width)

    monkeypatch.setattr(renderer, "_fit_font", fake_fit_font)

    renderer.show_text(
        ["Ready"],
        size=40,
        title=title,
        layout="center",
        show_exit_button=True,
        has_back=True,
    )
    renderer._render_once()

    assert renderer._display.get_at((expected_x, 60)) == title_color


def test_card_without_title_uses_no_header_image_rect_and_renders_no_title(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Title-less history cards expand the image upward and skip header text."""
    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()
    image_path = tmp_path / "blue.png"
    requested_rects: list[tuple[int, int, int, int]] = []
    title_render_calls: list[str] = []

    def fake_render_card_image(image_path: Path, target_rect: Any) -> None:
        del image_path
        requested_rects.append(
            (target_rect.left, target_rect.top, target_rect.width, target_rect.height)
        )

    original_fit_font = renderer._fit_font

    def fail_title_render(text: str, base_size: int, max_width: int) -> Any:
        if (base_size, max_width) == (30, _EXIT_BUTTON_RECT[2] - 28):
            return original_fit_font(text, base_size, max_width)
        title_render_calls.append(text)
        raise AssertionError("title font fitting should not run when title is absent")

    monkeypatch.setattr(renderer, "_render_card_image", fake_render_card_image)
    monkeypatch.setattr(renderer, "_fit_font", fail_title_render)

    request = _CardRequest(
        image_path=image_path,
        lines=(),
        highlight_index=None,
        title=None,
        sublabel=None,
        token="card-test",
        show_exit_button=True,
        show_back=False,
        show_prev_card=False,
        show_next_card=False,
    )

    hit_regions = renderer._render_card_request(request, layout_generation=1, pressed_target=None)
    renderer._hit_regions = tuple(hit_regions)
    renderer._displayed_layout_generation = 1

    assert requested_rects == [_CARD_IMAGE_RECT_NO_HEADER]
    assert title_render_calls == []
    assert len(hit_regions) == 1
    assert isinstance(hit_regions[0], _HitRegion)
    assert hit_regions[0].target == HitTarget("exit", None, 1)
    left, top, width, height = hit_regions[0].rect
    assert left <= 540 < left + width
    assert top <= 40 < top + height
    assert renderer.hit_test(540, 40) == HitTarget("exit", None, 1)


def test_card_with_title_uses_standard_image_rect_and_renders_title(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Title-present card path keeps the shared Funny English header layout."""
    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()
    image_path = tmp_path / "blue.png"
    requested_rects: list[tuple[int, int, int, int]] = []
    title_render_calls: list[str] = []

    def fake_render_card_image(image_path: Path, target_rect: Any) -> None:
        del image_path
        requested_rects.append(
            (target_rect.left, target_rect.top, target_rect.width, target_rect.height)
        )

    original_fit_font = renderer._fit_font

    def spy_fit_font(text: str, base_size: int, max_width: int) -> Any:
        if text == "Funny English":
            title_render_calls.append(text)
        return original_fit_font(text, base_size, max_width)

    monkeypatch.setattr(renderer, "_render_card_image", fake_render_card_image)
    monkeypatch.setattr(renderer, "_fit_font", spy_fit_font)

    request = _CardRequest(
        image_path=image_path,
        lines=(),
        highlight_index=None,
        title="Funny English",
        sublabel=None,
        token="card-test",
        show_exit_button=False,
        show_back=False,
        show_prev_card=False,
        show_next_card=False,
    )

    renderer._render_card_request(request, layout_generation=1, pressed_target=None)

    assert requested_rects == [_CARD_IMAGE_RECT]
    assert title_render_calls == ["Funny English"]


def test_history_card_header_rect_stays_left_of_exit_button() -> None:
    """The header chip has a hard layout gap before the exit button."""
    header_right = _CARD_HEADER_RECT[0] + _CARD_HEADER_RECT[2]

    assert header_right < _EXIT_BUTTON_RECT[0]


def test_card_caption_fit_uses_floor_and_two_line_clip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Long card captions shrink to the 30px floor and render at most two lines."""
    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()
    rendered_sizes: list[int] = []

    class FakeFont:
        """Font double exposing the subset of pygame font APIs used here."""

        def __init__(self, size: int) -> None:
            self._size = size
            self._height = 32 if size == 30 else size + 1

        def size(self, text: str) -> tuple[int, int]:
            """Return deterministic text bounds."""
            return (min(640, max(1, len(text) * 8)), self._height)

        def render(
            self,
            text: str,
            antialias: bool,
            color: tuple[int, int, int],
        ) -> pygame.Surface:
            """Return a surface and record the font size used for actual drawing."""
            del antialias
            rendered_sizes.append(self._size)
            width, height = self.size(text)
            surface = pygame.Surface((width, height), pygame.SRCALPHA)
            surface.fill((*color, 255))
            return surface

    def fake_get_font(size: int) -> FakeFont:
        return FakeFont(size)

    def fake_wrap_text(text: str, font: Any, max_width: int) -> list[str]:
        del text, font, max_width
        return ["첫 줄", "둘째 줄", "잘리는 줄"]

    monkeypatch.setattr(renderer, "_get_font", fake_get_font)
    monkeypatch.setattr(renderer, "_wrap_text", fake_wrap_text)

    panel_rect = pygame.Rect(*_CARD_CAPTION_RECT)
    request = _CardRequest(
        image_path=None,
        lines=("긴 설명 " * 40,),
        highlight_index=None,
        title="고조선",
        sublabel=None,
        token="card-test",
        show_exit_button=True,
        show_back=False,
        show_prev_card=False,
        show_next_card=False,
    )

    renderer._render_card_text(request, panel_rect)

    assert rendered_sizes == [30, 30]
    assert len(rendered_sizes) <= 2


def test_truncate_to_single_line_clamps_long_text_with_ellipsis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long title is clamped to one line that fits the reserved width."""
    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()
    font = renderer._get_font(26)
    max_width = _CARD_HEADER_RECT[2]

    long_title = "아주 길고 긴 우리 역사 이야기 제목 " * 6
    clamped = renderer._truncate_to_single_line(long_title, font, max_width)

    assert clamped.endswith("…")
    assert font.size(clamped)[0] <= max_width

    short_title = "고조선"
    assert renderer._truncate_to_single_line(short_title, font, max_width) == short_title


def test_card_title_with_exit_button_stays_left_of_exit_button(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long card title never bleeds into the top-right exit button column."""
    _init_dummy_display(monkeypatch)
    image_path = tmp_path / "blue.png"
    _write_rgba_png(image_path, (0, 0, 255, 255), size=(200, 200))
    renderer = _make_surface_renderer()

    title_left, title_right_limit = _top_title_span(
        has_back=False,
        show_exit_button=True,
        base_left=24,
    )
    title_width = title_right_limit - title_left
    assert title_right_limit < _EXIT_BUTTON_RECT[0]

    long_title = "아주 길고 긴 우리 역사 이야기 제목 " * 8
    renderer.show_card(
        image_path=image_path,
        lines=["설명"],
        title=long_title,
        show_exit_button=True,
    )
    renderer._render_once()

    # The title text band sits in the header chip and stays above the image region.
    title_y = _CARD_HEADER_RECT[1]
    title_font = renderer._fit_font(long_title, 26, title_width)
    title_line_height = title_font.get_height()
    assert title_y + title_line_height < _CARD_IMAGE_RECT[1]
    title_band = range(title_y, title_y + title_line_height)
    # The strip between the allowed title right edge and the exit button must stay
    # background-colored, i.e. no title glyph pixels overflowed past title_width.
    background = (*_TEXT_BG, 255)
    for x in range(title_right_limit, _EXIT_BUTTON_RECT[0]):
        for y in title_band:
            assert renderer._display.get_at((x, y)) == background


def test_card_title_with_back_button_starts_right_of_back_button(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A card title is centered in the inter-button span when back is shown."""
    _init_dummy_display(monkeypatch)
    image_path = tmp_path / "blue.png"
    _write_rgba_png(image_path, (0, 0, 255, 255), size=(200, 200))
    renderer = _make_surface_renderer()
    title = "Funny English Stage"
    title_color = (123, 45, 67, 255)
    original_fit_font = renderer._fit_font
    title_left, title_right = _top_title_span(
        has_back=True,
        show_exit_button=True,
        base_left=24,
    )
    expected_width = title_right - title_left

    class SolidFont:
        """Deterministic title font that makes rendered title pixels easy to inspect."""

        def size(self, text: str) -> tuple[int, int]:
            """Return deterministic text bounds."""
            return min(expected_width, len(text) * 12), 20

        def render(
            self,
            text: str,
            antialias: bool,
            color: tuple[int, int, int],
        ) -> pygame.Surface:
            """Render a solid title block."""
            del antialias, color
            width, height = self.size(text)
            surface = pygame.Surface((width, height), pygame.SRCALPHA)
            surface.fill(title_color)
            return surface

    solid_font = SolidFont()
    rendered_width = solid_font.size(title)[0]
    expected_title_x = _centered_title_x(rendered_width, title_left, title_right)

    def fake_fit_font(text: str, base_size: int, max_width: int) -> Any:
        if text == title and base_size == 26:
            assert max_width == expected_width
            return solid_font
        return original_fit_font(text, base_size, max_width)

    monkeypatch.setattr(renderer, "_fit_font", fake_fit_font)

    renderer.show_card(
        image_path=image_path,
        lines=["cat"],
        title=title,
        show_exit_button=True,
        show_back=True,
    )
    renderer._render_once()

    title_y = _CARD_HEADER_RECT[1] + 2
    background = (*_TEXT_BG, 255)
    for x in range(_BACK_BUTTON_RECT[0] + _BACK_BUTTON_RECT[2], expected_title_x):
        assert renderer._display.get_at((x, title_y)) == background
    assert renderer._display.get_at((expected_title_x, title_y)) == title_color


def test_scene_exit_target_stays_hittable_across_animation_only_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exit target survives animation-only re-renders during a SPEAKING scene."""
    _init_dummy_display(monkeypatch)
    image_path = tmp_path / "blue.png"
    _write_rgba_png(image_path, (0, 0, 255, 255), size=(200, 200))
    renderer = _make_surface_renderer()

    renderer.show_card(
        image_path=image_path,
        lines=["설명"],
        title=None,
        show_exit_button=True,
    )
    renderer._render_once()

    exit_target = renderer.hit_test(540, 40)
    assert exit_target is not None
    scene_generation = renderer._displayed_layout_generation

    # Simulate animation-only re-renders (SPEAKING frames, no new layout request).
    for expression in (
        CharacterExpression.SPEAKING,
        CharacterExpression.SPEAKING,
        CharacterExpression.SPEAKING,
    ):
        renderer._pending_expression = expression
        renderer._render_once()
        # Layout generation must stay stable so the exit target remains valid.
        assert renderer._displayed_layout_generation == scene_generation
        still_hittable = renderer.hit_test(540, 40)
        assert still_hittable == HitTarget("exit", None, scene_generation)

    # The exit target captured before any animation frame is still current, so the
    # show_press_feedback generation gate passes (it is not early-returned on a
    # generation mismatch) and the pressed style is queued.
    assert exit_target.generation == renderer._displayed_layout_generation
    renderer.show_press_feedback(exit_target, duration=0.0)
    assert renderer._pressed_hit_target is None  # cleared after the (zero) duration

    # Re-render with the pressed target set: the exit button must redraw in the
    # darker pressed style, proving feedback is actually applied.
    before = renderer._display.get_at((540, 40))
    renderer._pressed_hit_target = exit_target
    renderer._render_once()
    after = renderer._display.get_at((540, 40))
    assert after != before
    assert renderer._displayed_layout_generation == scene_generation


def test_layout_change_clears_stale_hit_regions_and_bumps_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuine layout change still mints a new generation and drops old hit regions."""
    _init_dummy_display(monkeypatch)
    image_path = tmp_path / "green.png"
    _write_rgba_png(image_path, (0, 255, 0, 255))
    renderer = _make_surface_renderer()

    renderer.show_card(
        image_path=image_path,
        lines=["설명"],
        title=None,
        show_exit_button=True,
    )
    renderer._render_once()
    first_target = renderer.hit_test(540, 40)
    assert first_target is not None
    first_generation = first_target.generation

    # Animation-only frame keeps the generation stable.
    renderer._pending_expression = CharacterExpression.SPEAKING
    renderer._render_once()
    assert renderer._displayed_layout_generation == first_generation

    # A real new card is a layout change: generation advances and the old target is
    # no longer current.
    renderer.show_card(
        image_path=image_path,
        lines=["다른 설명"],
        title=None,
        show_exit_button=True,
    )
    renderer._render_once()
    second_target = renderer.hit_test(540, 40)
    assert second_target is not None
    assert second_target.generation != first_generation

    # Switching to a full-screen image clears all direct-touch hit regions.
    renderer.show_image(image_path)
    renderer._render_once()
    assert renderer.hit_test(540, 40) is None


def test_missing_sprites_are_stored_as_none_and_logged(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    asset_dir = _create_sprites(tmp_path, [CharacterExpression.IDLE])

    renderer = PygameCharacterRenderer(asset_dir, sdl_driver="dummy")
    try:
        assert renderer._sprites[CharacterExpression.IDLE] is not None
        missing = [expr for expr in CharacterExpression if expr is not CharacterExpression.IDLE]
        assert all(renderer._sprites[expr] is None for expr in missing)
        assert "Missing character sprites (using NEUTRAL fallback)" in caplog.text
    finally:
        renderer.close()

    caplog.clear()
    invalid_asset_dir = tmp_path / "invalid"
    invalid_asset_dir.mkdir()
    (invalid_asset_dir / f"{CharacterExpression.NEUTRAL.value}.png").write_bytes(b"not a png")
    invalid_renderer = PygameCharacterRenderer(invalid_asset_dir, sdl_driver="dummy")
    try:
        assert invalid_renderer._sprites[CharacterExpression.NEUTRAL] is None
        assert "Failed to load sprite" in caplog.text
    finally:
        invalid_renderer.close()


def test_empty_asset_dir_falls_back_to_solid_black_neutral(tmp_path: Path) -> None:
    asset_dir = tmp_path / "empty"
    asset_dir.mkdir()
    renderer = PygameCharacterRenderer(asset_dir, sdl_driver="dummy")
    try:
        renderer.on_expression(CharacterExpression.LISTENING)
        _wait_for_expression(renderer, CharacterExpression.NEUTRAL)
        assert renderer._black_fallback is not None
    finally:
        renderer.close()


def test_missing_requested_expression_reports_effective_neutral(tmp_path: Path) -> None:
    asset_dir = _create_sprites(tmp_path, [CharacterExpression.NEUTRAL])
    renderer = PygameCharacterRenderer(asset_dir, sdl_driver="dummy")
    try:
        renderer.on_expression(CharacterExpression.LISTENING)
        _wait_for_expression(renderer, CharacterExpression.NEUTRAL)
    finally:
        renderer.close()


def test_init_discovers_frame_paths_without_loading_frame_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset_dir = _create_sprites(tmp_path)
    frame_dir = asset_dir / "frames" / CharacterExpression.LISTENING.value
    frame_dir.mkdir(parents=True)
    frame_paths = [frame_dir / "000.png", frame_dir / "001.png"]
    for frame_path in frame_paths:
        _write_rgba_png(frame_path, (255, 0, 0, 255), size=(2, 2))

    original_load = pygame.image.load
    loaded_paths: list[Path] = []

    def record_load(path: str | Path) -> pygame.Surface:
        loaded_paths.append(Path(path))
        return original_load(str(path))

    monkeypatch.setattr(pygame.image, "load", record_load)

    renderer = PygameCharacterRenderer(asset_dir, sdl_driver="dummy")
    try:
        static_paths = {asset_dir / f"{expression.value}.png" for expression in CharacterExpression}
        assert set(loaded_paths) == static_paths
        assert all(frame_path not in loaded_paths for frame_path in frame_paths)
        assert renderer._frame_paths[CharacterExpression.LISTENING] == frame_paths
        assert renderer._active_frames is None
        assert renderer._active_frames_expr is None
    finally:
        renderer.close()


def test_animation_frames_load_lazily_cycle_and_switch_single_active_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset_dir = _create_sprites(
        tmp_path,
        [CharacterExpression.NEUTRAL, CharacterExpression.LISTENING, CharacterExpression.HAPPY],
    )
    listening_frame_dir = asset_dir / "frames" / CharacterExpression.LISTENING.value
    listening_frame_dir.mkdir(parents=True)
    listening_frame_paths = [
        listening_frame_dir / "000.png",
        listening_frame_dir / "001.png",
    ]
    _write_rgba_png(listening_frame_paths[0], (255, 0, 0, 255), size=(2, 2))
    _write_rgba_png(listening_frame_paths[1], (0, 0, 255, 255), size=(2, 2))

    happy_frame_dir = asset_dir / "frames" / CharacterExpression.HAPPY.value
    happy_frame_dir.mkdir(parents=True)
    happy_frame_paths = [
        happy_frame_dir / "000.png",
        happy_frame_dir / "001.png",
    ]
    _write_rgba_png(happy_frame_paths[0], (0, 255, 0, 255), size=(2, 2))
    _write_rgba_png(happy_frame_paths[1], (255, 255, 0, 255), size=(2, 2))

    original_load = pygame.image.load
    loaded_paths: list[Path] = []

    def record_load(path: str | Path) -> pygame.Surface:
        loaded_paths.append(Path(path))
        return original_load(str(path))

    monkeypatch.setattr(pygame.image, "load", record_load)

    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()
    try:
        renderer._load_sprites(asset_dir)
        renderer._discover_frame_sequences(asset_dir)
        loaded_paths.clear()
        assert renderer._active_frames is None

        renderer._pending_expression = CharacterExpression.LISTENING
        renderer._render_once()
        assert renderer._display.get_at((0, 0)) == (255, 0, 0, 255)
        assert renderer.current_expression == CharacterExpression.LISTENING
        assert loaded_paths == listening_frame_paths
        assert renderer._active_frames_expr is CharacterExpression.LISTENING
        listening_frames = renderer._active_frames
        assert listening_frames is not None
        assert len(listening_frames) == 2
        assert renderer._frame_index == 1

        renderer._render_once()
        assert renderer._display.get_at((0, 0)) == (0, 0, 255, 255)
        assert renderer.current_expression == CharacterExpression.LISTENING
        assert renderer._frame_index == 2

        renderer._render_once()
        assert renderer._display.get_at((0, 0)) == (255, 0, 0, 255)
        assert renderer._frame_index == 3

        renderer._pending_expression = CharacterExpression.HAPPY
        renderer._render_once()
        assert renderer._display.get_at((0, 0)) == (0, 255, 0, 255)
        assert renderer.current_expression == CharacterExpression.HAPPY
        assert loaded_paths == [*listening_frame_paths, *happy_frame_paths]
        assert renderer._active_frames_expr is CharacterExpression.HAPPY
        assert renderer._active_frames is not listening_frames
        assert len(renderer._active_frames or []) == 2
        assert renderer._frame_index == 1
    finally:
        pygame.quit()


def test_static_sprite_renders_when_no_frame_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset_dir = _create_sprites(
        tmp_path, [CharacterExpression.NEUTRAL, CharacterExpression.LISTENING]
    )

    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()
    try:
        renderer._load_sprites(asset_dir)
        renderer._discover_frame_sequences(asset_dir)
        assert CharacterExpression.LISTENING not in renderer._frame_paths

        renderer._pending_expression = CharacterExpression.LISTENING
        renderer._render_once()

        assert renderer._display.get_at((0, 0)) == (44, 44, 44, 255)
        assert renderer.current_expression == CharacterExpression.LISTENING
        assert renderer._active_frames is None
        assert renderer._active_frames_expr is None
        assert renderer._frame_index == 0
    finally:
        pygame.quit()


def test_language_indicator_blits_inside_top_right_bezel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Language badge should stay in the bezel and outside the central character zone."""
    asset_dir = _create_sprites(tmp_path, [CharacterExpression.NEUTRAL])
    indicator_dir = asset_dir / "indicator"
    indicator_dir.mkdir()
    _write_rgba_png(indicator_dir / "flag_ko.png", (255, 0, 0, 255), size=(48, 48))

    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()
    try:
        renderer._load_sprites(asset_dir)
        renderer._load_language_indicators(asset_dir)
        renderer._pending_expression = CharacterExpression.NEUTRAL
        renderer._pending_language = "ko"

        renderer._render_once()

        assert renderer._display.get_at((690, 30)) == (255, 0, 0, 255)
        assert renderer._display.get_at((659, 60)) == (24, 24, 24, 255)
        assert renderer._current_language_value == "ko"
    finally:
        pygame.quit()


def test_missing_language_indicator_assets_are_tolerated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing language badges should render generated fallback indicators."""
    asset_dir = _create_sprites(tmp_path, [CharacterExpression.NEUTRAL])

    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()
    try:
        renderer._load_sprites(asset_dir)
        renderer._load_language_indicators(asset_dir)
        assert renderer._language_indicators["ko"] is not None
        renderer._pending_expression = CharacterExpression.NEUTRAL
        renderer._pending_language = "ko"

        renderer._render_once()

        assert renderer._display.get_at((690, 30)) != (24, 24, 24, 255)
        assert renderer._current_language_value == "ko"
        assert "Missing language indicator" in caplog.text
    finally:
        pygame.quit()


def test_first_language_indicator_set_does_not_animate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cold-wake initial language badge set should not start the pulse animation."""
    asset_dir = _create_sprites(tmp_path, [CharacterExpression.NEUTRAL])
    indicator_dir = asset_dir / "indicator"
    indicator_dir.mkdir()
    _write_rgba_png(indicator_dir / "flag_ko.png", (255, 0, 0, 255), size=(48, 48))

    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()
    try:
        renderer._load_sprites(asset_dir)
        renderer._load_language_indicators(asset_dir)
        renderer._pending_expression = CharacterExpression.NEUTRAL
        renderer._pending_language = "ko"

        renderer._render_once()

        assert renderer._current_language_value == "ko"
        assert renderer._indicator_anim_frames == 0
        assert not renderer._has_frame_to_render()
    finally:
        pygame.quit()


def test_real_language_indicator_change_animates_then_settles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real language change keeps the render loop awake until the pulse ends."""
    asset_dir = _create_sprites(tmp_path, [CharacterExpression.NEUTRAL])
    indicator_dir = asset_dir / "indicator"
    indicator_dir.mkdir()
    _write_rgba_png(indicator_dir / "flag_ko.png", (255, 0, 0, 255), size=(48, 48))
    _write_rgba_png(indicator_dir / "flag_en.png", (0, 0, 255, 255), size=(48, 48))

    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()
    try:
        renderer._load_sprites(asset_dir)
        renderer._load_language_indicators(asset_dir)
        renderer._pending_expression = CharacterExpression.NEUTRAL
        renderer._pending_language = "ko"
        renderer._render_once()
        assert renderer._indicator_anim_frames == 0

        renderer._pending_language = "en"
        assert renderer._has_frame_to_render()
        renderer._render_once()

        assert renderer._current_language_value == "en"
        assert renderer._indicator_anim_frames == _INDICATOR_ANIM_TOTAL_FRAMES - 1
        remaining_frames = _INDICATOR_ANIM_TOTAL_FRAMES - 1
        for _index in range(remaining_frames):
            assert renderer._has_frame_to_render()
            renderer._render_once()

        assert renderer._indicator_anim_frames == 0
        assert not renderer._has_frame_to_render()
    finally:
        pygame.quit()


def test_on_language_change_enqueues_indicator_update() -> None:
    """Language-change callback should wake the renderer like expression updates."""
    renderer = _make_surface_renderer()

    renderer.on_language_change("en")

    assert renderer._pending_language == "en"
    assert renderer._wake.is_set()


def test_idle_uses_neutral_frame_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset_dir = _create_sprites(tmp_path, [CharacterExpression.NEUTRAL, CharacterExpression.IDLE])
    frame_dir = asset_dir / "frames" / CharacterExpression.NEUTRAL.value
    frame_dir.mkdir(parents=True)
    _write_rgba_png(frame_dir / "000.png", (0, 255, 0, 255), size=(2, 2))

    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()
    try:
        renderer._load_sprites(asset_dir)
        renderer._discover_frame_sequences(asset_dir)

        renderer._pending_expression = CharacterExpression.IDLE
        renderer._render_once()

        assert renderer._display.get_at((0, 0)) == (0, 255, 0, 255)
        assert renderer.current_expression == CharacterExpression.NEUTRAL
        assert renderer._frame_index == 1
    finally:
        pygame.quit()


def test_idle_falls_back_to_neutral_static_sprite_when_idle_asset_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset_dir = _create_sprites(tmp_path, [CharacterExpression.NEUTRAL])

    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()
    try:
        renderer._load_sprites(asset_dir)
        renderer._discover_frame_sequences(asset_dir)

        renderer._pending_expression = CharacterExpression.IDLE
        renderer._render_once()

        assert renderer._display.get_at((0, 0)) == (24, 24, 24, 255)
        assert renderer.current_expression == CharacterExpression.NEUTRAL
        assert renderer._active_frames is None
        assert renderer._active_frames_expr is None
    finally:
        pygame.quit()


def test_failed_frame_load_logs_once_and_uses_static_sprite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    asset_dir = _create_sprites(
        tmp_path, [CharacterExpression.NEUTRAL, CharacterExpression.LISTENING]
    )
    frame_dir = asset_dir / "frames" / CharacterExpression.LISTENING.value
    frame_dir.mkdir(parents=True)
    _write_rgba_png(frame_dir / "000.png", (255, 0, 0, 255), size=(2, 2))
    (frame_dir / "001.png").write_bytes(b"not a png")

    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()
    try:
        renderer._load_sprites(asset_dir)
        renderer._discover_frame_sequences(asset_dir)

        renderer._pending_expression = CharacterExpression.LISTENING
        renderer._render_once()
        assert renderer._display.get_at((0, 0)) == (44, 44, 44, 255)
        assert renderer.current_expression == CharacterExpression.LISTENING
        assert renderer._active_frames is None
        assert renderer._active_frames_expr is None
        assert renderer._frame_index == 0

        renderer._pending_expression = CharacterExpression.LISTENING
        renderer._render_once()
        assert caplog.text.count("Failed to load animation frame") == 1
    finally:
        pygame.quit()


def test_on_state_change_preserves_current_expression(tmp_path: Path) -> None:
    asset_dir = _create_sprites(tmp_path)
    renderer = PygameCharacterRenderer(asset_dir, sdl_driver="dummy")
    try:
        renderer.on_expression(CharacterExpression.IDLE)
        _wait_for_expression(renderer, CharacterExpression.IDLE)
        renderer.on_state_change(SessionState.WAKING)
        time.sleep(0.05)
        assert renderer.current_expression == CharacterExpression.IDLE
    finally:
        renderer.close()


def test_close_joins_within_timeout_and_post_close_enqueue_is_noop(tmp_path: Path) -> None:
    asset_dir = _create_sprites(tmp_path)
    renderer = PygameCharacterRenderer(asset_dir, sdl_driver="dummy")
    renderer.on_expression(CharacterExpression.HAPPY)
    _wait_for_expression(renderer, CharacterExpression.HAPPY)

    start = time.monotonic()
    renderer.close()
    elapsed = time.monotonic() - start

    assert elapsed <= 2.5
    assert not renderer._ui_thread.is_alive()
    renderer.on_expression(CharacterExpression.SAD)
    assert renderer.current_expression == CharacterExpression.HAPPY


def test_close_is_idempotent(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    asset_dir = _create_sprites(tmp_path)
    renderer = PygameCharacterRenderer(asset_dir, sdl_driver="dummy")

    renderer.close()
    renderer.close()

    assert not renderer._ui_thread.is_alive()

    bare_renderer = PygameCharacterRenderer.__new__(PygameCharacterRenderer)
    bare_renderer._display = None
    with pytest.raises(RuntimeError, match="display is not initialized"):
        bare_renderer._render_once()

    class StuckThread:
        def __init__(self) -> None:
            self.join_timeout: float | None = None

        def join(self, timeout: float | None = None) -> None:
            self.join_timeout = timeout

        def is_alive(self) -> bool:
            return True

    stuck_thread = StuckThread()
    stuck_renderer = PygameCharacterRenderer.__new__(PygameCharacterRenderer)
    stuck_renderer._lock = threading.Lock()
    stuck_renderer._close_called = False
    stuck_renderer._render_events = {}
    stuck_renderer._stop_event = threading.Event()
    stuck_renderer._wake = threading.Event()
    stuck_renderer._ui_thread = stuck_thread

    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    stuck_renderer.close()

    assert stuck_thread.join_timeout == 2.5
    assert "degraded shutdown" in caplog.text


def test_init_failure_and_timeout_raise_runtime_error_and_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback_asset_dir = _create_sprites(tmp_path / "scaled-fallback")
    original_set_mode = pygame.display.set_mode
    set_mode_flags: list[int] = []

    def fail_once_then_set_mode(size: tuple[int, int], flags: int = 0) -> pygame.Surface:
        set_mode_flags.append(flags)
        if len(set_mode_flags) == 1:
            raise pygame.error("first mode failed")
        return original_set_mode(size, flags)

    monkeypatch.setattr(pygame.display, "list_modes", lambda: -1)
    monkeypatch.setattr(pygame.display, "set_mode", fail_once_then_set_mode)
    renderer = PygameCharacterRenderer(fallback_asset_dir, windowed=True, sdl_driver="dummy")
    renderer.close()
    assert len(set_mode_flags) == 2
    assert set_mode_flags[1] & pygame.SCALED
    monkeypatch.undo()

    asset_dir = _create_sprites(tmp_path / "set-mode-failure")
    original_error = pygame.error("display unavailable")

    def fail_set_mode(size: tuple[int, int], flags: int = 0) -> pygame.Surface:
        del size, flags
        raise original_error

    monkeypatch.setattr(pygame.display, "set_mode", fail_set_mode)
    with pytest.raises(RuntimeError, match="init failed") as failed_init:
        PygameCharacterRenderer(asset_dir, sdl_driver="dummy", ready_timeout=0.3)
    assert failed_init.value.__cause__ is original_error
    _wait_for_no_ui_threads()

    monkeypatch.undo()
    blocking_asset_dir = _create_sprites(tmp_path / "init-timeout")

    def block_until_stop(self: PygameCharacterRenderer, asset_dir: Path) -> None:
        del asset_dir
        self._stop_event.wait(timeout=5.0)

    monkeypatch.setattr(PygameCharacterRenderer, "_load_sprites", block_until_stop)
    with pytest.raises(RuntimeError, match="init timeout"):
        PygameCharacterRenderer(blocking_asset_dir, sdl_driver="dummy", ready_timeout=0.1)
    _wait_for_no_ui_threads()


def test_render_loop_exception_is_logged_and_next_expression_renders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    asset_dir = _create_sprites(tmp_path)
    renderer = PygameCharacterRenderer(asset_dir, sdl_driver="dummy")
    original_render_once = renderer._render_once
    injected = threading.Event()

    def raise_once_then_restore() -> None:
        if not injected.is_set():
            injected.set()
            raise RuntimeError("injected render failure")
        original_render_once()

    try:
        monkeypatch.setattr(renderer, "_render_once", raise_once_then_restore)
        renderer.on_expression(CharacterExpression.LISTENING)
        assert injected.wait(timeout=2.0)
        assert renderer._ui_thread.is_alive()
        _wait_for_log(caplog, "render loop iteration failed")

        renderer.on_expression(CharacterExpression.SURPRISED)
        _wait_for_expression(renderer, CharacterExpression.SURPRISED)
    finally:
        renderer.close()


def test_concurrent_expression_updates_and_close_do_not_deadlock(tmp_path: Path) -> None:
    asset_dir = _create_sprites(tmp_path)
    renderer = PygameCharacterRenderer(asset_dir, sdl_driver="dummy")
    barrier = threading.Barrier(6)
    errors: list[Exception] = []
    close_elapsed: list[float] = []
    expressions = list(CharacterExpression)

    def worker(worker_id: int) -> None:
        try:
            barrier.wait(timeout=5.0)
            for index in range(100):
                renderer.on_expression(expressions[(worker_id + index) % len(expressions)])
        except Exception as exc:
            errors.append(exc)

    def closer() -> None:
        try:
            barrier.wait(timeout=5.0)
            start = time.monotonic()
            renderer.close()
            close_elapsed.append(time.monotonic() - start)
        except Exception as exc:
            errors.append(exc)

    threads = [
        *[
            threading.Thread(target=worker, args=(worker_id,), name=f"renderer-worker-{worker_id}")
            for worker_id in range(5)
        ],
        threading.Thread(target=closer, name="renderer-closer"),
    ]

    for thread in threads:
        thread.start()

    deadline = time.monotonic() + 5.0
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))

    renderer.close()

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert close_elapsed
    assert close_elapsed[0] <= 2.5


def test_idle_registers_visible_top_left_portal_button(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The idle screen registers and draws the top-left portal button."""
    _init_dummy_display(monkeypatch)
    renderer = _make_surface_renderer()
    indicator = pygame.Surface((_INDICATOR_SIZE, _INDICATOR_SIZE), pygame.SRCALPHA)
    indicator.fill((255, 0, 0, 255))
    renderer._language_indicators["ko"] = indicator
    renderer.set_language_indicator("ko")
    renderer.on_state_change(SessionState.AWAITING_TAP)
    renderer._render_once()
    generation = renderer._displayed_layout_generation
    portal_center = (
        _PORTAL_HOTSPOT_POSITION[0] + _INDICATOR_SIZE // 2,
        _PORTAL_HOTSPOT_POSITION[1] + _INDICATOR_SIZE // 2,
    )
    toggle_center = (
        _INDICATOR_POSITION[0] + _INDICATOR_SIZE // 2,
        _INDICATOR_POSITION[1] + _INDICATOR_SIZE // 2,
    )
    assert renderer.hit_test(*portal_center) == HitTarget("portal_activate", None, generation)
    assert renderer._display.get_at(portal_center) != (0, 0, 0, 255)
    # The top-right language toggle is unaffected (opposite corner).
    assert renderer.hit_test(*toggle_center) == HitTarget("language_toggle", None, generation)
