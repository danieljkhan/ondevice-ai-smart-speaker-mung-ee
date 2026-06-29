"""Pygame-backed character renderer for the touchscreen display."""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from core.character_expression import CharacterExpression
from core.session_manager import CharacterRenderer, RenderHitTarget, SessionState

if TYPE_CHECKING:
    import pygame  # type: ignore[import-not-found, import-untyped]
else:
    pygame: Any | None = None

logger = logging.getLogger(__name__)

_CANVAS_SIZE = (720, 720)
_FRAME_INTERVAL_S = 1 / 12
_WAIT_EXPRESSION_CYCLE = (
    CharacterExpression.THINKING,
    CharacterExpression.HAPPY,
    CharacterExpression.JOYFUL,
    CharacterExpression.THINKING,
    CharacterExpression.EXCITED,
    CharacterExpression.SURPRISED,
    CharacterExpression.THINKING,
    CharacterExpression.GREETING,
    CharacterExpression.AFFECTIONATE,
    CharacterExpression.THINKING,
    CharacterExpression.WINKING,
    CharacterExpression.HAPPY,
)
_WAIT_EXPRESSION_CYCLE_FRAMES = max(1, round(2.5 / _FRAME_INTERVAL_S))
_DEFAULT_READY_TIMEOUT_S = 3.0
_CLOSE_JOIN_TIMEOUT_S = 2.5
_BLACK = (0, 0, 0)
_TEXT_BG = (16, 18, 22)
_TEXT_PANEL = (38, 42, 48)
_TEXT_PANEL_HIGHLIGHT = (255, 214, 102)
_TEXT_FG = (244, 246, 248)
_TEXT_MUTED = (176, 184, 192)
_TEXT_HIGHLIGHT_FG = (18, 20, 24)
_TEXT_SAFE_TOP_RIGHT_MARGIN = 72
_MENU_LEFT = 48
_MENU_WIDTH = 600
_MENU_ITEM_HEIGHT = 104
_MENU_ITEM_GAP = 14
_BACK_BUTTON_LABEL = "이전 단계"
_TOP_NAV_MARGIN = 20
_TOP_NAV_BUTTON_WIDTH = 190
_TOP_TITLE_BUTTON_GAP = 24
_BACK_BUTTON_RECT = (_TOP_NAV_MARGIN, 20, _TOP_NAV_BUTTON_WIDTH, 64)
_BACK_BUTTON_RADIUS = 8
_EXIT_BUTTON_LABEL = "나가기"
_EXIT_BUTTON_RECT = (
    _CANVAS_SIZE[0] - _TOP_NAV_MARGIN - _TOP_NAV_BUTTON_WIDTH,
    20,
    _TOP_NAV_BUTTON_WIDTH,
    64,
)
_EXIT_BUTTON_RADIUS = 8
# Touch target padding around the visual exit button (each side, px).
_EXIT_TOUCH_PAD = 20
# Page prev/next buttons are pushed to the screen sides and enlarged so a child
# hits them first-try. They mirror the generous card-nav sizing rather than the
# former small, centrally clustered chevrons that required 2-3 taps to register.
_PAGE_BUTTON_SIZE = (132, 76)
_PAGE_BUTTON_Y = 612
_PAGE_BUTTON_SIDE_MARGIN = 40
_PREV_PAGE_BUTTON_RECT = (_PAGE_BUTTON_SIDE_MARGIN, _PAGE_BUTTON_Y, *_PAGE_BUTTON_SIZE)
_NEXT_PAGE_BUTTON_RECT = (
    _CANVAS_SIZE[0] - _PAGE_BUTTON_SIDE_MARGIN - _PAGE_BUTTON_SIZE[0],
    _PAGE_BUTTON_Y,
    *_PAGE_BUTTON_SIZE,
)
_PAGE_INDICATOR_SIZE = (120, 56)
_PAGE_INDICATOR_RECT = (
    (_CANVAS_SIZE[0] - _PAGE_INDICATOR_SIZE[0]) // 2,
    _PAGE_BUTTON_Y + (_PAGE_BUTTON_SIZE[1] - _PAGE_INDICATOR_SIZE[1]) // 2,
    *_PAGE_INDICATOR_SIZE,
)
_PAGE_NAV_TOUCH_PAD = 16
_CARD_IMAGE_RECT = (20, 88, 680, 508)
_CARD_IMAGE_RECT_NO_HEADER = (20, 20, 680, 584)
_CARD_CAPTION_RECT = (20, 612, 680, 92)
# Reader cards (Funny English story/word cards that expose prev/next nav) carry full
# sentences; shrink the image and enlarge the caption so the whole sentence is shown.
_CARD_READER_IMAGE_RECT = (20, 88, 680, 400)
_CARD_READER_CAPTION_RECT = (20, 496, 680, 208)
_CARD_HEADER_RECT = (
    _BACK_BUTTON_RECT[0] + _BACK_BUTTON_RECT[2] + _TOP_TITLE_BUTTON_GAP,
    20,
    _EXIT_BUTTON_RECT[0]
    - (_BACK_BUTTON_RECT[0] + _BACK_BUTTON_RECT[2] + _TOP_TITLE_BUTTON_GAP)
    - _TOP_TITLE_BUTTON_GAP,
    44,
)
_CARD_PREV_BUTTON_LABEL = "이전 페이지"
_CARD_NEXT_BUTTON_LABEL = "다음 페이지"
_CARD_NAV_BUTTON_SIZE = (160, 48)
_CARD_NAV_BUTTON_Y = 632
_CARD_PREV_BUTTON_RECT = (32, _CARD_NAV_BUTTON_Y, *_CARD_NAV_BUTTON_SIZE)
_CARD_NEXT_BUTTON_RECT = (
    _CANVAS_SIZE[0] - 32 - _CARD_NAV_BUTTON_SIZE[0],
    _CARD_NAV_BUTTON_Y,
    *_CARD_NAV_BUTTON_SIZE,
)
_CARD_NAV_BUTTON_RADIUS = 8
_CARD_NAV_TOUCH_PAD = 10
_CARD_NAV_TEXT_SIDE_RESERVE = _CARD_NAV_BUTTON_SIZE[0] + 28
_PRESS_FEEDBACK_S = 0.15
_PRESS_FEEDBACK_RENDER_TIMEOUT_S = 0.5
_SAFE_BOTTOM = 690
_FONT_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / "fonts" / "pretendard-history-kr.subset.ttf"
)
_INDICATOR_SIZE = 48
_INDICATOR_INSET = 6
_INDICATOR_POSITION = (_CANVAS_SIZE[0] - _INDICATOR_INSET - _INDICATOR_SIZE, _INDICATOR_INSET)
# Touch-only padding for the idle language badge / portal hotspot. This inflates
# the tappable hit region WITHOUT changing the rendered 48px badge size, so the
# top-right toggle is easier to hit on the 4-inch panel without growing visually.
_INDICATOR_TOUCH_PAD = 60
# Top-left double-tap hotspot to activate the download portal. It mirrors the
# language badge to the opposite corner and is rendered as a subtle gray glyph.
_PORTAL_HOTSPOT_POSITION = (_INDICATOR_INSET, _INDICATOR_INSET)
_INDICATOR_PRESS_SHRINK_PX = 8
_INDICATOR_ANIM_TOTAL_FRAMES = 6
_INDICATOR_ANIM_MAX_SCALE = 1.35
_PORTAL_BUTTON_COLOR = (178, 184, 192)
_PORTAL_BUTTON_DISABLED_ALPHA = 88
_PORTAL_STATUS_GAP = 10
_PORTAL_STATUS_PADDING_X = 12
_PORTAL_STATUS_PADDING_Y = 7
_PORTAL_STATUS_RADIUS = 8
_PORTAL_STATUS_FONT_SIZE = 24
_PORTAL_STATUS_MAX_WIDTH = 260
_PORTAL_STATUS_LINE_GAP = 2
_INDICATOR_FALLBACK_COLORS = {
    "ko": (255, 112, 96, 255),
    "en": (72, 136, 255, 255),
}


@dataclass(frozen=True)
class HitTarget:
    """One interactive region from the currently displayed renderer layout."""

    # `str` (not a Literal) to structurally satisfy the invariant
    # `RenderHitTarget.kind: str` protocol attribute; values include
    # "menu_item", "exit", and "language_toggle".
    kind: str
    index: int | None
    generation: int


@dataclass(frozen=True)
class _HitRegion:
    """Screen rectangle for one interactive target."""

    rect: tuple[int, int, int, int]
    target: HitTarget


@dataclass(frozen=True)
class _ImageRequest:
    """Pending image layer request consumed by the UI thread."""

    path: Path
    letterbox: bool
    token: str


@dataclass(frozen=True)
class _TextRequest:
    """Pending text layer request consumed by the UI thread."""

    lines: tuple[str, ...]
    size: int
    highlight_index: int | None
    title: str | None
    layout: str
    token: str | None
    show_exit_button: bool
    has_back: bool
    page_index: int
    page_count: int


@dataclass(frozen=True)
class _CardRequest:
    """Pending image+text composite request consumed by the UI thread."""

    image_path: Path | None
    lines: tuple[str, ...]
    highlight_index: int | None
    title: str | None
    sublabel: str | None
    token: str
    show_exit_button: bool
    show_back: bool
    show_prev_card: bool
    show_next_card: bool


class _ClearImageRequest:
    """Sentinel for clearing the image layer."""


class _ClearTextRequest:
    """Sentinel for clearing the text layer."""


class _ClearCardRequest:
    """Sentinel for clearing the composite card layer."""


_CLEAR_IMAGE = _ClearImageRequest()
_CLEAR_TEXT = _ClearTextRequest()
_CLEAR_CARD = _ClearCardRequest()


def _load_pygame() -> Any:
    global pygame
    if pygame is None:
        import pygame as pygame_module  # type: ignore[import-not-found, import-untyped]

        pygame = pygame_module
    return pygame


def _darken_color(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    """Return ``color`` darkened by ``factor``."""
    channels = tuple(max(0, min(255, int(round(channel * factor)))) for channel in color)
    return cast(tuple[int, int, int], channels)


def _top_title_span(
    *,
    has_back: bool,
    show_exit_button: bool,
    base_left: int,
) -> tuple[int, int]:
    """Return the horizontal span available to a centered top title."""
    left = base_left
    right = _CANVAS_SIZE[0] - base_left
    if has_back:
        left = max(left, _BACK_BUTTON_RECT[0] + _BACK_BUTTON_RECT[2] + _TOP_TITLE_BUTTON_GAP)
    if show_exit_button:
        right = min(right, _EXIT_BUTTON_RECT[0] - _TOP_TITLE_BUTTON_GAP)
    return left, max(left + 1, right)


def _centered_title_x(rendered_width: int, span_left: int, span_right: int) -> int:
    """Return a canvas-centered x coordinate clamped inside a title span."""
    title_width = max(0, rendered_width)
    centered_x = (_CANVAS_SIZE[0] - title_width) // 2
    return min(max(centered_x, span_left), max(span_left, span_right - title_width))


def _normalize_page_state(page_index: int, page_count: int) -> tuple[int, int]:
    """Return a non-empty clamped page state for menu rendering."""
    clean_page_count = max(1, int(page_count))
    clean_page_index = min(max(0, int(page_index)), clean_page_count - 1)
    return clean_page_index, clean_page_count


class PygameCharacterRenderer(CharacterRenderer):
    """Pygame-backed implementation of the character renderer protocol.

    The renderer owns a dedicated UI thread. All Pygame display, event,
    image-load, blit, flip, and quit calls happen on that thread. Joiner
    threads that call close() never touch Pygame, including degraded shutdown.
    """

    def __init__(
        self,
        asset_dir: Path,
        *,
        windowed: bool = False,
        sdl_driver: str | None = None,
        ready_timeout: float = _DEFAULT_READY_TIMEOUT_S,
    ) -> None:
        """Start the renderer UI thread and wait for display initialization."""
        self._asset_dir = asset_dir
        self._windowed = windowed
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._pending_expression: CharacterExpression | None = None
        self._pending_language: str | None = None
        self._pending_image: _ImageRequest | _ClearImageRequest | None = None
        self._pending_text: _TextRequest | _ClearTextRequest | None = None
        self._pending_card: _CardRequest | _ClearCardRequest | None = None
        self._pending_feedback_token: str | None = None
        self._render_events: dict[str, threading.Event] = {}
        self._render_token_counter = 0
        self._layout_generation = 0
        self._displayed_layout_generation = 0
        self._hit_regions: tuple[_HitRegion, ...] = ()
        self._pressed_hit_target: HitTarget | None = None
        self._current_expression_value: CharacterExpression | None = None
        self._current_language_value: str | None = None
        self._current_image_request: _ImageRequest | None = None
        self._current_text_request: _TextRequest | None = None
        self._current_card_request: _CardRequest | None = None
        self._indicator_anim_frames = 0
        self._portal_button_active = False
        self._portal_button_icons: dict[bool, pygame.Surface] = {}
        self._portal_button_pressed_icon: pygame.Surface | None = None
        self._portal_status_text: str | None = None
        self._portal_status_timer: threading.Timer | None = None
        self._portal_status_generation = 0
        self._init_error: Exception | None = None
        self._sprites: dict[CharacterExpression, pygame.Surface | None] = {}
        self._language_indicators: dict[str, pygame.Surface | None] = {}
        self._font_cache: dict[int, pygame.font.Font] = {}
        self._frame_paths: dict[CharacterExpression, list[Path]] = {}
        self._active_frames: list[pygame.Surface] | None = None
        self._active_frames_expr: CharacterExpression | None = None
        self._failed_frame_expressions: set[CharacterExpression] = set()
        self._frame_index = 0
        self._wait_cycle_armed = False
        self._wait_cycle_index = 0
        self._wait_cycle_frames_elapsed = 0
        self._display: pygame.Surface | None = None
        self._black_fallback: pygame.Surface | None = None
        self._close_called = False

        if sdl_driver is not None:
            os.environ["SDL_VIDEODRIVER"] = sdl_driver

        self._ui_thread = threading.Thread(
            target=self._ui_main,
            name="PygameRendererUI",
            daemon=True,
        )
        self._ui_thread.start()

        if not self._ready_event.wait(timeout=ready_timeout):
            self._stop_event.set()
            self._wake.set()
            self._ui_thread.join(timeout=_CLOSE_JOIN_TIMEOUT_S)
            raise RuntimeError("Pygame renderer init timeout (UI thread did not signal ready)")

        if self._init_error is not None:
            raise RuntimeError("Pygame renderer init failed") from self._init_error

    def on_state_change(self, state: SessionState) -> None:
        """Track whether the download portal button should appear active."""
        with self._lock:
            if self._close_called:
                return
            self._portal_button_active = state is SessionState.AWAITING_TAP
        self._wake.set()

    def on_expression(self, expression: CharacterExpression) -> None:
        """Best-effort enqueue of a character expression update."""
        with self._lock:
            if self._close_called:
                return
            self._pending_expression = expression
            self._wait_cycle_armed = expression is CharacterExpression.THINKING
            if self._wait_cycle_armed:
                self._wait_cycle_index = 0
                self._wait_cycle_frames_elapsed = 0
        self._wake.set()

    def on_language_change(self, lang: str) -> None:
        """Best-effort enqueue of a session language indicator update."""
        self.set_language_indicator(lang)

    def set_language_indicator(self, lang: str) -> None:
        """Set the visible session language badge."""
        if lang not in {"ko", "en"}:
            logger.warning("Ignoring unsupported language indicator: %s", lang)
            return
        with self._lock:
            if self._close_called:
                return
            self._pending_language = lang
        self._wake.set()

    def show_image(self, path: str | Path, *, letterbox: bool = True) -> str | None:
        """Display a full-screen image on the next UI-thread render pass."""
        with self._lock:
            if self._close_called:
                return None
            token = self._next_render_token_locked()
            self._pending_image = _ImageRequest(path=Path(path), letterbox=letterbox, token=token)
            self._pending_text = _CLEAR_TEXT
            self._pending_card = _CLEAR_CARD
            self._clear_hit_regions_locked()
        self._wake.set()
        return token

    def clear_image(self) -> None:
        """Clear the active image layer on the next UI-thread render pass."""
        with self._lock:
            if self._close_called:
                return
            self._pending_image = _CLEAR_IMAGE
            self._pending_card = _CLEAR_CARD
            self._clear_hit_regions_locked()
        self._wake.set()

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
        """Display generic text or menu content on the next UI-thread render pass."""
        clean_lines = tuple(str(line) for line in lines)
        clean_page_index, clean_page_count = _normalize_page_state(page_index, page_count)
        with self._lock:
            if self._close_called:
                return
            self._pending_text = _TextRequest(
                lines=clean_lines,
                size=size,
                highlight_index=highlight_index,
                title=title,
                layout=layout,
                token=None,
                show_exit_button=show_exit_button,
                has_back=has_back,
                page_index=clean_page_index,
                page_count=clean_page_count,
            )
            self._pending_image = _CLEAR_IMAGE
            self._pending_card = _CLEAR_CARD
            self._clear_hit_regions_locked()
        self._wake.set()

    def clear_text(self) -> None:
        """Clear the active text layer on the next UI-thread render pass."""
        with self._lock:
            if self._close_called:
                return
            self._pending_text = _CLEAR_TEXT
            self._pending_card = _CLEAR_CARD
            self._clear_hit_regions_locked()
        self._wake.set()

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
        """Display a generic image+text card on the next UI-thread render pass."""
        clean_lines = tuple(str(line) for line in lines or ())
        with self._lock:
            if self._close_called:
                return None
            token = self._next_render_token_locked()
            self._pending_card = _CardRequest(
                image_path=Path(image_path) if image_path is not None else None,
                lines=clean_lines,
                highlight_index=highlight_index,
                title=title,
                sublabel=sublabel,
                token=token,
                show_exit_button=show_exit_button,
                show_back=show_back,
                show_prev_card=show_prev_card,
                show_next_card=show_next_card,
            )
            self._pending_image = _CLEAR_IMAGE
            self._pending_text = _CLEAR_TEXT
            self._clear_hit_regions_locked()
        self._wake.set()
        return token

    def wait_until_rendered(self, token: str | None, timeout: float) -> bool:
        """Wait until the UI thread flips the exact tokenized render request."""
        if token is None:
            return True
        with self._lock:
            event = self._render_events.get(token)
        if event is None:
            return False
        rendered = event.wait(timeout=max(0.0, timeout))
        if rendered:
            with self._lock:
                self._render_events.pop(token, None)
        return rendered

    def show_history_image(self, path: str | Path) -> str | None:
        """Display a history-mode image using the generic image primitive."""
        return self.show_image(path, letterbox=True)

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
        """Display a tokenized history menu with direct-touch hit regions."""
        clean_items = tuple(str(item) for item in items)
        clean_page_index, clean_page_count = _normalize_page_state(page_index, page_count)
        with self._lock:
            if self._close_called:
                return None
            token = self._next_render_token_locked()
            self._pending_text = _TextRequest(
                lines=clean_items,
                size=34,
                highlight_index=highlight,
                title=title,
                layout="menu",
                token=token,
                show_exit_button=True,
                has_back=has_back,
                page_index=clean_page_index,
                page_count=clean_page_count,
            )
            self._pending_image = _CLEAR_IMAGE
            self._pending_card = _CLEAR_CARD
            self._clear_hit_regions_locked()
        self._wake.set()
        return token

    def hit_test(self, x: int, y: int) -> RenderHitTarget | None:
        """Return the current direct-touch target at screen coordinate ``x,y``."""
        with self._lock:
            generation = self._displayed_layout_generation
            for region in self._hit_regions:
                if region.target.generation != generation:
                    continue
                left, top, width, height = region.rect
                if left <= x < left + width and top <= y < top + height:
                    return region.target
        return None

    def show_press_feedback(
        self,
        target: Any,
        duration: float = _PRESS_FEEDBACK_S,
    ) -> str | None:
        """Briefly redraw the current target in a pressed style on the UI thread."""
        token = self._queue_press_feedback(target)
        if token is None:
            return None
        self._wake.set()
        timer = threading.Timer(max(0.0, duration), self._clear_press_feedback, args=(target,))
        timer.daemon = True
        timer.start()
        return token

    def flash_press_feedback(
        self,
        target: Any,
        duration: float = _PRESS_FEEDBACK_S,
        timeout: float = _PRESS_FEEDBACK_RENDER_TIMEOUT_S,
    ) -> None:
        """Render and hold a pressed target before the caller changes layout."""
        token = self._queue_press_feedback(target)
        if token is None:
            return
        self._wake.set()
        self.wait_until_rendered(token, timeout=timeout)
        time.sleep(max(0.0, duration))
        self._clear_press_feedback(target)

    def show_portal_status(self, text: str, *, duration: float) -> None:
        """Render a transient status message next to the portal button."""
        clean_text = str(text).strip()
        timer: threading.Timer | None = None
        with self._lock:
            if self._close_called:
                return
            if self._portal_status_timer is not None:
                self._portal_status_timer.cancel()
                self._portal_status_timer = None
            self._portal_status_text = clean_text or None
            self._portal_status_generation += 1
            generation = self._portal_status_generation
            if clean_text:
                timer = threading.Timer(
                    max(0.0, duration),
                    self._clear_portal_status,
                    args=(generation,),
                )
                timer.daemon = True
                self._portal_status_timer = timer
        self._wake.set()
        if timer is not None:
            timer.start()

    def _queue_press_feedback(self, target: Any) -> str | None:
        """Queue a pressed-target redraw and return its render token."""
        with self._lock:
            target_generation = getattr(target, "generation", None)
            if self._close_called or target_generation != self._displayed_layout_generation:
                return None
            if (
                getattr(target, "kind", None) == "portal_activate"
                and not self._portal_button_active
            ):
                return None
            token = self._next_render_token_locked()
            self._pressed_hit_target = target
            self._pending_feedback_token = token
            return token

    def _clear_press_feedback(self, target: Any) -> None:
        with self._lock:
            if self._pressed_hit_target != target:
                return
            self._pressed_hit_target = None
            self._pending_feedback_token = None
        self._wake.set()

    def _clear_portal_status(self, generation: int) -> None:
        with self._lock:
            if self._portal_status_generation != generation:
                return
            self._portal_status_text = None
            self._portal_status_timer = None
        self._wake.set()

    def close(self) -> None:
        """Stop the UI thread without calling Pygame from the joiner thread."""
        with self._lock:
            if self._close_called:
                return
            self._close_called = True
            portal_status_timer = getattr(self, "_portal_status_timer", None)
            if portal_status_timer is not None:
                portal_status_timer.cancel()
                self._portal_status_timer = None
            render_events = list(self._render_events.values())

        self._stop_event.set()
        self._wake.set()
        for event in render_events:
            event.set()
        self._ui_thread.join(timeout=_CLOSE_JOIN_TIMEOUT_S)
        if self._ui_thread.is_alive():
            logger.warning(
                "Renderer UI thread did not exit within %.1fs; degraded shutdown",
                _CLOSE_JOIN_TIMEOUT_S,
            )

    @property
    def current_expression(self) -> CharacterExpression | None:
        """Return the last expression successfully displayed."""
        with self._lock:
            return self._current_expression_value

    def _next_render_token_locked(self) -> str:
        self._render_token_counter += 1
        token = f"render-{self._render_token_counter}"
        self._render_events[token] = threading.Event()
        return token

    def _clear_hit_regions_locked(self) -> None:
        """Drop stale direct-touch regions before a pending layout is displayed."""
        self._hit_regions = ()
        self._pressed_hit_target = None
        self._pending_feedback_token = None

    def _ui_main(self) -> None:
        pygame_module = _load_pygame()
        try:
            pygame_module.init()
            pygame_module.font.init()
            modes = pygame_module.display.list_modes()
            flags = 0 if self._windowed else pygame_module.FULLSCREEN
            if modes != -1 and _CANVAS_SIZE not in modes:
                flags |= pygame_module.SCALED
            try:
                self._display = pygame_module.display.set_mode(_CANVAS_SIZE, flags)
            except pygame_module.error:
                if flags & pygame_module.SCALED:
                    raise
                self._display = pygame_module.display.set_mode(
                    _CANVAS_SIZE, flags | pygame_module.SCALED
                )
            self._load_sprites(self._asset_dir)
            self._load_language_indicators(self._asset_dir)
            self._discover_frame_sequences(self._asset_dir)
            self._ready_event.set()
        except Exception as exc:
            self._init_error = exc
            with suppress(Exception):
                pygame_module.quit()
            self._ready_event.set()
            return

        try:
            while not self._stop_event.is_set():
                woke = self._wake.wait(timeout=_FRAME_INTERVAL_S)
                self._wake.clear()
                if self._stop_event.is_set():
                    break
                if not woke and not self._has_frame_to_render():
                    continue
                try:
                    self._render_once()
                except Exception:
                    logger.warning(
                        "PygameCharacterRenderer render loop iteration failed",
                        exc_info=True,
                    )
        finally:
            with suppress(Exception):
                pygame_module.quit()

    def _has_frame_to_render(self) -> bool:
        with self._lock:
            if self._pending_expression is not None:
                return True
            if self._pending_language is not None:
                return True
            if self._pending_image is not None:
                return True
            if self._pending_text is not None:
                return True
            if self._pending_card is not None:
                return True
            if self._pressed_hit_target is not None:
                return True
            if self._indicator_anim_frames > 0:
                return True
            if self._wait_cycle_armed:
                return True
            if self._portal_status_text is not None:
                return True
            current = self._current_expression_value

        if current is None:
            return False
        return (
            self._active_frames is not None
            and self._active_frames_expr is self._animation_expression(current)
        )

    def _render_once(self) -> None:
        if self._display is None:
            raise RuntimeError("Pygame display is not initialized")

        pygame_module = _load_pygame()
        pygame_module.event.pump()
        with self._lock:
            pending = self._pending_expression
            pending_language = self._pending_language
            pending_image = self._pending_image
            pending_text = self._pending_text
            pending_card = self._pending_card
            pending_feedback_token = self._pending_feedback_token
            pressed_target = self._pressed_hit_target
            self._pending_expression = None
            self._pending_language = None
            self._pending_image = None
            self._pending_text = None
            self._pending_card = None
            self._pending_feedback_token = None
            # Advance the layout generation only when the displayed LAYOUT actually
            # changes this frame (a card/text/image request was applied or cleared),
            # NOT on pure expression/animation frames. Bumping every frame would
            # invalidate the hit regions and press-feedback target captured for a
            # still-displayed layout, breaking direct-touch taps (e.g. the exit
            # button) while a scene's SPEAKING animation keeps the loop awake.
            layout_changed = (
                pending_card is not None or pending_image is not None or pending_text is not None
            )
            if layout_changed:
                self._layout_generation += 1
            layout_generation = self._layout_generation
            if pending is None:
                if self._wait_cycle_armed:
                    self._wait_cycle_frames_elapsed += 1
                    if self._wait_cycle_frames_elapsed >= _WAIT_EXPRESSION_CYCLE_FRAMES:
                        self._wait_cycle_frames_elapsed = 0
                        self._wait_cycle_index = (self._wait_cycle_index + 1) % len(
                            _WAIT_EXPRESSION_CYCLE
                        )
                    requested = _WAIT_EXPRESSION_CYCLE[self._wait_cycle_index]
                else:
                    requested = self._current_expression_value or CharacterExpression.NEUTRAL
            else:
                requested = pending
            if isinstance(pending_card, _CardRequest):
                self._current_card_request = pending_card
                self._current_image_request = None
                self._current_text_request = None
            elif pending_card is _CLEAR_CARD:
                self._current_card_request = None
            if isinstance(pending_image, _ImageRequest):
                self._current_image_request = pending_image
                self._current_text_request = None
                self._current_card_request = None
            elif pending_image is _CLEAR_IMAGE:
                self._current_image_request = None
            if isinstance(pending_text, _TextRequest):
                self._current_text_request = pending_text
                self._current_image_request = None
                self._current_card_request = None
            elif pending_text is _CLEAR_TEXT:
                self._current_text_request = None
            current_card = self._current_card_request
            current_image = self._current_image_request
            current_text = self._current_text_request
            current_language = self._current_language_value
            if (
                pending_language is not None
                and current_language is not None
                and pending_language != current_language
            ):
                self._indicator_anim_frames = _INDICATOR_ANIM_TOTAL_FRAMES
            indicator_language = pending_language or current_language
            indicator_anim_frames = self._indicator_anim_frames
            portal_button_active = self._portal_button_active
            portal_status_text = self._portal_status_text

        is_animation = False
        effective_expr = requested
        hit_regions: list[_HitRegion] = []
        if current_card is not None:
            hit_regions = self._render_card_request(
                current_card,
                layout_generation,
                pressed_target,
            )
        elif current_image is not None:
            self._render_image_request(current_image)
        elif current_text is not None:
            hit_regions = self._render_text_request(
                current_text,
                layout_generation,
                pressed_target,
            )
        else:
            effective_expr, surface, is_animation = self._resolve_render_surface(requested)
            self._display.fill(_BLACK)
            self._display.blit(surface, (0, 0))
        if current_image is None and current_card is None:
            indicator_hit_region = self._blit_language_indicator(
                indicator_language,
                animation_frames=indicator_anim_frames,
                layout_generation=layout_generation if current_text is None else None,
                pressed_target=pressed_target,
            )
            if indicator_hit_region is not None:
                hit_regions.append(indicator_hit_region)
            portal_hit_region = self._blit_portal_button(
                active=portal_button_active,
                layout_generation=layout_generation if current_text is None else None,
                pressed_target=pressed_target,
            )
            if portal_hit_region is not None:
                hit_regions.append(portal_hit_region)
            if portal_status_text is not None:
                self._draw_portal_status(portal_status_text)
        pygame_module.display.flip()
        rendered_token = None
        if current_card is not None:
            rendered_token = current_card.token
        elif current_image is not None:
            rendered_token = current_image.token
        elif current_text is not None:
            rendered_token = current_text.token
        rendered_tokens = [token for token in (rendered_token, pending_feedback_token) if token]
        if is_animation:
            self._frame_index += 1
        with self._lock:
            self._current_expression_value = effective_expr
            self._displayed_layout_generation = layout_generation
            # Refresh hit regions for the surface drawn this frame. Card/text
            # renderers return their own direct-touch regions; the idle surface
            # returns the language-toggle region; full-screen images return none.
            self._hit_regions = tuple(hit_regions)
            if pending_language is not None:
                self._current_language_value = pending_language
            if self._indicator_anim_frames > 0:
                self._indicator_anim_frames -= 1
            for token in rendered_tokens:
                event = self._render_events.get(token)
                if event is not None:
                    event.set()

    def _render_image_request(self, request: _ImageRequest) -> None:
        """Render one image request on the UI thread."""
        if self._display is None:
            raise RuntimeError("Pygame display is not initialized")
        pygame_module = _load_pygame()
        self._display.fill(_BLACK)
        try:
            image_surface = pygame_module.image.load(str(request.path)).convert()
        except Exception:
            logger.warning("Failed to load display image %s", request.path, exc_info=True)
            self._render_missing_image_fallback()
            return

        if request.letterbox:
            source_w, source_h = image_surface.get_size()
            if source_w <= 0 or source_h <= 0:
                logger.warning("Display image has invalid size: %s", request.path)
                self._render_missing_image_fallback()
                return
            scale = min(_CANVAS_SIZE[0] / source_w, _CANVAS_SIZE[1] / source_h)
            target_size = (
                max(1, int(round(source_w * scale))),
                max(1, int(round(source_h * scale))),
            )
        else:
            target_size = _CANVAS_SIZE
        if image_surface.get_size() != target_size:
            image_surface = pygame_module.transform.smoothscale(image_surface, target_size)
        x = (_CANVAS_SIZE[0] - target_size[0]) // 2
        y = (_CANVAS_SIZE[1] - target_size[1]) // 2
        self._display.blit(image_surface, (x, y))

    def _render_missing_image_fallback(self) -> None:
        """Render a stable fallback when an image is unavailable."""
        self._render_text_request(
            _TextRequest(
                lines=("이미지를 준비 중이에요.",),
                size=34,
                highlight_index=None,
                title=None,
                layout="center",
                token=None,
                show_exit_button=False,
                has_back=False,
                page_index=0,
                page_count=1,
            ),
            0,
            None,
        )

    def _draw_top_title(
        self,
        title: str,
        *,
        y: int,
        base_size: int,
        color: tuple[int, int, int],
        base_left: int,
        has_back: bool,
        show_exit_button: bool,
    ) -> None:
        """Draw a single-line top title centered and clamped between nav targets."""
        if self._display is None:
            raise RuntimeError("Pygame display is not initialized")
        span_left, span_right = _top_title_span(
            has_back=has_back,
            show_exit_button=show_exit_button,
            base_left=base_left,
        )
        max_width = max(1, span_right - span_left)
        title_font = self._fit_font(title, base_size, max_width)
        title_text = self._truncate_to_single_line(title, title_font, max_width)
        if not title_text:
            return
        title_surface = title_font.render(title_text, True, color)
        title_x = _centered_title_x(title_surface.get_width(), span_left, span_right)
        self._display.blit(title_surface, (title_x, y))

    def _render_card_request(
        self,
        request: _CardRequest,
        layout_generation: int,
        pressed_target: HitTarget | None,
    ) -> list[_HitRegion]:
        """Render one image+text card request on the UI thread."""
        if self._display is None:
            raise RuntimeError("Pygame display is not initialized")
        pygame_module = _load_pygame()
        self._display.fill(_TEXT_BG)
        if request.title:
            self._draw_top_title(
                request.title,
                y=_CARD_HEADER_RECT[1],
                base_size=26,
                color=_TEXT_MUTED,
                base_left=24,
                has_back=request.show_back,
                show_exit_button=request.show_exit_button,
            )
        is_reader_card = request.show_prev_card or request.show_next_card
        if is_reader_card:
            image_rect_values = _CARD_READER_IMAGE_RECT
            caption_rect_values = _CARD_READER_CAPTION_RECT
        else:
            image_rect_values = _CARD_IMAGE_RECT if request.title else _CARD_IMAGE_RECT_NO_HEADER
            caption_rect_values = _CARD_CAPTION_RECT
        image_rect = pygame_module.Rect(*image_rect_values)
        if request.image_path is not None:
            self._render_card_image(request.image_path, image_rect)
        elif not request.lines:
            self._render_card_missing_image(image_rect)

        panel_rect = pygame_module.Rect(*caption_rect_values)
        pygame_module.draw.rect(self._display, _TEXT_PANEL, panel_rect)
        hit_regions: list[_HitRegion] = []
        hit_regions.extend(self._draw_card_nav_buttons(request, layout_generation, pressed_target))
        self._render_card_text(request, panel_rect)
        if request.show_back:
            hit_regions.append(self._draw_back_button(layout_generation, pressed_target))
        if request.show_exit_button:
            hit_regions.append(self._draw_exit_button(layout_generation, pressed_target))
        return hit_regions

    def _render_card_image(self, image_path: Path, target_rect: Any) -> None:
        """Render one card image inside the reserved image region."""
        if self._display is None:
            raise RuntimeError("Pygame display is not initialized")
        pygame_module = _load_pygame()
        try:
            image_surface = pygame_module.image.load(str(image_path)).convert()
        except Exception:
            logger.warning("Failed to load card image %s", image_path, exc_info=True)
            self._render_card_missing_image(target_rect)
            return
        source_w, source_h = image_surface.get_size()
        if source_w <= 0 or source_h <= 0:
            logger.warning("Card image has invalid size: %s", image_path)
            self._render_card_missing_image(target_rect)
            return
        scale = min(target_rect.width / source_w, target_rect.height / source_h)
        target_size = (
            max(1, int(round(source_w * scale))),
            max(1, int(round(source_h * scale))),
        )
        if image_surface.get_size() != target_size:
            image_surface = pygame_module.transform.smoothscale(image_surface, target_size)
        x = target_rect.left + (target_rect.width - target_size[0]) // 2
        y = target_rect.top + (target_rect.height - target_size[1]) // 2
        self._display.blit(image_surface, (x, y))

    def _render_card_missing_image(self, target_rect: Any) -> None:
        """Render an image-region fallback inside a card."""
        if self._display is None:
            raise RuntimeError("Pygame display is not initialized")
        pygame_module = _load_pygame()
        pygame_module.draw.rect(self._display, _TEXT_PANEL, target_rect)
        font = self._get_font(30)
        label = "이미지를 준비 중이에요."
        surface = font.render(label, True, _TEXT_MUTED)
        self._display.blit(
            surface,
            (
                target_rect.left + (target_rect.width - surface.get_width()) // 2,
                target_rect.top + (target_rect.height - surface.get_height()) // 2,
            ),
        )

    def _render_card_text(self, request: _CardRequest, panel_rect: Any) -> None:
        """Render the card title/sublabel/body inside the text panel."""
        if self._display is None:
            raise RuntimeError("Pygame display is not initialized")
        lines = list(request.lines)
        if request.sublabel:
            lines.insert(0, request.sublabel)
        if not lines:
            return
        if request.show_prev_card or request.show_next_card:
            available_width = max(1, panel_rect.width - (_CARD_NAV_TEXT_SIDE_RESERVE * 2))
        else:
            available_width = panel_rect.width - 40
        text_left = panel_rect.left + (panel_rect.width - available_width) // 2
        max_height = panel_rect.height - 20
        body_size = 44
        body_floor = 30
        sublabel_size = 26
        rendered: list[tuple[Any, str, tuple[int, int, int]]] = []
        total_height = 0
        while True:
            body_font = self._get_font(body_size)
            sublabel_font = self._get_font(sublabel_size)
            rendered = []
            for index, line in enumerate(lines):
                if request.sublabel and index == 0:
                    font = sublabel_font
                    color = _TEXT_MUTED
                    max_parts = 1
                else:
                    font = body_font
                    original_index = index - 1 if request.sublabel else index
                    color = (
                        _TEXT_PANEL_HIGHLIGHT
                        if original_index == request.highlight_index
                        else _TEXT_FG
                    )
                    max_parts = 4 if (request.show_prev_card or request.show_next_card) else 2
                for part in self._wrap_text(line, font, available_width)[:max_parts]:
                    rendered.append((font, part, color))
            total_height = sum(font.size(text)[1] for font, text, _color in rendered)
            total_height += 8 * max(0, len(rendered) - 1)
            if total_height <= max_height or body_size == body_floor:
                break
            body_size = max(body_floor, body_size - 4)
            sublabel_size = max(20, sublabel_size - 2)

        y = panel_rect.top + max(10, (panel_rect.height - total_height) // 2)
        bottom = panel_rect.bottom - 10
        for font, text, color in rendered:
            surface = font.render(text, True, color)
            if y + surface.get_height() > bottom:
                break
            x = text_left + (available_width - surface.get_width()) // 2
            self._display.blit(surface, (x, y))
            y += surface.get_height() + 8

    def _draw_card_nav_buttons(
        self,
        request: _CardRequest,
        layout_generation: int,
        pressed_target: HitTarget | None,
    ) -> list[_HitRegion]:
        """Draw optional previous/next card buttons for Funny English cards."""
        hit_regions: list[_HitRegion] = []
        if request.show_prev_card:
            hit_regions.append(
                self._draw_text_button(
                    label=_CARD_PREV_BUTTON_LABEL,
                    rect_values=_CARD_PREV_BUTTON_RECT,
                    radius=_CARD_NAV_BUTTON_RADIUS,
                    target_kind="prev_card",
                    layout_generation=layout_generation,
                    pressed_target=pressed_target,
                    touch_pad=_CARD_NAV_TOUCH_PAD,
                )
            )
        if request.show_next_card:
            hit_regions.append(
                self._draw_text_button(
                    label=_CARD_NEXT_BUTTON_LABEL,
                    rect_values=_CARD_NEXT_BUTTON_RECT,
                    radius=_CARD_NAV_BUTTON_RADIUS,
                    target_kind="next_card",
                    layout_generation=layout_generation,
                    pressed_target=pressed_target,
                    touch_pad=_CARD_NAV_TOUCH_PAD,
                )
            )
        return hit_regions

    def _render_text_request(
        self,
        request: _TextRequest,
        layout_generation: int,
        pressed_target: HitTarget | None,
    ) -> list[_HitRegion]:
        """Render one text request on the UI thread."""
        if self._display is None:
            raise RuntimeError("Pygame display is not initialized")
        self._display.fill(_TEXT_BG)
        if request.layout == "menu":
            return self._render_menu_text(request, layout_generation, pressed_target)
        return self._render_center_text(request, layout_generation, pressed_target)

    def _render_menu_text(
        self,
        request: _TextRequest,
        layout_generation: int,
        pressed_target: HitTarget | None,
    ) -> list[_HitRegion]:
        """Render menu-style text rows."""
        if self._display is None:
            raise RuntimeError("Pygame display is not initialized")
        pygame_module = _load_pygame()
        title = request.title or ""
        if title:
            self._draw_top_title(
                title,
                y=58,
                base_size=38,
                color=_TEXT_FG,
                base_left=_MENU_LEFT,
                has_back=request.has_back,
                show_exit_button=request.show_exit_button,
            )
        y = 162

        hit_regions: list[_HitRegion] = []
        if request.has_back:
            hit_regions.append(self._draw_back_button(layout_generation, pressed_target))
        if request.show_exit_button:
            hit_regions.append(self._draw_exit_button(layout_generation, pressed_target))

        row_rects: list[Any] = []
        for index, line in enumerate(request.lines[:4]):
            highlighted = index == request.highlight_index
            rect = pygame_module.Rect(_MENU_LEFT, y, _MENU_WIDTH, _MENU_ITEM_HEIGHT)
            row_rects.append(rect)
            pressed = (
                pressed_target is not None
                and pressed_target.kind == "menu_item"
                and pressed_target.index == index
            )
            color = _TEXT_PANEL_HIGHLIGHT if highlighted else _TEXT_PANEL
            if pressed:
                color = _darken_color(color, 0.78)
                draw_rect = rect.inflate(-10, -8)
            else:
                draw_rect = rect
            pygame_module.draw.rect(self._display, color, draw_rect)
            text_color = _TEXT_HIGHLIGHT_FG if highlighted else _TEXT_FG
            fitted_font = self._fit_font(line, request.size, _MENU_WIDTH - 42)
            wrapped = self._wrap_text(line, fitted_font, _MENU_WIDTH - 42)[:2]
            total_height = sum(fitted_font.size(part)[1] for part in wrapped)
            text_y = y + max(0, (_MENU_ITEM_HEIGHT - total_height - 6 * (len(wrapped) - 1)) // 2)
            for part in wrapped:
                surface = fitted_font.render(part, True, text_color)
                self._display.blit(surface, (_MENU_LEFT + 22, text_y))
                text_y += surface.get_height() + 6
            y += _MENU_ITEM_HEIGHT + _MENU_ITEM_GAP
        hit_regions.extend(self._menu_item_hit_regions(row_rects, layout_generation))
        hit_regions.extend(self._draw_page_nav(request, layout_generation, pressed_target))
        return hit_regions

    def _menu_item_hit_regions(
        self,
        row_rects: list[Any],
        layout_generation: int,
    ) -> list[_HitRegion]:
        """Return full-width tiled hit regions for visible menu rows."""
        if not row_rects:
            return []
        page_nav_top = _PAGE_BUTTON_Y
        hit_regions: list[_HitRegion] = []
        for index, rect in enumerate(row_rects):
            if index == 0:
                top = rect.top
            else:
                previous = row_rects[index - 1]
                top = (previous.bottom + rect.top) // 2
            if index == len(row_rects) - 1:
                bottom = max(top, page_nav_top)
            else:
                following = row_rects[index + 1]
                bottom = (rect.bottom + following.top) // 2
            if bottom <= top:
                continue
            hit_regions.append(
                _HitRegion(
                    rect=(0, top, _CANVAS_SIZE[0], bottom - top),
                    target=HitTarget("menu_item", index, layout_generation),
                )
            )
        return hit_regions

    def _render_center_text(
        self,
        request: _TextRequest,
        layout_generation: int,
        pressed_target: HitTarget | None,
    ) -> list[_HitRegion]:
        """Render centered text content."""
        if self._display is None:
            raise RuntimeError("Pygame display is not initialized")
        content_top = _TEXT_SAFE_TOP_RIGHT_MARGIN
        if request.title:
            self._draw_top_title(
                request.title,
                y=58,
                base_size=34,
                color=_TEXT_MUTED,
                base_left=_TEXT_SAFE_TOP_RIGHT_MARGIN,
                has_back=request.has_back,
                show_exit_button=request.show_exit_button,
            )
            content_top = 162
        max_width = _CANVAS_SIZE[0] - 2 * _TEXT_SAFE_TOP_RIGHT_MARGIN
        max_height = _SAFE_BOTTOM - content_top
        body_size = request.size
        blocks: list[tuple[Any, str, tuple[int, int, int]]] = []
        total_height = 0
        while body_size >= 24:
            body_font = self._get_font(body_size)
            blocks = []
            for line in request.lines:
                for wrapped in self._wrap_text(line, body_font, max_width):
                    blocks.append((body_font, wrapped, _TEXT_FG))
            if not blocks:
                break
            total_height = sum(font.size(text)[1] for font, text, _color in blocks)
            total_height += 10 * max(0, len(blocks) - 1)
            if total_height <= max_height:
                break
            body_size -= 4
        y = max(content_top, (_CANVAS_SIZE[1] - total_height) // 2)
        bottom = _SAFE_BOTTOM
        for font, text, color in blocks:
            surface = font.render(text, True, color)
            if y + surface.get_height() > bottom:
                break
            x = (_CANVAS_SIZE[0] - surface.get_width()) // 2
            self._display.blit(surface, (x, y))
            y += surface.get_height() + 10
        hit_regions: list[_HitRegion] = []
        if request.has_back:
            hit_regions.append(self._draw_back_button(layout_generation, pressed_target))
        if request.show_exit_button:
            hit_regions.append(self._draw_exit_button(layout_generation, pressed_target))
        return hit_regions

    def _draw_back_button(
        self,
        layout_generation: int,
        pressed_target: HitTarget | None,
    ) -> _HitRegion:
        """Draw the menu parent-navigation target and return its hit region."""
        return self._draw_text_button(
            label=_BACK_BUTTON_LABEL,
            rect_values=_BACK_BUTTON_RECT,
            radius=_BACK_BUTTON_RADIUS,
            target_kind="back",
            layout_generation=layout_generation,
            pressed_target=pressed_target,
            touch_pad=_EXIT_TOUCH_PAD,
        )

    def _draw_exit_button(
        self,
        layout_generation: int,
        pressed_target: HitTarget | None,
    ) -> _HitRegion:
        """Draw the persistent history exit target and return its hit region."""
        return self._draw_text_button(
            label=_EXIT_BUTTON_LABEL,
            rect_values=_EXIT_BUTTON_RECT,
            radius=_EXIT_BUTTON_RADIUS,
            target_kind="exit",
            layout_generation=layout_generation,
            pressed_target=pressed_target,
            touch_pad=_EXIT_TOUCH_PAD,
        )

    def _draw_text_button(
        self,
        *,
        label: str,
        rect_values: tuple[int, int, int, int],
        radius: int,
        target_kind: str,
        layout_generation: int,
        pressed_target: HitTarget | None,
        touch_pad: int,
    ) -> _HitRegion:
        """Draw a labeled menu button and return its padded hit region."""
        if self._display is None:
            raise RuntimeError("Pygame display is not initialized")
        pygame_module = _load_pygame()
        rect = pygame_module.Rect(*rect_values)
        pressed = pressed_target is not None and pressed_target.kind == target_kind
        draw_rect = rect.inflate(-8, -6) if pressed else rect
        color = _darken_color(_TEXT_PANEL_HIGHLIGHT, 0.78) if pressed else _TEXT_PANEL_HIGHLIGHT
        pygame_module.draw.rect(
            self._display,
            color,
            draw_rect,
            border_radius=radius,
        )
        font = self._fit_font(label, 30, draw_rect.width - 28)
        surface = font.render(label, True, _TEXT_HIGHLIGHT_FG)
        self._display.blit(
            surface,
            (
                draw_rect.left + (draw_rect.width - surface.get_width()) // 2,
                draw_rect.top + (draw_rect.height - surface.get_height()) // 2,
            ),
        )
        touch_rect = rect.inflate(touch_pad * 2, touch_pad * 2).clip(
            pygame_module.Rect(0, 0, _CANVAS_SIZE[0], _CANVAS_SIZE[1])
        )
        return _HitRegion(
            rect=(touch_rect.left, touch_rect.top, touch_rect.width, touch_rect.height),
            target=HitTarget(target_kind, None, layout_generation),
        )

    def _draw_page_nav(
        self,
        request: _TextRequest,
        layout_generation: int,
        pressed_target: HitTarget | None,
    ) -> list[_HitRegion]:
        """Draw bottom page controls and return their optional hit regions."""
        if request.page_count <= 1:
            return []
        if self._display is None:
            raise RuntimeError("Pygame display is not initialized")
        pygame_module = _load_pygame()
        indicator_rect = pygame_module.Rect(*_PAGE_INDICATOR_RECT)
        label = f"{request.page_index + 1} / {request.page_count}"
        font = self._fit_font(label, 28, indicator_rect.width)
        surface = font.render(label, True, _TEXT_MUTED)
        self._display.blit(
            surface,
            (
                indicator_rect.left + (indicator_rect.width - surface.get_width()) // 2,
                indicator_rect.top + (indicator_rect.height - surface.get_height()) // 2,
            ),
        )

        hit_regions: list[_HitRegion] = []
        if request.page_index > 0:
            hit_regions.append(
                self._draw_page_button(
                    "prev_page",
                    _PREV_PAGE_BUTTON_RECT,
                    layout_generation,
                    pressed_target,
                )
            )
        if request.page_index + 1 < request.page_count:
            hit_regions.append(
                self._draw_page_button(
                    "next_page",
                    _NEXT_PAGE_BUTTON_RECT,
                    layout_generation,
                    pressed_target,
                )
            )
        return hit_regions

    def _draw_page_button(
        self,
        target_kind: str,
        rect_values: tuple[int, int, int, int],
        layout_generation: int,
        pressed_target: HitTarget | None,
    ) -> _HitRegion:
        """Draw a chevron page button and return its hit region."""
        if self._display is None:
            raise RuntimeError("Pygame display is not initialized")
        pygame_module = _load_pygame()
        rect = pygame_module.Rect(*rect_values)
        pressed = pressed_target is not None and pressed_target.kind == target_kind
        draw_rect = rect.inflate(-8, -6) if pressed else rect
        color = _darken_color(_TEXT_PANEL_HIGHLIGHT, 0.78) if pressed else _TEXT_PANEL_HIGHLIGHT
        pygame_module.draw.rect(
            self._display,
            color,
            draw_rect,
            border_radius=_BACK_BUTTON_RADIUS,
        )
        center_y = draw_rect.centery
        if target_kind == "prev_page":
            points = [
                (draw_rect.centerx + 18, center_y - 22),
                (draw_rect.centerx - 20, center_y),
                (draw_rect.centerx + 18, center_y + 22),
            ]
        else:
            points = [
                (draw_rect.centerx - 18, center_y - 22),
                (draw_rect.centerx + 20, center_y),
                (draw_rect.centerx - 18, center_y + 22),
            ]
        pygame_module.draw.polygon(self._display, _TEXT_HIGHLIGHT_FG, points)
        touch_rect = rect.inflate(_PAGE_NAV_TOUCH_PAD * 2, _PAGE_NAV_TOUCH_PAD * 2).clip(
            pygame_module.Rect(0, 0, _CANVAS_SIZE[0], _CANVAS_SIZE[1])
        )
        return _HitRegion(
            rect=(touch_rect.left, touch_rect.top, touch_rect.width, touch_rect.height),
            target=HitTarget(target_kind, None, layout_generation),
        )

    def _fit_font(self, text: str, base_size: int, max_width: int) -> Any:
        """Return the largest cached font that can fit a menu row reasonably."""
        size = base_size
        while size > 22:
            font = self._get_font(size)
            if all(
                font.size(part)[0] <= max_width for part in self._wrap_text(text, font, max_width)
            ):
                return font
            size -= 2
        return self._get_font(size)

    def _get_font(self, size: int) -> Any:
        """Return the cached committed Korean subset font at ``size``."""
        pygame_module = _load_pygame()
        cached = self._font_cache.get(size)
        if cached is not None:
            return cached
        try:
            font = pygame_module.font.Font(str(_FONT_PATH), size)
        except Exception:
            logger.warning("Failed to load history font subset %s", _FONT_PATH, exc_info=True)
            font = pygame_module.font.Font(None, size)
        self._font_cache[size] = font
        return font

    def _wrap_text(self, text: str, font: Any, max_width: int) -> list[str]:
        """Wrap text to fit within ``max_width`` pixels."""
        stripped = text.strip()
        if not stripped:
            return []
        words = stripped.split()
        if not words:
            return self._split_long_word(stripped, font, max_width)
        lines: list[str] = []
        current = ""
        for word in words:
            candidates = [word]
            if font.size(word)[0] > max_width:
                candidates = self._split_long_word(word, font, max_width)
            for candidate in candidates:
                trial = candidate if not current else f"{current} {candidate}"
                if font.size(trial)[0] <= max_width:
                    current = trial
                    continue
                if current:
                    lines.append(current)
                current = candidate
        if current:
            lines.append(current)
        return lines

    def _truncate_to_single_line(self, text: str, font: Any, max_width: int) -> str:
        """Return ``text`` clamped to one line no wider than ``max_width`` pixels.

        If the full text already fits it is returned unchanged; otherwise it is cut
        and an ellipsis is appended so the rendered surface never overflows the
        reserved width (e.g. into the top-right exit button region).
        """
        stripped = text.strip()
        if not stripped:
            return ""
        if font.size(stripped)[0] <= max_width:
            return stripped
        ellipsis = "…"
        ellipsis_width = font.size(ellipsis)[0]
        if ellipsis_width > max_width:
            return ""
        budget = max_width - ellipsis_width
        truncated = ""
        for char in stripped:
            trial = truncated + char
            if font.size(trial)[0] > budget:
                break
            truncated = trial
        return f"{truncated}{ellipsis}"

    def _split_long_word(self, word: str, font: Any, max_width: int) -> list[str]:
        """Split one long word by character so it cannot overflow its container."""
        lines: list[str] = []
        current = ""
        for char in word:
            trial = f"{current}{char}"
            if current and font.size(trial)[0] > max_width:
                lines.append(current)
                current = char
            else:
                current = trial
        if current:
            lines.append(current)
        return lines

    def _load_sprites(self, asset_dir: Path) -> None:
        pygame_module = _load_pygame()
        missing: list[str] = []
        for expression in CharacterExpression:
            path = asset_dir / f"{expression.value}.png"
            if path.exists():
                try:
                    self._sprites[expression] = pygame_module.image.load(str(path)).convert_alpha()
                except Exception:
                    logger.warning("Failed to load sprite %s", path, exc_info=True)
                    self._sprites[expression] = None
                    missing.append(expression.value)
            else:
                self._sprites[expression] = None
                missing.append(expression.value)

        if missing:
            logger.warning("Missing character sprites (using NEUTRAL fallback): %s", missing)

    def _load_language_indicators(self, asset_dir: Path) -> None:
        """Load optional session language badge images."""
        pygame_module = _load_pygame()
        self._language_indicators.clear()
        for lang in ("ko", "en"):
            path = asset_dir / "indicator" / f"flag_{lang}.png"
            if not path.exists():
                logger.warning("Missing language indicator %s; using generated fallback", path)
                self._language_indicators[lang] = self._build_language_indicator_fallback(lang)
                continue
            try:
                surface = pygame_module.image.load(str(path)).convert_alpha()
                if surface.get_size() != (_INDICATOR_SIZE, _INDICATOR_SIZE):
                    surface = pygame_module.transform.smoothscale(
                        surface,
                        (_INDICATOR_SIZE, _INDICATOR_SIZE),
                    )
                self._language_indicators[lang] = surface
            except Exception:
                logger.warning(
                    "Failed to load language indicator %s; using generated fallback",
                    path,
                    exc_info=True,
                )
                self._language_indicators[lang] = self._build_language_indicator_fallback(lang)

    def _build_language_indicator_fallback(self, lang: str) -> pygame.Surface:
        """Return a generated visible badge fallback for a missing language indicator."""
        pygame_module = _load_pygame()
        surface = pygame_module.Surface(
            (_INDICATOR_SIZE, _INDICATOR_SIZE),
            pygame_module.SRCALPHA,
        )
        color = _INDICATOR_FALLBACK_COLORS.get(lang, (220, 220, 220, 255))
        radius = _INDICATOR_SIZE // 2
        pygame_module.draw.circle(surface, color, (radius, radius), radius)
        return cast(pygame.Surface, surface)

    def _portal_button_icon(self, active: bool) -> pygame.Surface:
        """Return a cached download-glyph surface for the portal button state."""
        cached = self._portal_button_icons.get(active)
        if cached is not None:
            return cached
        icon = self._build_portal_button_icon(active)
        self._portal_button_icons[active] = icon
        return icon

    def _build_portal_button_icon(self, active: bool) -> pygame.Surface:
        """Build the generated gray download glyph for the portal button."""
        pygame_module = _load_pygame()
        surface = pygame_module.Surface(
            (_INDICATOR_SIZE, _INDICATOR_SIZE),
            pygame_module.SRCALPHA,
        )
        alpha = 255 if active else _PORTAL_BUTTON_DISABLED_ALPHA
        color = (*_PORTAL_BUTTON_COLOR, alpha)
        center_x = _INDICATOR_SIZE // 2
        pygame_module.draw.line(surface, color, (center_x, 12), (center_x, 28), width=4)
        pygame_module.draw.polygon(
            surface,
            color,
            [
                (center_x - 9, 25),
                (center_x, 34),
                (center_x + 9, 25),
            ],
        )
        pygame_module.draw.line(surface, color, (14, 36), (34, 36), width=4)
        pygame_module.draw.line(surface, color, (14, 32), (14, 36), width=4)
        pygame_module.draw.line(surface, color, (34, 32), (34, 36), width=4)
        return cast(pygame.Surface, surface)

    def _blit_portal_button(
        self,
        *,
        active: bool,
        layout_generation: int | None = None,
        pressed_target: HitTarget | None = None,
    ) -> _HitRegion | None:
        """Blit the download portal button and return its idle hit region."""
        if self._display is None:
            return None
        button = self._portal_button_icon(active)
        pressed = active and pressed_target is not None and pressed_target.kind == "portal_activate"
        if pressed:
            button = self._pressed_portal_button(button)
        self._display.blit(button, self._portal_button_blit_position(button))
        return self._portal_hotspot_hit_region(layout_generation)

    def _portal_button_blit_position(self, button: pygame.Surface) -> tuple[int, int]:
        """Return the centered, clipped top-left position for the portal button."""
        offset_x = (_INDICATOR_SIZE - button.get_width()) // 2
        offset_y = (_INDICATOR_SIZE - button.get_height()) // 2
        x = min(
            max(_PORTAL_HOTSPOT_POSITION[0] + offset_x, 0),
            _CANVAS_SIZE[0] - button.get_width(),
        )
        y = min(
            max(_PORTAL_HOTSPOT_POSITION[1] + offset_y, 0),
            _CANVAS_SIZE[1] - button.get_height(),
        )
        return (x, y)

    def _pressed_portal_button(self, button: pygame.Surface) -> pygame.Surface:
        """Return a shrunk and darkened portal button for press feedback."""
        if self._portal_button_pressed_icon is None:
            self._portal_button_pressed_icon = self._pressed_language_indicator(button)
        return self._portal_button_pressed_icon

    def _draw_portal_status(self, text: str) -> None:
        """Draw transient portal status text next to the top-left button."""
        if self._display is None:
            return
        pygame_module = _load_pygame()
        panel_left = _PORTAL_HOTSPOT_POSITION[0] + _INDICATOR_SIZE + _PORTAL_STATUS_GAP
        max_text_width = min(
            _PORTAL_STATUS_MAX_WIDTH,
            _CANVAS_SIZE[0] - panel_left - _PORTAL_STATUS_PADDING_X,
        )
        font = self._fit_font(text, _PORTAL_STATUS_FONT_SIZE, max_text_width)
        lines = self._wrap_text(text, font, max_text_width)
        if not lines:
            return
        surfaces = [font.render(line, True, _TEXT_FG) for line in lines]
        text_width = max(surface.get_width() for surface in surfaces)
        text_height = sum(surface.get_height() for surface in surfaces)
        text_height += _PORTAL_STATUS_LINE_GAP * max(0, len(surfaces) - 1)
        panel_width = text_width + (_PORTAL_STATUS_PADDING_X * 2)
        panel_height = text_height + (_PORTAL_STATUS_PADDING_Y * 2)
        panel_top = _PORTAL_HOTSPOT_POSITION[1] + (_INDICATOR_SIZE - panel_height) // 2
        panel_top = max(_INDICATOR_INSET, panel_top)
        panel_rect = pygame_module.Rect(panel_left, panel_top, panel_width, panel_height)
        pygame_module.draw.rect(
            self._display,
            _TEXT_BG,
            panel_rect,
            border_radius=_PORTAL_STATUS_RADIUS,
        )
        pygame_module.draw.rect(
            self._display,
            _TEXT_PANEL,
            panel_rect,
            width=1,
            border_radius=_PORTAL_STATUS_RADIUS,
        )
        text_y = panel_rect.top + _PORTAL_STATUS_PADDING_Y
        for surface in surfaces:
            self._display.blit(surface, (panel_rect.left + _PORTAL_STATUS_PADDING_X, text_y))
            text_y += surface.get_height() + _PORTAL_STATUS_LINE_GAP

    def _blit_language_indicator(
        self,
        lang: str | None,
        *,
        animation_frames: int = 0,
        layout_generation: int | None = None,
        pressed_target: HitTarget | None = None,
    ) -> _HitRegion | None:
        """Blit the current language badge and return its idle hit region."""
        if lang is None or self._display is None:
            return None
        indicator = self._language_indicators.get(lang)
        if indicator is None:
            logger.warning("Language indicator %s unavailable; using generated fallback", lang)
            indicator = self._build_language_indicator_fallback(lang)
            self._language_indicators[lang] = indicator
        pressed = pressed_target is not None and pressed_target.kind == "language_toggle"
        if animation_frames > 0:
            indicator = self._scale_indicator_for_animation(indicator, animation_frames)
        if pressed:
            indicator = self._pressed_language_indicator(indicator)
        self._display.blit(indicator, self._indicator_blit_position(indicator))
        if layout_generation is None:
            return None
        return self._language_indicator_hit_region(layout_generation)

    def _indicator_blit_position(self, indicator: pygame.Surface) -> tuple[int, int]:
        """Return the centered, clipped top-right position for an indicator surface."""
        offset_x = (_INDICATOR_SIZE - indicator.get_width()) // 2
        offset_y = (_INDICATOR_SIZE - indicator.get_height()) // 2
        x = min(max(_INDICATOR_POSITION[0] + offset_x, 0), _CANVAS_SIZE[0] - indicator.get_width())
        y = min(
            max(_INDICATOR_POSITION[1] + offset_y, 0),
            _CANVAS_SIZE[1] - indicator.get_height(),
        )
        return (x, y)

    def _pressed_language_indicator(self, indicator: pygame.Surface) -> pygame.Surface:
        """Return a shrunk and darkened indicator for press feedback."""
        pygame_module = _load_pygame()
        target_size = (
            max(1, indicator.get_width() - _INDICATOR_PRESS_SHRINK_PX),
            max(1, indicator.get_height() - _INDICATOR_PRESS_SHRINK_PX),
        )
        pressed = pygame_module.transform.smoothscale(indicator, target_size)
        darken_rgb = _darken_color((255, 255, 255), 0.78)
        pressed.fill((*darken_rgb, 255), special_flags=pygame_module.BLEND_RGBA_MULT)
        return cast(pygame.Surface, pressed)

    def _language_indicator_hit_region(self, layout_generation: int) -> _HitRegion:
        """Return the padded direct-touch region for the idle language badge."""
        pygame_module = _load_pygame()
        rect = pygame_module.Rect(*_INDICATOR_POSITION, _INDICATOR_SIZE, _INDICATOR_SIZE)
        touch_rect = rect.inflate(_INDICATOR_TOUCH_PAD * 2, _INDICATOR_TOUCH_PAD * 2).clip(
            pygame_module.Rect(0, 0, _CANVAS_SIZE[0], _CANVAS_SIZE[1])
        )
        return _HitRegion(
            rect=(touch_rect.left, touch_rect.top, touch_rect.width, touch_rect.height),
            target=HitTarget("language_toggle", None, layout_generation),
        )

    def _portal_hotspot_hit_region(self, layout_generation: int | None) -> _HitRegion | None:
        """Return the top-left double-tap hotspot for portal activation.

        The region mirrors the language badge's padded touch box into the
        top-left corner (the top-right corner is the language toggle), so it
        never overlaps it.
        """
        if layout_generation is None:
            return None
        pygame_module = _load_pygame()
        rect = pygame_module.Rect(*_PORTAL_HOTSPOT_POSITION, _INDICATOR_SIZE, _INDICATOR_SIZE)
        touch_rect = rect.inflate(_INDICATOR_TOUCH_PAD * 2, _INDICATOR_TOUCH_PAD * 2).clip(
            pygame_module.Rect(0, 0, _CANVAS_SIZE[0], _CANVAS_SIZE[1])
        )
        return _HitRegion(
            rect=(touch_rect.left, touch_rect.top, touch_rect.width, touch_rect.height),
            target=HitTarget("portal_activate", None, layout_generation),
        )

    def _scale_indicator_for_animation(
        self,
        indicator: pygame.Surface,
        animation_frames: int,
    ) -> pygame.Surface:
        """Return a centered pulse-scaled language indicator surface."""
        pygame_module = _load_pygame()
        elapsed_frames = _INDICATOR_ANIM_TOTAL_FRAMES - animation_frames + 1
        denominator = max(1, _INDICATOR_ANIM_TOTAL_FRAMES)
        phase = max(0.0, min(1.0, elapsed_frames / denominator))
        scale = 1.0 + (_INDICATOR_ANIM_MAX_SCALE - 1.0) * math.sin(math.pi * phase)
        size = max(1, int(round(_INDICATOR_SIZE * scale)))
        return cast(
            pygame.Surface,
            pygame_module.transform.smoothscale(indicator, (size, size)),
        )

    def _discover_frame_sequences(self, asset_dir: Path) -> None:
        frames_root = asset_dir / "frames"
        self._frame_paths.clear()
        for expression in CharacterExpression:
            frame_dir = frames_root / expression.value
            if not frame_dir.is_dir():
                continue
            self._frame_paths[expression] = sorted(frame_dir.glob("*.png"))

    def _ensure_active_frames(self, expression: CharacterExpression) -> bool:
        if self._active_frames is not None and self._active_frames_expr is expression:
            return True

        self._active_frames = None
        self._active_frames_expr = None

        if expression in self._failed_frame_expressions:
            return False

        frame_paths = self._frame_paths.get(expression)
        if frame_paths is None:
            return False
        if not frame_paths:
            logger.warning(
                "No animation frame files found for %s; using static sprite",
                expression.value,
            )
            self._failed_frame_expressions.add(expression)
            return False

        frames: list[pygame.Surface] = []
        pygame_module = _load_pygame()
        for path in frame_paths:
            try:
                frames.append(pygame_module.image.load(str(path)).convert_alpha())
            except Exception:
                logger.warning(
                    "Failed to load animation frame %s; using static sprite for %s",
                    path,
                    expression.value,
                    exc_info=True,
                )
                self._failed_frame_expressions.add(expression)
                return False

        self._active_frames = frames
        self._active_frames_expr = expression
        self._frame_index = 0
        return True

    def _resolve_render_surface(
        self,
        expression: CharacterExpression,
    ) -> tuple[CharacterExpression, pygame.Surface, bool]:
        animation_expression = self._animation_expression(expression)
        if self._ensure_active_frames(animation_expression):
            frames = self._active_frames
            assert frames is not None
            frame = frames[self._frame_index % len(frames)]
            return animation_expression, frame, True

        sprite = self._sprites.get(expression)
        if sprite is not None:
            return expression, sprite, False

        if animation_expression is not CharacterExpression.NEUTRAL and self._ensure_active_frames(
            CharacterExpression.NEUTRAL
        ):
            frames = self._active_frames
            assert frames is not None
            frame = frames[self._frame_index % len(frames)]
            return CharacterExpression.NEUTRAL, frame, True

        effective_expr, sprite = self._resolve_sprite(expression)
        return effective_expr, sprite, False

    def _animation_expression(self, expression: CharacterExpression) -> CharacterExpression:
        if expression is CharacterExpression.IDLE:
            return CharacterExpression.NEUTRAL
        return expression

    def _resolve_sprite(
        self,
        expression: CharacterExpression,
    ) -> tuple[CharacterExpression, pygame.Surface]:
        sprite = self._sprites.get(expression)
        if sprite is not None:
            return expression, sprite

        fallback = self._sprites.get(CharacterExpression.NEUTRAL)
        if fallback is not None:
            return CharacterExpression.NEUTRAL, fallback

        if self._black_fallback is None:
            pygame_module = _load_pygame()
            self._black_fallback = pygame_module.Surface(_CANVAS_SIZE)
            self._black_fallback.fill(_BLACK)
        return CharacterExpression.NEUTRAL, self._black_fallback
