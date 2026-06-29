# ADR 0095: Pygame Character Renderer Runtime Adoption (Phase 2 B-2)

- **Status**: Accepted (2026-05-27 — PR #140 squash merge `3105d07` integrated Plan v4 §10 P1/P2/P3 deliverables into dev; Jetson windowed + fullscreen smoke gates remain deferred to next session per Plan §10 P3 manual gate. Prior history: Proposed v3 — Codex r3 PM-closure n-01/m-01/p-01/m-03 ACCEPT.)
- **Date**: 2026-05-27
- **Authority**: `Dev_Plan/2026-05-27-phase2-b2-character-display-expression-plan.md` v2 final approval cycle (Codex `reviewer` r1 verdict NEEDS-FIX → r2 dispatch)
- **Related**: ADR 0080 (4inch HDMI Touch IPS LCD hardware adoption — H/W decision only, intentionally deferred runtime renderer to a later Plan), ADR 0079 (LED retirement — touchscreen is sole visual feedback channel), Touchscreen Plan v4 §4 Phase 2 (architecture skeleton this ADR realizes)

## Context

Phase 2 B-1 (PR #139, commit `a541405`) landed `CharacterRenderer(Protocol)` + `NullCharacterRenderer` + `CharacterExpression(NEUTRAL)` stub in `core/session_manager.py` + `core/character_expression.py`. The 4inch HDMI LCD currently shows a blank screen because no concrete renderer exists. Phase 2 B-2 introduces:

1. A new **runtime dependency** — `pygame>=2.6.0,<2.7` (binary SDL2 wrapper, not pure-Python). ADR 0080 §4 explicitly deferred this dependency: "pygame ... is deferred until the `hardware/display.py` runtime Plan (no orphan deps in Plan v4)." Phase 2 B-2 is that runtime Plan, but the implementation lives in **`core/character_renderer.py`** instead of `hardware/display.py` (j-04 — module placement decision: renderer is layer-internal to `core/` because it consumes `core.character_expression` + `core.session_manager.CharacterRenderer` Protocol, not a hardware abstraction). This ADR therefore **supersedes ADR 0080 §4's placeholder `hardware/display.py` runtime path** with `core/character_renderer.py`.
2. A new **rendering subsystem** — `PygameCharacterRenderer` class owning a UI thread, SDL2 display surface, sprite loading + caching, expression callback queue, and lifecycle.
3. A new **threading model** — single dedicated UI thread for `pygame.init` + `display.set_mode` + `event.pump` + `blit` + `flip` + `pygame.quit`. Only **`on_expression()`** callback enqueues into a `threading.Lock`-protected best-effort coalesced single-slot field (p-01 — `on_state_change()` is a renderer no-op; SessionManager._transition handles the state→expression mapping and forwards via `on_expression()`).
4. A new **runtime selection policy** — `MUNGI_RENDERER` env flag (`pygame` / `null` / `auto`); auto fallbacks to null when no display is available (headless SSH session, CI).

Per CLAUDE.md §7 rule 4, "An ADR is required for architecture / runtime-path / safety-policy changes." Each of items 1-4 is an architecture/runtime-path change. ADR 0080 covers hardware only; this ADR covers the runtime renderer that consumes the hardware.

Codex `reviewer` r1 (finding `g-01`, BLOCK) flagged this gap and required ADR creation before Plan implementation.

## Decision

### D1 — Renderer implementation = Pygame

Pygame 2.6.0–2.6.x is selected as the rendering backend.

- **Why Pygame, not raw SDL2 / pyglet / Kivy / Qt**:
  - Pygame is already deferred-listed in ADR 0080 §4 — adoption removes the orphan-dep state.
  - Mature aarch64 wheel on PyPI for CPython 3.10 (verified by Codex r1 axis b: `pygame 2.6.0` + `2.6.1` aarch64 wheels exist).
  - Single Python package — no separate compositor or windowing toolkit.
  - Sprite + Surface + display API matches our scope (static PNG fullscreen swap).
  - SDL2 backend selection (X11 / kmsdrm / wayland / dummy) supports both Jetson production and headless CI.
- **Why not raw SDL2 (pysdl2)**: extra C-level complexity, no asset-loading conveniences.
- **Why not pyglet/Kivy**: heavier, OpenGL-first, asset pipeline overkill for static sprite swap.
- **Why not Qt (PyQt/PySide)**: huge dependency footprint, license complexity (LGPL/Commercial split), aarch64 wheel availability less mature.

### D2 — Dependency placement = `Dev_Plan/requirements-core.txt`

Pygame is added to `Dev_Plan/requirements-core.txt` (the cross-platform common requirements), NOT `requirements-jetson.txt`.

- **Rationale**: tests must run on developer Windows PCs + Jetson + GitHub Actions Ubuntu CI. Splitting Pygame into `requirements-jetson.txt` would prevent developer-PC testing.
- **Pin policy**: `pygame>=2.6.0,<2.7` upper bound prevents accidental major-version drift that could break aarch64 wheel availability. The Jetson-verified lockfile (post-Phase 0 protocol per `Dev_Plan/requirements-jetson.txt`) pins the exact version.
- **Jetson source-build fallback**: if the aarch64 wheel install fails on a specific Jetson host, follow the official `https://www.pygame.org/wiki/CompileUbuntu` apt-dep list verbatim (single source of truth — Phase 2 B-2 Plan §8 R1 does NOT maintain an inline package list to avoid drift). m-01 closure.
- **Orphan-dep notice removal**: `Dev_Plan/requirements-jetson.txt:117-121` (the deferred pygame note from Plan v2.1+v2.7) is removed in the same PR. ADR 0080 §4 deferred-dep status updates to "adopted by ADR 0095."

### D3 — Threading model = single UI thread

`PygameCharacterRenderer` owns exactly one dedicated UI thread (`_ui_thread`).

- The UI thread is the **only** thread that calls Pygame display / event API: `pygame.init`, `pygame.display.set_mode`, `pygame.event.pump`, `surface.blit`, `pygame.display.flip`, `pygame.quit`. **No exceptions** — joiner thread (the thread that calls `close()`) MUST NOT invoke any Pygame API, even on join timeout (j-02 — degraded shutdown logs a warning instead).
- Callbacks called by `SessionManager`'s main thread:
  - **`on_state_change(state)` — no-op for PygameCharacterRenderer** (j-01). State→Expression mapping is owned by `SessionManager._transition`, which forwards the mapped expression via `on_expression()`. The renderer does not know about `SessionState`.
  - **`on_expression(expression)` — enqueues**: enters a `threading.Lock`, writes into a single-slot field `_pending_expression: CharacterExpression | None` (best-effort coalesced; latest lock write wins; no global ordering guarantee vs state transitions), notifies `_wake: threading.Event`, returns. Does not touch Pygame.
  - The UI thread loop: pump events → check `_wake` → if set, atomically read+clear `_pending_expression` under lock → resolve sprite (NEUTRAL fallback) → blit → flip → wait on `_wake` with timeout (1/60 sec) for next event or render heartbeat.
- `close()`: set `_stop_event` → notify `_wake` → join UI thread with timeout (≤ 2.5s, target ≤ 2.0s) → **`pygame.quit()` is invoked exclusively inside the UI thread before it exits**. If join times out, the joiner logs a warning ("degraded shutdown") and returns; the orphan UI thread is cleaned up at process exit. **The joiner never calls `pygame.quit()`** (j-02/h-02 — strict UI-thread-only invariant).
- **Why single UI thread, not daemon thread with pump in main**: SDL2 documentation (`SDL_PumpEvents`) requires events to be pumped in the thread that initialized the video subsystem. Pygame's official event-loop documentation echoes this. The v1.1 plan's daemon thread + main-thread pump model was rejected by Codex r1 (finding c-01, BLOCK).

### D4 — SDL backend selection

Backend is selected by environment variable resolution order:

1. `MUNGI_SDL_DRIVER` (user override, e.g., `kmsdrm`, `x11`, `fbdev`, `wayland`, `dummy`)
2. `SDL_VIDEODRIVER` (Pygame standard)
3. Platform default (Linux Pygame default = X11; Windows default = directx/windib; in headless CI, set `SDL_VIDEODRIVER=dummy`)

`MUNGI_RENDERER=auto` mode checks display availability across **all four** signals (j-03 — `SDL_VIDEODRIVER` included to match Plan §4 D14 + §6.4):

- `DISPLAY` env set → X11 available
- `WAYLAND_DISPLAY` env set → Wayland available
- `SDL_VIDEODRIVER` env in `{kmsdrm, fbdev, dummy}` → direct framebuffer or headless
- `MUNGI_SDL_DRIVER` env in `{kmsdrm, fbdev, dummy}` → direct framebuffer or headless
- None of the above → fall back to `NullCharacterRenderer` (no display)

### D5 — Display mode

- Default: 720×720 `pygame.FULLSCREEN` (matches Waveshare 4inch LCD per ADR 0080).
- `MUNGI_RENDERER_WINDOWED=1` → 720×720 windowed (developer PC debug).
- Init-time validation: `pygame.display.list_modes()` or `pygame.display.Info()` checked before `set_mode`. If 720×720 unavailable, attempt `pygame.SCALED` flag fallback; if still unavailable, fall back to `NullCharacterRenderer` with warning log.

### D6 — Renderer lifecycle ownership = `SessionManager`

- `SessionManager.shutdown()` (new public method, **TERMINAL** — call only after `run()` returns or before `run()` starts; raises `RuntimeError` if called during active run) owns ordered cleanup: **`audio_capture.stop()` → `renderer.close()` → `mm.cancel_preload()` + `mm.unload_stt(force=False)`** (n-01 align with Plan v4 §6.3: audio stop FIRST to block new InputStream callbacks before potentially blocking renderer join).
- `scripts/demo_live.py` finally block calls `session_manager.shutdown()`. Direct `renderer.close()` from the finally block is not required.
- `SessionManager.add_state_listener(new_renderer)` is **PRE-RUN ONLY** (n-01 runtime-swap ban): callable only before `run()` has been invoked. SessionManager tracks `_has_run: bool` latch; if `_has_run` is True, `add_state_listener()` raises `RuntimeError`. When called pre-run, `old_renderer.close()` is invoked BEFORE the new renderer assignment (single-owner invariant, close-before-replace).
- This consolidates lifecycle into one place (single owner principle) and resolves Codex r1 finding c-04 (MAJOR) + r3 finding n-01 (MAJOR — runtime-swap ban + ordering alignment).

### D7 — Asset spec

- Sprite format: 720×720 PNG RGBA. File name = lowercase enum value (e.g., `idle.png`, `listening.png`).
- Location: `assets/character/`.
- Fallback chain: requested expression → NEUTRAL → solid black surface + 1-time warning log.
- Placeholder asset generation: `scripts/generate_character_placeholders.py` (ASCII-only labels, no CJK / no emoji — avoids Jetson font availability dependency). External illustrator deliverable replaces placeholders in-place; no code change required.
- `assets/character/README.md` ships the dog-behavior ↔ expression mapping table and external-illustrator handoff guide.

### D8 — Verification gates

1. **Headless unit tests** (CI mandatory): `SDL_VIDEODRIVER=dummy` via `tests/conftest.py` autouse fixture. Cover renderer init + asset load + expression callback + lifecycle + concurrent-call + render-loop exception isolation. ≥ 70% coverage threshold (CLAUDE.md §7).
2. **Jetson windowed smoke** (manual, required gate per Plan §10 P3): `MUNGI_RENDERER=pygame MUNGI_RENDERER_WINDOWED=1 python -m scripts.demo_live` on Jetson with HDMI LCD connected; verify each state-driven sprite swap visible.
3. **Jetson fullscreen smoke** (manual, required gate per Plan §10 P3): `MUNGI_RENDERER=pygame python -m scripts.demo_live`; verify 720×720 fullscreen on the LCD.

## Consequences

### Positive

- Visual feedback channel becomes live (closes ADR 0080 deferred runtime gap).
- Single-thread SDL ownership eliminates thread-safety races at design time.
- Renderer lifecycle is centralized in SessionManager (single-owner invariant).
- Headless test path supports CI and developer PC without display.
- Asset spec + placeholder pipeline enables external illustrator handoff without blocking integration.

### Negative / trade-offs

- New binary runtime dependency (SDL2 native libs). Source-build fallback requires apt deps on Jetson (documented).
- Single-UI-thread invariant tightens callback contract — future renderers (e.g., parental UI overlay in 5-phase Phase 3) must reuse the same thread or define a new ownership model.
- Pygame's font subsystem cannot render Korean glyphs on stock Jetson without a CJK font installed — placeholder assets are intentionally ASCII-only to avoid the dependency. External illustrator assets carry visual style instead.
- Jetson smoke gates are manual — CI cannot fully validate the production rendering path.

### Out of scope (deferred to later ADRs / Plans)

- Frame-based animation / sprite-sheet rendering (Phase 2 B-2 v1 is static PNG only)
- TTS playback-start hook → `SPEAKING` expression emit (separate Phase 2 follow-up plan)
- LLM sentiment → expression auto-emit (HAPPY/SAD/CONCERNED) (separate Phase 2 follow-up plan)
- Parental mode PIN UI / learning mode UI (5-phase Phase 3, separate ADR + Plan)
- Touch input → expression interaction (out — touch is wake/parent trigger only)
- Multi-monitor / display hot-swap

## Validation criteria

- [x] Plan v3 (`Dev_Plan/2026-05-27-phase2-b2-character-display-expression-plan.md`) references this ADR in §4 D7 (explicit cite line) + §13 decision log
- [x] `Dev_Plan/requirements-core.txt` adds `pygame>=2.6.0,<2.7` with comment citing ADR 0095
- [x] `Dev_Plan/requirements-jetson.txt` removes the deferred-pygame orphan note (lines 117-121) and adds an ADR-0095-adopted marker
- [x] **ADR 0080 §4 status updated in the SAME PR** (j-05) — deferred-dep marker replaced with "(superseded by ADR 0095, adopted 2026-05-27)" pointer. No separate follow-up commit
- [x] `core/character_renderer.py` implements the single-UI-thread model per D3 (joiner-thread Pygame API calls forbidden, including degraded-shutdown path)
- [x] `tests/conftest.py` autouse fixture sets `SDL_VIDEODRIVER=dummy`
- [ ] Jetson windowed smoke gate passes (manual, recorded in Plan §10 P3 deliverable evidence) — **DEFERRED to next session per Plan §10 P3 manual gate**
- [ ] Jetson fullscreen smoke gate passes (manual, recorded in Plan §10 P3 deliverable evidence) — **DEFERRED to next session per Plan §10 P3 manual gate**

## Reference map

| Artifact | Role |
|---|---|
| ADR 0080 | Display H/W adoption (this ADR closes the deferred runtime portion) |
| ADR 0079 | LED retirement (touchscreen = sole visual channel) |
| Plan v2 (`Dev_Plan/2026-05-27-phase2-b2-character-display-expression-plan.md`) | Implementation plan |
| Codex r1 report (`docs/runbooks/2026-05-27-phase2-b2-character-display-plan-review-round1.md`) | r1 review that surfaced g-01 BLOCK requiring this ADR |
| Touchscreen Plan v4 §4 Phase 2 | Architecture skeleton |
| Pygame docs `event.pump` | Threading guidance source |
| SDL2 docs `SDL_PumpEvents` | Threading guidance source |
| `https://www.pygame.org/wiki/CompileUbuntu` | apt source-build fallback dep list |

## Update (2026-05-27 — Acceptance event)

- ADR status transitioned from `Proposed v3` to `Accepted` following PR #140 squash merge (`3105d07`) that landed Phase 2 B-2 Plan v4 §10 P1/P2/P3 deliverables into `dev`.
- 6 of 8 validation criteria are satisfied by the dev integration; 2 (Jetson windowed/fullscreen smoke) remain deferred to the next session per Plan §10 P3 manual gate (Jetson monitor/keyboard hookup + GUI session activation required).
- Implementation hotfix `fb7ddd4` switched `core/character_renderer.py` UI thread `daemon=False → daemon=True` to resolve Linux `Py_Finalize()` `threading.shutdown()` interpreter hang. This hardening is implementation-detail within the single-UI-thread model (D3) and the joiner-never-touches-pygame invariant (D3, h-02/j-02); original ADR decisions are unchanged.
- This Update note records the acceptance event for audit traceability; original Context / Decision / Consequences sections remain untouched per memory `feedback_adr_immutability`.

## Update (2026-05-29 — partial supersede by ADR 0097)

- This ADR's **D1** (engine = Pygame), **D3** (single-UI-thread Pygame model), **D5** (display mode set_mode), and **D7** (PNG sprite asset spec) are **superseded in part** by ADR 0097 (HTML+CSS Character Renderer).
- ADR 0097 introduces HtmlCharacterRenderer (WPE WebKit primary, WebKitGTK 4.1 fallback) as the new runtime primary; Pygame renderer is preserved as the fallback selector path (`MUNGI_RENDERER=pygame` explicit override + auto-cascade fallback).
- **D2** (dependency placement = `Dev_Plan/requirements-core.txt`), **D4** (SDL backend selection), **D6** (lifecycle ownership = SessionManager), **D8** (verification gates) remain **in force unchanged**. They continue to govern the Pygame fallback path.
- The 19 PNG assets from Phase 2 B-2 v5 (PR #141) are preserved as the visual base inside the new HTML wrappers (PNG-with-CSS-transform asset model per ADR 0097 D3); PNG investment is not lost.
- Pygame renderer + pygame dependency retirement is deferred to ADR 0097's P5 deferred phase (requires ≥ 2 weeks Jetson HTML soak first).
- Original Context / Decision / Consequences sections above remain untouched per `feedback_adr_immutability`. This Update note records the partial-supersede event for audit traceability.
- Authority: `Dev_Plan/2026-05-29-css-html-emoji-renderer-plan-v1.md` v3 (Codex-APPROVED 2026-05-29 r3 APPROVE WITH NOTES; 23/23 cumulative r1+r2 closures); user directive 2026-05-29 ("이모지는 css로 구현해서 html파일로 저장한 후 사용할꺼야"; "메모리 점유가 가장 적은 방법으로 랜더하는 걸 선택해").
