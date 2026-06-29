# ADR 0097: HTML+CSS Character Renderer (WPE WebKit primary, WebKitGTK fallback) — supersedes ADR 0095 D1/D3/D5/D7 in part

- **Status**: Accepted (2026-05-29 — user mini-approval 완료, Plan v3 §8 P-ADR gate 통과; P0 manual phase 진입 가능)
- **Date**: 2026-05-29
- **Authority**: `Dev_Plan/2026-05-29-css-html-emoji-renderer-plan-v1.md` v3 (Codex-APPROVED 2026-05-29 r3 APPROVE WITH NOTES; 23/23 cumulative r1+r2 closures)
- **Supersedes (in part)**: ADR 0095 D1 (engine = Pygame) / D3 (single-UI-thread Pygame model) / D5 (display mode set_mode) / D7 (PNG sprite asset spec). ADR 0095 D2 (dependency placement) / D4 (SDL backend selection) / D6 (lifecycle = SessionManager) / D8 (verification gates) remain in force unchanged.
- **Related**: ADR 0080 (4inch HDMI Touch IPS LCD adoption), ADR 0079 (LED retirement — touchscreen is sole visual feedback channel)
- **Memory refs**: `project_phase2_followup_emit_hooks` (Phase 2 B-2 v5 follow-up tracker; this Plan retires MP4 animation path in favor of CSS), `feedback_adr_immutability` (ADR 0095 modified only via Update section, never the original Decision)

## Context

Phase 2 B-2 v5 (PR #141, merged 2026-05-28) integrated 19 illustrator-produced 720×720 PNG character expressions via the Pygame renderer adopted in ADR 0095 (Accepted 2026-05-27). The illustrator also delivered 15 MP4 motion clips for animated states — but the Mungi product persona needs motion to feel alive, not a slideshow.

User directive (2026-05-29):
- "이모지는 css로 구현해서 html파일로 저장한 후 사용할꺼야"
- "PNG파일만 사용하면 움직임을 구현못하잖아 그래서 CSS로 움직임을 구현한 HTML을 사용하려는 거야"
- "메모리 점유가 가장 적은 방법으로 랜더하는 걸 선택해"

This pivots the renderer architecture from Pygame static PNG to an HTML+CSS animated renderer. The pivot triggers ADR 0095 partial supersede + a new ADR (this one) per CLAUDE.md §7 rule 4 and `feedback_adr_immutability`.

Plan v3 went through three Codex review rounds (r1 BLOCK with 26 findings → v2 → r2 PUSH BACK with 3 → v3 → r3 APPROVE WITH NOTES) and integrated 23/23 actionable findings before this ADR was drafted.

## Decision

### D1 — Renderer engine: WPE WebKit (candidate, pending Phase 0 confirmation)

- Primary: **WPE WebKit** (`libwpewebkit-1.0-3` + `wpebackend-fdo-1.0-1` + `gir1.2-wpewebkit-2.0` apt, Jetson Ubuntu 22.04 ARM64). Lowest-memory "real HTML+CSS3 animation" engine with stable aarch64 apt availability; designed for embedded set-top-box / kiosk use.
- Fallback: **WebKitGTK 4.1** (`libwebkit2gtk-4.1-0` + `gir1.2-webkit2-4.1` apt). Adopted if WPE Phase 0 A1/B1/B2 deliverables fail.
- "Candidate" until Phase 0 records the actually-working engine + backend + Python binding contract; final pinning happens at P0-closeout (Plan v3 §8 + §11 D11).
- This supersedes ADR 0095 D1 (engine = Pygame) — Pygame becomes the fallback selector path only.

### D2 — Python binding: PyGObject with lazy import

- `gi.require_version("WPEWebKit", "2.0")` + `from gi.repository import WPEWebKit, GLib` for WPE.
- `gi.require_version("WebKit2", "4.1")` + `from gi.repository import WebKit2, GLib` for fallback.
- Lazy import (inside `_HtmlEngineController` concrete impls, not at module top) with combined mypy ignore: `# type: ignore[import-not-found, import-untyped]` per CLAUDE.md §5 optional-dep rule.

### D3 — Asset model: PNG-with-CSS-transform (single `<img>` element)

Each HTML asset uses the existing 720×720 RGBA PNG (from Phase 2 B-2 v5 PR #141) as the visual base, with CSS3 `transform` / `opacity` / `filter` animations applied to the single `<img>` element. No SVG, no path morph, no eyelid/mouth/ear/tail sub-elements (visual identity stays in the PNG).

- Per-expression animation timing per Plan v3 §6.4 (19 expressions, all implementable in transform/opacity/filter only).
- CSP mandatory in every HTML asset: `<meta http-equiv="Content-Security-Policy" content="default-src 'self'; img-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'none';">`.
- This supersedes ADR 0095 D7 (PNG sprite asset spec): PNGs are still 720×720 RGBA under `assets/character/`, but the runtime now serves them via HTML wrappers under `assets/character/html/<expr>.html`.
- Pure CSS-layered shapes (eye / ear / mouth / tail / heart sub-elements) are deferred to a future Phase 5+ visual upgrade.

### D4 — Threading: HtmlCharacterRenderer with `_HtmlEngineController` Protocol seam

- HtmlCharacterRenderer owns one dedicated UI thread that hosts a GLib main context + WPE/WebKit2 WebView controller (UI process side).
- WebView automatically spawns a Web subprocess; lifecycle managed by upstream library.
- `_HtmlEngineController` Protocol (private) provides the implementation seam: production `_WPEEngineController` / `_WebKitGTKEngineController` are concrete impls; tests inject `_MockEngineController` (keeps `gi.repository` out of CI test path).
- `on_state_change()` no-op (parity with PygameCharacterRenderer); `on_expression()` enqueues + signals wake event under lock.
- `close()` captures child PIDs at init, joins UI thread, post-close orphan-PID check via `os.kill(pid, 0)` — orphan WARN if Web subprocess fails to exit within `shutdown_timeout` (default 3 s).
- This supersedes ADR 0095 D3 (single-UI-thread Pygame model) — the single-UI-thread invariant is preserved, but WebView's multi-process architecture is honored via the controller seam.

### D5 — Display mode: 720×720 fullscreen (default) / windowed (dev override)

- Default: fullscreen on Jetson 4inch LCD per ADR 0080.
- `MUNGI_RENDERER_WINDOWED=1` → windowed (developer PC debug).
- WPE backend selection: `WPE_BACKEND` env (`fdo` / `fdo-drm` / `headless`); Phase 0 B2 chooses the Jetson default backend.
- This supersedes ADR 0095 D5 (Pygame `display.set_mode`) — Pygame fallback retains the old D5 behavior; HTML primary uses WebView backend.

### D6 — Selector: `core/character_renderer.create_renderer()` 3-step capability check

- Resolution order: env override (`MUNGI_RENDERER=html|pygame|null|auto`) > `auto` cascade (HTML → Pygame → Null).
- HTML 3-step capability check: (a) `gi.repository` import succeeds, (b) backend init succeeds (instantiate engine controller + close), (c) asset root exists with at least one HTML file.
- Forced `MUNGI_RENDERER=html` raises on any failure; `auto` falls through to Pygame.

### D7 — Memory budget

- WebView process group RSS < 150 MB (Phase 0 O2 gate).
- Combined system peak ≤ 5500 MB (WebView + LLM resident + 1 voice turn; matches `docs/runbooks/baseline-stack-and-models.md:22-24` invariant).
- Critical guard: 6000 MB (CLAUDE.md §6).

### D8 — Codex implementation role split (per CLAUDE.md §8)

Plan v3 §8 rollout:
- P-ADR: PM direct (this ADR + ADR 0095 Update section)
- P0: PM + user manual (Jetson Phase 0 verification — apt + WPE smoke + memory measurement)
- P0-closeout: PM direct (plan/spec refresh with engine/backend pinning)
- P1a: Codex `feature` (core/character_renderer.py + selector + `_HtmlEngineController` Protocol)
- P1b: Codex `test` (renderer/selector unit tests with `_MockEngineController` — no asset-tree assertions)
- P2a: Codex `feature` (assets/character/html/*.html + _base.css + _animations.css — PM scope override authorized for new sub-path)
- P2b: Codex `test` (asset validator + 19-file enum-completeness assertion)
- P3: Codex `platform` (scripts/demo_live.py selector wiring + Dev_Plan/requirements-jetson.txt apt-dep note + docs/runbooks/jetson-html-renderer-setup.md)
- P4: PM + user manual (Jetson windowed + fullscreen smoke + load-changed signal capture + memory measurement)
- P5: deferred (Pygame renderer removal after ≥ 2 weeks Jetson HTML soak)

### D9 — ADR 0095 supersede mechanism

Per `feedback_adr_immutability`: ADR 0095's original Context / Decision / Consequences sections remain untouched. Append a `## Update (2026-05-29 — partial supersede)` section to ADR 0095 with:
- Pointer to this ADR
- Listing of which Decisions are superseded (D1/D3/D5/D7) and which remain in force (D2/D4/D6/D8)
- Statement that Pygame renderer is preserved as fallback selector path; not retired in v1/v2

## Consequences

### Positive

- Character motion becomes possible (PNG-only is static; user-rejected MP4 path retired)
- WPE WebKit is the lowest-memory real HTML+CSS3 engine that fits Jetson 8 GB combined budget
- PNG-with-CSS-transform asset model is immediately implementable (no SVG authoring labor)
- `_HtmlEngineController` Protocol isolates `gi.repository` from CI test path (mock-based unit tests)
- Pygame renderer preservation as fallback means zero blast-radius if HTML path fails on Jetson
- ADR 0095 Decision body unchanged → audit traceability preserved

### Negative / trade-offs

- WPE WebKit adds 80-150 MB RSS overhead vs ~30 MB Pygame baseline
- Multi-process WebView lifecycle complicates shutdown (mitigated by `_check_orphan_child_pids()`)
- WebKit native library security update cadence becomes part of operational maintenance (mitigated by no-network enforcement: CSP + WebView settings + optional UFW outbound)
- Phase 0 mandatory before P1 (avoid implementing against a non-working engine)
- P0-closeout adds an extra PM-owned phase before any Codex code dispatch

### Out of scope (deferred to Phase 5+)

- Pure CSS-layered shape authoring (eye / ear / mouth / tail / heart sub-elements)
- Pygame renderer + pygame dependency removal (P5; requires ≥ 2 weeks Jetson HTML soak first)
- MP4 frame-to-CSS transcription (MP4 path retired; CSS animations authored independently)
- Multi-character HTML composition / overlays
- Interactive HTML elements (touch / click / keyboard inside HTML)
- WebGL / Canvas 2D rendering
- LLM sentiment → CSS variable injection
- Audio sync inside HTML (TTS stays on existing pipeline)

## Validation criteria

- [ ] User mini-approval of this ADR (P-ADR gate)
- [ ] ADR 0095 Update section appended (see below; this ADR draft includes the proposed Update text for ADR 0095 in a sibling commit/PR)
- [ ] Plan v3 §7.2 Phase 0 verification completed (A1 package matrix + B1 minimal example + B2 display backend + D combined memory on Jetson)
- [ ] Phase 0 runbook authored at `docs/runbooks/jetson-html-renderer-phase0.md`
- [ ] P0-closeout plan/spec refresh stamps the actually-selected engine + backend + minimal-example snippet
- [ ] P1a-P1b-P2a-P2b-P3 Codex dispatches each pass §7.1.b phase gates + full repo gates
- [ ] P4 Jetson smoke gates (windowed + fullscreen) pass; load-changed signal arrives within 2 s per state; memory ≤ 5500 MB combined
- [ ] P4 child visual QA per Plan v3 §7.5 (contrast / motion-safety / flicker / scary expression check) user/designer sign-off

## ADR 0095 Update text (to append to `docs/adr/0095-pygame-character-renderer-runtime.md` in this same PR)

```markdown
## Update (2026-05-29 — partial supersede by ADR 0097)

- This ADR's D1 (engine = Pygame), D3 (single-UI-thread Pygame model), D5 (display mode set_mode), and D7 (PNG sprite asset spec) are **superseded in part** by ADR 0097 (HTML+CSS Character Renderer).
- ADR 0097 introduces HtmlCharacterRenderer (WPE WebKit primary, WebKitGTK 4.1 fallback) as the new runtime primary; Pygame renderer is preserved as the fallback selector path (`MUNGI_RENDERER=pygame` explicit override + auto-cascade fallback).
- D2 (dependency placement), D4 (SDL backend selection), D6 (lifecycle = SessionManager), D8 (verification gates) remain in force unchanged.
- The 19 PNG assets from Phase 2 B-2 v5 (PR #141) are preserved as the visual base inside the new HTML wrappers; PNG investment is not lost.
- Pygame renderer + pygame dependency retirement is deferred to ADR 0097's P5 deferred phase (requires ≥ 2 weeks Jetson HTML soak first).
- Original Context / Decision / Consequences sections above remain untouched per `feedback_adr_immutability`. This Update note records the supersede event for audit traceability.
- Authority: `Dev_Plan/2026-05-29-css-html-emoji-renderer-plan-v1.md` v3 (Codex-APPROVED 2026-05-29 r3 APPROVE WITH NOTES; 23/23 cumulative r1+r2 closures); user directive 2026-05-29 ("이모지는 css로 구현해서 html파일로 저장한 후 사용할꺼야"; "메모리 점유가 가장 적은 방법으로 랜더하는 걸 선택해").
```

## Reference map

| Artifact | Role |
|---|---|
| Plan v3 | Implementation plan + verification gates (23/23 Codex r1+r2 closures) |
| Plan v3 §5 | Locked decisions (engine candidate, asset model, CSP) |
| Plan v3 §6.1 | HtmlCharacterRenderer skeleton + `_HtmlEngineController` Protocol |
| Plan v3 §6.4 | Per-expression animation table (transform/opacity/filter only) |
| Plan v3 §7.2 | Phase 0 verification (A1 package matrix + B1 minimal example + B2 display backend + D combined memory) |
| Plan v3 §8 | Codex role split rollout (P-ADR / P0 / P0-closeout / P1a-P1b / P2a-P2b / P3 / P4 / P5) |
| Codex r1/r2/r3 reviews | `.codex/specs/css-html-emoji-renderer-plan-review-v{1-r1,2-r2,3-r3}.md` |
| r1/r2 discussion records | `Dev_Plan/2026-05-29-css-html-emoji-renderer-plan-codex-review-discussion-r{1,2}.md` |
| ADR 0095 | Predecessor (partially superseded by this ADR per `feedback_adr_immutability`) |
| ADR 0080 | 4inch HDMI Touch IPS LCD adoption (unchanged) |
| ADR 0079 | LED retirement (unchanged) |
| `feedback_adr_immutability` | Mandates Update section append, not Decision rewrite |
| CLAUDE.md §5 | Optional-dep mypy ignore convention |
| CLAUDE.md §6 | Jetson 8 GB memory budget |
| CLAUDE.md §7 rule 4 | ADR requirement for architecture/runtime-path change |
| CLAUDE.md §8 | Sub-agent scope ownership (P1/P2 role split per Codex r1 H1 BLOCK fix) |

## Update (2026-05-29 — Acceptance event)

- ADR status: Proposed → **Accepted** via user mini-approval (Session 86 start-of-work).
- ADR 0095 partial supersede (D1/D3/D5/D7) 효력 발생 — ADR 0095 Update section (2026-05-29 partial supersede) 활성.
- P-ADR gate (Plan v3 §8) 통과 — Plan v3 §8 P0 manual phase 진입 가능.
- Engine pinning은 P0 + P0-closeout에서 확정 (Plan v3 §5 D1 "candidate pending Phase 0").
- 원본 Context / Decision / Consequences 섹션 미변경 (`feedback_adr_immutability` 준수). 본 Update 노트는 acceptance event 감사 추적 목적.
