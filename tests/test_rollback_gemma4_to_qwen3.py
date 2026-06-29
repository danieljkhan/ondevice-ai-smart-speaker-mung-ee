"""Mock rollback contract tests for Gemma 4 to Qwen3 legacy recovery."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True)
class MockRollbackPlan:
    """Minimal rollback plan object used to verify command construction."""

    legacy_wheel: Path
    expected_sha256: str


class MockPipelineTurnGate:
    """Small accept-turn mock documenting the Phase 3 graceful-drain contract."""

    def __init__(self) -> None:
        self.in_flight = False
        self.marker_written = False

    def accept_turn(self) -> str:
        """Return a bridge phrase when rollback mode blocks new turn starts."""
        if os.getenv("MUNGI_ROLLBACK_IN_PROGRESS") == "1":
            return "Please wait while Mungi switches back safely."
        return "accepted"

    def drain_or_mark_interrupted(self, timeout_s: float, marker_path: Path) -> bool:
        """Drain in-flight work or write an interrupted-turn marker after timeout."""
        if not self.in_flight:
            return True
        if timeout_s >= 30.0:
            marker_path.write_text('{"interrupted": true}', encoding="utf-8")
            self.marker_written = True
            self.in_flight = False
            return False
        return False


@pytest.mark.skip(reason="graceful-drain landing in Phase 3 per Plan v3.2 section 11")
def test_graceful_drain_env_flag_blocks_new_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TODO(Phase 3): Live pipeline should reject new turns during rollback drain."""
    gate = MockPipelineTurnGate()
    monkeypatch.setenv("MUNGI_ROLLBACK_IN_PROGRESS", "1")

    assert gate.accept_turn() == "Please wait while Mungi switches back safely."


@pytest.mark.skip(reason="graceful-drain landing in Phase 3 per Plan v3.2 section 11")
def test_graceful_drain_writes_interrupted_turn_marker(tmp_path: Path) -> None:
    """TODO(Phase 3): Live pipeline should persist interrupted-turn markers after timeout."""
    gate = MockPipelineTurnGate()
    gate.in_flight = True
    marker_path = tmp_path / "interrupted_turn_marker.json"

    drained = gate.drain_or_mark_interrupted(30.0, marker_path)

    assert drained is False
    assert marker_path.exists()
    assert gate.marker_written is True


def _preserve_gemma4_conversation_dirs(root: Path, timestamp: str) -> list[Path]:
    """Rename Gemma 4 conversation directories with a preserved suffix."""
    renamed: list[Path] = []
    for path in root.iterdir():
        if path.is_dir() and "gemma4" in path.name and "_gemma4_preserved_" not in path.name:
            target = path.with_name(f"{path.name}_gemma4_preserved_{timestamp}")
            path.rename(target)
            renamed.append(target)
    return renamed


def test_rollback_preserves_gemma4_conversation_directories(tmp_path: Path) -> None:
    """Rollback renames Gemma 4 conversation dirs instead of deleting them."""
    gemma_dir = tmp_path / "session_gemma4_001"
    legacy_dir = tmp_path / "session_qwen3_001"
    gemma_dir.mkdir()
    legacy_dir.mkdir()
    (gemma_dir / "conversation.jsonl").write_text("kept", encoding="utf-8")

    renamed = _preserve_gemma4_conversation_dirs(tmp_path, "20260421T150000")

    assert len(renamed) == 1
    assert renamed[0].name == "session_gemma4_001_gemma4_preserved_20260421T150000"
    assert (renamed[0] / "conversation.jsonl").read_text(encoding="utf-8") == "kept"
    assert legacy_dir.exists()
    assert not gemma_dir.exists()


def _pip_force_reinstall_command(plan: MockRollbackPlan) -> list[str]:
    """Build the pinned local wheel reinstall command expected by rollback."""
    return ["python", "-m", "pip", "install", "--force-reinstall", plan.legacy_wheel.as_posix()]


def test_rollback_uses_pinned_local_legacy_wheel() -> None:
    """Wheel rollback command uses a pinned local path, not an unpinned PyPI package."""
    plan = MockRollbackPlan(
        legacy_wheel=Path(
            "/opt/mungi/wheels/llama_cpp_python-0.3.17-cp310-cp310-linux_aarch64.whl"
        ),
        expected_sha256="abc123",
    )

    command = _pip_force_reinstall_command(plan)

    assert command == [
        "python",
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        plan.legacy_wheel.as_posix(),
    ]
    assert command[-1].startswith("/opt/mungi/wheels/")
    assert "llama_cpp_python-0.3.17" in command[-1]


def _verify_sha256(path: Path, expected_sha256: str) -> bool:
    """Verify a file SHA256 digest for post-install smoke checks."""
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    return observed == expected_sha256


def test_post_install_smoke_verifies_expected_sha256(tmp_path: Path) -> None:
    """Post-install smoke must compare observed SHA256 against the pinned expected hash."""
    artifact = tmp_path / "libllama.so"
    artifact.write_bytes(b"legacy-lib")
    expected_sha256 = hashlib.sha256(b"legacy-lib").hexdigest()

    assert _verify_sha256(artifact, expected_sha256) is True
    assert _verify_sha256(artifact, "0" * 64) is False


def test_rollback_keeps_conversation_memory_cap_at_100() -> None:
    """Legacy Qwen3 rollback still honors the CLAUDE.md 100-token conversation cap."""
    from core.pipeline import PipelineConfig

    assert PipelineConfig().max_history_tokens == 100


def test_rollback_backend_env_value_selects_qwen3(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Setting the backend env var to qwen3_legacy selects the legacy config path."""
    from core.llm_backend_config import LLMBackendConfig

    monkeypatch.setenv("MUNGI_LLM_BACKEND", "qwen3_legacy")

    cfg = LLMBackendConfig.load(tmp_path / "missing.json")

    assert cfg.backend == "qwen3_legacy"


def test_preserve_operation_is_idempotent_for_already_preserved_dirs(tmp_path: Path) -> None:
    """Already preserved Gemma 4 directories are not renamed again."""
    preserved = tmp_path / "session_gemma4_001_gemma4_preserved_20260421T150000"
    preserved.mkdir()

    renamed = _preserve_gemma4_conversation_dirs(tmp_path, "20260421T160000")

    assert renamed == []
    assert preserved.exists()
