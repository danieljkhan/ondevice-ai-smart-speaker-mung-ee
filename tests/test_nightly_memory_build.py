"""Tests for the conversation-memory nightly builder."""

from __future__ import annotations

import json
import logging
import os
import stat
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from core.conversation_memory_schema import KST, parse_generation_pointer
from scripts import nightly_memory_build


def _kst(day: int, hour: int) -> datetime:
    return datetime(2026, 6, day, hour, 0, 0, tzinfo=KST)


def _record(
    text: str,
    timestamp: datetime,
    *,
    turn: int = 1,
    metrics: dict[str, object] | None = None,
    top_flags: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "timestamp": timestamp.isoformat(timespec="seconds"),
        "turn": turn,
        "user_text": text,
        "response_text": "응, 기억해.",
        "input_wav": None,
        "output_wav": None,
        "metrics": metrics or {},
    }
    payload.update(top_flags or {})
    return payload


def _write_session(root: Path, name: str, records: list[dict[str, object]]) -> Path:
    session_dir = root / "conversations" / name
    session_dir.mkdir(parents=True)
    conversation_path = session_dir / "conversation.jsonl"
    conversation_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return session_dir


def _current_generation(root: Path) -> Path:
    memory_root = root / "conversation_memory"
    generation_id = parse_generation_pointer((memory_root / "current").read_text(encoding="utf-8"))
    return memory_root / "generations" / generation_id


def test_main_flag_off_exits_zero_without_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The flag is checked before runtime work and disables the job."""
    monkeypatch.delenv("MUNGI_CONV_MEMORY", raising=False)

    rc = nightly_memory_build.main(["--mutable-root", str(tmp_path)])

    assert rc == 0
    assert not (tmp_path / "conversation_memory").exists()


def test_build_indexes_benign_turns_and_quarantines_crisis_session(tmp_path: Path) -> None:
    """Crisis sessions are wholly quarantined and marked by day."""
    _write_session(
        tmp_path,
        "2026-06-11_19-00-00",
        [
            _record("티라노사우루스 그림을 그렸어", _kst(11, 19)),
            _record("수영 혼자 해도 돼?", _kst(11, 19), turn=2, metrics={"template_matched": True}),
        ],
    )
    _write_session(
        tmp_path,
        "2026-06-11_20-00-00",
        [_record("죽고 싶어", _kst(11, 20), metrics={"crisis_matched": True})],
    )
    for session in (tmp_path / "conversations").iterdir():
        (session / "session_end.json").write_text("{}", encoding="utf-8")

    result = nightly_memory_build.run_build(mutable_root=tmp_path, now=_kst(12, 3))
    generation_dir = _current_generation(tmp_path)
    turns = (generation_dir / "turns.jsonl").read_text(encoding="utf-8")
    marker = json.loads((generation_dir / "quarantined_days.json").read_text(encoding="utf-8"))

    assert result.sessions_seen == 2
    assert result.sessions_quarantined == 1
    assert "티라노사우루스" in turns
    assert "수영 혼자" not in turns
    assert "죽고 싶어" not in turns
    assert marker == {"quarantined_days": ["2026-06-11"]}


def test_manifest_hash_updates_on_backfilled_session_and_idempotent_runs(tmp_path: Path) -> None:
    """Hash-based manifests catch backfilled or changed conversation files."""
    session_dir = _write_session(
        tmp_path,
        "2026-06-11_19-00-00",
        [_record("세종대왕은 한글 이야기를 했어", _kst(11, 19))],
    )
    (session_dir / "session_end.json").write_text("{}", encoding="utf-8")

    nightly_memory_build.run_build(mutable_root=tmp_path, now=_kst(12, 3))
    manifest1 = json.loads(
        (_current_generation(tmp_path) / "manifest.json").read_text(encoding="utf-8")
    )
    nightly_memory_build.run_build(mutable_root=tmp_path, now=_kst(12, 4))
    turns2 = (
        (_current_generation(tmp_path) / "turns.jsonl").read_text(encoding="utf-8").splitlines()
    )

    conversation_path = session_dir / "conversation.jsonl"
    with conversation_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(_record("새로 추가된 블록 놀이", _kst(11, 20), turn=2), ensure_ascii=False)
        )
        handle.write("\n")
    nightly_memory_build.run_build(mutable_root=tmp_path, now=_kst(12, 5))
    manifest3 = json.loads(
        (_current_generation(tmp_path) / "manifest.json").read_text(encoding="utf-8")
    )
    turns3 = (
        (_current_generation(tmp_path) / "turns.jsonl").read_text(encoding="utf-8").splitlines()
    )

    key = "2026-06-11_19-00-00"
    assert manifest1[key]["sha256"] != manifest3[key]["sha256"]
    assert len(turns2) == 1
    assert len(turns3) == 2


def test_active_session_defers_and_exits_zero(tmp_path: Path) -> None:
    """The job skips when the newest session is not sentinel-ended or quiescent."""
    session_dir = _write_session(
        tmp_path,
        "2026-06-12_02-55-00",
        [_record("아직 말하는 중이야", _kst(12, 2))],
    )
    recent = datetime.now().timestamp()
    os.utime(session_dir / "conversation.jsonl", (recent, recent))

    result = nightly_memory_build.run_build(
        mutable_root=tmp_path,
        now=datetime.now(KST),
        defer_attempts=1,
        defer_sleep_s=0,
    )

    assert result.skipped_active_session
    assert not (tmp_path / "conversation_memory").exists()


def test_active_session_rechecks_clock_between_defer_attempts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The defer loop refreshes now so mtime quiescence can clear after sleeping."""
    session_dir = _write_session(
        tmp_path,
        "2026-06-12_02-50-00",
        [_record("잠깐 쉬었다가 기억해", _kst(12, 2))],
    )
    base = datetime(2026, 6, 12, 2, 50, 0, tzinfo=KST)
    os.utime(session_dir / "conversation.jsonl", (base.timestamp(), base.timestamp()))
    sleeps: list[float] = []

    monkeypatch.setattr(
        nightly_memory_build.time,
        "sleep",
        lambda seconds: sleeps.append(seconds),
    )
    monkeypatch.setattr(
        nightly_memory_build,
        "_clock_kst",
        lambda: base + timedelta(minutes=11),
    )

    result = nightly_memory_build.run_build(
        mutable_root=tmp_path,
        now=base + timedelta(minutes=5),
        defer_attempts=2,
        defer_sleep_s=0,
    )

    assert sleeps == [0.0]
    assert not result.skipped_active_session
    assert (_current_generation(tmp_path) / "turns.jsonl").read_text(encoding="utf-8")


def test_retention_drops_raw_turns_older_than_90_days(tmp_path: Path) -> None:
    """Raw turn retention is enforced during generation build."""
    old_day = datetime(2026, 2, 1, 8, 0, 0, tzinfo=KST)
    session_dir = _write_session(
        tmp_path,
        "2026-02-01_08-00-00",
        [_record("오래된 이야기는 빠져야 해", old_day)],
    )
    (session_dir / "session_end.json").write_text("{}", encoding="utf-8")

    result = nightly_memory_build.run_build(mutable_root=tmp_path, now=_kst(12, 3))
    turns = (_current_generation(tmp_path) / "turns.jsonl").read_text(encoding="utf-8")

    assert result.turns_indexed == 0
    assert turns == ""


def test_retention_drops_reused_previous_generation_snippets(tmp_path: Path) -> None:
    """Raw turn retention also applies to unchanged sessions reused from current."""
    old_day = datetime(2026, 2, 1, 8, 0, 0, tzinfo=KST)
    session_dir = _write_session(
        tmp_path,
        "2026-02-01_08-00-00",
        [_record("재사용 경로의 오래된 이야기는 빠져야 해", old_day)],
    )
    (session_dir / "session_end.json").write_text("{}", encoding="utf-8")

    nightly_memory_build.run_build(
        mutable_root=tmp_path,
        now=datetime(2026, 2, 2, 3, 0, 0, tzinfo=KST),
    )
    assert (_current_generation(tmp_path) / "turns.jsonl").read_text(encoding="utf-8")

    nightly_memory_build.run_build(mutable_root=tmp_path, now=_kst(12, 3))
    generation_dir = _current_generation(tmp_path)
    turns = (generation_dir / "turns.jsonl").read_text(encoding="utf-8")
    index = json.loads((generation_dir / "index.json").read_text(encoding="utf-8"))

    assert turns == ""
    assert index == {}


def test_undecodable_session_is_logged_and_skipped(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """A corrupt UTF-8 session is contained to that session instead of aborting."""
    corrupt_dir = tmp_path / "conversations" / "2026-06-11_18-00-00"
    corrupt_dir.mkdir(parents=True)
    (corrupt_dir / "conversation.jsonl").write_bytes(b"\xff\xfe\xfa")
    (corrupt_dir / "session_end.json").write_text("{}", encoding="utf-8")
    valid_dir = _write_session(
        tmp_path,
        "2026-06-11_19-00-00",
        [_record("정상 세션은 계속 처리돼", _kst(11, 19))],
    )
    (valid_dir / "session_end.json").write_text("{}", encoding="utf-8")
    caplog.set_level(logging.WARNING, logger="mungi.scripts.nightly_memory_build")

    result = nightly_memory_build.run_build(mutable_root=tmp_path, now=_kst(12, 3))
    turns = (_current_generation(tmp_path) / "turns.jsonl").read_text(encoding="utf-8")

    assert result.sessions_seen == 2
    assert "정상 세션" in turns
    assert any("conversation_memory_session_skipped" in record.message for record in caplog.records)


def test_unreadable_session_is_logged_and_build_completes(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An OSError reading one session skips only it; other sessions still publish."""
    unreadable_dir = tmp_path / "conversations" / "2026-06-11_18-00-00"
    unreadable_dir.mkdir(parents=True)
    unreadable_path = unreadable_dir / "conversation.jsonl"
    unreadable_path.write_text("placeholder\n", encoding="utf-8")
    (unreadable_dir / "session_end.json").write_text("{}", encoding="utf-8")
    valid_dir = _write_session(
        tmp_path,
        "2026-06-11_19-00-00",
        [_record("정상 세션은 계속 처리돼", _kst(11, 19))],
    )
    (valid_dir / "session_end.json").write_text("{}", encoding="utf-8")

    real_read_bytes = Path.read_bytes

    def fake_read_bytes(self: Path) -> bytes:
        if self == unreadable_path:
            raise PermissionError("permission denied")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)
    caplog.set_level(logging.WARNING, logger="mungi.scripts.nightly_memory_build")

    result = nightly_memory_build.run_build(mutable_root=tmp_path, now=_kst(12, 3))
    turns = (_current_generation(tmp_path) / "turns.jsonl").read_text(encoding="utf-8")

    assert result.sessions_seen == 2
    assert "정상 세션" in turns
    assert any(
        "conversation_memory_session_skipped" in record.message and "read_failed" in record.message
        for record in caplog.records
    )


def test_default_defer_budget_stays_under_timeout_start_sec() -> None:
    """Worst-case defer wall-time must leave ample build headroom under the timeout."""
    service = Path("systemd/mungi-memory-nightly.service").read_text(encoding="utf-8")
    timeout_start_sec = next(
        int(line.split("=", 1)[1])
        for line in service.splitlines()
        if line.startswith("TimeoutStartSec=")
    )
    # The defer loop sleeps at most (attempts - 1) times (no sleep after the last attempt).
    worst_case_defer_s = (
        nightly_memory_build._DEFAULT_DEFER_ATTEMPTS - 1
    ) * nightly_memory_build._DEFAULT_DEFER_SLEEP_S
    build_headroom_s = timeout_start_sec - worst_case_defer_s

    assert worst_case_defer_s < timeout_start_sec
    # Build must retain at least ~20 minutes after worst-case deferral.
    assert build_headroom_s >= 1200


def test_generation_pointer_publish_and_gc_keep_last_two(tmp_path: Path) -> None:
    """Publishing updates current atomically and GC keeps two generations."""
    session_dir = _write_session(
        tmp_path,
        "2026-06-11_19-00-00",
        [_record("블록 놀이를 했어", _kst(11, 19))],
    )
    (session_dir / "session_end.json").write_text("{}", encoding="utf-8")

    for hour in (3, 4, 5):
        nightly_memory_build.run_build(mutable_root=tmp_path, now=_kst(12, hour))

    memory_root = tmp_path / "conversation_memory"
    (memory_root / "current.tmp-orphan").write_text("stale", encoding="utf-8")
    nightly_memory_build.run_build(mutable_root=tmp_path, now=_kst(12, 6))

    current = parse_generation_pointer((memory_root / "current").read_text(encoding="utf-8"))
    generations = sorted(path.name for path in (memory_root / "generations").iterdir())

    assert current.endswith("060000KST")
    assert len(generations) == 2
    assert current in generations
    assert not list(memory_root.glob("current.tmp-*"))


def test_versioned_reader_tolerates_old_rows_and_permissions_are_private(tmp_path: Path) -> None:
    """Old rows without optional flags parse and created files are private on POSIX."""
    session_dir = _write_session(
        tmp_path,
        "2026-06-11_19-00-00",
        [_record("깃발 없는 옛날 기록", _kst(11, 19))],
    )
    (session_dir / "session_end.json").write_text("{}", encoding="utf-8")

    nightly_memory_build.run_build(mutable_root=tmp_path, now=_kst(12, 3))
    generation_dir = _current_generation(tmp_path)

    assert (generation_dir / "turns.jsonl").read_text(encoding="utf-8")
    if os.name != "nt":
        assert stat.S_IMODE(generation_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE((generation_dir / "turns.jsonl").stat().st_mode) == 0o600


def test_static_systemd_units_have_required_directives() -> None:
    """Nightly timer/service ship inert but operationally bounded."""
    service = Path("systemd/mungi-memory-nightly.service").read_text(encoding="utf-8")
    timer = Path("systemd/mungi-memory-nightly.timer").read_text(encoding="utf-8")

    assert "User=mungi" in service
    assert "EnvironmentFile=-/var/lib/mungi/config/mungi.env" in service
    # Type=oneshot ignores RuntimeMaxSec; TimeoutStartSec is the correct runtime bound.
    assert "TimeoutStartSec=1800" in service
    assert "MemoryMax=512M" in service
    assert "ExecStart=/opt/mungi-repo/.venv/bin/python -m scripts.nightly_memory_build" in service
    assert "OnCalendar=*-*-* 03:00:00 Asia/Seoul" in timer
    assert "Persistent=true" in timer
