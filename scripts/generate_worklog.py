"""Generate a Korean daily worklog template from repository activity."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, cast

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class CommitEntry:
    """A single git commit entry."""

    commit_hash: str
    committed_at: str
    subject: str


@dataclass(slots=True)
class TaskEntry:
    """A single Codex task history entry."""

    task_id: str
    timestamp: str
    command: str
    status: str
    duration_s: float | None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Generate a daily worklog markdown template.")
    parser.add_argument(
        "--date",
        required=True,
        help="Target date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output markdown file path.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    """Configure structured logging for the CLI."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def parse_work_date(value: str) -> date:
    """Parse an ISO date string."""

    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid --date value: {value!r}") from exc


def get_repo_root() -> Path:
    """Return the repository root directory."""

    return Path(__file__).resolve().parents[1]


def run_command(args: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command and capture its text output."""

    return subprocess.run(
        list(args),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def git_branch(repo_root: Path) -> str:
    """Return the current git branch name, or a placeholder on failure."""

    result = run_command(("git", "rev-parse", "--abbrev-ref", "HEAD"), repo_root)
    branch = result.stdout.strip()
    if result.returncode == 0 and branch:
        return branch
    LOGGER.warning("Failed to detect git branch: %s", result.stderr.strip())
    return "[브랜치 미확인]"


def load_git_commits(repo_root: Path, work_date: date) -> list[CommitEntry]:
    """Load commits from git log for the requested day."""

    since = datetime.combine(work_date, time.min).isoformat(timespec="seconds")
    until = datetime.combine(work_date + timedelta(days=1), time.min).isoformat(timespec="seconds")
    command = (
        "git",
        "log",
        "--since",
        since,
        "--until",
        until,
        "--pretty=format:%H%x1f%ad%x1f%s",
        "--date=iso-strict",
        "--no-merges",
    )
    result = run_command(command, repo_root)
    if result.returncode != 0:
        LOGGER.warning("git log failed: %s", result.stderr.strip())
        return []

    commits: list[CommitEntry] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\x1f")
        if len(parts) != 3:
            LOGGER.warning("Skipping malformed git log line: %s", line)
            continue
        commits.append(
            CommitEntry(
                commit_hash=parts[0][:12],
                committed_at=parts[1],
                subject=parts[2],
            )
        )
    return commits


def load_codex_tasks(repo_root: Path, work_date: date) -> list[TaskEntry]:
    """Load Codex task history entries for the requested day."""

    history_path = repo_root / ".codex" / "chat" / "history.jsonl"
    if not history_path.exists():
        LOGGER.warning("Missing Codex history file: %s", history_path)
        return []

    tasks: list[TaskEntry] = []
    with history_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Skipping invalid JSONL line in %s", history_path)
                continue

            timestamp = str(entry.get("timestamp", ""))
            if not _matches_work_date(timestamp, work_date):
                continue

            tasks.append(
                TaskEntry(
                    task_id=str(entry.get("id", "")),
                    timestamp=timestamp,
                    command=str(entry.get("command", "")),
                    status=str(entry.get("status", "")),
                    duration_s=_as_float(entry.get("duration_s")),
                )
            )
    return tasks


def load_verification_status(repo_root: Path) -> dict[str, Any] | None:
    """Load the Claude verification status JSON if it exists."""

    status_path = repo_root / ".claude" / "verification-status.json"
    if not status_path.exists():
        LOGGER.warning("Missing verification status file: %s", status_path)
        return None

    try:
        with status_path.open("r", encoding="utf-8") as handle:
            return cast(dict[str, Any], json.load(handle))
    except json.JSONDecodeError as exc:
        LOGGER.warning("Failed to parse verification status JSON: %s", exc)
        return None


def get_git_diff_stat(repo_root: Path) -> str:
    """Return the current git diff --stat summary."""

    result = run_command(("git", "diff", "--stat"), repo_root)
    if result.returncode != 0:
        LOGGER.warning("git diff --stat failed: %s", result.stderr.strip())
        return "[git diff --stat 출력 실패]"

    diff_stat = result.stdout.strip()
    return diff_stat if diff_stat else "[변경 사항 없음]"


def build_markdown(
    work_date: date,
    branch: str,
    commits: list[CommitEntry],
    tasks: list[TaskEntry],
    verification_status: dict[str, Any] | None,
    diff_stat: str,
) -> str:
    """Build the final Korean markdown worklog."""

    lines: list[str] = []
    lines.append("# 일일 작업 로그")
    lines.append("")
    lines.append("## 기본 정보")
    lines.append(f"- 날짜: {work_date.isoformat()}")
    lines.append(f"- 브랜치: {branch}")
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    lines.append(f"- 생성 시각: {generated_at}")
    lines.append("")

    lines.append("## 커밋 기록")
    lines.extend(_render_commit_table(commits))
    lines.append("")

    lines.append("## Codex 작업 기록")
    lines.extend(_render_task_table(tasks))
    lines.append("")

    lines.append("## 검증 상태")
    lines.extend(_render_verification_status(verification_status))
    lines.append("")

    lines.append("## git diff --stat 요약")
    lines.append("```text")
    lines.append(diff_stat)
    lines.append("```")
    lines.append("")

    lines.append("## TODO")
    lines.append("- [TODO] 오늘 마무리되지 않은 작업을 적어 주세요.")
    lines.append("- [TODO] Claude Code가 후속 검토할 항목을 적어 주세요.")
    lines.append("- [TODO] 검증 결과와 실제 변경 사항의 차이를 정리해 주세요.")
    lines.append("")
    return "\n".join(lines)


def write_output(output_path: Path, content: str) -> None:
    """Write the generated markdown content to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def main() -> int:
    """Run the worklog generator CLI."""

    configure_logging()
    args = parse_args()
    repo_root = get_repo_root()

    try:
        work_date = parse_work_date(args.date)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 2

    LOGGER.info("Generating worklog for %s", work_date.isoformat())
    branch = git_branch(repo_root)
    commits = load_git_commits(repo_root, work_date)
    tasks = load_codex_tasks(repo_root, work_date)
    verification_status = load_verification_status(repo_root)
    diff_stat = get_git_diff_stat(repo_root)

    content = build_markdown(
        work_date=work_date,
        branch=branch,
        commits=commits,
        tasks=tasks,
        verification_status=verification_status,
        diff_stat=diff_stat,
    )
    write_output(args.output, content)
    LOGGER.info("Wrote worklog to %s", args.output)
    return 0


def _matches_work_date(timestamp: str, work_date: date) -> bool:
    """Return True when a timestamp falls on the requested work date."""

    if not timestamp:
        return False
    normalized = timestamp.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return timestamp.startswith(work_date.isoformat())
    return parsed.date() == work_date


def _as_float(value: Any) -> float | None:
    """Convert a JSON value to float when possible."""

    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _escape_table_cell(value: str) -> str:
    """Escape markdown table cell content."""

    return value.replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _render_commit_table(commits: list[CommitEntry]) -> list[str]:
    """Render the commit history as a markdown table."""

    lines = ["| 시간 | 커밋 | 메시지 |", "| --- | --- | --- |"]
    if not commits:
        lines.append("| [TODO] | [TODO] | 오늘 커밋 없음 또는 기록 미확인 |")
        return lines

    for commit in commits:
        lines.append(
            f"| {commit.committed_at} | `{commit.commit_hash}` | "
            f"{_escape_table_cell(commit.subject)} |"
        )
    return lines


def _render_task_table(tasks: list[TaskEntry]) -> list[str]:
    """Render Codex task history as a markdown table."""

    lines = ["| 시간 | 작업 ID | 상태 | 소요 시간(초) | 명령 |", "| --- | --- | --- | --- | --- |"]
    if not tasks:
        lines.append(
            "| [TODO] | [TODO] | [TODO] | [TODO] | Codex history.jsonl 항목 없음 또는 기록 미확인 |"
        )
        return lines

    for task in tasks:
        duration = "-" if task.duration_s is None else f"{task.duration_s:.3f}"
        command = _escape_table_cell(task.command)
        lines.append(
            f"| {task.timestamp} | `{task.task_id}` | "
            f"{_escape_table_cell(task.status)} | {duration} | {command} |"
        )
    return lines


def _render_verification_status(status: dict[str, Any] | None) -> list[str]:
    """Render the verification status section."""

    if status is None:
        return ["- [TODO] .claude/verification-status.json을 확인할 수 없습니다."]

    lines = [
        f"- verified: {status.get('verified', '[TODO]')}",
        f"- phase: {status.get('phase', '[TODO]')}",
        f"- rounds_completed: {status.get('rounds_completed', '[TODO]')}",
    ]

    polish_loop = status.get("polish_loop")
    if isinstance(polish_loop, dict):
        lines.append(f"- polish_loop.terminated: {polish_loop.get('terminated', '[TODO]')}")
        lines.append(
            f"- polish_loop.consecutive_zero_fix_cycles: "
            f"{polish_loop.get('consecutive_zero_fix_cycles', '[TODO]')}"
        )
    else:
        lines.append("- polish_loop: [TODO]")

    verifiers = status.get("verifiers")
    if isinstance(verifiers, dict) and verifiers:
        lines.append("- verifiers:")
        for name, payload in verifiers.items():
            result = payload.get("result", "[TODO]") if isinstance(payload, dict) else "[TODO]"
            lines.append(f"  - {name}: {result}")
    else:
        lines.append("- verifiers: [TODO]")

    return lines


if __name__ == "__main__":
    raise SystemExit(main())
