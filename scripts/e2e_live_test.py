"""Two-phase E2E runner for Jetson: scripted text and live voice."""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import random
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.e2e_60rounds as scripted_rounds  # noqa: E402
from core.model_manager import ManagerConfig, ModelManager  # noqa: E402
from core.pipeline import (  # noqa: E402
    ConversationPipeline,
    PipelineConfig,
    TurnMetrics,
    TurnResult,
)
from core.system_probe import (  # noqa: E402
    find_usb_input,
    read_max_thermal_c,
    read_meminfo_snapshot,
)
from scripts.e2e_60rounds_text_tts import (  # noqa: E402
    TegrastatsMonitor,
    TextTTSTurnRecord,
    _build_thermal_summary,
)
from scripts.test_tts import write_wav  # noqa: E402
from scripts.utils import get_peak_memory_kb  # noqa: E402

logger = logging.getLogger("mungi.scripts.e2e_live_test")

DEFAULT_MODEL_DIR = "/opt/mungi/ai_models"
DEFAULT_OUTPUT_DIR = Path("/tmp/mungi_e2e_live_test")
DEFAULT_SILENCE_TIMEOUT_S = 10.0
VOICE_CHUNK_SECONDS = 0.25
VOICE_PRE_ROLL_CHUNKS = 2
VOICE_QUEUE_TIMEOUT_S = 0.1
VOICE_NOISY_LOGGERS = (
    "mungi.core.pipeline",
    "mungi.core.model_manager",
    "mungi.models.vad_runner",
)


@dataclass(frozen=True)
class PhaseRoundPlan:
    """Fixed round plan for either E2E phase."""

    round_num: int
    max_turns: int
    purpose: str
    suggestions: tuple[str, ...] = ()


@dataclass(frozen=True)
class PhaseOutcome:
    """Persisted outcome for one phase run."""

    phase: str
    verdict: str
    completed_turns: int
    failure_count: int
    summary_path: Path
    table_path: Path
    rounds_path: Path


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the two-phase E2E runner."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a 2-phase Mungi E2E session: scripted text validation and "
            "live microphone conversation."
        ),
    )
    parser.add_argument(
        "--phase",
        choices=("text", "voice", "all"),
        default="all",
        help="Which phase to execute.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(DEFAULT_MODEL_DIR),
        help=f"Model root directory (default: {DEFAULT_MODEL_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for logs and WAVs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play TTS audio during the scripted text phase.",
    )
    parser.add_argument(
        "--output-device",
        type=str,
        default=os.getenv("MUNGI_AUDIO_OUTPUT_DEVICE", "").strip() or None,
        help="Optional sounddevice output device override.",
    )
    parser.add_argument(
        "--silence-timeout",
        type=float,
        default=DEFAULT_SILENCE_TIMEOUT_S,
        help=(
            "End a live round when no new speech arrives for this many seconds "
            f"(default: {DEFAULT_SILENCE_TIMEOUT_S})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to shuffle scripted topics.",
    )
    return parser


def _configure_logging() -> None:
    """Configure console logging once for this script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _configure_console_encoding() -> None:
    """Force UTF-8 output for interactive console text when supported."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


def _capture_logger_levels(logger_names: tuple[str, ...]) -> dict[str, int]:
    """Return the current level for each named logger."""
    return {name: logging.getLogger(name).level for name in logger_names}


def _set_logger_levels(logger_names: tuple[str, ...], level: int) -> None:
    """Set the level for each named logger."""
    for name in logger_names:
        logging.getLogger(name).setLevel(level)


def _restore_logger_levels(saved_levels: dict[str, int]) -> None:
    """Restore loggers to their previously captured levels."""
    for name, level in saved_levels.items():
        logging.getLogger(name).setLevel(level)


def _print_voice_round_banner(round_plan: PhaseRoundPlan, silence_timeout_s: float) -> None:
    """Print the round banner for the interactive voice phase."""
    print("\n" + "=" * 60, flush=True)
    print(f"  Round {round_plan.round_num}/7: {round_plan.purpose}", flush=True)
    print(f"  (max {round_plan.max_turns} turns)", flush=True)
    print(f"  End: {silence_timeout_s:g}s silence or Enter", flush=True)
    print("=" * 60, flush=True)
    for suggestion in round_plan.suggestions:
        print(f"  >> {suggestion}", flush=True)
    print(flush=True)
    print("  Listening... (speak now)", flush=True)
    print(flush=True)


def _ensure_output_dirs(output_root: Path) -> tuple[Path, Path]:
    """Create the output directory tree and return ``(root, wav_dir)``."""
    resolved_root = output_root.expanduser().resolve()
    wav_dir = resolved_root / "wav"
    resolved_root.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    return resolved_root, wav_dir


def _phase1_turn_count_for_round(round_num: int) -> int:
    """Return the fixed turn count required for the scripted phase."""
    if 1 <= round_num <= 3:
        return 3
    if 4 <= round_num <= 8:
        return 5
    if 9 <= round_num <= 12:
        return 8
    if 13 <= round_num <= 15:
        return 3
    msg = f"Unsupported phase-1 round number: {round_num}"
    raise ValueError(msg)


def _phase1_round_plans() -> list[PhaseRoundPlan]:
    """Return the fixed scripted round schedule."""
    plans: list[PhaseRoundPlan] = []
    for round_num in range(1, 16):
        if round_num <= 3:
            purpose = "워밍업 효과와 초기 OOM 예방 확인"
        elif round_num <= 8:
            purpose = "일반 대화 품질 확인"
        elif round_num <= 12:
            purpose = "장문 대화 안정성 확인"
        else:
            purpose = "긴 라운드 이후 회복력 확인"
        plans.append(
            PhaseRoundPlan(
                round_num=round_num,
                max_turns=_phase1_turn_count_for_round(round_num),
                purpose=purpose,
            ),
        )
    return plans


def _phase2_round_plans() -> list[PhaseRoundPlan]:
    """Return the fixed live-voice round schedule."""
    return [
        PhaseRoundPlan(1, 3, "인사와 워밍업", ("인사해 봐: 안녕, 뭉이?",)),
        PhaseRoundPlan(
            2,
            5,
            "자연스러운 일상 대화",
            ("자유 대화: 오늘 있었던 일 말하기", "좋아하는 것 물어보기"),
        ),
        PhaseRoundPlan(
            3,
            5,
            "자연스러운 일상 대화",
            ("자유 대화: 주말에 하고 싶은 일 말하기", "좋아하는 놀이 말하기"),
        ),
        PhaseRoundPlan(
            4,
            8,
            "긴 대화 유지",
            ("긴 대화로 계속 질문하기", "앞에서 한 이야기 이어서 말하기"),
        ),
        PhaseRoundPlan(
            5,
            8,
            "긴 대화 유지",
            ("같은 주제로 길게 이어 말하기", "여러 번 꼬리 질문하기"),
        ),
        PhaseRoundPlan(
            6,
            3,
            "감정 맥락 반응",
            ("감정 표현: 슬프거나 기쁜 이야기",),
        ),
        PhaseRoundPlan(
            7,
            3,
            "지식 경계 확인",
            ("어려운 질문: 공룡은 왜 멸종했어?",),
        ),
    ]


def _build_phase1_messages(
    topic_data: scripted_rounds.TopicData,
    round_num: int,
) -> list[str]:
    """Reuse the scripted round builder with a fixed turn schedule override."""
    original = scripted_rounds.turn_count_for_round
    scripted_rounds.turn_count_for_round = _phase1_turn_count_for_round
    try:
        return scripted_rounds.build_round_messages(topic_data, round_num)
    finally:
        scripted_rounds.turn_count_for_round = original


def _make_manager(model_dir: Path) -> ModelManager:
    """Create and initialize the shared model manager."""
    manager = ModelManager(ManagerConfig(model_dir=str(model_dir)))
    manager.initialize()
    return manager


def _make_pipeline(
    manager: ModelManager,
    *,
    play_audio: bool,
    output_device: str | None,
    enable_stt_preload: bool,
) -> ConversationPipeline:
    """Create the conversation pipeline for a phase run."""
    return ConversationPipeline(
        manager,
        PipelineConfig(
            play_tts_audio=play_audio,
            tts_output_device=output_device,
            enable_warmup=True,
            enable_stt_preload=enable_stt_preload,
        ),
    )


def _run_warmup(pipeline: ConversationPipeline) -> dict[str, Any]:
    """Run LLM warmup and return a JSON-serializable status dict."""
    started = time.monotonic()
    try:
        pipeline.warmup_llm()
    except Exception as exc:
        return {
            "success": False,
            "time_s": round(time.monotonic() - started, 3),
            "error": str(exc),
        }
    return {
        "success": True,
        "time_s": round(time.monotonic() - started, 3),
        "error": None,
    }


def _normalize_audio(audio_samples: Any) -> np.ndarray:
    """Normalize arbitrary audio samples to a mono ``float32`` vector."""
    audio = np.asarray(audio_samples, dtype=np.float32)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.reshape(-1)
    return audio.astype(np.float32, copy=False)


def _save_turn_audio(path: Path, audio_samples: Any, sample_rate: int) -> None:
    """Write synthesized audio to disk as a WAV file."""
    write_wav(path, _normalize_audio(audio_samples), sample_rate)


def _stt_total_s(metrics: TurnMetrics) -> float:
    """Return the combined STT load and inference time."""
    return metrics.stt_load_time_s + metrics.stt_time_s


def _first_sound_s(metrics: TurnMetrics) -> float:
    """Return latency until synthesized audio becomes available."""
    return (
        metrics.vad_time_s
        + _stt_total_s(metrics)
        + metrics.llm_load_time_s
        + metrics.llm_time_s
        + metrics.tts_load_time_s
        + metrics.tts_time_s
    )


def _turn_name(phase: str, round_num: int, exchange: int) -> str:
    """Return a stable turn identifier for logs and filenames."""
    return f"{phase}_r{round_num:02d}_t{exchange:02d}"


def _append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    """Append one record to a JSONL file."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_turn_record(
    *,
    phase: str,
    round_plan: PhaseRoundPlan,
    exchange: int,
    input_text: str,
    result: TurnResult,
    wav_path: Path | None,
    llm_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Build a JSON-serializable per-turn record with metrics and telemetry."""
    base_record = TextTTSTurnRecord(
        round_num=round_plan.round_num,
        topic=round_plan.purpose,
        exchange=exchange,
        user_text=input_text,
        assistant_text=result.response_text,
        tts_wav=str(wav_path) if wav_path is not None else None,
        llm_tokens=result.metrics.llm_tokens,
        llm_ttft_s=round(result.metrics.llm_ttft_s, 3),
        llm_time_s=round(result.metrics.llm_time_s, 3),
        llm_model_fallback_used=result.metrics.llm_model_fallback_used,
        llm_model_path_actual=result.metrics.llm_model_path_actual,
        llm_model_fallback_reason=result.metrics.llm_model_fallback_reason,
        tts_time_s=round(result.metrics.tts_time_s, 3),
        total_time_s=round(result.metrics.total_time_s, 3),
        peak_memory_kb=get_peak_memory_kb(),
        success=result.success,
        error=result.error,
    )
    record = asdict(base_record)
    record.update(
        {
            "phase": phase,
            "turn_id": _turn_name(phase, round_plan.round_num, exchange),
            "round_purpose": round_plan.purpose,
            "planned_turns": round_plan.max_turns,
            "suggestions": list(round_plan.suggestions),
            "state": result.state.value,
            "metrics": result.metrics.to_dict(),
            "stt_total_s": round(_stt_total_s(result.metrics), 3),
            "first_sound_s": round(_first_sound_s(result.metrics), 3),
            "meminfo": read_meminfo_snapshot(),
            "thermal_c": read_max_thermal_c(),
            "llm_load_diagnostics": llm_diagnostics,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    return record


def _avg(values: list[float]) -> float:
    """Return the average value, or 0.0 for an empty list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _coerce_float(value: Any) -> float:
    """Return a float for numeric-like values, or 0.0 otherwise."""
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _record_metric(record: dict[str, Any], key: str) -> float:
    """Read one numeric metric from a stored turn record."""
    metrics = record.get("metrics", {})
    value = metrics.get(key, 0.0)
    return _coerce_float(value)


def _write_latency_table(records: list[dict[str, Any]], path: Path) -> None:
    """Write the required markdown latency table."""
    header = (
        "| Turn | VAD | STT | LLM로드 | TTFT | LLM추론 | TTS로드 | "
        "TTS합성 | 재생 | 첫소리까지 | 전체 |"
    )
    divider = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    rows = [header, divider]

    def _row(turn_name: str, record: dict[str, Any]) -> str:
        return (
            f"| {turn_name} | "
            f"{_record_metric(record, 'vad_time_s'):.3f} | "
            f"{float(record.get('stt_total_s', 0.0)):.3f} | "
            f"{_record_metric(record, 'llm_load_time_s'):.3f} | "
            f"{_record_metric(record, 'llm_ttft_s'):.3f} | "
            f"{_record_metric(record, 'llm_time_s'):.3f} | "
            f"{_record_metric(record, 'tts_load_time_s'):.3f} | "
            f"{_record_metric(record, 'tts_time_s'):.3f} | "
            f"{_record_metric(record, 'playback_time_s'):.3f} | "
            f"{float(record.get('first_sound_s', 0.0)):.3f} | "
            f"{_record_metric(record, 'total_time_s'):.3f} |"
        )

    for record in records:
        rows.append(_row(str(record["turn_id"]), record))

    avg_record = {
        "metrics": {
            "vad_time_s": _avg([_record_metric(record, "vad_time_s") for record in records]),
            "llm_load_time_s": _avg(
                [_record_metric(record, "llm_load_time_s") for record in records],
            ),
            "llm_ttft_s": _avg([_record_metric(record, "llm_ttft_s") for record in records]),
            "llm_time_s": _avg([_record_metric(record, "llm_time_s") for record in records]),
            "tts_load_time_s": _avg(
                [_record_metric(record, "tts_load_time_s") for record in records],
            ),
            "tts_time_s": _avg([_record_metric(record, "tts_time_s") for record in records]),
            "playback_time_s": _avg(
                [_record_metric(record, "playback_time_s") for record in records],
            ),
            "total_time_s": _avg([_record_metric(record, "total_time_s") for record in records]),
        },
        "stt_total_s": _avg([float(record.get("stt_total_s", 0.0)) for record in records]),
        "first_sound_s": _avg([float(record.get("first_sound_s", 0.0)) for record in records]),
    }
    rows.append(_row("AVG", avg_record))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON summary file."""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _round_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    """Return completed turn counts per round."""
    counts: dict[str, int] = {}
    for record in records:
        key = f"R{int(record['round_num']):02d}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _summarize_phase1(
    *,
    records: list[dict[str, Any]],
    plans: list[PhaseRoundPlan],
    warmup: dict[str, Any],
    interrupted: bool,
    tegrastats_enabled: bool,
    tegrastats_snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the final summary payload for phase 1."""
    planned_turns = sum(plan.max_turns for plan in plans)
    success_count = sum(1 for record in records if bool(record.get("success")))
    failure_count = planned_turns - success_count
    success_rate = (success_count / planned_turns * 100.0) if planned_turns else 0.0
    avg_turn_time_s = _avg([_record_metric(record, "total_time_s") for record in records])
    long_round_records = [record for record in records if 9 <= int(record["round_num"]) <= 12]
    avg_long_llm_time_s = _avg(
        [_record_metric(record, "llm_time_s") for record in long_round_records],
    )
    checks = {
        "warmup_success": bool(warmup.get("success")),
        "success_rate_gte_99": success_rate >= 99.0,
        "avg_turn_time_lt_15": avg_turn_time_s < 15.0,
        "rounds_9_to_12_avg_llm_lt_8": avg_long_llm_time_s < 8.0,
        "not_interrupted": not interrupted,
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    return {
        "phase": "text",
        "verdict": verdict,
        "interrupted": interrupted,
        "warmup": warmup,
        "planned_rounds": len(plans),
        "planned_turns": planned_turns,
        "completed_turns": len(records),
        "successful_turns": success_count,
        "failure_count": failure_count,
        "success_rate_percent": round(success_rate, 3),
        "avg_turn_time_s": round(avg_turn_time_s, 3),
        "avg_rounds_9_to_12_llm_time_s": round(avg_long_llm_time_s, 3),
        "checks": checks,
        "round_turn_counts": _round_counts(records),
        "tegrastats_enabled": tegrastats_enabled,
        "thermal_summary": _build_thermal_summary(tegrastats_snapshots),
    }


def _summarize_phase2(
    *,
    records: list[dict[str, Any]],
    plans: list[PhaseRoundPlan],
    warmup: dict[str, Any],
    interrupted: bool,
    tegrastats_enabled: bool,
    tegrastats_snapshots: list[dict[str, Any]],
    rounds_completed: int,
) -> dict[str, Any]:
    """Build the final summary payload for phase 2."""
    success_count = sum(1 for record in records if bool(record.get("success")))
    failure_count = sum(1 for record in records if not bool(record.get("success")))
    success_rate = (success_count / len(records) * 100.0) if records else 100.0
    avg_turn_time_s = _avg([_record_metric(record, "total_time_s") for record in records])
    avg_llm_time_s = _avg([_record_metric(record, "llm_time_s") for record in records])
    verdict = "PASS"
    if not warmup.get("success") or interrupted or failure_count > 0:
        verdict = "FAIL"
    return {
        "phase": "voice",
        "verdict": verdict,
        "interrupted": interrupted,
        "warmup": warmup,
        "planned_rounds": len(plans),
        "completed_rounds": rounds_completed,
        "completed_turns": len(records),
        "successful_turns": success_count,
        "failure_count": failure_count,
        "success_rate_percent": round(success_rate, 3),
        "avg_turn_time_s": round(avg_turn_time_s, 3),
        "avg_llm_time_s": round(avg_llm_time_s, 3),
        "round_turn_counts": _round_counts(records),
        "tegrastats_enabled": tegrastats_enabled,
        "thermal_summary": _build_thermal_summary(tegrastats_snapshots),
    }


def _write_phase_outputs(
    *,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    summary_path: Path,
    table_path: Path,
) -> None:
    """Persist both summary JSON and latency markdown for one phase."""
    _write_summary(summary_path, summary)
    _write_latency_table(records, table_path)


def _drain_audio_queue(audio_queue: queue.Queue[np.ndarray]) -> None:
    """Discard any queued microphone chunks."""
    while True:
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            return


def _start_enter_listener(stop_event: threading.Event) -> threading.Thread:
    """Start a daemon thread that sets ``stop_event`` when Enter is pressed."""

    def _wait_for_enter() -> None:
        try:
            input()
        except EOFError:
            return
        stop_event.set()

    thread = threading.Thread(target=_wait_for_enter, daemon=True)
    thread.start()
    return thread


def _ensure_vad_ready(manager: ModelManager) -> None:
    """Reinitialize the manager if an earlier failure unloaded VAD."""
    if manager.vad is None:
        logger.warning("VAD was unloaded by a prior failure; reinitializing manager")
        manager.initialize()


def _run_phase1(
    args: argparse.Namespace,
    output_root: Path,
    wav_dir: Path,
) -> PhaseOutcome:
    """Run the unattended scripted text phase."""
    logger.info("1단계 텍스트 E2E를 시작합니다.")
    records: list[dict[str, Any]] = []
    plans = _phase1_round_plans()
    rounds_path = output_root / "phase1_rounds.jsonl"
    summary_path = output_root / "phase1_summary.json"
    table_path = output_root / "phase1_latency_table.md"
    tegrastats = TegrastatsMonitor(output_root / "phase1_tegrastats.log")
    tegrastats_enabled = tegrastats.start()
    interrupted = False
    manager: ModelManager | None = None
    warmup = {"success": False, "time_s": 0.0, "error": "warmup_not_started"}

    pool = list(scripted_rounds.TOPIC_POOL)
    rng = random.Random(args.seed)
    topic_cursor = 0

    try:
        manager = _make_manager(args.model_dir)
        pipeline = _make_pipeline(
            manager,
            play_audio=bool(args.play),
            output_device=args.output_device,
            enable_stt_preload=False,
        )
        warmup = _run_warmup(pipeline)
        warmup_time_s = _coerce_float(warmup.get("time_s"))
        logger.info(
            "텍스트 단계 워밍업 결과: success=%s, time=%.3fs",
            warmup["success"],
            warmup_time_s,
        )

        for plan in plans:
            logger.info(
                "1단계 Round %d/15: %s (%d턴)",
                plan.round_num,
                plan.purpose,
                plan.max_turns,
            )
            pipeline.clear_history()
            topic_data, topic_cursor = scripted_rounds.choose_round_topic(
                pool,
                cursor=topic_cursor,
                rng=rng,
            )
            messages = _build_phase1_messages(topic_data, plan.round_num)

            for exchange, user_text in enumerate(messages, start=1):
                result = pipeline.run_text_turn(str(user_text))
                wav_path: Path | None = None
                if result.audio_samples is not None and result.sample_rate > 0:
                    wav_path = wav_dir / f"{_turn_name('phase1', plan.round_num, exchange)}.wav"
                    _save_turn_audio(wav_path, result.audio_samples, result.sample_rate)
                record = _build_turn_record(
                    phase="phase1",
                    round_plan=plan,
                    exchange=exchange,
                    input_text=str(user_text),
                    result=result,
                    wav_path=wav_path,
                    llm_diagnostics=manager.latest_llm_load_diagnostics(),
                )
                record["topic"] = str(topic_data["topic"])
                _append_jsonl_record(rounds_path, record)
                records.append(record)
                logger.info(
                    "1단계 Round %d Turn %d/%d 완료: success=%s total=%.3fs",
                    plan.round_num,
                    exchange,
                    plan.max_turns,
                    result.success,
                    result.metrics.total_time_s,
                )
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("텍스트 단계가 Ctrl+C로 중단되었습니다. 부분 결과를 저장합니다.")
    finally:
        if manager is not None:
            try:
                manager.unload_all()
            except Exception:
                logger.warning("Failed to unload models after phase 1", exc_info=True)
        tegrastats.stop()

    summary = _summarize_phase1(
        records=records,
        plans=plans,
        warmup=warmup,
        interrupted=interrupted,
        tegrastats_enabled=tegrastats_enabled,
        tegrastats_snapshots=tegrastats.snapshots,
    )
    _write_phase_outputs(
        records=records,
        summary=summary,
        summary_path=summary_path,
        table_path=table_path,
    )
    logger.info(
        "1단계 완료: verdict=%s, completed=%d, failures=%d",
        summary["verdict"],
        len(records),
        int(summary["failure_count"]),
    )
    return PhaseOutcome(
        phase="text",
        verdict=str(summary["verdict"]),
        completed_turns=len(records),
        failure_count=int(summary["failure_count"]),
        summary_path=summary_path,
        table_path=table_path,
        rounds_path=rounds_path,
    )


def _run_voice_round(
    *,
    pipeline: ConversationPipeline,
    manager: ModelManager,
    round_plan: PhaseRoundPlan,
    rounds_path: Path,
    wav_dir: Path,
    input_device: int,
    input_sample_rate: int,
    input_channels: int,
    silence_timeout_s: float,
    output_phase_name: str,
    records: list[dict[str, Any]],
) -> int:
    """Run one continuous-listening live round and return completed turn count."""
    import sounddevice as sd  # type: ignore[import-not-found, import-untyped]

    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    stop_event = threading.Event()
    _ = _start_enter_listener(stop_event)
    pre_roll: deque[np.ndarray] = deque(maxlen=VOICE_PRE_ROLL_CHUNKS)
    captured_chunks: list[np.ndarray] = []
    exchange = 0
    capturing = False
    printed_listening = True
    silence_started_at: float | None = None
    last_activity_at = time.monotonic()
    end_of_speech_s = max(pipeline._config.vad_min_silence_ms / 1000.0, VOICE_CHUNK_SECONDS)
    saved_levels = _capture_logger_levels(VOICE_NOISY_LOGGERS)

    _print_voice_round_banner(round_plan, silence_timeout_s)
    _set_logger_levels(VOICE_NOISY_LOGGERS, logging.WARNING)

    def _audio_callback(
        indata: Any,
        frames: int,
        callback_time: Any,
        status: Any,
    ) -> None:
        del frames, callback_time
        if status:
            logger.warning("Input stream status: %s", status)
        audio_queue.put(np.array(indata, dtype=np.float32, copy=True))

    def _process_capture(stream: Any) -> None:
        nonlocal exchange, captured_chunks, capturing, silence_started_at, last_activity_at
        nonlocal printed_listening
        if not captured_chunks:
            return

        print("  Processing...", flush=True)
        stream.stop()
        _restore_logger_levels(saved_levels)
        turn_completed = False
        try:
            exchange += 1
            captured_audio = np.concatenate(captured_chunks, axis=0)
            _drain_audio_queue(audio_queue)
            result = pipeline.run_turn(captured_audio, sample_rate=input_sample_rate)
            wav_path: Path | None = None
            if result.audio_samples is not None and result.sample_rate > 0:
                wav_path = (
                    wav_dir / f"{_turn_name(output_phase_name, round_plan.round_num, exchange)}.wav"
                )
                _save_turn_audio(wav_path, result.audio_samples, result.sample_rate)
            record = _build_turn_record(
                phase=output_phase_name,
                round_plan=round_plan,
                exchange=exchange,
                input_text=result.user_text,
                result=result,
                wav_path=wav_path,
                llm_diagnostics=manager.latest_llm_load_diagnostics(),
            )
            _append_jsonl_record(rounds_path, record)
            records.append(record)
            logger.info(
                "2단계 Round %d Turn %d/%d 완료: success=%s total=%.3fs",
                round_plan.round_num,
                exchange,
                round_plan.max_turns,
                result.success,
                result.metrics.total_time_s,
            )
            if not result.success:
                _ensure_vad_ready(manager)
            last_activity_at = time.monotonic()
            print(f"  Child: {result.user_text}", flush=True)
            print(f"  Mungi: {result.response_text}", flush=True)
            print(f"  ({result.metrics.total_time_s:.1f}s)", flush=True)
            print(flush=True)
            print("  Listening...", flush=True)
            turn_completed = True
        finally:
            captured_chunks = []
            capturing = False
            silence_started_at = None
            pre_roll.clear()
            printed_listening = turn_completed
            _set_logger_levels(VOICE_NOISY_LOGGERS, logging.WARNING)
            stream.start()

    try:
        with sd.InputStream(
            device=input_device,
            samplerate=input_sample_rate,
            channels=input_channels,
            dtype="float32",
            blocksize=max(int(input_sample_rate * VOICE_CHUNK_SECONDS), 1),
            callback=_audio_callback,
        ) as stream:
            while exchange < round_plan.max_turns:
                if not capturing and not printed_listening:
                    print("  Listening...", flush=True)
                    printed_listening = True

                if stop_event.is_set() and not capturing:
                    logger.info("Enter 입력으로 현재 라운드를 종료합니다.")
                    break
                if not capturing and time.monotonic() - last_activity_at >= silence_timeout_s:
                    logger.info("지정한 무음 시간이 지나 현재 라운드를 종료합니다.")
                    break

                try:
                    chunk = audio_queue.get(timeout=VOICE_QUEUE_TIMEOUT_S)
                except queue.Empty:
                    continue

                pre_roll.append(chunk)
                prepared_audio = pipeline._prepare_input_audio(chunk, input_sample_rate)
                has_speech = bool(prepared_audio) and bool(pipeline._run_vad(prepared_audio))

                if has_speech:
                    last_activity_at = time.monotonic()
                    silence_started_at = None
                    if not capturing:
                        print("  Recording...", flush=True)
                        printed_listening = False
                        capturing = True
                        captured_chunks = list(pre_roll)
                    else:
                        captured_chunks.append(chunk)
                    continue

                if capturing:
                    captured_chunks.append(chunk)
                    if silence_started_at is None:
                        silence_started_at = time.monotonic()
                        continue
                    if time.monotonic() - silence_started_at >= end_of_speech_s:
                        _process_capture(stream)

            if capturing and captured_chunks and exchange < round_plan.max_turns:
                _process_capture(stream)
    finally:
        _restore_logger_levels(saved_levels)

    print(flush=True)
    print(f"  Round {round_plan.round_num} complete: {exchange} turns", flush=True)
    print("-" * 60, flush=True)

    return exchange


def _run_phase2(
    args: argparse.Namespace,
    output_root: Path,
    wav_dir: Path,
) -> PhaseOutcome:
    """Run the live microphone phase."""
    logger.info("2단계 라이브 음성 E2E를 시작합니다.")
    records: list[dict[str, Any]] = []
    plans = _phase2_round_plans()
    rounds_path = output_root / "phase2_rounds.jsonl"
    summary_path = output_root / "phase2_summary.json"
    table_path = output_root / "phase2_latency_table.md"
    tegrastats = TegrastatsMonitor(output_root / "phase2_tegrastats.log")
    tegrastats_enabled = tegrastats.start()
    interrupted = False
    rounds_completed = 0
    manager: ModelManager | None = None
    warmup = {"success": False, "time_s": 0.0, "error": "warmup_not_started"}

    try:
        input_info = find_usb_input()
        if input_info is None:
            msg = "USB input device not found"
            raise RuntimeError(msg)
        input_device = int(input_info["index"])
        input_sample_rate = int(input_info["sample_rate"])
        input_channels = int(input_info["channels"])
        manager = _make_manager(args.model_dir)
        pipeline = _make_pipeline(
            manager,
            play_audio=True,
            output_device=args.output_device,
            enable_stt_preload=True,
        )
        warmup = _run_warmup(pipeline)
        warmup_time_s = _coerce_float(warmup.get("time_s"))
        logger.info(
            "음성 단계 워밍업 결과: success=%s, time=%.3fs",
            warmup["success"],
            warmup_time_s,
        )

        for plan in plans:
            pipeline.clear_history()
            completed_turns = _run_voice_round(
                pipeline=pipeline,
                manager=manager,
                round_plan=plan,
                rounds_path=rounds_path,
                wav_dir=wav_dir,
                input_device=input_device,
                input_sample_rate=input_sample_rate,
                input_channels=input_channels,
                silence_timeout_s=float(args.silence_timeout),
                output_phase_name="phase2",
                records=records,
            )
            rounds_completed += 1
            logger.info(
                "2단계 Round %d 종료: completed_turns=%d",
                plan.round_num,
                completed_turns,
            )
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("음성 단계가 Ctrl+C로 중단되었습니다. 부분 결과를 저장합니다.")
    finally:
        if manager is not None:
            try:
                manager.unload_all()
            except Exception:
                logger.warning("Failed to unload models after phase 2", exc_info=True)
        tegrastats.stop()

    summary = _summarize_phase2(
        records=records,
        plans=plans,
        warmup=warmup,
        interrupted=interrupted,
        tegrastats_enabled=tegrastats_enabled,
        tegrastats_snapshots=tegrastats.snapshots,
        rounds_completed=rounds_completed,
    )
    _write_phase_outputs(
        records=records,
        summary=summary,
        summary_path=summary_path,
        table_path=table_path,
    )
    logger.info(
        "2단계 완료: verdict=%s, completed=%d, failures=%d",
        summary["verdict"],
        len(records),
        int(summary["failure_count"]),
    )
    return PhaseOutcome(
        phase="voice",
        verdict=str(summary["verdict"]),
        completed_turns=len(records),
        failure_count=int(summary["failure_count"]),
        summary_path=summary_path,
        table_path=table_path,
        rounds_path=rounds_path,
    )


def _run_selected_phases(
    args: argparse.Namespace,
    output_root: Path,
    wav_dir: Path,
) -> list[PhaseOutcome]:
    """Execute the selected phases and return their outcomes."""
    outcomes: list[PhaseOutcome] = []
    if args.phase in {"text", "all"}:
        phase1 = _run_phase1(args, output_root, wav_dir)
        outcomes.append(phase1)
        if args.phase == "all" and phase1.verdict != "PASS":
            logger.warning("1단계가 FAIL이어서 2단계는 실행하지 않습니다.")
            return outcomes
    if args.phase in {"voice", "all"}:
        outcomes.append(_run_phase2(args, output_root, wav_dir))
    return outcomes


def main() -> int:
    """CLI entry point."""
    _configure_console_encoding()
    _configure_logging()
    args = build_parser().parse_args()
    output_root, wav_dir = _ensure_output_dirs(args.output_dir)
    logger.info("출력 디렉터리: %s", output_root)
    logger.info("실행 모드: %s", args.phase)
    outcomes = _run_selected_phases(args, output_root, wav_dir)
    if not outcomes:
        logger.error("실행된 단계가 없습니다.")
        return 1
    return 0 if all(outcome.verdict == "PASS" for outcome in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
