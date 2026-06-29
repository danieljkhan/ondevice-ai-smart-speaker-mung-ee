"""E2E pipeline test + memory profiling for Jetson Orin Nano.

Runs all 4 model stages (VAD, STT, LLM, TTS) sequentially with
memory snapshots at each step. Uses a synthetic WAV since no test
audio files exist yet.

Usage:
    cd /opt/mungi-repo
    python scripts/test_e2e_pipeline.py
"""

from __future__ import annotations

import gc
import math
import os
import struct
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any


def get_memory_info() -> dict[str, int]:
    """Read /proc/meminfo and return selected fields in kB."""
    info: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                key = parts[0].rstrip(":")
                if key in (
                    "MemTotal",
                    "MemAvailable",
                    "MemFree",
                    "SwapTotal",
                    "SwapFree",
                ):
                    info[key] = int(parts[1])
    except OSError:
        pass
    return info


def get_rss_mb() -> int:
    """Get current process RSS in MB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def get_cuda_mb() -> int:
    """Get CUDA allocated memory in MB."""
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.memory_allocated() / (1024 * 1024))
    except ImportError:
        pass
    return 0


def snapshot(label: str) -> dict[str, int | str]:
    """Take a memory snapshot and print it."""
    info = get_memory_info()
    rss = get_rss_mb()
    cuda = get_cuda_mb()
    avail_mb = info.get("MemAvailable", 0) // 1024
    total_mb = info.get("MemTotal", 0) // 1024
    used_mb = total_mb - avail_mb
    swap_used = (info.get("SwapTotal", 0) - info.get("SwapFree", 0)) // 1024
    print(f"\n  [{label}]")
    print(
        f"    RAM: {used_mb}/{total_mb} MB (avail: {avail_mb} MB)"
        f" | Swap: {swap_used} MB | RSS: {rss} MB | CUDA: {cuda} MB"
    )
    return {
        "label": label,
        "used_mb": used_mb,
        "avail_mb": avail_mb,
        "total_mb": total_mb,
        "rss_mb": rss,
        "cuda_mb": cuda,
        "swap_mb": swap_used,
    }


def generate_test_wav(
    path: Path,
    duration_s: float = 3.0,
    sample_rate: int = 16000,
) -> list[float]:
    """Generate a synthetic WAV with speech-like tones."""
    n_samples = int(duration_s * sample_rate)
    samples: list[float] = []
    for i in range(n_samples):
        t = i / sample_rate
        val = (
            0.3 * math.sin(2 * math.pi * 200 * t)
            + 0.2 * math.sin(2 * math.pi * 400 * t)
            + 0.1 * math.sin(2 * math.pi * 800 * t)
        )
        val *= 0.5 + 0.5 * math.sin(2 * math.pi * 3 * t)
        samples.append(val)

    pcm = struct.pack(
        f"<{len(samples)}h",
        *(max(-32768, min(32767, int(s * 32768))) for s in samples),
    )
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return samples


def clear_gpu() -> None:
    """Run GC and clear CUDA cache."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main() -> int:
    """Run E2E pipeline test with memory profiling."""
    print("=" * 60)
    print("Task 2 & 3: E2E Pipeline Test + Memory Profiling")
    print("=" * 60)

    snapshots: list[dict[str, Any]] = []
    timings: dict[str, float] = {}

    snap = snapshot("0. Baseline")
    snapshots.append(snap)

    # ---- Stage 1: VAD ----
    print("\n" + "=" * 50)
    print("Stage 1: VAD Load + Inference")
    from models.vad_runner import SAMPLE_RATE, load_vad_model, run_vad

    t0 = time.monotonic()
    vad_model = load_vad_model()
    timings["vad_load"] = time.monotonic() - t0
    print(f"  VAD load: {timings['vad_load']:.3f}s")

    snap = snapshot("1. After VAD load")
    snapshots.append(snap)

    fd, tmp_name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    test_wav = Path(tmp_name)
    audio_samples = generate_test_wav(test_wav)
    print(f"  Test WAV: {len(audio_samples)} samples, {len(audio_samples) / SAMPLE_RATE:.1f}s")

    t0 = time.monotonic()
    vad_segments = run_vad(audio_samples, vad_model)
    timings["vad_infer"] = time.monotonic() - t0
    print(f"  VAD inference: {timings['vad_infer']:.3f}s, {len(vad_segments)} segments")
    for i, seg in enumerate(vad_segments):
        print(f"    Seg {i + 1}: {seg.start:.3f}s - {seg.end:.3f}s ({seg.duration_ms():.0f}ms)")

    # ---- Stage 2: STT ----
    print("\n" + "=" * 50)
    print("Stage 2: STT Load + Inference")
    from models.stt_runner import DEFAULT_MODEL_DIR as STT_MODEL_DIR
    from models.stt_runner import load_stt_model, run_stt

    t0 = time.monotonic()
    stt_model = load_stt_model("small", "cuda", "float16", STT_MODEL_DIR)
    timings["stt_load"] = time.monotonic() - t0
    print(f"  STT load: {timings['stt_load']:.3f}s")

    snap = snapshot("2. After STT load")
    snapshots.append(snap)

    t0 = time.monotonic()
    stt_segments, stt_info = run_stt(stt_model, test_wav, language="ko", beam_size=5)
    timings["stt_infer"] = time.monotonic() - t0
    transcribed = " ".join(seg.text for seg in stt_segments)
    print(f"  STT inference: {timings['stt_infer']:.3f}s")
    print(f"  Transcription: '{transcribed}'")
    print(f"  STT info: {stt_info}")

    # ---- Stage 3: Unload STT -> Load LLM ----
    print("\n" + "=" * 50)
    print("Stage 3: Unload STT -> Load LLM")

    del stt_model
    # Force ctranslate2 to release GPU memory
    try:
        import ctranslate2  # type: ignore[import-not-found]

        if hasattr(ctranslate2, "unload_backends"):
            ctranslate2.unload_backends()
    except (ImportError, AttributeError):
        pass
    clear_gpu()
    time.sleep(3)

    snap = snapshot("3. After STT unload")
    snapshots.append(snap)

    from models.llm_runner import DEFAULT_MODEL_DIR as LLM_MODEL_DIR
    from models.llm_runner import find_gguf_model, load_llm_model, run_generation

    gguf_path = find_gguf_model(LLM_MODEL_DIR)
    if gguf_path is None:
        print(f"  FATAL: No GGUF model found in {LLM_MODEL_DIR}")
        return 1

    # Drop page cache to free CUDA memory before LLM load
    import subprocess

    try:
        subprocess.run(
            ["sudo", "tee", "/proc/sys/vm/drop_caches"],
            input=b"1",
            check=True,
            timeout=5,
            capture_output=True,
        )
        print("  Page cache dropped for CUDA memory reclaim")
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        print("  Page cache drop skipped (no root permission)")

    # Try GPU layers from high to low, fall back if ENOMEM
    llm_model = None
    for try_layers in [-1, 15, 10, 5, 0]:
        try:
            t0 = time.monotonic()
            llm_model = load_llm_model(
                str(gguf_path),
                n_gpu_layers=try_layers,
                n_ctx=2048,
            )
            timings["llm_load"] = time.monotonic() - t0
            print(f"  LLM load: {timings['llm_load']:.3f}s (n_gpu_layers={try_layers})")
            break
        except (ValueError, RuntimeError) as e:
            print(f"  LLM n_gpu_layers={try_layers} failed: {e}")
            clear_gpu()
            time.sleep(1)

    if llm_model is None:
        print("  FATAL: LLM could not load with any GPU config")
        return 1

    snap = snapshot("4. After LLM load")
    snapshots.append(snap)

    prompt = (
        "<|im_start|>system\n"
        "너는 뭉이야. 10살 미만 아이들의 첫 번째 AI 친구야.\n"
        "존댓말은 쓰지 말고, 친절하고 따뜻한 반말만 써. 짧고 쉬운 단어만 써.\n"
        "아이의 감정에 공감해주고 칭찬해줘. 대답은 2~3문장으로 짧게 해.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "뭉이야 안녕! 나랑 놀자!<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    t0 = time.monotonic()
    llm_text, llm_tokens, llm_ttft, llm_gen_time = run_generation(
        llm_model,
        prompt,
        max_tokens=200,
        stop=["<|im_end|>", "<|im_start|>"],
        enable_thinking=True,
    )
    timings["llm_infer"] = time.monotonic() - t0

    from models.llm_runner import strip_think_tags

    raw_text = llm_text.strip()
    clean_text = strip_think_tags(raw_text)
    print(f"  LLM inference: {timings['llm_infer']:.3f}s")
    print(f"  TTFT: {llm_ttft:.3f}s, tokens: {llm_tokens}")
    print(f"  [Thinking]: '{raw_text[:150]}...'") if len(raw_text) > 150 else None
    print("  아이: 뭉이야 안녕! 나랑 놀자!")
    print(f"  뭉이: {clean_text}")
    llm_text = clean_text

    # ---- Stage 4: Unload LLM -> Load TTS ----
    print("\n" + "=" * 50)
    print("Stage 4: Unload LLM -> Load TTS")

    del llm_model
    clear_gpu()
    time.sleep(1)

    snap = snapshot("5. After LLM unload")
    snapshots.append(snap)

    from models.tts_runner import SupertonicEngine

    tts_model_dir = os.path.join(LLM_MODEL_DIR, "supertonic-2")
    tts_engine = SupertonicEngine(tts_model_dir, voice_style="F1")
    t0 = time.monotonic()
    tts_engine.load()
    timings["tts_load"] = time.monotonic() - t0
    print(f"  TTS load: {timings['tts_load']:.3f}s")

    snap = snapshot("6. After TTS load")
    snapshots.append(snap)

    tts_text = llm_text.strip() if llm_text.strip() else "안녕! 나는 뭉이야. 같이 놀자!"
    t0 = time.monotonic()
    try:
        audio_out, sr = tts_engine.synthesize(tts_text)
        timings["tts_infer"] = time.monotonic() - t0
        print(f"  TTS synthesis: {timings['tts_infer']:.3f}s")
        print(f"  Output: {len(audio_out)} samples, {sr} Hz, {len(audio_out) / sr:.2f}s")
    except Exception as e:
        timings["tts_infer"] = time.monotonic() - t0
        print(f"  TTS synthesis FAILED: {e}")

    test_wav.unlink(missing_ok=True)

    # ========== SUMMARY ==========
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print("\n--- Memory Profile ---")
    header = f"  {'Stage':<28} {'RAM(MB)':>8} {'RSS(MB)':>8} {'CUDA(MB)':>9} {'Swap(MB)':>9}"
    print(header)
    print("  " + "-" * 68)
    for s in snapshots:
        print(
            f"  {s['label']:<28}"
            f" {s['used_mb']:>8}"
            f" {s['rss_mb']:>8}"
            f" {s['cuda_mb']:>9}"
            f" {s['swap_mb']:>9}"
        )

    print("\n--- Memory Deltas ---")
    base = snapshots[0]["used_mb"]
    for i in range(1, len(snapshots)):
        delta = snapshots[i]["used_mb"] - snapshots[i - 1]["used_mb"]
        total_delta = snapshots[i]["used_mb"] - base
        print(f"  {snapshots[i]['label']}: prev {delta:+d} MB, baseline {total_delta:+d} MB")

    peak = max(s["used_mb"] for s in snapshots)
    total_ram = snapshots[0]["total_mb"]
    print(f"\n  Peak RAM: {peak} / {total_ram} MB")
    fits = "YES" if peak < 7500 else "NO"
    print(f"  Fits 8GB: {fits}")

    print("\n--- Timing ---")
    total = 0.0
    for key, val in timings.items():
        print(f"  {key:<15}: {val:.3f}s")
        total += val
    print(f"  {'TOTAL':<15}: {total:.3f}s")

    e2e_keys = [
        "stt_load",
        "stt_infer",
        "llm_load",
        "llm_infer",
        "tts_load",
        "tts_infer",
    ]
    e2e = sum(timings.get(k, 0) for k in e2e_keys)
    print(f"  E2E turn (STT+LLM+TTS): {e2e:.3f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
