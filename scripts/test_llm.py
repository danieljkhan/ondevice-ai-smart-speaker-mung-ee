"""Qwen LLM (llama.cpp) standalone test script for Jetson Orin Nano.

Loads a GGUF model via llama-cpp-python, runs text generation with a
Korean prompt, and reports timing metrics (load time, TTFT, tokens/s)
and memory usage.

Usage:
    python scripts/test_llm.py
    python scripts/test_llm.py --model-path /opt/mungi/ai_models/qwen3-1.7b-q5_k_m.gguf
    python scripts/test_llm.py --model-dir /opt/mungi/ai_models --prompt "안녕!"
    python scripts/test_llm.py --max-tokens 256 --n-ctx 4096
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

from models.llm_runner import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_DIR,
    DEFAULT_N_CTX,
    DEFAULT_N_GPU_LAYERS,
    DEFAULT_STOP_SEQUENCES,
    find_gguf_model,
    load_llm_model,
    run_generation,
)
from scripts.utils import get_peak_memory_kb

logger = logging.getLogger("mungi.scripts.test_llm")

# Re-export for backward compatibility (tests, bench_model)
STOP_SEQUENCES = DEFAULT_STOP_SEQUENCES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PROMPT: str = "사용자: 안녕하세요, 제 이름은 뭉이에요. 오늘 뭐 하고 놀까?\n뭉이:"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMResult:
    """Full result of an LLM generation run including metrics."""

    model_path: str
    prompt: str
    generated_text: str
    completion_tokens: int
    model_load_time_s: float
    ttft_s: float
    generation_time_s: float
    tokens_per_s: float
    peak_memory_kb: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to a plain dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the LLM test script.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Qwen LLM (llama.cpp) standalone test -- load GGUF model and run text generation."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=("Direct path to a GGUF model file. Overrides --model-dir auto-discovery."),
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=(f"Directory to auto-discover *.gguf files (default: {DEFAULT_MODEL_DIR})."),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt text for generation.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=(f"Maximum tokens to generate (default: {DEFAULT_MAX_TOKENS})."),
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=DEFAULT_N_GPU_LAYERS,
        help=(f"Number of layers to offload to GPU. -1 = all (default: {DEFAULT_N_GPU_LAYERS})."),
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=DEFAULT_N_CTX,
        help=(f"Context window size in tokens (default: {DEFAULT_N_CTX})."),
    )
    return parser


def main() -> int:
    """Run the Qwen LLM test and report results.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args()

    # Resolve model path
    resolved_model_path: str | None = args.model_path
    if resolved_model_path is None:
        logger.info(
            "No --model-path given, auto-discovering in: %s",
            args.model_dir,
        )
        discovered = find_gguf_model(args.model_dir)
        if discovered is None:
            logger.error(
                "No GGUF files found in %s. Provide --model-path explicitly.",
                args.model_dir,
            )
            return 1
        resolved_model_path = str(discovered)
        logger.info("Using discovered model: %s", resolved_model_path)

    # Load model
    try:
        mem_before = get_peak_memory_kb()
        t0 = time.monotonic()
        llm = load_llm_model(
            model_path=resolved_model_path,
            n_gpu_layers=args.n_gpu_layers,
            n_ctx=args.n_ctx,
        )
        load_time = time.monotonic() - t0
        logger.info("Model loaded in %.3f seconds", load_time)
    except (ImportError, FileNotFoundError, RuntimeError) as exc:
        logger.error("Model load failed: %s", exc)
        return 1

    # Run generation
    logger.info("Running generation with prompt: %s", args.prompt)
    logger.info("  max_tokens=%d", args.max_tokens)

    try:
        generated_text, token_count, ttft, generation_time = run_generation(
            llm=llm,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
        )
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        return 1

    peak_memory = get_peak_memory_kb()

    # Compute tokens/s
    tokens_per_s: float = 0.0
    if generation_time > 0:
        tokens_per_s = token_count / generation_time

    # Build result
    result = LLMResult(
        model_path=resolved_model_path,
        prompt=args.prompt,
        generated_text=generated_text,
        completion_tokens=token_count,
        model_load_time_s=round(load_time, 4),
        ttft_s=round(ttft, 4),
        generation_time_s=round(generation_time, 4),
        tokens_per_s=round(tokens_per_s, 2),
        peak_memory_kb=max(peak_memory - mem_before, 0),
    )

    # Report
    output = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    logger.info("LLM result:\n%s", output)

    logger.info("--- Summary ---")
    logger.info("Model:           %s", resolved_model_path)
    logger.info("Load time:       %.3f s", load_time)
    logger.info("TTFT:            %.4f s", ttft)
    logger.info("Generation time: %.3f s", generation_time)
    logger.info("Tokens generated: %d", token_count)
    logger.info("Tokens/s:        %.2f", tokens_per_s)
    logger.info("Peak memory:     %d KB", result.peak_memory_kb)
    logger.info("Generated text:\n%s", generated_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
