"""Preflight check for ADR 0073 default Gemma 4 backend on Jetson.

Verifies that the resolved LLMBackendConfig, Gemma 4 GGUF, and llama.cpp loader
are all consistent before promoting the new default to a Jetson runtime. Run as
a gate before merging the LLM primary swap PR to main.

Exit codes:
    0: All checks pass; default Gemma backend will boot.
    2: Backend resolved to non-gemma4_text; default flip is not in effect.
    3: load_gemma4_text_llm raised; underlying exception is logged.
    4: Wrong model family at resolved path; architecture mismatch.
    5: Resolved path does not exist or is not a .gguf file.
"""

from __future__ import annotations

import logging
import struct
import sys
from pathlib import Path
from typing import BinaryIO

logger = logging.getLogger("mungi.scripts.preflight_gemma4_default")

GGUF_METADATA_VALUE_TYPE_UINT8 = 0
GGUF_METADATA_VALUE_TYPE_INT8 = 1
GGUF_METADATA_VALUE_TYPE_UINT16 = 2
GGUF_METADATA_VALUE_TYPE_INT16 = 3
GGUF_METADATA_VALUE_TYPE_UINT32 = 4
GGUF_METADATA_VALUE_TYPE_INT32 = 5
GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6
GGUF_METADATA_VALUE_TYPE_BOOL = 7
GGUF_METADATA_VALUE_TYPE_STRING = 8
GGUF_METADATA_VALUE_TYPE_ARRAY = 9
GGUF_METADATA_VALUE_TYPE_UINT64 = 10
GGUF_METADATA_VALUE_TYPE_INT64 = 11
GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12

GGUF_VERSION = 3
GGUF_MAGIC = b"GGUF"
GGUF_ARCHITECTURE_KEY = "general.architecture"
GGUF_KEY_MAX_BYTES = 65_535
GGUF_ARCHITECTURE_MAX_BYTES = 256

_GGUF_SCALAR_VALUE_SIZES: dict[int, int] = {
    GGUF_METADATA_VALUE_TYPE_UINT8: 1,
    GGUF_METADATA_VALUE_TYPE_INT8: 1,
    GGUF_METADATA_VALUE_TYPE_UINT16: 2,
    GGUF_METADATA_VALUE_TYPE_INT16: 2,
    GGUF_METADATA_VALUE_TYPE_UINT32: 4,
    GGUF_METADATA_VALUE_TYPE_INT32: 4,
    GGUF_METADATA_VALUE_TYPE_FLOAT32: 4,
    GGUF_METADATA_VALUE_TYPE_BOOL: 1,
    GGUF_METADATA_VALUE_TYPE_UINT64: 8,
    GGUF_METADATA_VALUE_TYPE_INT64: 8,
    GGUF_METADATA_VALUE_TYPE_FLOAT64: 8,
}

# Allow-list per Plan v4 section 6.2 step 5 and risk R7. The current Gemma 4
# GGUF family marker is expected to be "gemma4".
GEMMA_FAMILY_ARCHITECTURES: frozenset[str] = frozenset({"gemma4"})

# Observation paths reported on miss.
_OBSERVATION_PATHS: tuple[str, ...] = ("/home/mungi/.cache/huggingface/gemma-4-E2B-it-Q5_K_M.gguf",)


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    """Read exactly size bytes from a binary handle or raise ValueError."""
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("malformed GGUF: unexpected end of file")
    return data


def _read_u32(handle: BinaryIO) -> int:
    """Read one little-endian uint32 from a binary handle."""
    return int(struct.unpack("<I", _read_exact(handle, 4))[0])


def _read_u64(handle: BinaryIO) -> int:
    """Read one little-endian uint64 from a binary handle."""
    return int(struct.unpack("<Q", _read_exact(handle, 8))[0])


def _skip_bytes(handle: BinaryIO, size: int, file_size: int) -> None:
    """Move forward exactly size bytes without reading the skipped payload."""
    if size < 0:
        raise ValueError("malformed GGUF: negative skip size")

    current = handle.tell()
    if current + size > file_size:
        raise ValueError("malformed GGUF: value extends beyond end of file")
    handle.seek(size, 1)


def _read_gguf_string(
    handle: BinaryIO,
    file_size: int,
    *,
    encoding: str = "utf-8",
    max_size: int | None = None,
) -> str:
    """Read a GGUF string and decode it with the requested encoding."""
    length = _read_u64(handle)
    if max_size is not None and length > max_size:
        raise ValueError(f"malformed GGUF: string length exceeds {max_size} bytes")

    current = handle.tell()
    if current + length > file_size:
        raise ValueError("malformed GGUF: string extends beyond end of file")

    try:
        return _read_exact(handle, length).decode(encoding)
    except UnicodeDecodeError as exc:
        raise ValueError("malformed GGUF: string is not valid text") from exc


def _skip_gguf_string(handle: BinaryIO, file_size: int) -> None:
    """Skip a GGUF string payload after reading its byte length."""
    length = _read_u64(handle)
    _skip_bytes(handle, length, file_size)


def _skip_gguf_value(handle: BinaryIO, value_type: int, file_size: int) -> None:
    """Skip a GGUF metadata value of the supplied value type."""
    if value_type == GGUF_METADATA_VALUE_TYPE_STRING:
        _skip_gguf_string(handle, file_size)
        return

    if value_type == GGUF_METADATA_VALUE_TYPE_ARRAY:
        nested_value_type = _read_u32(handle)
        array_length = _read_u64(handle)
        for _ in range(array_length):
            _skip_gguf_value(handle, nested_value_type, file_size)
        return

    value_size = _GGUF_SCALAR_VALUE_SIZES.get(value_type)
    if value_size is None:
        raise ValueError(f"malformed GGUF: unknown metadata value type {value_type}")
    _skip_bytes(handle, value_size, file_size)


def _read_gguf_architecture(gguf_path: Path) -> str:
    """Parse a GGUF v3 header and return the general.architecture value.

    This minimal reader validates the GGUF magic bytes, verifies the v3 header,
    then walks the metadata KV table until it finds `general.architecture`.

    Raises:
        ValueError: If the file is malformed or lacks general.architecture.
    """
    file_size = gguf_path.stat().st_size
    with gguf_path.open("rb") as handle:
        magic = _read_exact(handle, 4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"malformed GGUF: invalid magic bytes {magic!r}")

        version = _read_u32(handle)
        if version != GGUF_VERSION:
            raise ValueError(f"unsupported GGUF version {version}; expected {GGUF_VERSION}")

        _ = _read_u64(handle)
        metadata_kv_count = _read_u64(handle)

        for _ in range(metadata_kv_count):
            key = _read_gguf_string(
                handle,
                file_size,
                encoding="ascii",
                max_size=GGUF_KEY_MAX_BYTES,
            )
            value_type = _read_u32(handle)

            if key == GGUF_ARCHITECTURE_KEY:
                if value_type != GGUF_METADATA_VALUE_TYPE_STRING:
                    raise ValueError("malformed GGUF: general.architecture is not a string")
                return _read_gguf_string(
                    handle,
                    file_size,
                    max_size=GGUF_ARCHITECTURE_MAX_BYTES,
                )

            _skip_gguf_value(handle, value_type, file_size)

    raise ValueError("malformed GGUF: general.architecture metadata is missing")


def main() -> int:
    """Run the Gemma 4 default preflight check."""
    from core.llm_backend_config import LLMBackendConfig

    cfg = LLMBackendConfig.load()
    logger.info("Resolved backend=%s model_path=%s", cfg.backend, cfg.model_path)

    if cfg.backend != "gemma4_text":
        logger.error("default flip not in effect (resolved backend=%s)", cfg.backend)
        return 2

    from models import llm_runner

    effective_path = Path(cfg.model_path or llm_runner.DEFAULT_GEMMA4_TEXT_MODEL_PATH)

    if not effective_path.exists() or effective_path.suffix != ".gguf":
        logger.error("resolved Gemma GGUF missing or non-.gguf at %s", effective_path)
        candidates = (
            llm_runner.DEFAULT_GEMMA4_TEXT_MODEL_PATH,
            *_OBSERVATION_PATHS,
        )
        present = [path for path in candidates if Path(path).exists()]
        if present:
            logger.error("Found candidate(s) at: %s", present)
            logger.error("Recommended fix: ln -s %s %s", present[0], effective_path)
        return 5

    try:
        arch = _read_gguf_architecture(effective_path)
    except ValueError as exc:
        logger.error("wrong model family at %s: %s", effective_path, exc)
        return 4

    if arch not in GEMMA_FAMILY_ARCHITECTURES:
        logger.error(
            "wrong model family at %s: expected one of %s, found %r",
            effective_path,
            sorted(GEMMA_FAMILY_ARCHITECTURES),
            arch,
        )
        return 4

    llm: object | None = None
    try:
        llm = llm_runner.load_gemma4_text_llm(
            str(effective_path),
            n_gpu_layers=99,
            n_ctx=2048,
        )
    except Exception as exc:
        logger.exception("load_gemma4_text_llm raised: %s", exc)
        return 3
    finally:
        if llm is not None:
            del llm

    logger.info("Gemma 4 default preflight PASS at %s", effective_path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    sys.exit(main())
