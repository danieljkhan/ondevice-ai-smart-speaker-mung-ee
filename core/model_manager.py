"""Model lifecycle manager for the Mungi AI pipeline.

Manages loading, unloading, and status tracking of all 4 AI models
(VAD, STT, LLM, TTS) with sequential GPU loading for Jetson 8GB.

Sequential loading keeps at most one large model on the GPU at a time,
preventing ENOMEM on unified-memory devices.

Usage::

    from core.model_manager import ModelManager, ManagerConfig, ModelType

    mm = ModelManager(ManagerConfig(model_dir="/opt/mungi/ai_models"))
    mm.initialize()          # loads VAD (CPU-resident) only
    mm.load(ModelType.STT)   # GPU: load STT
    mm.load(ModelType.LLM)   # GPU: unload STT → load LLM
"""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import subprocess
import sys
import threading
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Any

from core.runtime import detect_runtime_paths

logger = logging.getLogger("mungi.core.model_manager")
PRELOAD_JOIN_TIMEOUT = 2.0
_TTS_LOAD_HEADROOM_MB = 512


# ---------------------------------------------------------------------------
# Enums and data structures
# ---------------------------------------------------------------------------


class ModelType(Enum):
    """Identifies which model to load in sequential GPU mode."""

    NONE = "none"
    STT = "stt"
    LLM = "llm"
    TTS = "tts"
    VAD = "vad"


class MemoryHealth(Enum):
    """3-level memory health classification for Jetson 8GB."""

    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


class ModelState(Enum):
    """Lifecycle state of an individual model."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


@dataclass
class ModelStatus:
    """Observable status of a single model slot."""

    name: str
    state: ModelState = ModelState.UNLOADED
    load_time_s: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for logging/JSON."""
        return {
            "name": self.name,
            "state": self.state.value,
            "load_time_s": round(self.load_time_s, 3),
            "error": self.error,
        }


@dataclass
class LlmLoadDiagnostics:
    """Runtime diagnostics for the most recent LLM load attempt."""

    selected_n_gpu_layers: int | None = None
    loaded_n_gpu_layers: int | None = None
    attempted_n_gpu_layers: list[int] = field(default_factory=list)
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for logging/JSONL output."""
        return {
            "selected_n_gpu_layers": self.selected_n_gpu_layers,
            "loaded_n_gpu_layers": self.loaded_n_gpu_layers,
            "attempted_n_gpu_layers": list(self.attempted_n_gpu_layers),
            "fallback_used": self.fallback_used,
        }


@dataclass
class GemmaModelLoadResult:
    """Result metadata for a Gemma primary/fallback model load."""

    model: Any
    model_path_actual: str
    fallback_used: bool = False
    fallback_reason: str | None = None


@dataclass
class ManagerConfig:
    """Configuration for :class:`ModelManager`.

    Attributes:
        vad_model_path: Path to ``silero_vad.jit``. ``None`` → torch.hub.
        stt_model_size: Sherpa-ONNX STT selector or legacy alias. Override
            via MUNGI_STT_MODEL_SIZE env; invalid values keep the current
            configured/default selector.
        stt_device: ``"cpu"`` by default to reserve GPU memory for the LLM.
            Override via MUNGI_STT_PROVIDER env (accepted values: ``"cuda"``,
            ``"cpu"``; invalid values silently ignored). Wave 3 T3.1 GPU build
            available but not default-on until memory-safety verified via
            M1 measurement.
        stt_compute_type: ``"float16"``, ``"int8"``, or ``"float32"``.
        model_dir: Root directory for AI model files.
        llm_model_path: Explicit GGUF path. ``None`` → auto-discover the
            active GGUF model from ``model_dir``.
        llm_n_gpu_layers: GPU offload layers (``-1`` = all).
        llm_n_ctx: LLM context window size in tokens.
        llm_full_offload_memfree_mb: MemFree threshold for ``-1`` selection.
        llm_partial_offload_memfree_mb: MemFree threshold for 20-layer selection.
        stt_resident: Keep the STT model loaded between turns when safe.
        stt_resident_min_memavailable_mb: Minimum ``MemAvailable`` required to
            keep resident STT loaded after a turn.
        llm_resident: Keep the LLM loaded between turns.
        tts_resident: Keep the TTS engine loaded between turns when safe.
        llm_model_family: Model family for stop-sequence selection.
            ``"auto"`` detects from GGUF filename. ``"qwen"`` or ``"gemma"``
            for explicit override.
        tts_engine: ``"supertonic"`` as the only supported TTS engine.
        tts_model_dir: Path to TTS model directory.
        tts_voice_style: Supertonic voice selector — either a preset name
            shipped with the model (e.g. "F2", "M1") or an absolute path to
            a custom voice JSON file (e.g. "/var/lib/mungi/voices/tobi.json").
            Overridable via the ``MUNGI_TTS_VOICE_STYLE`` env var.
        tts_voice_style_ko: Korean Supertonic voice selector. Overridable via
            the ``MUNGI_TTS_VOICE_STYLE_KO`` env var.
        tts_voice_style_en: English Supertonic voice selector. Overridable via
            the ``MUNGI_TTS_VOICE_STYLE_EN`` env var.
        memory_limit_mb: RSS alarm threshold in MB (default 6000).
    """

    vad_model_path: str | None = None
    stt_model_size: str = "small"
    stt_device: str = "cpu"
    stt_compute_type: str = "float16"
    model_dir: str = ""
    llm_model_path: str | None = None
    llm_n_gpu_layers: int = -1  # Full GPU offload (page cache drop enables this)
    llm_n_ctx: int = 2048
    llm_full_offload_memfree_mb: int = 3000
    llm_partial_offload_memfree_mb: int = 2500
    stt_resident: bool = False
    stt_resident_min_memavailable_mb: int = 1024
    llm_resident: bool = True  # Keep LLM loaded between turns (skip unload). L1 default.
    tts_resident: bool = False  # Keep TTS loaded between turns (skip unload)
    llm_model_family: str = "auto"  # "auto", "qwen", or "gemma"
    tts_engine: str = "supertonic"
    tts_model_dir: str = ""
    tts_voice_style: str = "F2"
    tts_voice_style_ko: str | None = None
    tts_voice_style_en: str | None = None
    memory_limit_mb: int = 6000

    def __post_init__(self) -> None:
        """Fill defaults from runtime paths if not explicitly set."""
        self.llm_full_offload_memfree_mb = self._read_env_int(
            "MUNGI_LLM_FULL_OFFLOAD_MEMFREE_MB",
            self.llm_full_offload_memfree_mb,
        )
        self.llm_partial_offload_memfree_mb = self._read_env_int(
            "MUNGI_LLM_PARTIAL_OFFLOAD_MEMFREE_MB",
            self.llm_partial_offload_memfree_mb,
        )
        self.stt_resident = self._read_env_bool("MUNGI_STT_RESIDENT", self.stt_resident)
        self.llm_resident = self._read_env_bool("MUNGI_LLM_RESIDENT", self.llm_resident)
        self.tts_resident = self._read_env_bool("MUNGI_TTS_RESIDENT", self.tts_resident)
        env_tts_voice_legacy = os.environ.get("MUNGI_TTS_VOICE_STYLE", "").strip()
        env_tts_voice_ko = os.environ.get("MUNGI_TTS_VOICE_STYLE_KO", "").strip()
        env_tts_voice_en = os.environ.get("MUNGI_TTS_VOICE_STYLE_EN", "").strip()
        if env_tts_voice_legacy:
            self.tts_voice_style = env_tts_voice_legacy
        if env_tts_voice_ko:
            self.tts_voice_style_ko = env_tts_voice_ko
        elif self.tts_voice_style_ko is None and env_tts_voice_legacy:
            self.tts_voice_style_ko = env_tts_voice_legacy
        if env_tts_voice_en:
            self.tts_voice_style_en = env_tts_voice_en
        elif self.tts_voice_style_en is None and env_tts_voice_legacy:
            self.tts_voice_style_en = env_tts_voice_legacy
        env_provider = os.environ.get("MUNGI_STT_PROVIDER", "").strip().lower()
        if env_provider in ("cuda", "cpu"):
            self.stt_device = env_provider
        env_stt_model = os.environ.get("MUNGI_STT_MODEL_SIZE", "").strip()
        if env_stt_model:
            try:
                from models.stt_runner import resolve_model_size

                resolve_model_size(env_stt_model)
                self.stt_model_size = env_stt_model
            except (ImportError, ValueError) as exc:
                logger.warning(
                    "Invalid MUNGI_STT_MODEL_SIZE=%r; keeping default %r: %s",
                    env_stt_model,
                    self.stt_model_size,
                    exc,
                )
        if self.llm_partial_offload_memfree_mb > self.llm_full_offload_memfree_mb:
            warnings.warn(
                "llm_partial_offload_memfree_mb exceeded "
                "llm_full_offload_memfree_mb; clamping partial threshold",
                stacklevel=2,
            )
            self.llm_partial_offload_memfree_mb = self.llm_full_offload_memfree_mb
        if not self.model_dir:
            self.model_dir = detect_runtime_paths().model_root
        if not self.tts_model_dir:
            self.tts_model_dir = f"{self.model_dir}/supertonic-2"
        env_family = os.environ.get("MUNGI_LLM_MODEL_FAMILY", "").strip().lower()
        if env_family in ("qwen", "gemma", "auto"):
            self.llm_model_family = env_family

    @staticmethod
    def _read_env_int(name: str, default: int) -> int:
        """Read an integer override from the environment."""
        raw = os.getenv(name, "").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            warnings.warn(
                f"Ignoring invalid integer value for {name}: {raw!r}",
                stacklevel=2,
            )
            return default

    @staticmethod
    def _read_env_bool(name: str, default: bool) -> bool:
        """Read a boolean override from the environment."""
        raw = os.getenv(name, "").strip()
        if not raw:
            return default

        normalized = raw.lower()
        if normalized in {"1", "true", "yes"}:
            return True
        if normalized in {"0", "false", "no"}:
            return False

        warnings.warn(
            f"Ignoring invalid boolean value for {name}: {raw!r}; treating it as False",
            stacklevel=2,
        )
        return False


# ---------------------------------------------------------------------------
# ModelManager
# ---------------------------------------------------------------------------


class ModelManager:
    """Orchestrates loading/unloading of all AI models.

    All heavy imports (torch, sherpa_onnx, llama_cpp, supertonic) are
    deferred to load-time so this module can be imported on any platform.
    """

    _MODEL_NAMES: tuple[str, ...] = ("vad", "stt", "llm", "tts")

    def __init__(self, config: ManagerConfig | None = None) -> None:
        self._config = config or ManagerConfig()
        self._models: dict[str, Any] = {n: None for n in self._MODEL_NAMES}
        self._status: dict[str, ModelStatus] = {n: ModelStatus(name=n) for n in self._MODEL_NAMES}
        # Sequential GPU loading state
        self._current_gpu_model: ModelType = ModelType.NONE
        self._load_lock = threading.RLock()
        self._preload_lock = threading.Lock()
        self._preload_cancelled = threading.Event()
        self._preload_thread: threading.Thread | None = None
        self._llm_load_diagnostics = LlmLoadDiagnostics()
        self._latest_gemma_model_load_result: GemmaModelLoadResult | None = None
        self._active_stt_hotwords_csv: str | None = None
        self._sudo_check_lock = threading.Lock()
        self._passwordless_sudo_available: bool | None = None
        self._sudo_warning_once: set[str] = set()
        self._detected_model_family: str = "qwen"

    # ---- Public properties ------------------------------------------------

    @property
    def vad(self) -> Any:
        """Loaded Silero VAD model, or ``None``."""
        return self._models["vad"]

    @property
    def stt(self) -> Any:
        """Loaded Sherpa-ONNX STT recognizer wrapper, or ``None``."""
        return self._models["stt"]

    @property
    def llm(self) -> Any:
        """Loaded llama-cpp Llama instance, or ``None``."""
        return self._models["llm"]

    @property
    def llm_model_family(self) -> str:
        """Return the detected or configured model family (``"qwen"`` or ``"gemma"``)."""
        return getattr(self, "_detected_model_family", "qwen")

    @property
    def tts(self) -> Any:
        """Loaded TTSEngine instance, or ``None``."""
        return self._models["tts"]

    @property
    def current_gpu_model(self) -> ModelType:
        """Which model currently occupies the GPU slot."""
        return self._current_gpu_model

    @property
    def config(self) -> ManagerConfig:
        """Return the active manager configuration."""
        return self._config

    def latest_llm_load_diagnostics(self) -> dict[str, Any]:
        """Return diagnostics for the latest LLM load attempt."""
        return self._llm_load_diagnostics.to_dict()

    def latest_gemma_model_load_result(self) -> GemmaModelLoadResult | None:
        """Return the latest Gemma model-path fallback result, if any."""
        return self._latest_gemma_model_load_result

    # ---- Status -----------------------------------------------------------

    def status(self) -> dict[str, ModelStatus]:
        """Return a snapshot of all model statuses."""
        return dict(self._status)

    def is_ready(self, name: str) -> bool:
        """Check if a specific model is loaded and ready."""
        return self._status[name].state == ModelState.READY

    def all_ready(self) -> bool:
        """Check if all 4 models are loaded and ready."""
        return all(s.state == ModelState.READY for s in self._status.values())

    # ---- Sequential GPU loading API ---------------------------------------

    def initialize(self) -> None:
        """Start the manager by loading only CPU-resident models.

        Loads VAD (CPU-resident). STT/LLM/TTS are loaded on-demand
        so each stage can release unified memory before the next
        heavyweight model is needed.
        """
        logger.info("Initializing ModelManager (CPU models only)")
        self.load_vad()
        logger.info("ModelManager initialized. VAD ready, transient models on-demand.")

    def load(self, model_type: ModelType, *, stt_hotwords_csv: str | None = None) -> None:
        """Load a model using the sequential memory-management protocol.

        Jetson unified memory means CPU-only models still compete with
        the LLM's GPU offload for the same physical RAM. STT therefore
        participates in sequential stage loading even though it runs on
        CPU. TTS is loaded on-demand unless resident mode keeps it warm.

        Args:
            model_type: Which model to load.
            stt_hotwords_csv: Optional Qwen3-ASR hotword CSV for STT loads.
        """
        cancel_preload_for_stt = (
            model_type == ModelType.STT and not self._is_preload_worker_thread()
        )
        if cancel_preload_for_stt:
            self.cancel_preload_and_join()
        try:
            with self._load_lock:
                self._load_unlocked(model_type, stt_hotwords_csv=stt_hotwords_csv)
        finally:
            if cancel_preload_for_stt:
                self._join_preload_thread(context="foreground STT cleanup")
                self._clear_preload_cancel_if_idle()

    def _load_unlocked(
        self,
        model_type: ModelType,
        *,
        stt_hotwords_csv: str | None = None,
    ) -> None:
        """Load a model while the lifecycle lock is already held."""
        effective_stt_hotwords_csv: str | None = None
        if model_type == ModelType.STT:
            effective_stt_hotwords_csv = self._resolve_stt_hotwords_csv(stt_hotwords_csv)
            if self._stt_hotwords_changed(effective_stt_hotwords_csv):
                logger.info("STT hotwords changed; reloading recognizer")
                self.unload_stt(force=True)
        # Skip reload if LLM is already loaded (resident mode)
        if (
            model_type == ModelType.LLM
            and self._config.llm_resident
            and self._models.get("llm") is not None
        ):
            logger.info("LLM already resident, skipping reload")
            return
        if (
            model_type == ModelType.STT
            and self._config.stt_resident
            and self._models.get("stt") is not None
            and self.is_ready("stt")
            and self._active_stt_hotwords_csv == effective_stt_hotwords_csv
        ):
            logger.info("STT already resident, skipping reload")
            self._current_gpu_model = ModelType.STT
            return
        if model_type == ModelType.NONE:
            return

        name = model_type.value

        # Skip if already the active sequential stage and ready.
        if model_type in (ModelType.STT, ModelType.LLM):
            if self._current_gpu_model == model_type and self.is_ready(name):
                if (
                    model_type == ModelType.STT
                    and self._active_stt_hotwords_csv != effective_stt_hotwords_csv
                ):
                    logger.info("STT current stage has stale hotwords; reloading")
                else:
                    logger.debug("Model '%s' already loaded for this stage, skipping", name)
                    return
        elif self.is_ready(name):
            logger.debug("Model '%s' already loaded, skipping", name)
            return

        # Memory-critical stages use sequential loading.
        if model_type == ModelType.STT:
            self._unload_current_gpu()
            if stt_hotwords_csv is None:
                self.load_stt()
            else:
                self.load_stt(stt_hotwords_csv=effective_stt_hotwords_csv)
            self._active_stt_hotwords_csv = effective_stt_hotwords_csv
            self._current_gpu_model = ModelType.STT
        elif model_type == ModelType.LLM:
            self._unload_current_gpu()
            self._load_llm_full_gpu()
            self._current_gpu_model = ModelType.LLM
        elif model_type == ModelType.VAD:
            self.load_vad()
        elif model_type == ModelType.TTS:
            self._trim_resident_llm_before_tts_if_needed()
            self.load_tts()

    def _resolve_stt_hotwords_csv(self, stt_hotwords_csv: str | None) -> str:
        """Resolve the effective Qwen3-ASR hotword CSV used by an STT load."""
        from models.stt_runner import _resolve_qwen3_asr_hotwords

        return _resolve_qwen3_asr_hotwords(stt_hotwords_csv)

    def _stt_hotwords_changed(self, requested_csv: str) -> bool:
        """Return whether a ready STT recognizer has different hotwords."""
        return (
            self._models.get("stt") is not None
            and self.is_ready("stt")
            and self._active_stt_hotwords_csv != requested_csv
        )

    def _trim_resident_llm_before_tts_if_needed(self) -> bool:
        """Unload resident LLM when projected TTS load would exceed the memory limit."""
        if not self._config.llm_resident:
            return False
        if self._models.get("llm") is None:
            return False

        usage_mb = self.check_memory_mb()
        if usage_mb == 0:
            return False

        projected_mb = usage_mb + _TTS_LOAD_HEADROOM_MB
        limit_mb = self._config.memory_limit_mb
        if projected_mb <= limit_mb:
            logger.debug(
                "Pre-TTS memory gate kept resident LLM: usage=%d MB, "
                "tts_headroom=%d MB, projected=%d MB, limit=%d MB",
                usage_mb,
                _TTS_LOAD_HEADROOM_MB,
                projected_mb,
                limit_mb,
            )
            return False

        logger.info(
            "Pre-TTS memory gate unloading resident LLM: usage=%d MB, "
            "tts_headroom=%d MB, projected=%d MB, limit=%d MB",
            usage_mb,
            _TTS_LOAD_HEADROOM_MB,
            projected_mb,
            limit_mb,
        )
        self.unload_llm()
        return True

    @staticmethod
    def _get_meminfo_mb(field: str) -> int:
        """Read a field from ``/proc/meminfo`` in MB.

        Args:
            field: Field name (e.g. ``"MemFree"``, ``"MemAvailable"``).

        Returns 0 on non-Linux platforms or read failure.
        """
        if sys.platform != "linux":
            return 0
        target = f"{field}:"
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith(target):
                        return int(line.split()[1]) // 1024
        except (OSError, ValueError, IndexError):
            pass
        return 0

    def _get_available_memory_mb(self) -> int:
        """Read MemAvailable from ``/proc/meminfo``."""
        return self._get_meminfo_mb("MemAvailable")

    def _ensure_cuda_memory(self, required_mb: int = 4000) -> int:
        """Ensure sufficient free memory for CUDA allocation.

        On Jetson unified-memory devices, CUDA's ``cudaMalloc`` can only
        use ``MemFree`` pages — it cannot reclaim kernel page cache.
        This method checks ``MemFree`` and drops page cache if needed.

        Args:
            required_mb: Minimum ``MemFree`` in MB for full GPU offload.

        Returns:
            ``MemFree`` in MB after any reclamation attempt.
        """
        free_mb = self._get_meminfo_mb("MemFree")
        if free_mb == 0:
            return 0  # Non-Linux

        if free_mb >= required_mb:
            logger.info("MemFree=%d MB — sufficient for CUDA", free_mb)
            return free_mb

        logger.info(
            "MemFree=%d MB < %d MB required — dropping page cache",
            free_mb,
            required_mb,
        )
        self._drop_page_cache()

        free_mb = self._get_meminfo_mb("MemFree")
        logger.info("MemFree=%d MB after cache drop", free_mb)
        return free_mb

    def _select_gpu_layers(self) -> int:
        """Choose ``n_gpu_layers`` based on CUDA-usable free memory.

        Uses ``MemFree`` (not ``MemAvailable``) because CUDA cannot
        reclaim kernel page cache on Jetson unified-memory devices.
        Automatically drops page cache if ``MemFree`` is insufficient.

        Thresholds are configurable via ``ManagerConfig`` and default to:
        - >= 3000 MB → ``-1`` (full offload)
        - >= 2500 MB → 20 layers
        - < 2500 MB  → 10 layers (safe fallback)
        """
        full_threshold = self._config.llm_full_offload_memfree_mb
        partial_threshold = self._config.llm_partial_offload_memfree_mb
        free_mb = self._ensure_cuda_memory(required_mb=full_threshold)
        if free_mb == 0:
            # Non-Linux: use full offload (dev machine)
            return -1

        # Full offload is safe ONLY in resident mode (no load/unload cycles).
        # Non-resident mode causes GPU memory fragmentation → KV cache corruption.
        if free_mb >= full_threshold:
            # Full offload safe with llama-cpp-python >= 0.3.17:
            # - ggml_cpy() legacy KV path fully removed
            # - Jetson iGPU buffer detection fixed
            # - Verified: 0 garbage in both resident and non-resident modes
            logger.info(
                "MemFree=%d MB — using n_gpu_layers=-1 (full offload)",
                free_mb,
            )
            return -1
        if free_mb >= partial_threshold:
            logger.info(
                "MemFree=%d MB — using n_gpu_layers=20 (partial offload)",
                free_mb,
            )
            return 20

        fallback = max(self._config.llm_n_gpu_layers, 10)
        logger.warning(
            "MemFree=%d MB — using n_gpu_layers=%d (safe fallback)",
            free_mb,
            fallback,
        )
        return fallback

    @staticmethod
    def _is_cuda_oom_error(exc: Exception) -> bool:
        """Return ``True`` when *exc* looks like a recoverable llama-cpp load failure."""
        message = str(exc).lower()
        oom_markers = (
            "out of memory",
            "cuda out of memory",
            "cudamalloc failed",
            "enomem",
            "nvmapmemallocinternaltagged",
            # llama-cpp-python often collapses Jetson load failures to this ValueError.
            "failed to load model from file",
            "failed to create llama_context",
        )
        return any(marker in message for marker in oom_markers)

    def _can_use_passwordless_sudo(self) -> bool:
        """Return True when ``sudo -n`` is available for non-interactive helpers."""
        if sys.platform != "linux":
            return False

        with self._sudo_check_lock:
            if self._passwordless_sudo_available is None:
                try:
                    result = subprocess.run(
                        ["sudo", "-n", "true"],
                        check=False,
                        timeout=1,
                        capture_output=True,
                    )
                    self._passwordless_sudo_available = result.returncode == 0
                except (subprocess.SubprocessError, OSError):
                    self._passwordless_sudo_available = False
            return self._passwordless_sudo_available

    def _warn_passwordless_sudo_once(self, action: str) -> None:
        """Log a passwordless-sudo warning at most once per action."""
        if action in self._sudo_warning_once:
            return
        self._sudo_warning_once.add(action)
        logger.warning(
            "%s skipped: passwordless sudo is unavailable. "
            "Configure sudoers for faster Jetson recovery.",
            action,
        )

    def _compact_memory(self) -> None:
        """Best-effort Linux memory compaction for Jetson unified memory."""
        if sys.platform != "linux":
            return
        if not self._can_use_passwordless_sudo():
            self._warn_passwordless_sudo_once("Memory compaction")
            return
        try:
            subprocess.run(
                ["sudo", "-n", "tee", "/proc/sys/vm/compact_memory"],
                input=b"1",
                check=True,
                capture_output=True,
            )
            logger.info("Triggered Linux memory compaction for CUDA allocation recovery")
        except (subprocess.CalledProcessError, OSError) as exc:
            logger.warning("Memory compaction failed: %s", exc)

    def _recover_cuda_memory_after_oom(self) -> None:
        """Run best-effort recovery steps before a lower-offload retry."""
        self._gc_collect()
        self._drop_page_cache()
        self._compact_memory()

    def _llm_gpu_layer_candidates(self, chosen_layers: int) -> list[int]:
        """Return descending ``n_gpu_layers`` candidates for OOM recovery."""
        candidates = [chosen_layers]
        if chosen_layers == -1 or chosen_layers > 20:
            candidates.append(20)
        if chosen_layers == -1 or chosen_layers > 10:
            candidates.append(10)
        if chosen_layers != 0:
            candidates.append(0)

        # Preserve order while removing duplicates.
        unique_candidates: list[int] = []
        for candidate in candidates:
            if candidate not in unique_candidates:
                unique_candidates.append(candidate)
        return unique_candidates

    def _log_llm_load_diagnostics(self, attempted_layers: int, exc: Exception) -> None:
        """Log lightweight memory diagnostics for Jetson LLM load failures."""
        mem_free_mb = self._get_meminfo_mb("MemFree")
        mem_available_mb = self._get_available_memory_mb()
        logger.warning(
            "LLM load failed with n_gpu_layers=%d: %s (MemFree=%d MB, MemAvailable=%d MB)",
            attempted_layers,
            exc,
            mem_free_mb,
            mem_available_mb,
        )

        if sys.platform != "linux":
            return

        try:
            with open("/proc/buddyinfo", encoding="utf-8") as handle:
                buddyinfo = " || ".join(line.strip() for line in handle)
        except OSError:
            return

        if buddyinfo:
            logger.warning("BuddyInfo snapshot: %s", buddyinfo)

    def _load_llm_full_gpu(self) -> None:
        """Load LLM with dynamic ``n_gpu_layers`` for GPU offload.

        Selects the initial layer count based on available memory, then
        retries with progressively smaller offload targets when loading
        fails due to CUDA/Jetson ENOMEM.
        """
        from models.llm_runner import (
            MODEL_FAMILY_AUTO,
            detect_model_family,
            find_gguf_model,
            load_llm_model,
        )

        cfg = self._config

        if cfg.llm_model_path:
            gguf_path = cfg.llm_model_path
        else:
            found = find_gguf_model(cfg.model_dir)
            if found is None:
                msg = f"No GGUF model found in {cfg.model_dir}"
                raise FileNotFoundError(msg)
            gguf_path = str(found)

        candidates = self._llm_gpu_layer_candidates(self._select_gpu_layers())
        self._llm_load_diagnostics = LlmLoadDiagnostics(
            selected_n_gpu_layers=candidates[0] if candidates else None,
            attempted_n_gpu_layers=[],
            loaded_n_gpu_layers=None,
            fallback_used=False,
        )

        last_exc: Exception | None = None
        for idx, n_gpu_layers in enumerate(candidates):
            self._llm_load_diagnostics.attempted_n_gpu_layers.append(n_gpu_layers)
            try:
                self._do_load(
                    "llm",
                    partial(
                        load_llm_model,
                        gguf_path,
                        n_gpu_layers=n_gpu_layers,
                        n_ctx=cfg.llm_n_ctx,
                    ),
                )
                self._llm_load_diagnostics.loaded_n_gpu_layers = n_gpu_layers
                if cfg.llm_model_family == MODEL_FAMILY_AUTO:
                    self._detected_model_family = detect_model_family(gguf_path)
                else:
                    self._detected_model_family = cfg.llm_model_family
                logger.info("LLM model family: %s", self._detected_model_family)
                self._llm_load_diagnostics.fallback_used = idx > 0
                return
            except Exception as exc:
                last_exc = exc
                self._log_llm_load_diagnostics(n_gpu_layers, exc)
                if not self._is_cuda_oom_error(exc) or idx == len(candidates) - 1:
                    raise

                retry_layers = candidates[idx + 1]
                logger.warning(
                    "Recovering after LLM OOM with n_gpu_layers=%d; retrying with n_gpu_layers=%d",
                    n_gpu_layers,
                    retry_layers,
                )
                self._recover_cuda_memory_after_oom()

        if last_exc is not None:
            raise last_exc

    def _release_model_resources(self, name: str, model: Any, *, force: bool = False) -> None:
        """Run model-specific unload hooks before clearing references."""
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "unload_model"):
            try:
                inner.unload_model()
                if not force:
                    logger.info(
                        "Called unload_model() on '%s' inner model",
                        name,
                    )
            except Exception:
                if not force:
                    logger.warning(
                        "'%s' inner unload_model() failed",
                        name,
                        exc_info=True,
                    )

        unload = getattr(model, "unload", None)
        if callable(unload):
            try:
                unload()
            except Exception:
                if not force:
                    logger.warning(
                        "Model '%s' unload() raised an exception",
                        name,
                        exc_info=True,
                    )

    def _force_clear_llm_slot_unlocked(self) -> None:
        """Release any resident LLM and mark the GPU slot empty."""
        model = self._models.get("llm")
        if model is not None:
            self._release_model_resources("llm", model)

        self._do_unload("llm")
        self._current_gpu_model = ModelType.NONE
        self._gc_collect()
        self._drop_page_cache()

    def _unload_current_gpu(self) -> None:
        """Unload the current sequential stage and verify memory release."""
        with self._load_lock:
            self._unload_current_gpu_unlocked()

    def _unload_current_gpu_unlocked(self) -> None:
        """Unload the current sequential stage while the lifecycle lock is held."""
        if self._current_gpu_model == ModelType.NONE:
            return

        # Preserve LLM in memory when resident mode is active.
        if self._current_gpu_model == ModelType.LLM and self._config.llm_resident:
            logger.info("LLM resident mode: skipping GPU unload")
            return
        if self._current_gpu_model == ModelType.STT and self._config.stt_resident:
            logger.info("STT resident mode: skipping GPU unload")
            return
        if self._current_gpu_model == ModelType.TTS and self._config.tts_resident:
            logger.info("TTS resident mode: skipping GPU unload")
            return

        name = self._current_gpu_model.value
        model = self._models.get(name)

        if model is not None:
            self._release_model_resources(name, model)

        self._do_unload(name)
        self._gc_collect()

        if not self._verify_vram_released():
            logger.warning("VRAM not fully released after unloading '%s'", name)

        self._current_gpu_model = ModelType.NONE

    def _verify_vram_released(self, timeout_s: float = 3.0) -> bool:
        """Poll until VRAM is confirmed released.

        Only checks ``/proc/meminfo MemAvailable >= 1.0 GB``.
        The previous ``torch.cuda.memory_allocated()`` check was
        removed because ctranslate2 (used by STT) does not use
        the PyTorch CUDA allocator, making that metric meaningless
        for STT unload verification.

        Returns ``True`` immediately on non-Linux (Windows dev env).
        """
        if sys.platform != "linux":
            return True

        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            available_mb = self._get_available_memory_mb()
            if available_mb == 0 or available_mb >= 1024:
                return True
            time.sleep(0.05)

        return False

    # ---- Preloading -------------------------------------------------------

    def _is_preload_worker_thread(self) -> bool:
        """Return True when called from the currently registered preload worker."""
        return threading.current_thread() is self._preload_thread

    def preload_stt(self, *, stt_hotwords_csv: str | None = None) -> None:
        """Pre-load STT model in a background thread.

        Designed to be called on VAD ``speech_start`` to hide STT
        load latency. The method is single-flight: if a preload worker is
        already alive, later calls leave that worker in place and return.
        """
        requested_hotwords_csv = self._resolve_stt_hotwords_csv(stt_hotwords_csv)
        with self._preload_lock:
            if self._preload_cancelled.is_set():
                return
            if self._preload_thread is not None and self._preload_thread.is_alive():
                logger.debug("STT preload already running, skipping duplicate request")
                return
            if (
                self._config.stt_resident
                and self.is_ready("stt")
                and self._active_stt_hotwords_csv == requested_hotwords_csv
            ):
                logger.debug("STT already resident, skipping preload")
                return
            if (
                self.is_ready("stt")
                and self._current_gpu_model == ModelType.STT
                and self._active_stt_hotwords_csv == requested_hotwords_csv
            ):
                return

            def _preload() -> None:
                try:
                    if not self._preload_cancelled.is_set():
                        self.load(ModelType.STT, stt_hotwords_csv=stt_hotwords_csv)
                except Exception:
                    logger.error("STT preload failed", exc_info=True)

            self._preload_thread = threading.Thread(
                target=_preload,
                daemon=True,
                name="mungi-stt-preload",
            )
            self._preload_thread.start()

    def cancel_preload(self) -> None:
        """Cancel any pending STT preload."""
        self._preload_cancelled.set()

    def cancel_preload_and_join(self) -> None:
        """Cancel any STT preload worker and wait briefly for it to finish."""
        self.cancel_preload()
        self._join_preload_thread(context="foreground STT load")

    def is_preload_running(self) -> bool:
        """Return True when the background STT preload worker is alive."""
        thread = self._preload_thread
        return thread is not None and thread.is_alive()

    def reset_preload_state(self) -> None:
        """Wait briefly for an in-flight preload and reopen preload requests."""
        self._join_preload_thread(context="reset")
        self._clear_preload_cancel_if_idle()

    def _join_preload_thread(self, *, context: str) -> None:
        """Join the registered preload thread without holding manager locks."""
        with self._preload_lock:
            thread = self._preload_thread

        if thread is None or not thread.is_alive():
            return
        if thread is threading.current_thread():
            return

        thread.join(timeout=PRELOAD_JOIN_TIMEOUT)
        if thread.is_alive():
            logger.warning(
                "STT preload thread still running after %.1fs %s wait",
                PRELOAD_JOIN_TIMEOUT,
                context,
            )

    def _clear_preload_cancel_if_idle(self) -> None:
        """Clear preload cancellation only when no preload worker remains alive."""
        with self._preload_lock:
            thread = self._preload_thread
            if thread is not None and thread.is_alive():
                return
            self._preload_thread = None
            self._preload_cancelled.clear()

    def _clear_finished_preload_thread(self) -> None:
        """Drop the preload thread reference after it has exited."""
        with self._preload_lock:
            thread = self._preload_thread
            if thread is None or thread.is_alive():
                return
            self._preload_thread = None

    def _join_preload_before_unload(self) -> None:
        """Wait briefly for an in-flight preload before unloading STT."""
        self._join_preload_thread(context="unload")
        self._clear_finished_preload_thread()

    # ---- Memory health ----------------------------------------------------

    def check_memory_health(self) -> MemoryHealth:
        """Assess memory health with 3-level classification.

        - **NORMAL**: < 4500 MB — no action.
        - **WARNING**: 4500–6500 MB — forces GC.
        - **CRITICAL**: > 6500 MB — unloads transient models, retains VAD.

        Returns ``NORMAL`` on non-Linux where ``/proc`` is unavailable.
        """
        usage_mb = self.check_memory_mb()

        if usage_mb == 0:
            return MemoryHealth.NORMAL

        # ADR 0013: VAD retained here is intentional so the permanent
        # resident model survives while transient models are rebuilt lazily.
        if usage_mb > 6500:
            logger.critical(
                "CRITICAL memory: %d MB — unloading transient models "
                "(VAD retained as permanent resident)",
                usage_mb,
            )
            self.unload_stt(force=True)
            self.unload_llm()
            self.unload_tts(force=True)
            self._drop_page_cache()
            self._gc_collect()
            self._current_gpu_model = ModelType.NONE
            return MemoryHealth.CRITICAL

        if usage_mb > 4500:
            logger.warning("WARNING memory: %d MB — running GC", usage_mb)
            self._gc_collect()
            return MemoryHealth.WARNING

        return MemoryHealth.NORMAL

    def guard_stt_resident_memory(self) -> bool:
        """Run the post-turn memory guard for resident STT mode.

        Returns:
            ``True`` when next-turn STT preload may proceed. ``False`` when a
            CRITICAL memory event or ``MemAvailable`` guard forces resident STT
            cleanup for this turn.
        """
        health = self.check_memory_health()
        if health == MemoryHealth.CRITICAL:
            logger.info("Memory-health guard entered CRITICAL state; skipping STT preload")
            return False

        if not self._config.stt_resident or self._models["stt"] is None:
            return True

        available_mb = self._get_available_memory_mb()
        threshold_mb = self._config.stt_resident_min_memavailable_mb
        if available_mb == 0 or available_mb >= threshold_mb:
            return True

        # Layered eviction (ADR 0013 pattern): before forcing the
        # expensive STT reload, run the LLM-adjacent reclaim step and
        # re-check. Only fall back to STT unload if memory is still
        # below threshold.
        if self.flush_llm_prompt_cache():
            recovered_mb = self._get_available_memory_mb()
            if recovered_mb != 0 and recovered_mb >= threshold_mb:
                logger.info(
                    "MemAvailable=%d MB recovered to %d MB after "
                    "LLM-adjacent memory reclaim; STT retained",
                    available_mb,
                    recovered_mb,
                )
                return True
            logger.info(
                "MemAvailable=%d MB after LLM-adjacent memory reclaim; falling back "
                "to STT unload (threshold=%d MB)",
                recovered_mb,
                threshold_mb,
            )

        self.unload_stt(force=True)
        return False

    def flush_llm_prompt_cache(self) -> bool:
        """Reclaim LLM-adjacent memory via page-cache drop and GC.

        Layer D of the memory guard cascade (ADR 0060). The method name is
        preserved for caller compatibility, but the body no longer interacts
        with any LlamaCache; the T3.5 cleanup removed that path.
        """
        self._drop_page_cache()
        self._gc_collect()
        return True

    def guard_tts_resident_memory(self) -> bool:
        """Unload resident TTS when free system memory drops below the safety floor.

        Returns:
            ``True`` when the guard forced a resident TTS unload for this turn.
            ``False`` when no action was required.
        """
        if not self._config.tts_resident or self._models["tts"] is None:
            return False

        available_mb = self._get_available_memory_mb()
        # Keep a larger safety floor than resident STT because resident TTS
        # is held specifically to accelerate the next synthesis stage.
        threshold_mb = 1500
        if available_mb != 0 and available_mb < threshold_mb:
            logger.info(
                "MemAvailable=%d MB below resident TTS threshold=%d MB; forcing TTS unload",
                available_mb,
                threshold_mb,
            )
            self.unload_tts(force=True)
            return True

        return False

    # ---- Load helpers -----------------------------------------------------

    def _do_load(self, name: str, loader: Callable[[], Any]) -> Any:
        """Run *loader* callable, track timing and status.

        Args:
            name: Model slot name (``"vad"``, ``"stt"``, etc.).
            loader: Zero-argument callable that returns the loaded model.

        Returns:
            The loaded model object.
        """
        with self._load_lock:
            st = self._status[name]
            st.state = ModelState.LOADING
            st.error = None
            t0 = time.monotonic()
            try:
                model = loader()
                st.state = ModelState.READY
                st.load_time_s = time.monotonic() - t0
                logger.info("Model '%s' loaded in %.2fs", name, st.load_time_s)
                self._models[name] = model
                return model
            except Exception as exc:
                st.state = ModelState.ERROR
                st.error = str(exc)
                st.load_time_s = time.monotonic() - t0
                logger.error("Failed to load '%s': %s", name, exc)
                raise

    # ---- Individual loaders -----------------------------------------------

    @property
    def is_llm_loaded(self) -> bool:
        """Return True if the LLM model is currently loaded."""
        return self._models.get("llm") is not None

    def load_vad(self) -> None:
        """Load Silero VAD model."""
        from models.vad_runner import load_vad_model

        self._do_load(
            "vad",
            lambda: load_vad_model(self._config.vad_model_path),
        )

    def load_stt(self, *, stt_hotwords_csv: str | None = None) -> None:
        """Load Sherpa-ONNX STT model."""
        from models.stt_runner import load_stt_model

        cfg = self._config
        self._do_load(
            "stt",
            lambda: load_stt_model(
                cfg.stt_model_size,
                cfg.stt_device,
                cfg.stt_compute_type,
                cfg.model_dir,
                qwen3_asr_hotwords=stt_hotwords_csv,
            ),
        )

    def load_llm(self) -> None:
        """Load GGUF LLM via llama-cpp-python."""
        from models.llm_runner import find_gguf_model, load_llm_model

        cfg = self._config

        if cfg.llm_model_path:
            gguf_path = cfg.llm_model_path
        else:
            found = find_gguf_model(cfg.model_dir)
            if found is None:
                msg = f"No GGUF model found in {cfg.model_dir}"
                raise FileNotFoundError(msg)
            gguf_path = str(found)

        self._do_load(
            "llm",
            lambda: load_llm_model(
                gguf_path,
                cfg.llm_n_gpu_layers,
                cfg.llm_n_ctx,
            ),
        )

    def load_gemma_with_fallback(
        self,
        primary_path: str,
        fallback_path: str,
        *,
        n_gpu_layers: int,
        n_ctx: int,
    ) -> GemmaModelLoadResult:
        """Load Gemma primary GGUF, retrying the fallback path on load failure.

        The fallback is attempted only for model-load exceptions. Output
        validation and response sanitization remain in the pipeline generation
        path and do not trigger a model-path fallback.
        """
        from models.llm_runner import MODEL_FAMILY_GEMMA, load_gemma4_text_llm

        result: GemmaModelLoadResult | None = None
        primary = str(primary_path)
        fallback = str(fallback_path)

        def _load_gemma_text() -> Any:
            nonlocal result
            self._force_clear_llm_slot_unlocked()
            primary_error: str | None = None
            try:
                model = load_gemma4_text_llm(
                    primary,
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=n_ctx,
                )
            except Exception as exc:
                if primary == fallback:
                    raise
                primary_error = str(exc)
            else:
                result = GemmaModelLoadResult(
                    model=model,
                    model_path_actual=primary,
                    fallback_used=False,
                    fallback_reason=None,
                )
                return model

            logger.warning(
                "Gemma primary load failed; recovering before fallback load: %s",
                primary_error,
            )
            self._force_clear_llm_slot_unlocked()
            self._recover_cuda_memory_after_oom()
            try:
                model = load_gemma4_text_llm(
                    fallback,
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=n_ctx,
                )
            except Exception:
                self._force_clear_llm_slot_unlocked()
                raise

            result = GemmaModelLoadResult(
                model=model,
                model_path_actual=fallback,
                fallback_used=True,
                fallback_reason=primary_error,
            )
            return model

        try:
            loaded_model = self._do_load("llm", _load_gemma_text)
        except Exception:
            with self._load_lock:
                self._current_gpu_model = ModelType.NONE
                self._latest_gemma_model_load_result = None
            raise

        if result is None:
            result = GemmaModelLoadResult(
                model=loaded_model,
                model_path_actual=primary,
                fallback_used=False,
                fallback_reason=None,
            )
        with self._load_lock:
            self._current_gpu_model = ModelType.LLM
            self._detected_model_family = MODEL_FAMILY_GEMMA
            self._latest_gemma_model_load_result = result
        logger.info(
            "Gemma model loaded: actual_path=%s model_fallback_used=%s",
            result.model_path_actual,
            result.fallback_used,
        )
        return result

    def load_tts(self) -> None:
        """Load the Supertonic TTS engine."""
        from models.tts_runner import SupertonicEngine, _set_active_supertonic_engine

        cfg = self._config

        if cfg.tts_engine == "supertonic":
            ko = cfg.tts_voice_style_ko
            en = cfg.tts_voice_style_en
            if (ko is None) ^ (en is None):
                if ko is None:
                    ko = cfg.tts_voice_style
                else:
                    en = cfg.tts_voice_style

            if ko is not None and en is not None:
                engine = SupertonicEngine(
                    cfg.tts_model_dir,
                    voice_style_ko=ko,
                    voice_style_en=en,
                )
            else:
                engine = SupertonicEngine(
                    cfg.tts_model_dir,
                    voice_style=cfg.tts_voice_style,
                )
            _set_active_supertonic_engine(engine)
        else:
            msg = f"Unsupported TTS engine: {cfg.tts_engine}"
            raise ValueError(msg)

        def _load_engine() -> Any:
            try:
                engine.load()
            except Exception:
                _set_active_supertonic_engine(None)
                raise
            return engine

        self._do_load("tts", _load_engine)

    def load_all(self) -> None:
        """Load all 4 models sequentially.

        .. deprecated::
            Use :meth:`initialize` + :meth:`load` for sequential GPU
            management that prevents Jetson 8 GB ENOMEM.

        On failure, unloads any already-loaded models to prevent
        partial load state.
        """
        warnings.warn(
            "load_all() is deprecated. Use initialize() + "
            "load(ModelType) for sequential GPU loading.",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.info("Loading all models (deprecated path)...")
        try:
            self.load_vad()
            self.load_stt()
            self.load_llm()
            self.load_tts()
        except Exception:
            logger.error("load_all() failed, rolling back partial loads")
            self.unload_all()
            raise
        logger.info(
            "All models loaded. Total: %.2fs",
            sum(s.load_time_s for s in self._status.values()),
        )

    # ---- Unload -----------------------------------------------------------

    def _do_unload(self, name: str, *, force: bool = False) -> None:
        """Unload a model and reclaim memory."""
        with self._load_lock:
            if self._models[name] is not None:
                self._models[name] = None
            self._status[name].state = ModelState.UNLOADED
            self._status[name].error = None
            if not force:
                logger.info("Unloaded '%s'", name)

    def unload_vad(self) -> None:
        """Unload the VAD model."""
        with self._load_lock:
            self._do_unload("vad")
            self._gc_collect()

    def unload_stt(self, *, force: bool = False) -> None:
        """Unload the STT model and drop page cache.

        Jetson unified memory requires page cache drop after STT
        unload to reclaim CUDA-allocatable memory for LLM loading.
        """
        self._join_preload_before_unload()
        with self._load_lock:
            if self._config.stt_resident and not force:
                logger.debug("STT resident mode: skipping unload")
                return

            if force:
                logger.info("Forcing STT unload")

            model = self._models["stt"]
            if model is not None:
                self._release_model_resources("stt", model)
            self._do_unload("stt")
            self._active_stt_hotwords_csv = None
            if self._current_gpu_model == ModelType.STT:
                self._current_gpu_model = ModelType.NONE
            self._gc_collect()
            self._drop_page_cache()

    def unload_llm(self) -> None:
        """Unload the LLM and drop page cache.

        LLM uses the most GPU memory (~2GB+). Page cache drop
        reclaims CUDA-allocatable memory on Jetson unified memory.
        """
        with self._load_lock:
            self._do_unload("llm")
            self._latest_gemma_model_load_result = None
            if self._current_gpu_model == ModelType.LLM:
                self._current_gpu_model = ModelType.NONE
            self._gc_collect()
            self._drop_page_cache()

    def unload_tts(self, *, force: bool = False) -> None:
        """Unload the TTS engine and drop page cache after playback.

        Args:
            force: Override resident-mode retention and unload immediately.
        """
        with self._load_lock:
            cfg = self._config
            if cfg.tts_resident and not force:
                logger.info("unload_tts skipped: tts_resident=True. Use force=True to override.")
                return

            model = self._models["tts"]
            if model is not None:
                self._release_model_resources("tts", model)
            self._do_unload("tts")
            if cfg.tts_engine == "supertonic":
                from models.tts_runner import _set_active_supertonic_engine

                _set_active_supertonic_engine(None)
            if self._current_gpu_model == ModelType.TTS:
                self._current_gpu_model = ModelType.NONE
            self._gc_collect()
            self._drop_page_cache()

    def unload_all(self, *, force: bool = False) -> None:
        """Unload all models and reclaim memory.

        Args:
            force: Skip unload logging for low-friction shutdown paths such as
                signal handlers or partially initialized teardown.
        """
        with self._load_lock:
            for name in self._MODEL_NAMES:
                model = self._models.get(name)
                if model is not None:
                    self._release_model_resources(name, model, force=force)
                self._do_unload(name, force=force)
            self._current_gpu_model = ModelType.NONE
            self._gc_collect()
            if not force:
                logger.info("All models unloaded")

    # ---- Memory -----------------------------------------------------------

    def check_memory_mb(self) -> int:
        """Return current memory usage in MB (RSS + GPU VRAM).

        On Linux, reads VmRSS from ``/proc/self/status``.  When torch
        is available, adds ``torch.cuda.memory_allocated()`` to account
        for GPU VRAM on Jetson unified-memory systems.  Returns 0 on
        non-Linux platforms where ``/proc`` is unavailable.
        """
        rss_mb = 0
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_mb = int(line.split()[1]) // 1024
                        break
        except (OSError, ValueError, IndexError):
            pass

        gpu_mb = 0
        try:
            import torch

            if torch.cuda.is_available():
                gpu_mb = int(torch.cuda.memory_allocated() / (1024 * 1024))
        except ImportError:
            pass

        return rss_mb + gpu_mb

    def memory_ok(self) -> bool:
        """Return ``True`` if combined memory is below the alarm threshold."""
        usage = self.check_memory_mb()
        if usage == 0:
            return True  # Non-Linux: skip check
        return usage < self._config.memory_limit_mb

    # ---- Internal ---------------------------------------------------------

    def _drop_page_cache(self) -> None:
        """Drop kernel page cache to free memory for CUDA allocation.

        On Jetson unified-memory devices, the kernel page cache competes
        with CUDA for the same physical RAM. ``cudaMalloc`` cannot reclaim
        page cache, so we must explicitly drop it before GPU model loading.

        Uses ``echo 1`` (page cache only; preserves dentry/inode cache).
        Requires root or appropriate permissions. Logs a warning and
        continues silently if permission is denied.
        """
        if sys.platform != "linux":
            return
        if not self._can_use_passwordless_sudo():
            self._warn_passwordless_sudo_once("Page cache drop")
            return
        try:
            subprocess.run(
                ["sudo", "-n", "tee", "/proc/sys/vm/drop_caches"],
                input=b"1",
                check=True,
                capture_output=True,
            )
            logger.info("Page cache dropped for CUDA memory reclaim")
        except (subprocess.CalledProcessError, OSError) as exc:
            logger.warning("Page cache drop failed: %s", exc)

    @staticmethod
    def _gc_collect() -> None:
        """Run Python GC, clear CUDA cache, and release glibc arena.

        On Linux (Jetson), calls ``malloc_trim(0)`` via ctypes to
        return freed C/C++ heap pages to the OS.  This reclaims
        ~50 MB that glibc retains in its arena after ctranslate2
        and other native libraries free memory.
        """
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

        # Return freed C heap to OS (Linux/glibc only)
        if sys.platform == "linux":
            try:
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except (OSError, AttributeError):
                pass
