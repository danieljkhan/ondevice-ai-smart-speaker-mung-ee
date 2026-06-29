from __future__ import annotations

from importlib import import_module
from typing import Any


def _import_ok(module_name: str) -> tuple[bool, str | None]:
    try:
        import_module(module_name)
        return True, None
    except Exception as exc:  # pragma: no cover - diagnostic path
        return False, f"{type(exc).__name__}: {exc}"


def probe_torch_cuda() -> dict[str, Any]:
    """Check PyTorch installation and CUDA availability."""
    try:
        import torch

        return {
            "installed": True,
            "version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
        }
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {
            "installed": False,
            "version": None,
            "cuda_available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def probe_onnxruntime() -> dict[str, Any]:
    """Check ONNX Runtime installation and available providers."""
    try:
        import onnxruntime as ort

        providers = list(ort.get_available_providers())
        return {
            "installed": True,
            "version": str(ort.__version__),
            "providers": providers,
            "has_cuda_provider": "CUDAExecutionProvider" in providers,
        }
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {
            "installed": False,
            "version": None,
            "providers": [],
            "has_cuda_provider": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def probe_optional_inference_libs() -> dict[str, Any]:
    """Check availability of optional inference libraries (sherpa_onnx, llama_cpp)."""
    sherpa_ok, sherpa_error = _import_ok("sherpa_onnx")
    ll_ok, ll_error = _import_ok("llama_cpp")
    return {
        "sherpa_onnx": {"installed": sherpa_ok, "error": sherpa_error},
        "llama_cpp": {"installed": ll_ok, "error": ll_error},
    }
