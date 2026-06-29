"""Live-runtime content filter wiring tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.llm_backend_config import LLMBackendConfig
from core.pipeline import ConversationPipeline, PipelineConfig
from safety.content_filter import SAFE_FALLBACK_RESPONSE, ContentFilter


def _legacy_backend_config() -> LLMBackendConfig:
    """Return a lightweight backend config that avoids Gemma prompt loading."""
    return LLMBackendConfig(
        backend="qwen3_legacy",
        model_path=None,
        n_ctx=2048,
        max_tokens=256,
        temperature=0.4,
        n_gpu_layers=99,
    )


def test_from_default_eagerly_loads_production_assets() -> None:
    """The default factory returns a loaded production filter."""
    content_filter = ContentFilter.from_default()

    assert content_filter.category_count > 0
    assert content_filter.pattern_count > 0


def test_default_session_factory_injects_loaded_filter_and_blocks_input() -> None:
    """The SessionManager default live factory installs an active content filter."""
    from core import session_manager

    with patch("core.pipeline.LLMBackendConfig.load", return_value=_legacy_backend_config()):
        pipeline = session_manager._default_pipeline_factory(MagicMock())

    assert isinstance(pipeline._content_filter, ContentFilter)
    assert pipeline._content_filter.category_count > 0
    assert pipeline._content_filter.pattern_count > 0

    filter_result = pipeline._filter_text("나는 너를 죽이다")

    assert filter_result is not None
    assert filter_result.allowed is False
    assert filter_result.filtered == SAFE_FALLBACK_RESPONSE


def test_demo_live_factory_injects_loaded_filter_and_blocks_output() -> None:
    """The demo live factory installs a filter that blocks unsafe LLM output."""
    from scripts import demo_live

    with patch("core.pipeline.LLMBackendConfig.load", return_value=_legacy_backend_config()):
        pipeline = demo_live._make_pipeline_factory()(MagicMock())

    with patch.object(pipeline, "_run_llm", return_value=("kill", 1, 0.0)):
        response_text, _, _, _, _, output_filtered = pipeline._generate_response_candidate(
            [],
            "hello",
        )

    assert isinstance(pipeline._content_filter, ContentFilter)
    assert output_filtered is True
    assert response_text == SAFE_FALLBACK_RESPONSE


def test_demo_live_factory_propagates_filter_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live construction fails closed when default filter assets cannot load."""
    from scripts import demo_live

    def raise_missing_assets(self: ContentFilter) -> None:
        """Simulate missing production filter assets."""
        del self
        raise FileNotFoundError("missing filter config")

    monkeypatch.setattr(ContentFilter, "load", raise_missing_assets)

    with pytest.raises(FileNotFoundError, match="missing filter config"):
        demo_live._make_pipeline_factory()(MagicMock())


def test_default_session_factory_propagates_filter_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SessionManager default live construction also fails closed."""
    from core import session_manager

    def raise_missing_assets(self: ContentFilter) -> None:
        """Simulate missing production filter assets."""
        del self
        raise FileNotFoundError("missing filter config")

    monkeypatch.setattr(ContentFilter, "load", raise_missing_assets)

    with pytest.raises(FileNotFoundError, match="missing filter config"):
        session_manager._default_pipeline_factory(MagicMock())


def test_pipeline_without_filter_keeps_none_semantics() -> None:
    """Direct pipeline construction without a filter still returns None."""
    with patch("core.pipeline.LLMBackendConfig.load", return_value=_legacy_backend_config()):
        pipeline = ConversationPipeline(
            MagicMock(),
            PipelineConfig(enable_content_filter=True),
        )

    assert pipeline._filter_text("kill") is None
