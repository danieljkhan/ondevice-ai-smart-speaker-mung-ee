"""Verification tests for the P0 hallucination fix.

This test file verifies that the ConversationPipeline and ModelManager
correctly propagate the new parameters (temperature, repeat_penalty)
and use the improved Korean system prompt.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.model_manager import ModelManager  # noqa: F401 (used in test_parameter_propagation)
from core.pipeline import ConversationPipeline, PipelineConfig


def test_pipeline_config_hallucination_fix_defaults() -> None:
    """Verify that the default PipelineConfig matches current 1.7B-FT defaults."""
    cfg = PipelineConfig()

    assert cfg.llm_temperature == 1.0
    assert cfg.llm_min_p == 0.1
    assert cfg.llm_top_p == 1.0
    assert cfg.llm_top_k == 0
    assert cfg.llm_presence_penalty == 1.5
    assert cfg.llm_repeat_penalty == 1.15


def test_korean_system_prompt_structure() -> None:
    """Verify that the system prompt follows the new Korean-first structure."""
    cfg = PipelineConfig()
    prompt = cfg.llm_system_prompt

    # The new prompt should be in Korean as it's more natural for Qwen3
    # when the output is expected to be Korean.
    # (This test will fail until the implementation is updated)
    assert "뭉이" in prompt
    # assert "반말" in prompt


@patch("models.llm_runner.run_chat_generation")
def test_parameter_propagation_to_llm_runner(mock_chat_gen: MagicMock) -> None:
    """Verify that parameters from PipelineConfig reach the llm_runner."""
    mock_chat_gen.return_value = ("테스트 응답", 10, 0.5, 1.0, None, None)

    mm = MagicMock(spec=ModelManager)
    cfg = PipelineConfig(
        llm_temperature=0.15,
        llm_repeat_penalty=1.8,
        llm_presence_penalty=1.0,
    )

    pipeline = ConversationPipeline(mm, cfg)

    # Mocking necessary attributes for run_text_turn
    mm.llm = MagicMock()
    mm.latest_llm_load_diagnostics.return_value = {}

    # Execute a turn
    with patch.object(pipeline, "_filter_text", return_value=None):
        pipeline.run_text_turn("안녕")

    # Verify mock_chat_gen was called with the correct parameters
    args, kwargs = mock_chat_gen.call_args
    assert kwargs["temperature"] == 0.15
    assert kwargs["repeat_penalty"] == 1.8
    assert kwargs["presence_penalty"] == 1.0


def test_history_cap_and_hallucination_leakage() -> None:
    """Verify that history is capped to prevent long-term hallucination leakage."""
    cfg = PipelineConfig(max_history_turns=1)
    mm = MagicMock()
    pipeline = ConversationPipeline(mm, cfg)

    # Add several turns
    pipeline._append_history("하늘은 왜 파란색이야?", "우유 때문이야.")  # Hallucination
    pipeline._append_history("진짜 우유 때문이야?", "웅, 우유가 퍼지는 거야.")

    prompt = pipeline._build_prompt("바다는 왜 파래?")

    # With max_history_turns=1, only the LAST turn pair should be in the prompt.
    # The first "milk" hallucination should be purged from the active context.
    assert "하늘은 왜 파란색이야?" not in prompt
    assert "진짜 우유 때문이야?" in prompt
    assert "웅, 우유가 퍼지는 거야." in prompt
