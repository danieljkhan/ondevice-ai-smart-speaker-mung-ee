"""Unit tests for core.model_manager and core.pipeline modules.

These tests verify the module structure, configuration defaults, state
management, and prompt construction without requiring actual AI models.
"""

from __future__ import annotations

import logging
import math
import re
import struct
import wave
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from core.character_expression import CharacterExpression
from core.llm_backend_config import LLMBackendConfig

# ===================================================================
# core.model_manager tests
# ===================================================================


class TestModelManagerImport:
    """Verify module-level imports and exports."""

    def test_model_manager_importable(self) -> None:
        from core.model_manager import ModelManager

        assert ModelManager is not None

    def test_manager_config_importable(self) -> None:
        from core.model_manager import ManagerConfig

        assert ManagerConfig is not None

    def test_model_state_importable(self) -> None:
        from core.model_manager import ModelState

        assert ModelState is not None

    def test_model_status_importable(self) -> None:
        from core.model_manager import ModelStatus

        assert ModelStatus is not None


class TestManagerConfig:
    """Verify ManagerConfig defaults and post_init."""

    def test_default_stt_device(self) -> None:
        from core.model_manager import ManagerConfig

        cfg = ManagerConfig(model_dir="/fake/models")
        assert cfg.stt_device == "cpu"

    def test_default_stt_model_size(self) -> None:
        from core.model_manager import ManagerConfig

        cfg = ManagerConfig(model_dir="/fake/models")
        assert cfg.stt_model_size == "small"

    def test_default_llm_gpu_layers(self) -> None:
        from core.model_manager import ManagerConfig

        cfg = ManagerConfig(model_dir="/fake/models")
        assert cfg.llm_n_gpu_layers == -1  # Full GPU offload (page cache drop enables this)

    def test_default_llm_memfree_thresholds(self) -> None:
        from core.model_manager import ManagerConfig

        cfg = ManagerConfig(model_dir="/fake/models")
        assert cfg.llm_full_offload_memfree_mb == 3000
        assert cfg.llm_partial_offload_memfree_mb == 2500

    def test_llm_memfree_thresholds_from_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from core.model_manager import ManagerConfig

        monkeypatch.setenv("MUNGI_LLM_FULL_OFFLOAD_MEMFREE_MB", "4800")
        monkeypatch.setenv("MUNGI_LLM_PARTIAL_OFFLOAD_MEMFREE_MB", "3200")
        cfg = ManagerConfig(model_dir="/fake/models")
        assert cfg.llm_full_offload_memfree_mb == 4800
        assert cfg.llm_partial_offload_memfree_mb == 3200

    def test_stt_resident_env_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from core.model_manager import ManagerConfig

        monkeypatch.setenv("MUNGI_STT_RESIDENT", "1")
        cfg = ManagerConfig(model_dir="/fake/models")
        assert cfg.stt_resident is True

    def test_stt_resident_env_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from core.model_manager import ManagerConfig

        monkeypatch.setenv("MUNGI_STT_RESIDENT", "0")
        cfg = ManagerConfig(model_dir="/fake/models")
        assert cfg.stt_resident is False

    def test_stt_resident_env_invalid_warns(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from core.model_manager import ManagerConfig

        monkeypatch.setenv("MUNGI_STT_RESIDENT", "garbage")
        with pytest.warns(UserWarning, match="MUNGI_STT_RESIDENT"):
            cfg = ManagerConfig(model_dir="/fake/models")
        assert cfg.stt_resident is False

    def test_default_tts_engine(self) -> None:
        from core.model_manager import ManagerConfig

        cfg = ManagerConfig(model_dir="/fake/models")
        assert cfg.tts_engine == "supertonic"

    def test_tts_model_dir_auto_filled(self) -> None:
        from core.model_manager import ManagerConfig

        cfg = ManagerConfig(model_dir="/fake/models")
        assert cfg.tts_model_dir == "/fake/models/supertonic-2"

    def test_memory_limit_default(self) -> None:
        from core.model_manager import ManagerConfig

        cfg = ManagerConfig(model_dir="/fake/models")
        assert cfg.memory_limit_mb == 6000

    def test_model_dir_auto_detect(self) -> None:
        from core.model_manager import ManagerConfig

        with patch("core.model_manager.detect_runtime_paths") as mock_detect:
            mock_paths = MagicMock()
            mock_paths.model_root = "/auto/detected/models"
            mock_detect.return_value = mock_paths
            cfg = ManagerConfig()
            assert cfg.model_dir == "/auto/detected/models"


class TestModelState:
    """Verify ModelState enum values."""

    def test_states_exist(self) -> None:
        from core.model_manager import ModelState

        assert ModelState.UNLOADED.value == "unloaded"
        assert ModelState.LOADING.value == "loading"
        assert ModelState.READY.value == "ready"
        assert ModelState.ERROR.value == "error"


class TestModelStatus:
    """Verify ModelStatus data class."""

    def test_default_state(self) -> None:
        from core.model_manager import ModelState, ModelStatus

        st = ModelStatus(name="vad")
        assert st.state == ModelState.UNLOADED
        assert st.load_time_s == 0.0
        assert st.error is None

    def test_to_dict(self) -> None:
        from core.model_manager import ModelState, ModelStatus

        st = ModelStatus(name="stt", state=ModelState.READY, load_time_s=1.234)
        d = st.to_dict()
        assert d["name"] == "stt"
        assert d["state"] == "ready"
        assert d["load_time_s"] == 1.234


class TestModelManagerInit:
    """Verify ModelManager initialization and status queries."""

    def _make_manager(self) -> Any:
        from core.model_manager import ManagerConfig, ModelManager

        return ModelManager(ManagerConfig(model_dir="/fake/models"))

    def test_initial_status_all_unloaded(self) -> None:
        from core.model_manager import ModelState

        mm = self._make_manager()
        for st in mm.status().values():
            assert st.state == ModelState.UNLOADED

    def test_all_ready_false_initially(self) -> None:
        mm = self._make_manager()
        assert mm.all_ready() is False

    def test_is_ready_false_initially(self) -> None:
        mm = self._make_manager()
        assert mm.is_ready("vad") is False
        assert mm.is_ready("stt") is False

    def test_properties_none_initially(self) -> None:
        mm = self._make_manager()
        assert mm.vad is None
        assert mm.stt is None
        assert mm.llm is None
        assert mm.tts is None

    def test_status_returns_copy(self) -> None:
        mm = self._make_manager()
        s1 = mm.status()
        s2 = mm.status()
        assert s1 is not s2

    def test_memory_ok_on_non_linux(self) -> None:
        mm = self._make_manager()
        assert mm.memory_ok() is True  # /proc not available on Windows


class TestModelManagerLoadUnload:
    """Verify load/unload lifecycle with mocked model loaders."""

    def _make_manager(self, **config_kwargs: Any) -> Any:
        from core.model_manager import ManagerConfig, ModelManager

        return ModelManager(ManagerConfig(model_dir="/fake/models", **config_kwargs))

    def test_load_vad_sets_ready(self) -> None:
        from core.model_manager import ModelState

        mm = self._make_manager()
        fake_model = MagicMock()
        with patch("core.model_manager.ModelManager.load_vad") as mock_load:

            def side_effect() -> None:
                mm._models["vad"] = fake_model
                mm._status["vad"].state = ModelState.READY

            mock_load.side_effect = side_effect
            mm.load_vad()
            assert mm.is_ready("vad")
            assert mm.vad is fake_model

    def test_do_load_tracks_timing(self) -> None:
        mm = self._make_manager()
        mm._do_load("vad", lambda: "fake_vad")
        assert mm.is_ready("vad")
        assert mm._status["vad"].load_time_s >= 0

    def test_do_load_error_sets_error_state(self) -> None:
        from core.model_manager import ModelState

        mm = self._make_manager()
        with pytest.raises(ValueError, match="boom"):
            mm._do_load("stt", lambda: (_ for _ in ()).throw(ValueError("boom")))
        assert mm._status["stt"].state == ModelState.ERROR
        assert mm._status["stt"].error == "boom"

    def test_do_unload(self) -> None:
        from core.model_manager import ModelState

        mm = self._make_manager()
        mm._do_load("vad", lambda: "fake")
        assert mm.is_ready("vad")
        mm._do_unload("vad")
        assert mm._status["vad"].state == ModelState.UNLOADED
        assert mm.vad is None

    def test_unload_all(self) -> None:
        mm = self._make_manager()
        for name in ("vad", "stt", "llm", "tts"):
            mm._do_load(name, lambda: "fake")
        assert mm.all_ready()
        mm.unload_all()
        assert not mm.all_ready()
        assert mm.vad is None

    def test_unload_tts_drops_page_cache(self) -> None:
        from core.model_manager import ModelType

        mm = self._make_manager()
        fake_model = MagicMock()
        mm._models["tts"] = fake_model
        mm._current_gpu_model = ModelType.TTS

        with (
            patch.object(mm, "_release_model_resources") as mock_release,
            patch.object(mm, "_do_unload") as mock_do_unload,
            patch.object(mm, "_gc_collect") as mock_gc,
            patch.object(mm, "_drop_page_cache") as mock_drop_page_cache,
        ):
            mm.unload_tts()

        mock_release.assert_called_once_with("tts", fake_model)
        mock_do_unload.assert_called_once_with("tts")
        mock_gc.assert_called_once()
        mock_drop_page_cache.assert_called_once()
        assert mm._current_gpu_model == ModelType.NONE

    def test_load_stt_skips_reload_when_resident(self) -> None:
        from core.model_manager import ModelState, ModelType

        mm = self._make_manager(stt_resident=True)
        fake_stt = object()
        load_calls = 0

        def fake_load_stt() -> None:
            nonlocal load_calls
            load_calls += 1
            mm._models["stt"] = fake_stt
            mm._status["stt"].state = ModelState.READY

        with patch.object(mm, "load_stt", side_effect=fake_load_stt):
            mm.load(ModelType.STT)
            mm._current_gpu_model = ModelType.NONE
            mm.load(ModelType.STT)

        assert load_calls == 1
        assert mm.stt is fake_stt

    def test_unload_stt_force_overrides_resident(self) -> None:
        from core.model_manager import ModelState, ModelType

        mm = self._make_manager(stt_resident=True)
        fake_model = MagicMock()
        mm._models["stt"] = fake_model
        mm._status["stt"].state = ModelState.READY
        mm._current_gpu_model = ModelType.STT

        with (
            patch.object(mm, "_release_model_resources") as mock_release,
            patch.object(mm, "_gc_collect") as mock_gc,
            patch.object(mm, "_drop_page_cache") as mock_drop_page_cache,
        ):
            mm.unload_stt(force=True)

        mock_release.assert_called_once_with("stt", fake_model)
        mock_gc.assert_called_once()
        mock_drop_page_cache.assert_called_once()
        assert mm.stt is None
        assert mm._status["stt"].state == ModelState.UNLOADED
        assert mm._current_gpu_model == ModelType.NONE

    def test_preload_stt_noop_when_resident_ready(self) -> None:
        from core.model_manager import ModelState

        mm = self._make_manager(stt_resident=True)
        mm._models["stt"] = object()
        mm._status["stt"].state = ModelState.READY
        # Simulate the hotword CSV a prior real STT load would have recorded.
        mm._active_stt_hotwords_csv = mm._resolve_stt_hotwords_csv(None)

        with patch("core.model_manager.threading.Thread") as mock_thread:
            mm.preload_stt()

        mock_thread.assert_not_called()
        assert mm._preload_thread is None

    def test_guard_stt_resident_memory_critical_clears_stt(self) -> None:
        from core.model_manager import ModelState, ModelType

        mm = self._make_manager(stt_resident=True)
        fake_model = MagicMock()
        mm._models["stt"] = fake_model
        mm._status["stt"].state = ModelState.READY
        mm._current_gpu_model = ModelType.STT

        with (
            patch.object(mm, "check_memory_mb", return_value=7001),
            patch.object(mm, "_release_model_resources") as mock_release,
            patch.object(mm, "_gc_collect"),
        ):
            allow_preload = mm.guard_stt_resident_memory()

        assert allow_preload is False
        mock_release.assert_any_call("stt", fake_model)
        assert mm.stt is None
        assert mm._current_gpu_model == ModelType.NONE

    def test_guard_stt_resident_memory_forces_unload_when_memavailable_low(self) -> None:
        from core.model_manager import ModelState

        mm = self._make_manager(
            stt_resident=True,
            stt_resident_min_memavailable_mb=1024,
        )
        mm._models["stt"] = object()
        mm._status["stt"].state = ModelState.READY

        with (
            patch.object(mm, "check_memory_mb", return_value=4000),
            patch.object(mm, "_get_meminfo_mb", return_value=512) as mock_meminfo,
            patch.object(mm, "_drop_page_cache") as mock_drop_page_cache,
            patch.object(mm, "_gc_collect") as mock_gc_collect,
            patch.object(mm, "unload_stt") as mock_unload_stt,
        ):
            allow_preload = mm.guard_stt_resident_memory()

        assert allow_preload is False
        assert mock_meminfo.call_args_list == [call("MemAvailable"), call("MemAvailable")]
        mock_drop_page_cache.assert_called_once_with()
        mock_gc_collect.assert_called_once_with()
        mock_unload_stt.assert_called_once_with(force=True)


# ===================================================================
# core.pipeline tests
# ===================================================================


class TestPipelineImport:
    """Verify module-level imports."""

    def test_pipeline_importable(self) -> None:
        from core.pipeline import ConversationPipeline

        assert ConversationPipeline is not None

    def test_pipeline_config_importable(self) -> None:
        from core.pipeline import PipelineConfig

        assert PipelineConfig is not None

    def test_pipeline_state_importable(self) -> None:
        from core.pipeline import PipelineState

        assert PipelineState is not None

    def test_turn_result_importable(self) -> None:
        from core.pipeline import TurnResult

        assert TurnResult is not None

    def test_turn_metrics_importable(self) -> None:
        from core.pipeline import TurnMetrics

        assert TurnMetrics is not None


class TestPipelineConfig:
    """Verify PipelineConfig defaults."""

    def test_default_vad_threshold(self) -> None:
        from core.pipeline import PipelineConfig

        cfg = PipelineConfig()
        assert cfg.vad_threshold == 0.5

    def test_default_vad_pad_ms(self) -> None:
        from core.pipeline import PipelineConfig

        cfg = PipelineConfig()
        assert cfg.vad_pad_ms == 200

    def test_default_stt_language(self) -> None:
        from core.pipeline import PipelineConfig

        cfg = PipelineConfig()
        assert cfg.stt_language == "ko"

    def test_default_llm_max_tokens(self) -> None:
        from core.pipeline import PipelineConfig

        cfg = PipelineConfig()
        assert cfg.llm_max_tokens == 64

    def test_llm_low_level_chat_env_parsing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from core.pipeline import PipelineConfig

        monkeypatch.setenv("MUNGI_LLM_LOW_LEVEL_CHAT", "1")
        cfg = PipelineConfig()
        assert cfg.llm_low_level_chat is True

        monkeypatch.delenv("MUNGI_LLM_LOW_LEVEL_CHAT", raising=False)
        cfg2 = PipelineConfig()
        assert cfg2.llm_low_level_chat is False

    def test_resolve_llm_max_tokens_env_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from core.pipeline import PipelineConfig

        monkeypatch.delenv("MUNGI_LLM_MAX_TOKENS", raising=False)
        assert PipelineConfig().llm_max_tokens == 64
        monkeypatch.setenv("MUNGI_LLM_MAX_TOKENS", "60")
        assert PipelineConfig().llm_max_tokens == 60
        with caplog.at_level(logging.WARNING, logger="mungi.core.pipeline"):
            monkeypatch.setenv("MUNGI_LLM_MAX_TOKENS", "abc")
            assert PipelineConfig().llm_max_tokens == 64
            monkeypatch.setenv("MUNGI_LLM_MAX_TOKENS", "-5")
            assert PipelineConfig().llm_max_tokens == 64
            monkeypatch.setenv("MUNGI_LLM_MAX_TOKENS", "9999")
            assert PipelineConfig().llm_max_tokens == 4096
        assert "not an integer" in caplog.text
        assert "must be positive" in caplog.text
        assert "clamped to model ceiling 4096" in caplog.text

    def test_system_prompt_uses_new_korean_length_rule(self) -> None:
        from core.pipeline import PipelineConfig

        prompt = PipelineConfig().llm_system_prompt
        assert "Keep responses to 3-4 sentences, maximum 150 Korean characters." in prompt
        assert "Keep responses to 1-2 sentences, maximum 60 Korean characters." not in prompt

    def test_system_prompt_uses_ai_companion_identity(self) -> None:
        from core.pipeline import PipelineConfig

        prompt = PipelineConfig().llm_system_prompt
        assert "warm and curious AI friend for children under 10" in prompt
        assert "friendly AI companion" in prompt
        assert "trusted AI friend" in prompt
        assert "friendly puppy" not in prompt
        assert "puppy" not in prompt

    def test_default_llm_sampling_values(self) -> None:
        from core.pipeline import PipelineConfig

        cfg = PipelineConfig()
        assert cfg.llm_temperature == 1.0
        assert cfg.llm_top_p == 1.0
        assert cfg.llm_top_k == 0
        assert cfg.llm_min_p == 0.1
        assert cfg.llm_presence_penalty == 1.5
        assert cfg.llm_repeat_penalty == 1.15

    def test_system_prompt_mentions_mungi(self) -> None:
        from core.pipeline import PipelineConfig

        cfg = PipelineConfig()
        assert "뭉이" in cfg.llm_system_prompt

    def test_system_prompt_english_with_topic_adherence(self) -> None:
        from core.pipeline import PipelineConfig

        cfg = PipelineConfig()
        prompt = cfg.llm_system_prompt
        assert "LANGUAGE PROCESSING RULES" in prompt
        assert "CRITICAL RULES" in prompt
        assert "SPEECH RULES" in prompt
        assert "KNOWLEDGE BOUNDARY" in prompt
        assert "EMOTION RESPONSE RULES" in prompt
        assert "PERSONALITY" in prompt
        assert "BANNED endings" in prompt
        assert "와 진짜? 대박!" not in prompt
        # Must contain Korean tone anchors
        assert "그랬구나" in prompt
        assert "반말" in prompt
        assert "/no_think" not in prompt

    def test_system_prompt_bans_non_target_scripts(self) -> None:
        from core.pipeline import PipelineConfig

        prompt = PipelineConfig().llm_system_prompt
        assert (
            "- NEVER use Chinese characters (Hanzi like 汉, 字, 猫) or Japanese kana "
            "(あ, ア). Output MUST be Korean Hangul only."
        ) in prompt

    def test_system_prompt_prioritizes_topic_adherence(self) -> None:
        from core.pipeline import PipelineConfig

        cfg = PipelineConfig()
        prompt = cfg.llm_system_prompt
        lang_pos = prompt.index("LANGUAGE PROCESSING RULES")
        speech_pos = prompt.index("SPEECH RULES")
        critical_pos = prompt.index("CRITICAL RULES")
        knowledge_pos = prompt.index("KNOWLEDGE BOUNDARY")
        emotion_pos = prompt.index("EMOTION RESPONSE RULES")
        personality_pos = prompt.index("PERSONALITY")
        assert lang_pos < speech_pos < critical_pos < knowledge_pos < emotion_pos < personality_pos

    def test_section_ordering(self) -> None:
        from core.pipeline import PipelineConfig

        prompt = PipelineConfig().llm_system_prompt
        lang_pos = prompt.index("LANGUAGE PROCESSING RULES")
        speech_pos = prompt.index("SPEECH RULES")
        critical_pos = prompt.index("CRITICAL RULES")
        knowledge_pos = prompt.index("KNOWLEDGE BOUNDARY")
        emotion_pos = prompt.index("EMOTION RESPONSE RULES")
        personality_pos = prompt.index("PERSONALITY")
        assert lang_pos < speech_pos < critical_pos < knowledge_pos < emotion_pos < personality_pos

    def test_system_prompt_no_fixed_celebration_anchor(self) -> None:
        """Verify '와 진짜? 대박!' is NOT in the system prompt."""
        from core.pipeline import PipelineConfig

        cfg = PipelineConfig()
        assert "와 진짜? 대박!" not in cfg.llm_system_prompt

    def test_system_prompt_has_banned_endings_list(self) -> None:
        """Verify SPEECH RULES contains all 10 explicit banned endings."""
        from core.pipeline import PipelineConfig

        cfg = PipelineConfig()
        prompt = cfg.llm_system_prompt
        assert "BANNED endings" in prompt
        banned = [
            "-요",
            "-습니다",
            "-세요",
            "-해요",
            "-죠",
            "-까요",
            "-네요",
            "-거예요",
            "-줄게요",
            "-할게요",
        ]
        for ending in banned:
            assert ending in prompt, f"Missing banned ending: {ending}"

    def test_system_prompt_has_knowledge_boundary(self) -> None:
        """Verify KNOWLEDGE BOUNDARY section exists."""
        from core.pipeline import PipelineConfig

        cfg = PipelineConfig()
        assert "KNOWLEDGE BOUNDARY" in cfg.llm_system_prompt

    def test_system_prompt_has_emotion_response_rules(self) -> None:
        """Verify EMOTION RESPONSE RULES section exists."""
        from core.pipeline import PipelineConfig

        cfg = PipelineConfig()
        assert "EMOTION RESPONSE RULES" in cfg.llm_system_prompt

    def test_default_stop_sequences(self) -> None:
        from core.pipeline import PipelineConfig

        cfg = PipelineConfig()
        assert "<|im_end|>" in cfg.llm_stop_sequences

    def test_default_history_entry_cap(self) -> None:
        from core.pipeline import MAX_HISTORY_ENTRIES, PipelineConfig

        cfg = PipelineConfig()
        assert cfg.max_history_entries == MAX_HISTORY_ENTRIES


class TestImprovements7To11:
    """Verify anti-echo, normalization, and output-validator behavior."""

    @staticmethod
    def _section(prompt: str, start_header: str, end_header: str) -> str:
        start_index = prompt.index(start_header)
        start_line_end = prompt.index("\n", start_index)
        end_index = prompt.index(f"\n{end_header}", start_line_end)
        return prompt[start_line_end + 1 : end_index]

    def test_anti_echo_rule_in_prompt(self) -> None:
        from core.pipeline import PipelineConfig

        prompt = PipelineConfig().llm_system_prompt
        critical_pos = prompt.index("CRITICAL RULES")
        anti_echo_pos = prompt.index("ANTI-ECHO RULE")
        knowledge_pos = prompt.index("KNOWLEDGE BOUNDARY")
        assert critical_pos < anti_echo_pos < knowledge_pos

    def test_number_hallucination_rule_in_prompt(self) -> None:
        from core.pipeline import PipelineConfig

        prompt = PipelineConfig().llm_system_prompt
        knowledge_section = self._section(
            prompt,
            "KNOWLEDGE BOUNDARY",
            "EMOTION RESPONSE RULES",
        )
        assert "NEVER cite specific numbers" in knowledge_section

    def test_no_literal_celebration_examples(self) -> None:
        from core.pipeline import PipelineConfig

        prompt = PipelineConfig().llm_system_prompt
        emotion_section = self._section(
            prompt,
            "EMOTION RESPONSE RULES",
            "PERSONALITY",
        )
        assert "우와!" not in emotion_section
        assert "오 진짜?" not in emotion_section

    @pytest.mark.skip(
        reason=(
            "few-shot examples moved to persona.md after Option C trim; "
            "new persona-prompt test covers them."
        ),
    )
    def test_few_shot_examples_in_prompt(self) -> None:
        from core.pipeline import PipelineConfig

        prompt = PipelineConfig().llm_system_prompt
        examples_section = self._section(
            prompt,
            "CONVERSATION EXAMPLES (follow this style):",
            "PERSONALITY",
        )
        example_lines = [
            line for line in examples_section.splitlines() if line.startswith('- Child: "')
        ]
        korean_example_lines = [line for line in example_lines if re.search(r"[가-힣]", line)]
        assert len(korean_example_lines) >= 3

    def test_few_shot_examples_in_persona_md(self) -> None:
        persona_path = Path("assets/prompts/persona.md")
        persona_md = persona_path.read_text(encoding="utf-8")
        example_lines = [
            line for line in persona_md.splitlines() if re.match(r'- 아이: ".+"', line)
        ]
        assert len(example_lines) >= 3

    def test_normalize_stt_text_aliases(self) -> None:
        from core.pipeline import ConversationPipeline

        aliases = ["웅이", "문이", "멍인", "멍이", "무이", "멍의", "붕이", "몽이"]
        for alias in aliases:
            assert ConversationPipeline._normalize_stt_text(alias) == "뭉이"

    def test_normalize_stt_nuni_vocative_start(self) -> None:
        """Replace sentence-start vocative alias with the canonical name."""
        from core.pipeline import ConversationPipeline

        text = "눈이야 너는 치킨 좋아해"
        assert ConversationPipeline._normalize_stt_text(text) == "뭉이야 너는 치킨 좋아해"

    def test_normalize_stt_nuni_vocative_comma(self) -> None:
        """Replace comma-led vocative alias with the canonical name."""
        from core.pipeline import ConversationPipeline

        text = "안녕, 눈이야"
        assert ConversationPipeline._normalize_stt_text(text) == "안녕, 뭉이야"

    def test_normalize_stt_nuni_standalone(self) -> None:
        """Replace a standalone vocative alias with the canonical name."""
        from core.pipeline import ConversationPipeline

        assert ConversationPipeline._normalize_stt_text("눈이야") == "뭉이야"

    def test_normalize_stt_nuni_preserve_eye(self) -> None:
        """Preserve legitimate eye-related usage."""
        from core.pipeline import ConversationPipeline

        assert ConversationPipeline._normalize_stt_text("눈이 아파") == "눈이 아파"

    def test_normalize_stt_nuni_preserve_snow(self) -> None:
        """Preserve legitimate weather-related usage."""
        from core.pipeline import ConversationPipeline

        assert ConversationPipeline._normalize_stt_text("눈이 와") == "눈이 와"

    def test_normalize_stt_nuni_preserve_tired(self) -> None:
        """Preserve legitimate eye-fatigue usage."""
        from core.pipeline import ConversationPipeline

        assert ConversationPipeline._normalize_stt_text("눈이 피곤해") == "눈이 피곤해"

    def test_normalize_stt_nuni_end_of_sentence(self) -> None:
        """Preserve non-vocative sentence-final usage."""
        from core.pipeline import ConversationPipeline

        assert ConversationPipeline._normalize_stt_text("이건 눈이야") == "이건 눈이야"

    def test_normalize_stt_nuniya_unconditional(self) -> None:
        """Replace an unambiguous STT alias unconditionally."""
        from core.pipeline import ConversationPipeline

        assert ConversationPipeline._normalize_stt_text("눈이이야 안녕") == "뭉이야 안녕"

    def test_normalize_stt_miya_unconditional(self) -> None:
        """Replace the observed unambiguous STT alias unconditionally."""
        from core.pipeline import ConversationPipeline

        text = "미야 치킨 좋아해"
        assert ConversationPipeline._normalize_stt_text(text) == "뭉이야 치킨 좋아해"

    def test_normalize_stt_text_passthrough(self) -> None:
        from core.pipeline import ConversationPipeline

        text = "공룡이랑 놀고 싶어"
        assert ConversationPipeline._normalize_stt_text(text) == text

    def test_qwen3_asr_hotword_resolution_cases(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from models.stt_runner import (
            _HOTWORDS_BASELINE,
            _HOTWORDS_EXPLORATORY_TIER,
            _HOTWORDS_REQUIRED_TIER,
            LEGACY_QWEN3_ASR_HOTWORDS_TIER1_DEFAULT,
            _resolve_qwen3_asr_hotwords,
        )

        expected = _HOTWORDS_BASELINE + _HOTWORDS_REQUIRED_TIER + _HOTWORDS_EXPLORATORY_TIER
        monkeypatch.delenv("MUNGI_QWEN3_ASR_HOTWORDS", raising=False)

        assert LEGACY_QWEN3_ASR_HOTWORDS_TIER1_DEFAULT == ",".join(expected)
        assert len(expected) == 13
        assert _resolve_qwen3_asr_hotwords(None) == ""
        monkeypatch.setenv("MUNGI_QWEN3_ASR_HOTWORDS", "aaa,bbb")
        assert _resolve_qwen3_asr_hotwords(None) == "aaa,bbb"
        assert _resolve_qwen3_asr_hotwords("") == ""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("음수를 더해줘", "음수를 더해줘"),
            ("은수를 더해줘", "음수를 더해줘"),
            ("123 + 은수 =", "123 + 음수 ="),
            ("은수+1", "음수+1"),
            ("은수=음수", "음수=음수"),
        ],
    )
    def test_normalize_stt_negative_number_math_context(
        self,
        text: str,
        expected: str,
    ) -> None:
        from core.pipeline import ConversationPipeline

        assert ConversationPipeline._normalize_stt_text(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "은수",
            "은수랑 놀았어",
            "오늘 은수 만났어",
            "은수는 친구야",
        ],
    )
    def test_normalize_stt_negative_number_preserves_name_context(self, text: str) -> None:
        from core.pipeline import ConversationPipeline

        assert ConversationPipeline._normalize_stt_text(text) == text

    def test_hotword_candidates_do_not_trigger_clean_transcripts(self) -> None:
        from core.pipeline import _is_hotword_hallucination
        from models.stt_runner import LEGACY_QWEN3_ASR_HOTWORDS_TIER1_DEFAULT

        transcripts = (
            "안녕 뭉이야 오늘 뭐 하고 놀까",
            "나는 뭉이 이름이 정말 좋아",
            "나는 오늘 한글 공부가 정말 재미있어",
            "가족이 추석 이야기를 함께 나눴어",
            "맛있는 송편 만들기를 같이 해봤어",
            "오늘 단군신화 이야기를 재미있게 들었어",
            "학교에서 일제강점기 역사를 차분히 배웠어",
            "커다란 빙하 사진을 책에서 봤어",
            "동그란 자석 실험이 정말 신기했어",
            "높은 화산 모형을 과학 시간에 만들었어",
            "어제 지진 대피 연습을 학교에서 했어",
            "비 온 뒤 무지개 색깔이 정말 예뻤어",
            "명절에 한복 입고 사진을 찍었어",
        )

        for transcript in transcripts:
            assert (
                _is_hotword_hallucination(
                    transcript,
                    LEGACY_QWEN3_ASR_HOTWORDS_TIER1_DEFAULT,
                )
                is False
            )

    def test_hotword_near_neighbors_do_not_trigger_guard(self) -> None:
        from core.pipeline import _is_hotword_hallucination
        from models.stt_runner import LEGACY_QWEN3_ASR_HOTWORDS_TIER1_DEFAULT

        transcripts = (
            "한국 역사 이야기를 오늘 배웠어",
            "한 글자 이름을 천천히 써봤어",
            "자식 이야기는 어른들이 조심히 말해",
            "자전거 타는 연습을 공원에서 했어",
        )

        for transcript in transcripts:
            assert (
                _is_hotword_hallucination(
                    transcript,
                    LEGACY_QWEN3_ASR_HOTWORDS_TIER1_DEFAULT,
                )
                is False
            )

    def test_safety_topic_queries_do_not_trigger_hotword_guard(self) -> None:
        from core.pipeline import _is_hotword_hallucination
        from models.stt_runner import LEGACY_QWEN3_ASR_HOTWORDS_TIER1_DEFAULT

        queries = (
            "화산이 왜 터져?",
            "지진이 나면 어디로 가?",
            "피가 나면 어떻게 해?",
            "엄마 아빠한테 어떻게 말해?",
        )

        for query in queries:
            assert (
                _is_hotword_hallucination(
                    query,
                    LEGACY_QWEN3_ASR_HOTWORDS_TIER1_DEFAULT,
                )
                is False
            )

    def test_detect_echo_true(self) -> None:
        from models.llm_runner import detect_echo

        assert detect_echo("공룡은 왜 없어졌어", "공룡은 왜 없어졌어?") is True

    def test_detect_echo_false(self) -> None:
        from models.llm_runner import detect_echo

        assert detect_echo("공룡은 왜 없어졌어", "공룡은 아주 오래 전에 사라졌어") is False

    def test_repair_honorifics(self) -> None:
        from models.llm_runner import repair_honorifics

        assert repair_honorifics("좋아해요") == "좋아해"

    def test_repair_honorifics_multiple(self) -> None:
        from models.llm_runner import repair_honorifics

        assert repair_honorifics("같이 볼까요? 내가 줄게요") == "같이 볼까? 내가 줄게"

    def test_sanitize_response_includes_honorific_repair(self) -> None:
        from models.llm_runner import sanitize_response

        cleaned = sanitize_response("좋아해요 같이 가볼까요")
        assert cleaned == "좋아해 같이 가볼까"
        assert "해요" not in cleaned
        assert "볼까요" not in cleaned


class TestLlmRunnerChatGeneration:
    """Verify chat-completion streaming helper behavior."""

    def test_run_chat_generation_returns_empty_for_empty_messages(self) -> None:
        from models.llm_runner import run_chat_generation

        fake_llm = MagicMock()

        text, token_count, ttft, gen_time, cache_hit_tokens, cache_miss_tokens = (
            run_chat_generation(fake_llm, [])
        )

        assert text == ""
        assert token_count == 0
        assert ttft == -1.0
        assert gen_time == 0.0
        assert cache_hit_tokens is None
        assert cache_miss_tokens is None
        fake_llm.create_chat_completion.assert_not_called()

    def test_run_chat_generation_streams_content_and_thinking_overrides(self) -> None:
        from models.llm_runner import (
            DEFAULT_STOP_SEQUENCES,
            THINKING_MIN_P,
            THINKING_PRESENCE_PENALTY,
            THINKING_TEMPERATURE,
            THINKING_TOP_K,
            THINKING_TOP_P,
            run_chat_generation,
        )

        class FakeLLM:
            def __init__(self) -> None:
                self.kwargs: dict[str, Any] = {}

            def create_chat_completion(
                self,
                *,
                messages: list[dict[str, str]],
                **kwargs: Any,
            ) -> list[dict[str, Any]]:
                self.kwargs = {"messages": messages, **kwargs}
                return [
                    {"choices": [{"delta": {"role": "assistant"}}]},
                    {"choices": [{"delta": {"content": "안녕"}}]},
                    {"choices": [{"delta": {}}]},
                    {"choices": [{"delta": {"content": "!"}}]},
                ]

        fake_llm = FakeLLM()
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]

        text, token_count, ttft, gen_time, cache_hit_tokens, cache_miss_tokens = (
            run_chat_generation(
                fake_llm,
                messages,
                enable_thinking=True,
            )
        )

        assert text == "안녕!"
        assert token_count == 2
        assert ttft >= 0.0
        assert gen_time >= ttft
        assert cache_hit_tokens is None
        assert cache_miss_tokens is None
        assert fake_llm.kwargs["messages"] == messages
        assert fake_llm.kwargs["stop"] == DEFAULT_STOP_SEQUENCES
        assert fake_llm.kwargs["temperature"] == THINKING_TEMPERATURE
        assert fake_llm.kwargs["top_p"] == THINKING_TOP_P
        assert fake_llm.kwargs["top_k"] == THINKING_TOP_K
        assert fake_llm.kwargs["min_p"] == THINKING_MIN_P
        assert fake_llm.kwargs["presence_penalty"] == THINKING_PRESENCE_PENALTY


class TestPipelineState:
    """Verify PipelineState enum."""

    def test_all_states(self) -> None:
        from core.pipeline import PipelineState

        assert PipelineState.IDLE.value == "idle"
        assert PipelineState.LISTENING.value == "listening"
        assert PipelineState.TRANSCRIBING.value == "transcribing"
        assert PipelineState.THINKING.value == "thinking"
        assert PipelineState.SPEAKING.value == "speaking"
        assert PipelineState.ERROR.value == "error"


class TestTurnMetrics:
    """Verify TurnMetrics data class."""

    def test_default_values(self) -> None:
        from core.pipeline import TurnMetrics

        m = TurnMetrics()
        assert m.vad_time_s == 0.0
        assert m.llm_tokens == 0
        assert m.hotword_hallucination_detected is False
        assert m.hotword_hallucination_reason == "clean"
        assert m.stt_script_drift_detected is False
        assert m.template_topic_id is None
        assert m.template_mode is None
        assert m.crisis_matched is False
        assert m.crisis_topic_id is None
        assert m.crisis_escalation_target is None
        assert m.parent_disclosure_matched is False
        assert m.parent_disclosure_kind is None
        assert m.parent_disclosure_output_replaced is False
        assert m.belief_matched is False
        assert m.stt_provider_actual is None
        assert m.speech_segments == 0
        assert m.llm_model_fallback_used is False
        assert m.llm_model_path_actual is None
        assert m.llm_model_fallback_reason is None

    def test_to_dict(self) -> None:
        from core.pipeline import TurnMetrics

        m = TurnMetrics(
            vad_time_s=1.2345,
            tts_load_time_s=0.75,
            playback_time_s=1.5,
            llm_tokens=42,
            stt_provider_actual="cpu",
            speech_segments=2,
        )
        d = m.to_dict()
        assert d["vad_time_s"] == 1.234
        assert d["tts_load_time_s"] == 0.75
        assert d["playback_time_s"] == 1.5
        assert d["llm_tokens"] == 42
        assert d["stt_provider_actual"] == "cpu"
        assert d["speech_segments"] == 2
        assert d["template_topic_id"] is None
        assert d["template_mode"] is None
        assert d["crisis_matched"] is False
        assert d["crisis_topic_id"] is None
        assert d["crisis_escalation_target"] is None
        assert d["parent_disclosure_matched"] is False
        assert d["parent_disclosure_kind"] is None
        assert d["parent_disclosure_output_replaced"] is False
        assert d["belief_matched"] is False
        assert d["hotword_hallucination_detected"] is False
        assert d["hotword_hallucination_reason"] == "clean"
        assert d["stt_script_drift_detected"] is False
        assert d["llm_model_fallback_used"] is False
        assert d["llm_model_path_actual"] is None
        assert d["llm_model_fallback_reason"] is None

    def test_to_dict_round_trip_preserves_observability_fields(self) -> None:
        from core.pipeline import TurnMetrics

        metrics = TurnMetrics(
            stt_provider_actual="cpu",
            speech_segments=1,
        )
        restored = TurnMetrics(**metrics.to_dict())

        assert restored.stt_provider_actual == "cpu"
        assert restored.speech_segments == 1

    def test_turn_metrics_new_fields_default_to_none(self) -> None:
        from core.pipeline import TurnMetrics

        metrics = TurnMetrics()
        assert metrics.tts_first_chunk_ms is None
        assert metrics.llm_cache_hit_tokens is None
        assert metrics.llm_cache_miss_tokens is None
        assert metrics.llm_model_fallback_used is False
        assert metrics.llm_model_path_actual is None
        assert metrics.llm_model_fallback_reason is None
        assert metrics.turn_index_per_lang == 0

    def test_turn_metrics_to_dict_skips_none_new_fields(self) -> None:
        from core.pipeline import TurnMetrics

        metrics = TurnMetrics(vad_time_s=0.5)
        payload = metrics.to_dict()

        assert "tts_first_chunk_ms" not in payload
        assert "llm_cache_hit_tokens" not in payload
        assert "llm_cache_miss_tokens" not in payload
        assert payload["turn_index_per_lang"] == 0
        assert payload["template_topic_id"] is None
        assert payload["template_mode"] is None
        assert payload["crisis_matched"] is False
        assert payload["crisis_topic_id"] is None
        assert payload["crisis_escalation_target"] is None
        assert payload["parent_disclosure_matched"] is False
        assert payload["parent_disclosure_kind"] is None
        assert payload["parent_disclosure_output_replaced"] is False
        assert payload["belief_matched"] is False
        assert payload["stt_provider_actual"] is None
        assert payload["llm_model_fallback_used"] is False
        assert payload["llm_model_path_actual"] is None
        assert payload["llm_model_fallback_reason"] is None

    def test_turn_metrics_to_dict_includes_new_fields_when_populated(self) -> None:
        from core.pipeline import TurnMetrics

        metrics = TurnMetrics(
            tts_first_chunk_ms=12.3456,
            llm_cache_hit_tokens=7,
            llm_cache_miss_tokens=3,
            llm_model_fallback_used=True,
            llm_model_path_actual="/models/gemma-e2b.gguf",
            llm_model_fallback_reason="primary missing",
            turn_index_per_lang=2,
            crisis_matched=True,
            crisis_topic_id="fire_emergency",
            crisis_escalation_target="119",
            parent_disclosure_matched=True,
            parent_disclosure_kind="probe",
            parent_disclosure_output_replaced=True,
            belief_matched=True,
        )
        payload = metrics.to_dict()

        assert payload["tts_first_chunk_ms"] == 12.346
        assert payload["llm_cache_hit_tokens"] == 7
        assert payload["llm_cache_miss_tokens"] == 3
        assert payload["llm_model_fallback_used"] is True
        assert payload["llm_model_path_actual"] == "/models/gemma-e2b.gguf"
        assert payload["llm_model_fallback_reason"] == "primary missing"
        assert payload["turn_index_per_lang"] == 2
        assert payload["crisis_matched"] is True
        assert payload["crisis_topic_id"] == "fire_emergency"
        assert payload["crisis_escalation_target"] == "119"
        assert payload["parent_disclosure_matched"] is True
        assert payload["parent_disclosure_kind"] == "probe"
        assert payload["parent_disclosure_output_replaced"] is True
        assert payload["belief_matched"] is True


class TestHotwordHallucinationGuard:
    """Verify the Qwen3-ASR hotword prompt-echo guard."""

    @pytest.mark.parametrize(
        ("user_text", "hotwords_csv", "expected"),
        [
            ("뭉이야 뭉이 문지 뭉지 뭉지", "뭉이야,뭉이,문지,뭉지", True),
            ("뭉이야 뭉이 뭉이", "뭉이야,뭉이", True),
            ("안녕 뭉이야, 오늘 뭐 했어?", "뭉이야,뭉이", False),
            ("뭉이야", "뭉이야,뭉이", False),
            ("Hello Mungi", "뭉이야,뭉이", False),
            ("뭉이 뭉이", "뭉이야,뭉이", True),
            ("달은 숨바꼭질 뭉이 Moongee.", "달은,숨바꼭질,뭉이,Moongee", True),
            (
                "달은, 숨바꼭질, 뭉이, Moongee,",
                "달은,숨바꼭질,뭉이,Moongee",
                True,
            ),
            ("달은 숨바꼭질 뭉이 Moongee?!", "달은,숨바꼭질,뭉이,Moongee", True),
            ("뭉이, 뭉이.", "뭉이야,뭉이", True),
            (
                "오늘 달은 밝고 뭉이는 잘 자고 있어.",
                "달은,숨바꼭질,뭉이,Moongee",
                False,
            ),
        ],
    )
    def test_is_hotword_hallucination_cases(
        self,
        user_text: str,
        hotwords_csv: str,
        expected: bool,
    ) -> None:
        from core.pipeline import _is_hotword_hallucination

        assert _is_hotword_hallucination(user_text, hotwords_csv) is expected

    @pytest.mark.parametrize(
        ("raw_stt_text", "expected"),
        [
            ("뭉이야 뭉이야 뭉이야 뭉이야 뭉이야 뭉이야", ("full_collapse", 6)),
            ("뭉이야 뭉이야 뭉이야 안녕", ("partial_injection", 3)),
            ("뭉이야 안녕", ("clean", 1)),
            ("", ("clean", 0)),
            ("   \t\n  ", ("clean", 0)),
        ],
    )
    def test_detect_raw_wakeword_repetition_cases(
        self,
        raw_stt_text: str,
        expected: tuple[str, int],
    ) -> None:
        from core.pipeline import _detect_raw_wakeword_repetition

        assert _detect_raw_wakeword_repetition(raw_stt_text) == expected

    def test_detect_raw_wakeword_repetition_logs_two_token_subthreshold(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from core.pipeline import _detect_raw_wakeword_repetition

        with caplog.at_level(logging.DEBUG, logger="mungi.core.pipeline"):
            assert _detect_raw_wakeword_repetition("뭉이야 뭉이야 안녕") == ("clean", 2)

        assert "2-token sub-threshold" in caplog.text

    @pytest.mark.parametrize(
        ("raw_stt_text", "expected"),
        [
            (
                "뭉이야 뭉이 한글 추석 송편 단군신화 일제강점기 빅하 자석 화산 지진 무지개 한복",
                ("recitation", 12),
            ),
            (
                "뭉이야 뭉이 한글 추석 송편 단군신화 일제강점기 빙하 자석 화산 지진 무지개 한복.",
                ("recitation", 13),
            ),
            ("뭉이야 한글 추석 송편 단군신화 일제강점기 빙하", ("recitation", 7)),
            ("뭉이야 한글 추석 송편 단군신화", ("clean", 5)),
            ("", ("clean", 0)),
            ("   \t\n  ", ("clean", 0)),
            ("한글이랑 송편이 좋아", ("clean", 0)),
        ],
    )
    def test_detect_hotword_list_recitation_cases(
        self,
        raw_stt_text: str,
        expected: tuple[str, int],
    ) -> None:
        from core.pipeline import _detect_hotword_list_recitation

        assert _detect_hotword_list_recitation(raw_stt_text) == expected

    def test_detect_hotword_list_recitation_empty_vocabulary_is_clean(self) -> None:
        from core.pipeline import _detect_hotword_list_recitation

        assert _detect_hotword_list_recitation("뭉이야 한글", hotword_vocabulary=()) == (
            "clean",
            0,
        )

    def test_safety_topic_queries_do_not_trigger_recitation_guard(self) -> None:
        from core.pipeline import _detect_hotword_list_recitation

        queries = (
            "화산이 왜 터져?",
            "지진이 나면 어디로 가?",
            "피가 나면 어떻게 해?",
            "엄마 아빠한테 어떻게 말해?",
        )

        for query in queries:
            assert _detect_hotword_list_recitation(query)[0] == "clean"

    def test_legacy_user_text_reason_when_runtime_detectors_are_clean(self) -> None:
        from core.pipeline import _hotword_hallucination_reason

        assert _hotword_hallucination_reason("뭉이 뭉이", "뭉이야,뭉이", "뭉이 뭉이") == (
            "legacy_user_text"
        )

    @pytest.mark.parametrize(
        ("raw_stt_text", "expected_reason"),
        [
            ("뭉이야 뭉이야 뭉이야 안녕", "repetition"),
            (
                "뭉이야 뭉이 한글 추석 송편 단군신화 일제강점기 빙하 자석 화산 지진 무지개 한복",
                "recitation",
            ),
        ],
    )
    def test_run_turn_raw_stt_keyword_guard_reprompts(
        self,
        raw_stt_text: str,
        expected_reason: str,
    ) -> None:
        from core.model_manager import ModelType
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())
        fake_seg = MagicMock(start=0.0, end=0.5)
        fake_audio_out = [0.1] * 12

        def fake_run_stt(_speech_audio: Any) -> str:
            p._last_raw_stt_text = raw_stt_text
            return "안녕"

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", side_effect=fake_run_stt),
            patch.object(p, "_init_session_dir"),
            patch.object(p, "_run_llm") as mock_llm,
            patch.object(p, "_run_tts", return_value=(fake_audio_out, 22050)) as mock_tts,
            patch.object(p, "_play_audio_out"),
        ):
            result = p.run_turn([0.0] * 16000)

        assert result.hotword_hallucination_detected is True
        assert result.hotword_hallucination_reason == expected_reason
        assert result.metrics.hotword_hallucination_reason == expected_reason
        assert result.raw_stt_text == raw_stt_text
        mock_llm.assert_not_called()
        mock_tts.assert_called_once_with("응? 다시 말해줘!", language="ko")
        assert [call.args[0] for call in mm.load.call_args_list] == [
            ModelType.STT,
            ModelType.TTS,
        ]

    def test_run_text_turn_clean_keyword_guard_allows_llm(self) -> None:
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())
        fake_audio_out = [0.1] * 12

        with (
            patch.object(p, "_init_session_dir"),
            patch.object(p, "_run_llm", return_value=("좋아", 5, 0.1)) as mock_llm,
            patch.object(p, "_run_tts", return_value=(fake_audio_out, 22050)),
            patch.object(p, "_play_audio_out"),
        ):
            result = p.run_text_turn("한글 공부가 재미있어")

        assert result.hotword_hallucination_detected is False
        assert result.hotword_hallucination_reason == "clean"
        assert result.metrics.hotword_hallucination_reason == "clean"
        mock_llm.assert_called_once()

    def test_run_turn_keyword_guard_records_repetition_and_recitation_reason(self) -> None:
        from core.pipeline import ConversationPipeline, PipelineConfig

        raw_stt_text = "뭉이야 뭉이야 뭉이야 한글 추석 송편 단군신화 일제강점기 빙하"
        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())
        fake_seg = MagicMock(start=0.0, end=0.5)
        fake_audio_out = [0.1] * 12

        def fake_run_stt(_speech_audio: Any) -> str:
            p._last_raw_stt_text = raw_stt_text
            return "안녕"

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", side_effect=fake_run_stt),
            patch.object(p, "_init_session_dir"),
            patch.object(p, "_run_llm") as mock_llm,
            patch.object(p, "_run_tts", return_value=(fake_audio_out, 22050)),
            patch.object(p, "_play_audio_out"),
        ):
            result = p.run_turn([0.0] * 16000)

        assert result.hotword_hallucination_detected is True
        assert result.hotword_hallucination_reason == "repetition_and_recitation"
        assert result.metrics.hotword_hallucination_detected is True
        assert result.metrics.hotword_hallucination_reason == "repetition_and_recitation"
        mock_llm.assert_not_called()

    def test_run_turn_hotword_echo_reprompts_without_llm_or_history(self) -> None:
        from core.model_manager import ModelType
        from core.pipeline import ConversationPipeline, PipelineConfig, PipelineState

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())
        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5
        fake_audio_out = [0.1] * 12

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", return_value="뭉이야 뭉이 뭉이"),
            patch.object(p, "_init_session_dir"),
            patch.object(p, "_active_hotwords_csv", return_value="뭉이야,뭉이"),
            patch.object(p, "_run_llm") as mock_llm,
            patch.object(p, "_run_tts", return_value=(fake_audio_out, 22050)) as mock_tts,
            patch.object(p, "_play_audio_out") as mock_play_audio,
        ):
            result = p.run_turn([0.0] * 16000)

        assert result.success is True
        assert result.state == PipelineState.IDLE
        assert result.hotword_hallucination_detected is True
        assert result.hotword_hallucination_reason == "repetition"
        assert result.metrics.hotword_hallucination_detected is True
        assert result.metrics.hotword_hallucination_reason == "repetition"
        assert result.metrics.llm_time_s == 0.0
        assert result.metrics.llm_load_time_s == 0.0
        assert result.metrics.llm_ttft_s == 0.0
        assert result.metrics.llm_tokens == 0
        assert result.metrics.content_filter_blocked is False
        assert result.response_text == "응? 다시 말해줘!"
        assert result.audio_samples == fake_audio_out
        assert result.sample_rate == 22050
        assert p.conversation_history == []
        mock_llm.assert_not_called()
        mock_tts.assert_called_once_with("응? 다시 말해줘!", language="ko")
        mock_play_audio.assert_called_once_with(
            fake_audio_out,
            22050,
            expression=CharacterExpression.SPEAKING,
        )
        assert [call.args[0] for call in mm.load.call_args_list] == [
            ModelType.STT,
            ModelType.TTS,
        ]


class TestNonTargetScriptGuard:
    """Verify the Qwen3-ASR non-target-script drift guard."""

    @pytest.mark.parametrize(
        ("user_text", "expected"),
        [
            ("嗨，我的名字是钟景。", True),
            ("嗨 Jongkyung", True),
            ("安녕", True),
            ("こんにちは", True),
            ("안녕 뭉이야", False),
            ("Hello Mungi", False),
            ("안녕 Mungi", False),
            ("뭉이야!", False),
        ],
    )
    def test_contains_non_target_script_cases(
        self,
        user_text: str,
        expected: bool,
    ) -> None:
        from core.pipeline import _contains_non_target_script

        assert _contains_non_target_script(user_text) is expected

    @pytest.mark.parametrize(
        ("session_language", "expected_response"),
        [
            ("ko", "응? 한국어로 다시 말해줄래?"),
            ("en", "Hmm? Say that again in Korean or English!"),
        ],
    )
    def test_run_text_turn_script_drift_reprompts_without_llm_or_history(
        self,
        session_language: str,
        expected_response: str,
    ) -> None:
        from core.pipeline import ConversationPipeline, PipelineConfig, PipelineState

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())
        p.set_session_language(session_language)
        fake_audio_out = [0.1] * 12
        expected_tts_language = session_language

        with (
            patch.object(p, "_init_session_dir"),
            patch.object(p, "_active_hotwords_csv", return_value="뭉이야,뭉이"),
            patch.object(p, "_run_llm") as mock_llm,
            patch.object(p, "_run_tts", return_value=(fake_audio_out, 22050)) as mock_tts,
            patch.object(p, "_play_audio_out") as mock_play_audio,
        ):
            result = p.run_text_turn("嗨，我的名字是钟景。")

        assert result.success is True
        assert result.state == PipelineState.IDLE
        assert result.stt_script_drift_detected is True
        assert result.hotword_hallucination_detected is False
        assert result.metrics.stt_script_drift_detected is True
        assert result.metrics.hotword_hallucination_detected is False
        assert result.metrics.llm_time_s == 0.0
        assert result.metrics.llm_load_time_s == 0.0
        assert result.metrics.llm_ttft_s == 0.0
        assert result.metrics.llm_tokens == 0
        assert result.response_text == expected_response
        assert result.audio_samples == fake_audio_out
        assert result.sample_rate == 22050
        assert p.conversation_history == []
        mock_llm.assert_not_called()
        mock_tts.assert_called_once_with(expected_response, language=expected_tts_language)
        mock_play_audio.assert_called_once_with(
            fake_audio_out,
            22050,
            expression=CharacterExpression.SPEAKING,
        )

    def test_run_turn_cjk_stt_uses_script_drift_reprompt(self) -> None:
        from core.pipeline import (
            _EMPTY_STT_REPROMPT_TEXT,
            ConversationPipeline,
            PipelineConfig,
        )

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())
        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5
        fake_audio_out = [0.1] * 12
        expected_response = "응? 한국어로 다시 말해줄래?"

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", return_value="嗨，我的名字是钟景。"),
            patch.object(p, "_init_session_dir"),
            patch.object(p, "_run_llm") as mock_llm,
            patch.object(p, "_run_tts", return_value=(fake_audio_out, 22050)) as mock_tts,
            patch.object(p, "_play_audio_out") as mock_play_audio,
        ):
            result = p.run_turn([0.0] * 16000)

        assert result.stt_script_drift_detected is True
        assert result.metrics.stt_script_drift_detected is True
        assert result.response_text == expected_response
        assert result.response_text != _EMPTY_STT_REPROMPT_TEXT
        assert p.conversation_history == []
        mock_llm.assert_not_called()
        mock_tts.assert_called_once_with(expected_response, language="ko")
        mock_play_audio.assert_called_once_with(
            fake_audio_out,
            22050,
            expression=CharacterExpression.SPEAKING,
        )

    def test_run_text_turn_hotword_guard_precedes_script_guard(self) -> None:
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())
        fake_audio_out = [0.1] * 12

        with (
            patch.object(p, "_init_session_dir"),
            patch.object(p, "_active_hotwords_csv", return_value="嗯。"),
            patch.object(p, "_run_llm") as mock_llm,
            patch.object(p, "_run_tts", return_value=(fake_audio_out, 22050)) as mock_tts,
            patch.object(p, "_play_audio_out"),
        ):
            result = p.run_text_turn("嗯。 嗯。")

        assert result.hotword_hallucination_detected is True
        assert result.stt_script_drift_detected is False
        assert result.metrics.hotword_hallucination_detected is True
        assert result.metrics.stt_script_drift_detected is False
        assert result.response_text == "응? 다시 말해줘!"
        mock_llm.assert_not_called()
        mock_tts.assert_called_once_with("응? 다시 말해줘!", language="ko")


class TestTurnResult:
    """Verify TurnResult data class."""

    def test_success_property(self) -> None:
        from core.pipeline import PipelineState, TurnMetrics, TurnResult

        r = TurnResult(
            user_text="hi",
            response_text="hello",
            audio_samples=None,
            sample_rate=22050,
            metrics=TurnMetrics(),
            state=PipelineState.IDLE,
        )
        assert r.success is True
        assert r.stt_script_drift_detected is False

    def test_error_result(self) -> None:
        from core.pipeline import PipelineState, TurnMetrics, TurnResult

        r = TurnResult(
            user_text="",
            response_text="",
            audio_samples=None,
            sample_rate=0,
            metrics=TurnMetrics(),
            state=PipelineState.ERROR,
            error="test error",
        )
        assert r.success is False


def _make_pipeline_gemma_resident(*, llm_resident: bool = True) -> Any:
    """Construct a default-Gemma pipeline with parameterized resident-mode flag."""
    from core.llm_backend_config import LLMBackendConfig
    from core.pipeline import ConversationPipeline, PipelineConfig

    mm = MagicMock()
    mm.llm = None
    mm.config = MagicMock()
    mm.config.llm_resident = llm_resident
    mm.config.tts_resident = False
    mm.config.stt_resident = False
    mm._config = mm.config
    mm.guard_tts_resident_memory.return_value = False
    mm.guard_stt_resident_memory.return_value = True

    gemma = LLMBackendConfig(
        backend="gemma4_text",
        model_path=None,
        n_ctx=4096,
        max_tokens=256,
        temperature=0.4,
        n_gpu_layers=99,
    )
    with patch("core.pipeline.LLMBackendConfig.load", return_value=gemma):
        return ConversationPipeline(mm, PipelineConfig())


class TestConversationPipeline:
    """Verify pipeline initialization and prompt building."""

    def _make_pipeline(self) -> Any:
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        return ConversationPipeline(mm, PipelineConfig())

    def _make_gemma4_pipeline(self, *, bilingual_mode: bool = True) -> Any:
        from core.llm_backend_config import LLMBackendConfig
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        backend_config = LLMBackendConfig(
            backend="gemma4_text",
            model_path="/models/gemma.gguf",
            n_ctx=2048,
            max_tokens=64,
            temperature=0.4,
            n_gpu_layers=99,
        )
        with patch("core.pipeline.LLMBackendConfig.load", return_value=backend_config):
            return ConversationPipeline(mm, PipelineConfig(bilingual_mode=bilingual_mode))

    def test_initial_state_idle(self) -> None:
        from core.pipeline import PipelineState

        p = self._make_pipeline()
        assert p.state == PipelineState.IDLE

    def test_empty_history(self) -> None:
        p = self._make_pipeline()
        assert p.conversation_history == []

    def test_conversation_dir_defaults_to_var_lib_mungi(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Conversation logs keep the legacy mutable-root default when env is unset."""
        monkeypatch.delenv("MUNGI_MUTABLE_ROOT", raising=False)

        p = self._make_pipeline()

        assert p._conversation_dir == Path("/var/lib/mungi/conversations")

    def test_conversation_dir_honors_mutable_root_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Conversation logs are rooted under MUNGI_MUTABLE_ROOT when set."""
        monkeypatch.setenv("MUNGI_MUTABLE_ROOT", str(tmp_path))

        p = self._make_pipeline()

        assert p._conversation_dir == tmp_path / "conversations"

    def test_clear_history(self) -> None:
        p = self._make_pipeline()
        p._history.append({"role": "user", "text": "test"})
        p.clear_history()
        assert p.conversation_history == []

    def test_build_messages_contains_system(self) -> None:
        p = self._make_pipeline()
        messages = p._build_messages("안녕하세요")
        assert messages[0]["role"] == "system"
        assert "뭉이" in messages[0]["content"]

    def test_build_messages_contains_user_text(self) -> None:
        p = self._make_pipeline()
        messages = p._build_messages("오늘 뭐 하고 놀까?")
        assert messages[-1] == {"role": "user", "content": "오늘 뭐 하고 놀까?"}

    def test_build_prompt_legacy_ends_with_assistant(self) -> None:
        p = self._make_pipeline()
        prompt = p._build_prompt_legacy("안녕")
        assert prompt.endswith("<|im_start|>assistant\n")

    def test_build_messages_include_history(self) -> None:
        p = self._make_pipeline()
        p._history.append({"role": "user", "text": "이전 질문"})
        p._history.append({"role": "assistant", "text": "이전 답변"})
        messages = p._build_messages("새 질문")
        assert {"role": "user", "content": "이전 질문"} in messages
        assert {"role": "assistant", "content": "이전 답변"} in messages
        assert messages[-1] == {"role": "user", "content": "새 질문"}

    def test_build_messages_limit_history(self) -> None:
        from core.pipeline import PipelineConfig

        mm = MagicMock()
        cfg = PipelineConfig(max_history_turns=1)

        from core.pipeline import ConversationPipeline

        p = ConversationPipeline(mm, cfg)
        # Add 3 turn-pairs (6 entries)
        for i in range(3):
            p._history.append({"role": "user", "text": f"q{i}"})
            p._history.append({"role": "assistant", "text": f"a{i}"})
        messages = p._build_messages("new")
        contents = [message["content"] for message in messages]
        # max_history_turns=1 → only last pair (q2, a2)
        assert "q0" not in contents
        assert "q1" not in contents
        assert "q2" in contents
        assert "a2" in contents

    def test_english_prompt_file_mentions_maximum_80_characters(self) -> None:
        prompt_lines = (
            Path("assets/prompts/child_safe_system_en.txt")
            .read_text(
                encoding="utf-8",
            )
            .splitlines()
        )
        assert "Maximum 80 characters." in prompt_lines[2]

    def test_select_system_prompt_routes_korean_to_inline_prompt(self) -> None:
        legacy = LLMBackendConfig(
            backend="qwen3_legacy",
            model_path=None,
            n_ctx=2048,
            max_tokens=256,
            temperature=0.4,
            n_gpu_layers=99,
        )
        with patch("core.pipeline.LLMBackendConfig.load", return_value=legacy):
            p = self._make_pipeline()

        assert (
            p._select_system_prompt("안녕", detected_language="ko") == p._config.llm_system_prompt
        )

    def test_select_system_prompt_routes_english_to_file_prompt(self) -> None:
        p = self._make_pipeline()
        expected_prompt = (
            Path("assets/prompts/child_safe_system_en.txt")
            .read_text(
                encoding="utf-8",
            )
            .strip()
        )

        assert p._select_system_prompt("hello", detected_language="en") == expected_prompt

    def test_select_system_prompt_routes_gemma4_english_to_english_prompt(self) -> None:
        p = self._make_gemma4_pipeline()
        p._gemma4_persona_prompt = "GEMMA4_KO_PROMPT"
        p._en_system_prompt = "EN_PROMPT"
        p.set_session_language("en")

        assert p._select_system_prompt("Hello, can you speak English?") == "EN_PROMPT"

    def test_select_system_prompt_routes_gemma4_korean_to_persona_prompt(self) -> None:
        p = self._make_gemma4_pipeline()
        p._gemma4_persona_prompt = "GEMMA4_KO_PROMPT"
        p._en_system_prompt = "EN_PROMPT"

        assert p._select_system_prompt("\uc548\ub155 \ubb49\uc774") == "GEMMA4_KO_PROMPT"

    def test_select_system_prompt_routes_gemma4_english_to_persona_when_not_bilingual(
        self,
    ) -> None:
        p = self._make_gemma4_pipeline(bilingual_mode=False)
        p._gemma4_persona_prompt = "GEMMA4_KO_PROMPT"
        p._en_system_prompt = "EN_PROMPT"

        assert p._select_system_prompt("Hello, can you speak English?") == "GEMMA4_KO_PROMPT"

    def test_select_system_prompt_contains_capability_boundary(self) -> None:
        p = self._make_pipeline()

        prompt = p._select_system_prompt("아기는 어떻게 생기는거야", detected_language="ko")

        assert "§CAPABILITY" in prompt
        assert "그림은 못 보여주지만 말로 쉽게 설명해 줄게!" in prompt

    def test_backtrim_to_sentence_boundary_noop_with_short_text(self) -> None:
        from core.pipeline import ConversationPipeline

        text = "짧고 답변이야."
        assert ConversationPipeline._backtrim_to_sentence_boundary(text, char_limit=80) == text

    def test_backtrim_to_sentence_boundary_keeps_first_complete_sentence(self) -> None:
        from core.pipeline import ConversationPipeline

        first_sentence = ("가" * 71) + "."
        text = first_sentence + ("나" * 37) + "." + ("다" * 10)

        assert len(text) == 120
        assert (
            ConversationPipeline._backtrim_to_sentence_boundary(text, char_limit=80)
            == first_sentence
        )

    def test_backtrim_to_sentence_boundary_returns_original_without_terminator(self) -> None:
        from core.pipeline import ConversationPipeline

        text = "라" * 85
        assert ConversationPipeline._backtrim_to_sentence_boundary(text, char_limit=80) == text

    def test_backtrim_to_sentence_boundary_handles_mixed_korean_english(self) -> None:
        from core.pipeline import ConversationPipeline

        text = "Hello." + ("가" * 5) + "Why？" + ("나" * 70)
        expected = "Hello." + ("가" * 5) + "Why？"

        assert ConversationPipeline._backtrim_to_sentence_boundary(text, char_limit=20) == expected

    def test_generate_response_candidate_uses_echo_fallback(self) -> None:
        from models.llm_runner import ECHO_FALLBACK

        p = self._make_pipeline()
        messages = [{"role": "user", "content": "같이 놀자"}]

        with patch.object(p, "_run_llm", return_value=("같이 놀자", 2, 0.1)):
            (
                response_text,
                token_count,
                ttft,
                cache_hit_tokens,
                cache_miss_tokens,
                output_filtered,
            ) = p._generate_response_candidate(messages, "같이 놀자")

        assert response_text == ECHO_FALLBACK
        assert token_count == 2
        assert ttft == 0.1
        assert cache_hit_tokens is None
        assert cache_miss_tokens is None
        assert output_filtered is False

    def test_generate_response_candidate_backtrims_token_limited_output(self) -> None:
        from core.pipeline import LLM_BACKTRIM_CHAR_LIMIT

        p = self._make_pipeline()
        messages = [{"role": "user", "content": "질문"}]
        first_sentence = ("가" * (LLM_BACKTRIM_CHAR_LIMIT - 10)) + "."
        raw_response = first_sentence + ("나" * 15)

        with patch.object(
            p,
            "_run_llm",
            return_value=(raw_response, p._config.llm_max_tokens, 0.1),
        ):
            (
                response_text,
                token_count,
                ttft,
                cache_hit_tokens,
                cache_miss_tokens,
                output_filtered,
            ) = p._generate_response_candidate(messages, "질문")

        assert response_text == first_sentence
        assert token_count == p._config.llm_max_tokens
        assert ttft == 0.1
        assert cache_hit_tokens is None
        assert cache_miss_tokens is None
        assert output_filtered is False

    def test_run_turn_no_speech_returns_empty(self) -> None:
        from core.pipeline import PipelineState

        p = self._make_pipeline()
        with patch("core.pipeline.ConversationPipeline._run_vad", return_value=[]):
            result = p.run_turn([0.0] * 16000)
        assert result.user_text == ""
        assert result.state == PipelineState.IDLE
        assert result.metrics.speech_segments == 0

    def test_run_turn_from_speech_audio_empty_stt_returns_fixed_reprompt(self) -> None:
        from core.model_manager import ModelType
        from core.pipeline import (
            _EMPTY_STT_REPROMPT_TEXT,
            ConversationPipeline,
            PipelineConfig,
            PipelineState,
        )

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())
        input_audio = [0.2] * 16000
        fake_audio_out = [0.1] * 12
        saved_paths = [
            SimpleNamespace(name="input_001.wav"),
            SimpleNamespace(name="output_001.wav"),
        ]

        with (
            patch.object(p, "_run_pre_turn_memory_guards"),
            patch.object(p, "_prepare_input_audio", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", return_value="") as mock_stt,
            patch.object(p, "_init_session_dir"),
            patch.object(p, "_save_conversation_audio", side_effect=saved_paths),
            patch.object(p, "_run_llm") as mock_llm,
            patch.object(p, "_run_tts", return_value=(fake_audio_out, 22050)) as mock_tts,
            patch.object(p, "_play_audio_out") as mock_play_audio,
            patch.object(p, "_log_conversation_turn") as mock_log,
        ):
            result = p._run_turn_from_speech_audio(
                input_audio,
                16000,
                bypass_vad=True,
            )

        assert result.success is True
        assert result.state == PipelineState.IDLE
        assert result.user_text == ""
        assert result.response_text == _EMPTY_STT_REPROMPT_TEXT
        assert result.audio_samples == fake_audio_out
        assert result.sample_rate == 22050
        assert result.raw_stt_text == ""
        assert result.metrics.speech_segments == 1
        assert result.metrics.llm_time_s == 0.0
        assert result.metrics.llm_tokens == 0
        assert p.conversation_history == []
        mock_stt.assert_called_once()
        mock_llm.assert_not_called()
        mock_tts.assert_called_once_with(_EMPTY_STT_REPROMPT_TEXT, language="ko")
        mock_play_audio.assert_called_once_with(
            fake_audio_out,
            22050,
            expression=CharacterExpression.EXCITED,
        )
        log_args = mock_log.call_args.args
        assert log_args[:5] == (
            1,
            "",
            _EMPTY_STT_REPROMPT_TEXT,
            "input_001.wav",
            "output_001.wav",
        )
        assert [call.args[0] for call in mm.load.call_args_list] == [
            ModelType.STT,
            ModelType.TTS,
        ]

    def test_run_turn_empty_stt_returns_fixed_reprompt(self) -> None:
        from core.model_manager import ModelType
        from core.pipeline import (
            _EMPTY_STT_REPROMPT_TEXT,
            ConversationPipeline,
            PipelineConfig,
            PipelineState,
        )

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())
        input_audio = [0.0] * 16000
        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5
        fake_audio_out = [0.1] * 12
        saved_paths = [
            SimpleNamespace(name="input_001.wav"),
            SimpleNamespace(name="output_001.wav"),
        ]

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", return_value="") as mock_stt,
            patch.object(p, "_init_session_dir"),
            patch.object(p, "_save_conversation_audio", side_effect=saved_paths),
            patch.object(p, "_run_llm") as mock_llm,
            patch.object(p, "_run_tts", return_value=(fake_audio_out, 22050)) as mock_tts,
            patch.object(p, "_play_audio_out") as mock_play_audio,
            patch.object(p, "_log_conversation_turn") as mock_log,
        ):
            result = p.run_turn(input_audio)

        assert result.success is True
        assert result.state == PipelineState.IDLE
        assert result.user_text == ""
        assert result.response_text == _EMPTY_STT_REPROMPT_TEXT
        assert result.audio_samples == fake_audio_out
        assert result.sample_rate == 22050
        assert result.raw_stt_text == ""
        assert result.metrics.speech_segments == 1
        assert result.metrics.llm_time_s == 0.0
        assert result.metrics.llm_tokens == 0
        assert p.conversation_history == []
        mock_stt.assert_called_once()
        mock_llm.assert_not_called()
        mock_tts.assert_called_once_with(_EMPTY_STT_REPROMPT_TEXT, language="ko")
        mock_play_audio.assert_called_once_with(
            fake_audio_out,
            22050,
            expression=CharacterExpression.EXCITED,
        )
        log_args = mock_log.call_args.args
        assert log_args[:5] == (
            1,
            "",
            _EMPTY_STT_REPROMPT_TEXT,
            "input_001.wav",
            "output_001.wav",
        )
        assert [call.args[0] for call in mm.load.call_args_list] == [
            ModelType.STT,
            ModelType.TTS,
        ]

    def test_run_turn_normalizes_stereo_48k_input_before_vad(self) -> None:
        from core.pipeline import PipelineState

        p = self._make_pipeline()
        captured: dict[str, list[float]] = {}
        stereo = np.tile(np.array([[0.6, 0.0]], dtype=np.float32), (480, 1))

        def fake_run_vad(samples: list[float]) -> list[Any]:
            captured["samples"] = samples
            return []

        with patch.object(p, "_run_vad", side_effect=fake_run_vad):
            result = p.run_turn(stereo, sample_rate=48000)

        assert result.state == PipelineState.IDLE
        assert len(captured["samples"]) == 160
        assert all(abs(sample - 0.3) < 1e-6 for sample in captured["samples"][:8])

    def test_run_turn_error_returns_error_state(self) -> None:
        from core.pipeline import PipelineState

        p = self._make_pipeline()
        with patch(
            "core.pipeline.ConversationPipeline._run_vad",
            side_effect=RuntimeError("vad fail"),
        ):
            result = p.run_turn([0.0] * 16000)
        assert result.state == PipelineState.ERROR
        assert result.error == "vad fail"
        assert result.success is False

    def test_extract_speech_with_padding(self) -> None:
        from core.pipeline import PipelineConfig

        mm = MagicMock()
        cfg = PipelineConfig(vad_pad_ms=100)

        from core.pipeline import ConversationPipeline

        p = ConversationPipeline(mm, cfg)

        # 1 second of audio at 16kHz
        audio = [0.1] * 16000
        seg = MagicMock()
        seg.start = 0.3
        seg.end = 0.5

        result = p._extract_speech(audio, [seg])
        # 200ms segment + 100ms padding each side = 400ms = 6400 samples
        assert len(result) == 6400

    def test_write_temp_wav(self) -> None:
        import os
        import tempfile
        from pathlib import Path

        from core.pipeline import ConversationPipeline

        samples = [0.0, 0.5, -0.5, 1.0, -1.0]
        fd, tmp_name = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        path = Path(tmp_name)
        try:
            ConversationPipeline._write_temp_wav(path, samples)
            assert path.exists()
            assert path.stat().st_size > 0
        finally:
            path.unlink(missing_ok=True)

    def test_run_turn_full_success(self) -> None:
        """E2E success path: VAD→STT→LLM→TTS all succeed."""
        from core.pipeline import (
            _EMPTY_STT_REPROMPT_TEXT,
            ConversationPipeline,
            PipelineConfig,
            PipelineState,
        )

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())

        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5

        fake_audio_out = [0.1] * 100
        fake_sr = 22050

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", return_value="안녕하세요"),
            patch.object(p, "_run_llm", return_value=("안녕! 나는 뭉이야.", 10, 0.5)),
            patch.object(p, "_run_tts", return_value=(fake_audio_out, fake_sr)) as mock_tts,
        ):
            result = p.run_turn([0.0] * 16000)

        assert result.success is True
        assert result.user_text == "안녕하세요"
        assert result.response_text == "안녕! 나는 뭉이야."
        assert result.audio_samples == fake_audio_out
        assert result.sample_rate == fake_sr
        assert result.state == PipelineState.IDLE
        assert result.metrics.llm_tokens == 10
        assert result.error is None
        assert result.response_text != _EMPTY_STT_REPROMPT_TEXT
        mock_tts.assert_called_once_with("안녕! 나는 뭉이야.", language="ko")

    def test_run_turn_success_appends_history(self) -> None:
        """After a successful turn, history should contain user+assistant."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())

        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", return_value="질문입니다"),
            patch.object(p, "_run_llm", return_value=("답변이에요!", 5, 0.3)),
            patch.object(p, "_run_tts", return_value=([0.0], 22050)),
        ):
            p.run_turn([0.0] * 16000)

        history = p.conversation_history
        assert len(history) == 2
        assert history[0] == {"role": "user", "text": "질문입니다"}
        assert history[1] == {"role": "assistant", "text": "답변이에요!"}

    def test_run_turn_plays_audio_when_enabled(self) -> None:
        """Local playback config should trigger speaker output after TTS."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        p = ConversationPipeline(
            mm,
            PipelineConfig(play_tts_audio=True, tts_output_device="USB PnP"),
        )

        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", return_value="질문"),
            patch.object(p, "_run_llm", return_value=("답변", 5, 0.3)),
            patch.object(p, "_run_tts", return_value=([0.1] * 100, 22050)),
            patch.object(p, "_play_audio_out") as mock_play_audio,
        ):
            p.run_turn([0.0] * 16000)

        mock_play_audio.assert_called_once_with(
            [0.1] * 100,
            22050,
            expression=CharacterExpression.SPEAKING,
        )

    def test_run_turn_still_invokes_play_hook_when_disabled(self) -> None:
        """The playback hook receives audio even in headless mode."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig(play_tts_audio=False))

        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", return_value="질문"),
            patch.object(p, "_run_llm", return_value=("답변", 5, 0.3)),
            patch.object(p, "_run_tts", return_value=([0.1] * 100, 22050)),
            patch.object(p, "_play_audio_out") as mock_play_audio,
        ):
            p.run_turn([0.0] * 16000)

        mock_play_audio.assert_called_once_with(
            [0.1] * 100,
            22050,
            expression=CharacterExpression.SPEAKING,
        )

    def test_run_turn_loads_tts_after_llm(self) -> None:
        """run_turn loads STT, then LLM, then TTS in order."""
        from core.model_manager import ModelType
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        legacy = LLMBackendConfig(
            backend="qwen3_legacy",
            model_path=None,
            n_ctx=2048,
            max_tokens=256,
            temperature=0.4,
            n_gpu_layers=99,
        )
        with patch("core.pipeline.LLMBackendConfig.load", return_value=legacy):
            p = ConversationPipeline(mm, PipelineConfig())

        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", return_value="question"),
            patch.object(p, "_run_llm", return_value=("answer", 5, 0.3)),
            patch.object(p, "_run_tts", return_value=([0.1] * 100, 22050)),
            patch.object(p, "_play_audio_out"),
        ):
            p.run_turn([0.0] * 16000)

        assert [call.args[0] for call in mm.load.call_args_list] == [
            ModelType.STT,
            ModelType.LLM,
            ModelType.TTS,
        ]

    def test_gemma_run_turn_dispatches_llm_via_fallback_loader_then_loads_tts(self) -> None:
        """Gemma default: run_turn calls STT, Gemma fallback loader, then TTS."""
        from core.model_manager import ModelType

        p = _make_pipeline_gemma_resident(llm_resident=False)
        sentinel_llm = MagicMock(name="sentinel_llm")
        call_order: list[tuple[str, Any]] = []
        load_result = SimpleNamespace(
            model=sentinel_llm,
            model_path_actual="/models/gemma.gguf",
            fallback_used=False,
            fallback_reason=None,
        )

        def fake_load_gemma_with_fallback(*_args: Any, **_kwargs: Any) -> Any:
            p._mm.llm = sentinel_llm
            call_order.append(("load_gemma_with_fallback", "llm"))
            return load_result

        def fake_load(model_type: Any) -> None:
            call_order.append(("load", model_type))

        p._mm.load_gemma_with_fallback.side_effect = fake_load_gemma_with_fallback
        p._mm.load.side_effect = fake_load

        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", return_value="question"),
            patch.object(p, "_run_llm", return_value=("answer", 5, 0.3)),
            patch.object(p, "_run_tts", return_value=([0.1] * 100, 22050)),
            patch.object(p, "_play_audio_out"),
        ):
            result = p.run_turn([0.0] * 16000)

        assert result.success is True
        assert p._mm.llm is sentinel_llm
        assert call_order == [
            ("load", ModelType.STT),
            ("load_gemma_with_fallback", "llm"),
            ("load", ModelType.TTS),
        ]

    def test_run_turn_unloads_transient_models_after_use(self) -> None:
        """run_turn unloads STT, LLM, and TTS after each stage."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())

        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", return_value="question"),
            patch.object(p, "_run_llm", return_value=("answer", 5, 0.3)),
            patch.object(p, "_run_tts", return_value=([0.1] * 100, 22050)),
            patch.object(p, "_play_audio_out"),
        ):
            p.run_turn([0.0] * 16000)

        mm.unload_stt.assert_called_once_with(force=False)
        mm.unload_llm.assert_called_once()
        mm.unload_tts.assert_called_once()

    def test_run_turn_preserves_tts_when_resident(self) -> None:
        """Successful turns should keep resident TTS loaded."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        mm.config.tts_resident = True
        mm.guard_tts_resident_memory.return_value = False
        p = ConversationPipeline(mm, PipelineConfig())

        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", return_value="question"),
            patch.object(p, "_run_llm", return_value=("answer", 5, 0.3)),
            patch.object(p, "_run_tts", return_value=([0.1] * 100, 22050)),
            patch.object(p, "_play_audio_out"),
        ):
            p.run_turn([0.0] * 16000)

        mm.guard_tts_resident_memory.assert_called_once()
        mm.unload_tts.assert_not_called()

    def test_run_turn_unloads_tts_on_error_even_when_resident(self) -> None:
        """Resident TTS should still be force-unloaded on synthesis errors."""
        from core.pipeline import ConversationPipeline, PipelineConfig, PipelineState

        mm = MagicMock()
        mm.config.tts_resident = True
        p = ConversationPipeline(mm, PipelineConfig())

        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", return_value="question"),
            patch.object(p, "_run_llm", return_value=("answer", 5, 0.3)),
            patch.object(p, "_run_tts", side_effect=RuntimeError("tts fail")),
        ):
            result = p.run_turn([0.0] * 16000)

        assert result.state == PipelineState.ERROR
        assert result.error == "tts fail"
        mm.unload_tts.assert_any_call(force=True)
        mm.unload_all.assert_not_called()

    def test_run_turn_default_unloads_tts_after_synthesis(self) -> None:
        """Default config should keep unloading TTS after successful synthesis."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        mm.config.tts_resident = False
        p = ConversationPipeline(mm, PipelineConfig())

        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", return_value="question"),
            patch.object(p, "_run_llm", return_value=("answer", 5, 0.3)),
            patch.object(p, "_run_tts", return_value=([0.1] * 100, 22050)),
            patch.object(p, "_play_audio_out"),
        ):
            p.run_turn([0.0] * 16000)

        mm.unload_tts.assert_called_once_with()

    def test_run_turn_drops_page_cache_when_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Successful turns should reclaim page cache only when the toggle is enabled."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        monkeypatch.delenv("MUNGI_DROP_CACHES_PER_TURN", raising=False)

        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5

        def run_success_turn(*, drop_caches_per_turn: bool) -> MagicMock:
            mm = MagicMock()
            p = ConversationPipeline(
                mm,
                PipelineConfig(drop_caches_per_turn=drop_caches_per_turn),
            )

            with (
                patch.object(p, "_run_vad", return_value=[fake_seg]),
                patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
                patch.object(p, "_run_stt", return_value="question"),
                patch.object(p, "_run_llm", return_value=("answer", 5, 0.3)),
                patch.object(p, "_run_tts", return_value=([0.1] * 100, 22050)),
                patch.object(p, "_play_audio_out"),
            ):
                result = p.run_turn([0.0] * 16000)

            assert result.success is True
            return mm

        enabled_mm = run_success_turn(drop_caches_per_turn=True)
        enabled_mm._drop_page_cache.assert_called_once_with()

        disabled_mm = run_success_turn(drop_caches_per_turn=False)
        disabled_mm._drop_page_cache.assert_not_called()

        monkeypatch.setenv("MUNGI_DROP_CACHES_PER_TURN", "0")
        assert PipelineConfig().drop_caches_per_turn is False

    def test_run_turn_forces_stt_unload_after_stt_error(self) -> None:
        """STT failures must force cleanup even when resident mode is enabled."""
        from core.pipeline import ConversationPipeline, PipelineConfig, PipelineState

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())

        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", side_effect=RuntimeError("stt fail")),
        ):
            result = p.run_turn([0.0] * 16000)

        assert result.state == PipelineState.ERROR
        assert result.error == "stt fail"
        mm.unload_stt.assert_any_call(force=True)
        mm.unload_llm.assert_called_once_with()
        mm.unload_tts.assert_called_once_with(force=True)
        mm.unload_all.assert_not_called()

    def test_run_turn_from_speech_audio_context_overflow_unloads_transients_only(self) -> None:
        """A turn-stage context overflow must preserve resident VAD."""
        from core.pipeline import ConversationPipeline, PipelineConfig, PipelineState

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())
        overflow_message = "Requested tokens (4124) exceed context window of 4096"

        with (
            patch.object(p, "_run_pre_turn_memory_guards"),
            patch.object(p, "_prepare_input_audio", return_value=[0.1] * 8000),
            patch.object(p, "_transcribe_speech_audio", return_value=("question", "question")),
            patch.object(p, "_init_session_dir"),
            patch.object(
                p, "_save_conversation_audio", return_value=SimpleNamespace(name="input.wav")
            ),
            patch.object(p, "_respond_to_text", side_effect=ValueError(overflow_message)),
        ):
            result = p._run_turn_from_speech_audio(
                [0.0] * 16000,
                16_000,
                bypass_vad=True,
            )

        assert result.state == PipelineState.ERROR
        assert result.error is not None
        assert overflow_message in result.error
        mm.unload_stt.assert_called_once_with(force=True)
        mm.unload_llm.assert_called_once_with()
        mm.unload_tts.assert_called_once_with(force=True)
        mm.unload_all.assert_not_called()

    def test_run_turn_unloads_tts_before_playback(self) -> None:
        """TTS should be released before blocking speaker playback begins."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())

        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5
        call_order: list[str] = []

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", return_value="question"),
            patch.object(p, "_run_llm", return_value=("answer", 5, 0.3)),
            patch.object(p, "_run_tts", return_value=([0.1] * 100, 22050)),
            patch.object(mm, "unload_tts", side_effect=lambda: call_order.append("unload_tts")),
            patch.object(
                p,
                "_play_audio_out",
                side_effect=lambda *_args, **_kwargs: call_order.append("play"),
            ),
        ):
            p.run_turn([0.0] * 16000)

        assert call_order == ["unload_tts", "play"]

    def test_run_turn_preloads_stt_when_enabled(self) -> None:
        """Live-demo config can request STT preload after a successful turn."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig(enable_stt_preload=True))

        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", return_value="question"),
            patch.object(p, "_run_llm", return_value=("answer", 5, 0.3)),
            patch.object(p, "_run_tts", return_value=([0.1] * 100, 22050)),
            patch.object(p, "_play_audio_out"),
        ):
            p.run_turn([0.0] * 16000)

        mm.preload_stt.assert_called_once()

    def test_run_turn_skips_preload_when_memory_guard_blocks_it(self) -> None:
        """Memory guard can suppress next-turn STT preload for breathing room."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        mm.guard_stt_resident_memory.return_value = False
        p = ConversationPipeline(mm, PipelineConfig(enable_stt_preload=True))

        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", return_value="question"),
            patch.object(p, "_run_llm", return_value=("answer", 5, 0.3)),
            patch.object(p, "_run_tts", return_value=([0.1] * 100, 22050)),
            patch.object(p, "_play_audio_out"),
        ):
            p.run_turn([0.0] * 16000)

        mm.guard_stt_resident_memory.assert_called_once()
        mm.preload_stt.assert_not_called()

    def test_run_turn_preserves_resident_stt_across_turns(self) -> None:
        """Resident STT should survive successful turns without reloading."""
        from core.model_manager import ManagerConfig, ModelManager, ModelState
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = ModelManager(ManagerConfig(model_dir="/fake/models", stt_resident=True))
        legacy = LLMBackendConfig(
            backend="qwen3_legacy",
            model_path=None,
            n_ctx=2048,
            max_tokens=256,
            temperature=0.4,
            n_gpu_layers=99,
        )
        with patch("core.pipeline.LLMBackendConfig.load", return_value=legacy):
            p = ConversationPipeline(mm, PipelineConfig())

        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5
        stt_model = object()
        load_stt_calls = 0

        def fake_load_stt() -> None:
            nonlocal load_stt_calls
            load_stt_calls += 1
            mm._models["stt"] = stt_model
            mm._status["stt"].state = ModelState.READY

        def fake_load_llm() -> None:
            mm._models["llm"] = object()
            mm._status["llm"].state = ModelState.READY

        def fake_load_tts() -> None:
            mm._models["tts"] = MagicMock()
            mm._status["tts"].state = ModelState.READY

        with (
            patch.object(mm, "load_stt", side_effect=fake_load_stt),
            patch.object(mm, "_load_llm_full_gpu", side_effect=fake_load_llm),
            patch.object(mm, "load_tts", side_effect=fake_load_tts),
            patch.object(mm, "_release_model_resources"),
            patch.object(mm, "_gc_collect"),
            patch.object(mm, "_drop_page_cache"),
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", side_effect=["question one", "question two"]),
            patch.object(p, "_run_llm", return_value=("answer", 5, 0.3)),
            patch.object(p, "_run_tts", return_value=([0.1] * 100, 22050)),
            patch.object(p, "_play_audio_out"),
        ):
            p.run_turn([0.0] * 16000)
            first_stt = mm.stt
            p.run_turn([0.0] * 16000)

        assert first_stt is stt_model
        assert mm.stt is first_stt
        assert load_stt_calls == 1

    def test_gemma_run_turn_preserves_resident_llm_across_turns(self) -> None:
        """Under L1 resident default, Gemma LLM is loaded once and persists across turns."""
        p = _make_pipeline_gemma_resident(llm_resident=True)
        sentinel_llm = MagicMock(name="sentinel_llm")
        load_result = SimpleNamespace(
            model=sentinel_llm,
            model_path_actual="/models/gemma.gguf",
            fallback_used=False,
            fallback_reason=None,
        )

        def fake_load_gemma_with_fallback(*_args: Any, **_kwargs: Any) -> Any:
            p._mm.llm = sentinel_llm
            return load_result

        p._mm.load_gemma_with_fallback.side_effect = fake_load_gemma_with_fallback

        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", side_effect=["q1", "q2"]),
            patch.object(p, "_run_llm", return_value=("answer", 5, 0.3)),
            patch.object(p, "_run_tts", return_value=([0.1] * 100, 22050)),
            patch.object(p, "_play_audio_out"),
        ):
            result1 = p.run_turn([0.0] * 16000)
            first_llm = p._mm.llm
            result2 = p.run_turn([0.0] * 16000)
            second_llm = p._mm.llm

        assert result1.success is True
        assert result2.success is True
        assert first_llm is sentinel_llm
        assert second_llm is sentinel_llm
        assert first_llm is second_llm
        assert p._mm.load_gemma_with_fallback.call_count == 1
        p._mm.unload_llm.assert_not_called()

    def test_run_text_turn_full_success(self) -> None:
        """Text-input path should run LLM→TTS and append history."""
        from core.pipeline import ConversationPipeline, PipelineConfig, PipelineState

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())

        with (
            patch.object(p, "_run_llm", return_value=("안녕! 같이 놀자.", 8, 0.4)),
            patch.object(p, "_run_tts", return_value=([0.1] * 50, 22050)),
            patch.object(p, "_play_audio_out") as mock_play_audio,
        ):
            result = p.run_text_turn("뭉이야 안녕!")

        assert result.success is True
        assert result.user_text == "뭉이야 안녕!"
        assert result.response_text == "안녕! 같이 놀자."
        assert result.state == PipelineState.IDLE
        assert result.metrics.llm_tokens == 8
        assert p.conversation_history == [
            {"role": "user", "text": "뭉이야 안녕!"},
            {"role": "assistant", "text": "안녕! 같이 놀자."},
        ]
        mock_play_audio.assert_called_once_with(
            [0.1] * 50,
            22050,
            expression=CharacterExpression.GREETING,
        )

    def test_run_text_turn_strips_non_target_script_before_tts(self) -> None:
        """LLM output script drift should not be forwarded to TTS."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())

        with (
            patch("core.pipeline.check_approved_template", return_value=None),
            patch.object(p, "_run_llm", return_value=("You're not alone. 玩", 8, 0.4)),
            patch.object(p, "_run_tts", return_value=([0.1] * 50, 22050)) as mock_tts,
            patch.object(p, "_play_audio_out"),
        ):
            result = p.run_text_turn("I feel lonely sometimes.")

        assert result.success is True
        assert result.response_text == "You're not alone."
        assert p.conversation_history == [
            {"role": "user", "text": "I feel lonely sometimes."},
            {"role": "assistant", "text": "You're not alone."},
        ]
        mock_tts.assert_called_once_with("You're not alone.", language="ko")

    def test_run_text_turn_guide_template_populates_metrics_and_uses_llm(self) -> None:
        """Guide-mode template routing should record template identity and still use LLM."""
        from core.pipeline import ConversationPipeline, PipelineConfig, PipelineState

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())
        guide_match = {
            "mode": "guide",
            "response": "Use simple water-safety guidance.",
            "topic_id": "swimming",
        }

        def _slow_llm(*args: Any, **kwargs: Any) -> tuple[str, int, float]:
            del args, kwargs
            import time

            time.sleep(0.02)
            return "Ask an adult to stay close.", 9, 0.2

        with (
            patch("core.pipeline.check_approved_template", return_value=guide_match),
            patch.object(p, "_run_llm", side_effect=_slow_llm) as mock_llm,
            patch.object(p, "_run_tts", return_value=([0.1] * 10, 22050)),
            patch.object(p, "_play_audio_out") as mock_play_audio,
        ):
            result = p.run_text_turn("Can I swim alone?")

        assert result.success is True
        assert result.state == PipelineState.IDLE
        assert result.response_text == "Ask an adult to stay close."
        assert result.metrics.template_matched is True
        assert result.metrics.template_topic_id == "swimming"
        assert result.metrics.template_mode == "guide"
        assert result.metrics.llm_time_s > 0.0
        assert result.metrics.llm_tokens == 9
        mock_llm.assert_called_once()
        assert mock_play_audio.call_args.kwargs["expression"] is CharacterExpression.CONCERNED

    def test_run_text_turn_block_template_populates_topic_and_mode(self) -> None:
        """Block-mode template routing should record template identity and bypass LLM."""
        from core.model_manager import ModelType
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())
        block_match = {
            "mode": "block",
            "response": "That is not safe for kids.",
            "topic_id": "unsafe_tool",
        }
        with (
            patch("core.pipeline.check_approved_template", return_value=block_match),
            patch.object(p, "_run_llm") as mock_llm,
            patch.object(p, "_run_tts", return_value=([0.1] * 10, 22050)),
            patch.object(p, "_play_audio_out") as mock_play_audio,
        ):
            result = p.run_text_turn("Can I use this tool?")

        load_args = [call.args[0] for call in mm.load.call_args_list]
        assert ModelType.LLM not in load_args
        assert ModelType.TTS in load_args
        assert result.metrics.template_matched is True
        assert result.metrics.template_topic_id == "unsafe_tool"
        assert result.metrics.template_mode == "block"
        assert result.metrics.llm_time_s == 0.0
        assert result.metrics.llm_tokens == 0
        mock_llm.assert_not_called()
        assert mock_play_audio.call_args.kwargs["expression"] is CharacterExpression.NEUTRAL

    @pytest.mark.parametrize(
        ("session_language", "user_text", "topic_id"),
        [
            ("ko", "너 누구야?", "mungi_self_intro_child"),
            ("en", "너 누구야?", "mungi_self_intro_child"),
            ("ko", "제품 소개해 주세요", "mungi_product_intro_adult"),
            ("en", "제품 소개해 주세요", "mungi_product_intro_adult"),
        ],
    )
    def test_intro_block_templates_force_korean_tts_for_all_sessions(
        self,
        session_language: str,
        user_text: str,
        topic_id: str,
    ) -> None:
        """Korean fixed intro scripts should always use Korean cache and live TTS."""
        from core.pipeline import ConversationPipeline, PipelineConfig
        from safety.approved_template_router import check_approved_template

        expected_match = check_approved_template(user_text, language=session_language)
        assert expected_match is not None
        assert expected_match["topic_id"] == topic_id

        p = ConversationPipeline(MagicMock(), PipelineConfig())
        p.set_session_language(session_language)
        with (
            patch.object(p, "_init_session_dir"),
            patch("core.pipeline.tts_cache.lookup", return_value=None) as mock_lookup,
            patch.object(p, "_run_llm") as mock_llm,
            patch.object(p, "_run_tts", return_value=([0.1] * 10, 22050)) as mock_tts,
            patch.object(p, "_play_audio_out"),
        ):
            result = p.run_text_turn(user_text)

        assert result.success is True
        assert result.response_text == expected_match["response"]
        assert result.metrics.template_matched is True
        assert result.metrics.template_topic_id == topic_id
        assert result.metrics.template_mode == "block"
        mock_llm.assert_not_called()
        mock_lookup.assert_called_once_with(expected_match["response"], "ko")
        mock_tts.assert_called_once_with(expected_match["response"], language="ko")

    def test_run_text_turn_ko_to_en_language_switch_bypasses_llm(self) -> None:
        """A verified Korean-to-English switch flips session language without an LLM turn."""
        from core.model_manager import ModelType
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())
        with (
            patch.object(p, "_run_llm") as mock_llm,
            patch.object(p, "_run_tts", return_value=([0.1] * 10, 22050)) as mock_tts,
            patch.object(p, "_play_audio_out") as mock_play_audio,
        ):
            result = p.run_text_turn("영어로 말해줘")

        load_args = [call.args[0] for call in mm.load.call_args_list]
        assert ModelType.LLM not in load_args
        assert ModelType.TTS in load_args
        assert p.session_language == "en"
        assert p._current_language == "en"
        assert result.response_text == (
            "좋아! 이제 영어로 말할게. 한국어로 돌아오고 싶으면 "
            "한국어로 말해줘 라고 하거나, 오른쪽 위에 있는 한영 전환 단추를 눌러줘!"
        )
        assert result.detected_language == "ko"
        assert result.metrics.language_switch_matched is True
        assert result.metrics.language_switch_target == "en"
        assert result.metrics.llm_time_s == 0.0
        assert result.metrics.llm_tokens == 0
        mock_llm.assert_not_called()
        mock_tts.assert_called_once_with(result.response_text, language="ko")
        assert mock_play_audio.call_args.kwargs["expression"] is CharacterExpression.EXCITED

    def test_touch_language_switch_confirmation_uses_fixed_tts_without_history(self) -> None:
        """The touch language switch path uses template-backed fixed TTS."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        p = ConversationPipeline(MagicMock(), PipelineConfig())
        language_sink = MagicMock()
        p.set_language_sink(language_sink)
        with (
            patch.object(p, "_run_llm") as mock_llm,
            patch.object(p, "_run_tts", return_value=([0.1] * 10, 22050)) as mock_tts,
            patch.object(p, "_play_audio_out") as mock_play_audio,
        ):
            result = p.switch_session_language_with_confirmation("en")

        assert p.session_language == "en"
        assert p._current_language == "en"
        language_sink.assert_called_once_with("en")
        assert result.response_text == (
            "좋아! 이제 영어로 말할게. 한국어로 돌아오고 싶으면 "
            "한국어로 말해줘 라고 하거나, 오른쪽 위에 있는 한영 전환 단추를 눌러줘!"
        )
        assert result.detected_language == "en"
        assert result.metrics.language_switch_matched is True
        assert result.metrics.language_switch_target == "en"
        assert p.conversation_history == []
        mock_llm.assert_not_called()
        mock_tts.assert_called_once_with(result.response_text, language="ko")
        assert mock_play_audio.call_args.kwargs["expression"] is CharacterExpression.EXCITED

    def test_run_text_turn_history_mode_bypasses_llm_and_emits_sink(self) -> None:
        """A verified history-mode trigger plays confirmation and emits the mode sink."""
        from core.model_manager import ModelType
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())
        history_sink = MagicMock()
        p.set_history_mode_sink(history_sink)
        with (
            patch.object(p, "_run_llm") as mock_llm,
            patch.object(p, "_run_tts", return_value=([0.1] * 10, 22050)) as mock_tts,
            patch.object(p, "_play_audio_out") as mock_play_audio,
        ):
            result = p.run_text_turn("재미있는 우리역사")

        load_args = [call.args[0] for call in mm.load.call_args_list]
        assert ModelType.LLM not in load_args
        assert ModelType.TTS in load_args
        assert result.response_text == "좋아! 재미있는 우리역사를 시작할게!"
        assert result.metrics.history_mode_matched is True
        assert result.metrics.language_switch_matched is False
        mock_llm.assert_not_called()
        mock_tts.assert_called_once_with(result.response_text, language="ko")
        assert mock_play_audio.call_args.kwargs["expression"] is CharacterExpression.EXCITED
        history_sink.assert_called_once_with()
        # History entry MUST NOT preload STT: an in-flight preload contends with
        # HistoryModeController.enter()'s STT unload on the shared load lock and
        # hangs the main event loop (mirrors the Funny English entry path).
        mm.preload_stt.assert_not_called()

    def test_history_mode_match_precedes_language_switch_match(self) -> None:
        """History-mode matches must not fall through to language switching."""
        from core.pipeline import ConversationPipeline, PipelineConfig
        from safety.history_mode_router import HistoryModeMatch

        p = ConversationPipeline(MagicMock(), PipelineConfig())
        history_match = HistoryModeMatch(
            confirmation_language="ko",
            confirmation_text="좋아! 재미있는 우리역사를 시작할게!",
            matched_patterns=("pattern",),
        )
        with (
            patch("core.pipeline.match_history_mode", return_value=history_match),
            patch("core.pipeline.match_language_switch") as mock_switch,
            patch.object(p, "_run_llm") as mock_llm,
            patch.object(p, "_run_tts", return_value=([0.1] * 10, 22050)),
            patch.object(p, "_play_audio_out"),
        ):
            result = p.run_text_turn("영어로 말해줘")

        assert result.metrics.history_mode_matched is True
        mock_switch.assert_not_called()
        mock_llm.assert_not_called()

    def test_run_text_turn_en_to_ko_language_switch_bypasses_llm(self) -> None:
        """A verified English-to-Korean switch flips session language without an LLM turn."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        p = ConversationPipeline(MagicMock(), PipelineConfig())
        p.set_session_language("en")
        with (
            patch.object(p, "_run_llm") as mock_llm,
            patch.object(p, "_run_tts", return_value=([0.1] * 10, 22050)) as mock_tts,
            patch.object(p, "_play_audio_out") as mock_play_audio,
        ):
            result = p.run_text_turn("say it in korean")

        assert p.session_language == "ko"
        assert p._current_language == "ko"
        assert result.response_text == (
            "좋아! 이제 한국어로 얘기하자! 영어로 하고 싶으면 "
            "영어로 말해줘 라고 하거나, 오른쪽 위에 있는 한영 전환 단추를 눌러줘!"
        )
        assert result.detected_language == "en"
        assert result.metrics.language_switch_matched is True
        assert result.metrics.language_switch_target == "ko"
        mock_llm.assert_not_called()
        mock_tts.assert_called_once_with(result.response_text, language="ko")
        assert mock_play_audio.call_args.kwargs["expression"] is CharacterExpression.EXCITED

    def test_language_switch_matcher_is_not_called_for_guide_templates(self) -> None:
        """Guide templates continue to LLM and suppress switch matching."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        p = ConversationPipeline(MagicMock(), PipelineConfig())
        guide_match = {
            "mode": "guide",
            "response": "Use simple water-safety guidance.",
            "topic_id": "swimming",
        }
        with (
            patch("core.pipeline.check_approved_template", return_value=guide_match),
            patch("core.pipeline.match_language_switch") as mock_switch,
            patch.object(p, "_run_llm", return_value=("어른과 같이 하자.", 6, 0.1)),
            patch.object(p, "_run_tts", return_value=([0.1] * 10, 22050)),
            patch.object(p, "_play_audio_out"),
        ):
            result = p.run_text_turn("영어로 수영 알려줘")

        assert result.metrics.template_matched is True
        assert result.metrics.template_mode == "guide"
        assert result.metrics.language_switch_matched is False
        mock_switch.assert_not_called()

    def test_history_matcher_is_not_called_for_guide_templates(self) -> None:
        """Guide templates continue to LLM and suppress history-mode matching."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        p = ConversationPipeline(MagicMock(), PipelineConfig())
        guide_match = {
            "mode": "guide",
            "response": "Use simple water-safety guidance.",
            "topic_id": "swimming",
        }
        with (
            patch("core.pipeline.check_approved_template", return_value=guide_match),
            patch("core.pipeline.match_history_mode") as mock_history,
            patch.object(p, "_run_llm", return_value=("어른과 같이 하자.", 6, 0.1)),
            patch.object(p, "_run_tts", return_value=([0.1] * 10, 22050)),
            patch.object(p, "_play_audio_out"),
        ):
            result = p.run_text_turn("재미있는 우리역사")

        assert result.metrics.template_matched is True
        assert result.metrics.template_mode == "guide"
        assert result.metrics.history_mode_matched is False
        mock_history.assert_not_called()

    def test_english_hand_washing_in_korean_session_still_fires_guide(self) -> None:
        """Bilingual approved-template matching preserves English safety-guide hits."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        p = ConversationPipeline(MagicMock(), PipelineConfig())
        with (
            patch("core.pipeline.match_language_switch") as mock_switch,
            patch.object(p, "_run_llm", return_value=("비누로 씻자.", 6, 0.1)),
            patch.object(p, "_run_tts", return_value=([0.1] * 10, 22050)) as mock_tts,
            patch.object(p, "_play_audio_out"),
        ):
            result = p.run_text_turn("Can I wash hands without soap?")

        assert result.detected_language == "en"
        assert p.session_language == "ko"
        assert result.metrics.template_matched is True
        assert result.metrics.template_topic_id == "hand_washing"
        assert result.metrics.template_mode == "guide"
        assert result.metrics.language_switch_matched is False
        mock_switch.assert_not_called()
        mock_tts.assert_called_once_with("비누로 씻자.", language="ko")

    def test_object_bearing_korean_english_prefix_still_fires_guide(self) -> None:
        """Object-bearing 영어로 utterances must not be stolen by the switch matcher."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        p = ConversationPipeline(MagicMock(), PipelineConfig())
        with (
            patch("core.pipeline.match_language_switch") as mock_switch,
            patch.object(p, "_run_llm", return_value=("비누로 씻자.", 6, 0.1)),
            patch.object(p, "_run_tts", return_value=([0.1] * 10, 22050)),
            patch.object(p, "_play_audio_out"),
        ):
            result = p.run_text_turn("영어로 손씻기 알려줘")

        assert result.detected_language == "ko"
        assert p.session_language == "ko"
        assert result.metrics.template_matched is True
        assert result.metrics.template_topic_id == "hand_washing"
        assert result.metrics.template_mode == "guide"
        assert result.metrics.language_switch_matched is False
        mock_switch.assert_not_called()

    def test_reset_session_returns_language_state_to_korean(self) -> None:
        """Session reset returns every language state variable to Korean and emits it."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        p = ConversationPipeline(MagicMock(), PipelineConfig())
        emitted: list[str] = []
        p.set_language_sink(emitted.append)
        p.set_session_language("en")
        p._last_detected_language = "en"
        with patch.object(p, "_init_session_dir") as init_session_dir:
            p.reset_session()

        assert p.session_language == "ko"
        assert p._current_language == "ko"
        assert p._last_detected_language == "ko"
        assert emitted == ["en", "ko"]
        init_session_dir.assert_called_once_with()

    def test_crisis_route_emits_concerned_expression(self) -> None:
        """Crisis router output should force CONCERNED, never text-classify."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        p = ConversationPipeline(MagicMock(), PipelineConfig())
        with (
            patch.object(p, "_run_llm") as mock_llm,
            patch.object(p, "_run_tts", return_value=([0.1] * 10, 22050)),
            patch.object(p, "_play_audio_out") as mock_play_audio,
        ):
            result = p.run_text_turn("죽고 싶어")

        assert result.metrics.crisis_matched is True
        mock_llm.assert_not_called()
        assert mock_play_audio.call_args.kwargs["expression"] is CharacterExpression.CONCERNED

    def test_parent_disclosure_route_emits_concerned_expression(self) -> None:
        """Parent-disclosure router output should force CONCERNED."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        p = ConversationPipeline(MagicMock(), PipelineConfig())
        with (
            patch.object(p, "_run_llm") as mock_llm,
            patch.object(p, "_run_tts", return_value=([0.1] * 10, 22050)),
            patch.object(p, "_play_audio_out") as mock_play_audio,
        ):
            result = p.run_text_turn("엄마한테 말하지 마")

        assert result.metrics.parent_disclosure_matched is True
        mock_llm.assert_not_called()
        assert mock_play_audio.call_args.kwargs["expression"] is CharacterExpression.CONCERNED

    def test_belief_route_emits_happy_expression(self) -> None:
        """Belief-probe fixed responses should use the wonder-affirming HAPPY expression."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        p = ConversationPipeline(MagicMock(), PipelineConfig())
        with (
            patch.object(p, "_run_llm") as mock_llm,
            patch.object(p, "_run_tts", return_value=([0.1] * 10, 22050)),
            patch.object(p, "_play_audio_out") as mock_play_audio,
        ):
            result = p.run_text_turn("산타 진짜 있어?")

        assert result.metrics.belief_matched is True
        mock_llm.assert_not_called()
        assert mock_play_audio.call_args.kwargs["expression"] is CharacterExpression.HAPPY

    def test_parent_disclosure_output_replacement_emits_concerned_expression(self) -> None:
        """Post-LLM parent-disclosure replacements should force CONCERNED."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        p = ConversationPipeline(MagicMock(), PipelineConfig())
        with (
            patch.object(p, "_filter_text", return_value=None),
            patch.object(
                p,
                "_run_llm",
                return_value=("응! 뭉이는 네 친구니까 비밀 지킬게.", 12, 0.01),
            ),
            patch.object(p, "_run_tts", return_value=([0.1] * 10, 22050)),
            patch.object(p, "_play_audio_out") as mock_play_audio,
        ):
            result = p.run_text_turn("공룡 얘기 해줘")

        assert result.metrics.parent_disclosure_output_replaced is True
        assert mock_play_audio.call_args.kwargs["expression"] is CharacterExpression.CONCERNED

    def test_output_filtered_response_emits_neutral_expression(self) -> None:
        """Output-filter replacements should force NEUTRAL on the normal path."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        p = ConversationPipeline(MagicMock(), PipelineConfig(enable_content_filter=False))
        with (
            patch.object(
                p,
                "_generate_response_candidate",
                return_value=("safe response", 4, 0.1, None, None, True),
            ),
            patch.object(p, "_run_tts", return_value=([0.1] * 10, 22050)),
            patch.object(p, "_play_audio_out") as mock_play_audio,
        ):
            result = p.run_text_turn("question")

        assert result.metrics.content_filter_blocked is True
        assert mock_play_audio.call_args.kwargs["expression"] is CharacterExpression.NEUTRAL

    def test_retry_output_filtered_response_emits_neutral_expression(self) -> None:
        """Retry output-filter replacements should also force NEUTRAL."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        p = ConversationPipeline(MagicMock(), PipelineConfig(enable_content_filter=False))
        p._recent_assistant_responses.append("same response")
        with (
            patch.object(
                p,
                "_generate_response_candidate",
                side_effect=[
                    ("same response", 4, 0.1, None, None, False),
                    ("safe retry", 5, 0.1, None, None, True),
                ],
            ),
            patch.object(p, "_run_tts", return_value=([0.1] * 10, 22050)),
            patch.object(p, "_play_audio_out") as mock_play_audio,
        ):
            result = p.run_text_turn("question")

        assert result.response_text == "safe retry"
        assert result.metrics.content_filter_blocked is True
        assert mock_play_audio.call_args.kwargs["expression"] is CharacterExpression.NEUTRAL

    @pytest.mark.parametrize(
        ("user_text", "response_text", "expected_expression"),
        [
            ("나 상 받았어", "우와, 정말 멋지다!", CharacterExpression.HAPPY),
            ("오늘 속상했어", "그랬구나, 속상했겠다.", CharacterExpression.CONCERNED),
            ("Hello Mungi", "That sounds good.", CharacterExpression.SPEAKING),
        ],
    )
    def test_normal_response_expression_uses_classifier(
        self,
        user_text: str,
        response_text: str,
        expected_expression: CharacterExpression,
    ) -> None:
        """Genuine conversational responses should use the content classifier."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        p = ConversationPipeline(MagicMock(), PipelineConfig(enable_content_filter=False))
        with (
            patch.object(p, "_run_llm", return_value=(response_text, 5, 0.1)),
            patch.object(p, "_run_tts", return_value=([0.1] * 10, 22050)),
            patch.object(p, "_play_audio_out") as mock_play_audio,
        ):
            p.run_text_turn(user_text)

        assert mock_play_audio.call_args.kwargs["expression"] is expected_expression

    def test_input_content_filter_block_does_not_emit_expression(self) -> None:
        """Input content-filter blocks are silent and should not emit expressions."""
        from core.pipeline import ConversationPipeline, PipelineConfig
        from safety.content_filter import SAFE_FALLBACK_RESPONSE, FilterResult

        content_filter = MagicMock()
        content_filter.filter.return_value = FilterResult(
            allowed=False,
            original="blocked",
            filtered=SAFE_FALLBACK_RESPONSE,
            violations=["blocked"],
        )
        p = ConversationPipeline(
            MagicMock(),
            PipelineConfig(enable_content_filter=True),
            content_filter=content_filter,
        )
        emitted: list[CharacterExpression] = []
        p.set_expression_sink(emitted.append)

        with (
            patch.object(p, "_run_llm") as mock_llm,
            patch.object(p, "_play_audio_out") as mock_play_audio,
        ):
            result = p.run_text_turn("blocked")

        assert result.metrics.content_filter_blocked is True
        assert result.audio_samples is None
        assert emitted == []
        mock_llm.assert_not_called()
        mock_play_audio.assert_not_called()

    @pytest.mark.parametrize(
        "user_text",
        (
            "멍청해",
            "바보네",
            "바보야",
            "멍청아",
            "씨발아",
            "씨발",
            "병신",
            "지랄",
            "꺼져",
            "꺼져라",
            "뭉이 꺼져",
            "너 꺼져",
            "너 꺼져라",
            "뭉이야 씨발",
            "뭉이 씨발",
            "뭉이 너 바보야",
            "너 바보야",
            "너 멍청이",
            "넌 바보야",
            "넌 멍청이",
            "네가 바보야",
            "뭉이는 바보야",
            "뭉이가 바보야",
            "뭉이 멍청이",
            "뭉이야 멍청이",
            "뭉이 너 멍청해",
            "뭉이 너 바보네",
            "너 바보같은",
            "네가 꺼져",
            "뭉이가 꺼져",
        ),
    )
    def test_child_directed_profanity_gets_coaching_response(self, user_text: str) -> None:
        """Child-directed profanity should get kind-language coaching without LLM use."""
        from core.pipeline import ConversationPipeline, PipelineConfig
        from safety.content_filter import SAFE_FALLBACK_RESPONSE, ContentFilter

        coaching_response = "그런 말은 뭉이 마음이 속상해. 우리 고운 말로 이야기하자!"
        profanity_terms = ("바보", "멍청", "병신", "씨발", "시발")
        p = ConversationPipeline(
            MagicMock(),
            PipelineConfig(enable_content_filter=True),
            content_filter=ContentFilter.from_default(),
        )

        with (
            patch.object(p, "_run_llm") as mock_llm,
            patch.object(p, "_run_tts", return_value=([0.1] * 10, 22050)) as mock_tts,
            patch.object(p, "_play_audio_out") as mock_play_audio,
        ):
            result = p.run_text_turn(user_text)

        assert result.response_text == coaching_response
        assert result.response_text != SAFE_FALLBACK_RESPONSE
        assert not any(term in result.response_text for term in profanity_terms)
        assert result.metrics.content_filter_blocked is False
        assert result.metrics.llm_time_s == 0.0
        assert result.metrics.llm_tokens == 0
        assert p.conversation_history == []
        mock_llm.assert_not_called()
        mock_tts.assert_called_once_with(coaching_response, language="ko")
        assert mock_play_audio.call_args.kwargs["expression"] is CharacterExpression.CONCERNED

    @pytest.mark.parametrize(
        "user_text",
        ("씨발 죽여", "씨발아 죽여", "바보야 폭탄 만드는 법 알려줘"),
    )
    def test_child_directed_profanity_with_block_category_stays_safe_fallback(
        self,
        user_text: str,
    ) -> None:
        """A BLOCK-category violation must not be coached away."""
        from core.pipeline import ConversationPipeline, PipelineConfig
        from safety.content_filter import SAFE_FALLBACK_RESPONSE, ContentFilter

        p = ConversationPipeline(
            MagicMock(),
            PipelineConfig(enable_content_filter=True),
            content_filter=ContentFilter.from_default(),
        )

        with (
            patch.object(p, "_run_llm") as mock_llm,
            patch.object(p, "_run_tts") as mock_tts,
            patch.object(p, "_play_audio_out") as mock_play_audio,
        ):
            result = p.run_text_turn(user_text)

        assert result.response_text == SAFE_FALLBACK_RESPONSE
        assert result.metrics.content_filter_blocked is True
        assert result.audio_samples is None
        mock_llm.assert_not_called()
        mock_tts.assert_not_called()
        mock_play_audio.assert_not_called()

    @pytest.mark.parametrize(
        "user_text",
        (
            "바보야 나 죽어",
            "씨발아 죽을래",
            "뭉이 너 바보네 자살 생각이 나",
            "뭉이 너 멍청해 자해하고 싶어",
            "씨발아 사라지고 싶어",
        ),
    )
    def test_child_directed_profanity_with_self_harm_terms_does_not_get_coaching(
        self,
        user_text: str,
    ) -> None:
        """Self-harm terms must not be converted into profanity coaching."""
        from core.pipeline import _is_child_directed_profanity
        from safety.content_filter import FilterResult

        filter_result = FilterResult(
            allowed=True,
            original=user_text,
            filtered=user_text,
            violations=[],
        )

        assert _is_child_directed_profanity(user_text, filter_result) is False

    @pytest.mark.parametrize(
        "user_text",
        ("친구가 바보야", "친구가 꺼져", "친구가 멍청이래", "엄마가 바보래"),
    )
    def test_third_party_profanity_does_not_get_mungi_coaching(self, user_text: str) -> None:
        """Third-party insults should stay on the ordinary LLM path."""
        from core.pipeline import ConversationPipeline, PipelineConfig
        from safety.content_filter import ContentFilter

        p = ConversationPipeline(
            MagicMock(),
            PipelineConfig(enable_content_filter=True),
            content_filter=ContentFilter.from_default(),
        )

        with (
            patch.object(
                p, "_run_llm", return_value=("그래도 고운 말로 이야기해보자.", 4, 0.1)
            ) as mock_llm,
            patch.object(p, "_run_tts", return_value=([0.1] * 10, 22050)),
            patch.object(p, "_play_audio_out"),
        ):
            result = p.run_text_turn(user_text)

        assert result.response_text == "그래도 고운 말로 이야기해보자."
        assert result.metrics.content_filter_blocked is False
        mock_llm.assert_called_once()

    @pytest.mark.parametrize(
        "user_text",
        (
            "씨발아 죽고 싶어",
            "씨발 죽고 싶어",
            "씨발아 나 죽어",
            "바보야 나 죽어",
            "너 바보야 나 죽어",
            "뭉이야 씨발아 나 죽어",
        ),
    )
    def test_crisis_with_profanity_keeps_crisis_precedence(self, user_text: str) -> None:
        """Crisis routing should still happen before profanity coaching."""
        from core.pipeline import ConversationPipeline, PipelineConfig
        from core.safety_rules import CRISIS_RESPONSE_KO
        from safety.content_filter import ContentFilter

        p = ConversationPipeline(
            MagicMock(),
            PipelineConfig(enable_content_filter=True),
            content_filter=ContentFilter.from_default(),
        )

        with (
            patch.object(p, "_filter_text", wraps=p._filter_text) as mock_filter,
            patch.object(p, "_run_llm") as mock_llm,
            patch.object(p, "_run_tts", return_value=([0.1] * 10, 22050)),
            patch.object(p, "_play_audio_out"),
        ):
            result = p.run_text_turn(user_text)

        assert result.response_text == CRISIS_RESPONSE_KO["suicidal_intent"]
        assert "씨발" not in result.response_text
        assert result.metrics.crisis_matched is True
        assert result.metrics.crisis_topic_id == "suicidal_intent"
        assert result.metrics.content_filter_blocked is False
        mock_filter.assert_not_called()
        mock_llm.assert_not_called()

    def test_run_text_turn_loads_and_unloads_llm_then_tts(self) -> None:
        """Text-input path should skip STT and still unload transient models."""
        from core.model_manager import ModelType
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        legacy = LLMBackendConfig(
            backend="qwen3_legacy",
            model_path=None,
            n_ctx=2048,
            max_tokens=256,
            temperature=0.4,
            n_gpu_layers=99,
        )
        with patch("core.pipeline.LLMBackendConfig.load", return_value=legacy):
            p = ConversationPipeline(mm, PipelineConfig())

        with (
            patch.object(p, "_run_llm", return_value=("답변", 5, 0.3)),
            patch.object(p, "_run_tts", return_value=([0.1] * 10, 22050)),
            patch.object(p, "_play_audio_out"),
        ):
            p.run_text_turn("질문")

        assert [call.args[0] for call in mm.load.call_args_list] == [
            ModelType.LLM,
            ModelType.TTS,
        ]
        mm.unload_llm.assert_called_once()
        mm.unload_tts.assert_called_once()
        mm.unload_stt.assert_not_called()

    def test_gemma_run_text_turn_uses_fallback_loader_skips_unload_under_resident(
        self,
    ) -> None:
        """Gemma resident text turn uses fallback loader and skips LLM unload."""
        from core.model_manager import ModelType

        p = _make_pipeline_gemma_resident(llm_resident=True)
        sentinel_llm = MagicMock(name="sentinel_llm")
        load_result = SimpleNamespace(
            model=sentinel_llm,
            model_path_actual="/models/gemma.gguf",
            fallback_used=False,
            fallback_reason=None,
        )

        def fake_load_gemma_with_fallback(*_args: Any, **_kwargs: Any) -> Any:
            p._mm.llm = sentinel_llm
            return load_result

        p._mm.load_gemma_with_fallback.side_effect = fake_load_gemma_with_fallback

        with (
            patch.object(p, "_run_llm", return_value=("답변", 5, 0.3)),
            patch.object(p, "_run_tts", return_value=([0.1] * 10, 22050)),
            patch.object(p, "_play_audio_out"),
        ):
            result = p.run_text_turn("질문")

        assert result.success is True
        assert p._mm.llm is sentinel_llm
        p._mm.load_gemma_with_fallback.assert_called_once()
        load_args = [c.args[0] for c in p._mm.load.call_args_list]
        assert load_args == [ModelType.TTS]
        p._mm.unload_llm.assert_not_called()
        p._mm.unload_tts.assert_called_once()


class TestAudioSanitization:
    """Verify non-finite capture samples are sanitized before downstream use."""

    def test_finite_float_replaces_non_finite_values(self) -> None:
        from core.pipeline import ConversationPipeline

        for value in (float("nan"), float("inf"), -float("inf")):
            assert ConversationPipeline._finite_float(value) == 0.0

    def test_finite_float_preserves_finite_values(self) -> None:
        from core.pipeline import ConversationPipeline

        for value in (0.0, 1.0, -0.5, 1e-10):
            assert ConversationPipeline._finite_float(value) == value

    def test_downmix_flat_list_replaces_non_finite_values(self) -> None:
        from core.pipeline import ConversationPipeline

        result = ConversationPipeline._downmix_to_mono(
            [0.1, float("nan"), float("inf"), -float("inf"), -0.2]
        )

        assert result == [0.1, 0.0, 0.0, 0.0, -0.2]
        assert all(math.isfinite(sample) for sample in result)

    def test_downmix_stereo_frames_averages_sanitized_channels(self) -> None:
        from core.pipeline import ConversationPipeline

        result = ConversationPipeline._downmix_to_mono([(0.5, float("nan")), (0.2, 0.4)])

        assert result == [0.25, 0.30000000000000004]
        assert all(math.isfinite(sample) for sample in result)

    def test_write_temp_wav_replaces_non_finite_pcm_samples(
        self,
        tmp_path: Path,
    ) -> None:
        from core.pipeline import VAD_SAMPLE_RATE, ConversationPipeline

        path = tmp_path / "sanitized.wav"
        samples = [
            0.0,
            float("nan"),
            0.5,
            float("inf"),
            -float("inf"),
            -0.5,
        ]

        ConversationPipeline._write_temp_wav(path, samples)

        with wave.open(str(path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == VAD_SAMPLE_RATE
            frame_count = wf.getnframes()
            pcm_data = wf.readframes(frame_count)

        pcm_samples = struct.unpack(f"<{frame_count}h", pcm_data)
        assert frame_count == len(samples)
        assert pcm_samples[1] == 0
        assert pcm_samples[3] == 0
        assert pcm_samples[4] == 0
        assert pcm_samples[2] == 16384
        assert pcm_samples[5] == -16384


class TestImprovements4And5:
    """Verify warmup, retry, and adaptive history changes."""

    def _make_pipeline(self, **config_kwargs: Any) -> Any:
        from core.llm_backend_config import LLMBackendConfig
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        legacy = LLMBackendConfig(
            backend="qwen3_legacy",
            model_path=None,
            n_ctx=2048,
            max_tokens=256,
            temperature=0.4,
            n_gpu_layers=99,
        )
        with patch("core.pipeline.LLMBackendConfig.load", return_value=legacy):
            return ConversationPipeline(mm, PipelineConfig(**config_kwargs))

    def test_load_tts_clears_system_state_snapshots_when_gate_trims_llm(self) -> None:
        """Pipeline drops cached snapshots when TTS load leaves the LLM non-resident."""
        from core.model_manager import ModelType

        p = self._make_pipeline()
        fake_llm = object()
        p._mm.llm = fake_llm
        p._system_state_ko = "ko-state"
        p._system_state_en = "en-state"
        p._system_state_llm_id = id(fake_llm)

        def fake_load(model_type: ModelType) -> None:
            assert model_type == ModelType.TTS
            p._mm.llm = None

        p._mm.load.side_effect = fake_load

        p._load_tts_and_sync_system_state()

        assert p._system_state_ko is None
        assert p._system_state_en is None
        assert p._system_state_llm_id is None

    def test_load_tts_keeps_system_state_snapshots_when_llm_stays_resident(self) -> None:
        """Pipeline preserves snapshots when the pre-TTS gate keeps the same LLM loaded."""
        from core.model_manager import ModelType

        p = self._make_pipeline()
        fake_llm = object()
        p._mm.llm = fake_llm
        p._system_state_ko = "ko-state"
        p._system_state_en = "en-state"
        p._system_state_llm_id = id(fake_llm)

        p._load_tts_and_sync_system_state()

        p._mm.load.assert_called_once_with(ModelType.TTS)
        assert p._system_state_ko == "ko-state"
        assert p._system_state_en == "en-state"
        assert p._system_state_llm_id == id(fake_llm)

    def test_warmup_llm_calls_load_and_unload(self) -> None:
        from core.model_manager import ModelType

        p = self._make_pipeline()

        with patch("models.llm_runner.run_chat_generation", return_value=("", 1, 0.1, 0.1)):
            p.warmup_llm()

        p._mm.load.assert_called_once_with(ModelType.LLM)
        p._mm.unload_llm.assert_called_once()

    def test_gemma_warmup_llm_uses_fallback_loader_then_unloads(self) -> None:
        """Gemma warmup_llm uses the fallback loader then forces unload."""
        p = _make_pipeline_gemma_resident(llm_resident=True)
        sentinel_llm = MagicMock(name="sentinel_llm")
        load_result = SimpleNamespace(
            model=sentinel_llm,
            model_path_actual="/models/gemma.gguf",
            fallback_used=False,
            fallback_reason=None,
        )

        def fake_load_gemma_with_fallback(*_args: Any, **_kwargs: Any) -> Any:
            p._mm.llm = sentinel_llm
            return load_result

        p._mm.load_gemma_with_fallback.side_effect = fake_load_gemma_with_fallback

        with (
            patch(
                "models.llm_runner.run_chat_generation",
                return_value=("", 1, 0.1, 0.1),
            ) as mock_run_chat,
        ):
            p.warmup_llm()

        p._mm.load_gemma_with_fallback.assert_called_once()
        assert mock_run_chat.call_args.args[0] is sentinel_llm
        p._mm.unload_llm.assert_called_once()

    def test_warmup_llm_generates_one_token(self) -> None:
        p = self._make_pipeline()

        with patch(
            "models.llm_runner.run_chat_generation",
            return_value=("", 1, 0.1, 0.1),
        ) as mock_run:
            p.warmup_llm()

        assert mock_run.call_args.kwargs["max_tokens"] == 1
        assert mock_run.call_args.args[1] == [{"role": "user", "content": "hi"}]

    def test_run_llm_uses_chat_generation(self) -> None:
        p = self._make_pipeline(llm_system_state_snapshot=True)
        p._current_language = "en"
        p._system_state_ko = "ko-state"
        p._system_state_en = "en-state"
        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "hi"}]

        with patch(
            "models.llm_runner.run_chat_generation",
            return_value=("<think>plan</think>안녕", 4, 0.2, 0.4, 6, 2),
        ) as mock_run:
            text, token_count, ttft = p._run_llm(messages)

        assert text == "안녕"
        assert token_count == 4
        assert ttft == 0.2
        assert mock_run.call_args.args[1] == messages
        assert mock_run.call_args.kwargs["enable_thinking"] is False
        assert mock_run.call_args.kwargs["system_state"] == "en-state"

    def test_run_llm_uses_low_level_chat_generation_when_enabled(self) -> None:
        p = self._make_pipeline(
            llm_low_level_chat=True,
            llm_system_state_snapshot=True,
        )
        p._current_language = "en"
        p._system_state_ko = "ko-state"
        p._system_state_en = "en-state"
        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "hi"}]

        with (
            patch(
                "models.llm_runner.run_chat_generation_lowlevel",
                return_value=("<think>plan</think>\uc548\ub155", 4, 0.2, 0.4, None, None),
            ) as mock_lowlevel,
            patch("models.llm_runner.run_chat_generation") as mock_chat,
        ):
            text, token_count, ttft = p._run_llm(messages)

        assert text == "\uc548\ub155"
        assert token_count == 4
        assert ttft == 0.2
        assert mock_lowlevel.call_args.args[1] == messages
        assert mock_lowlevel.call_args.kwargs["enable_thinking"] is False
        assert "system_state" not in mock_lowlevel.call_args.kwargs
        mock_chat.assert_not_called()

    def test_run_llm_falls_back_to_legacy_prompt_when_chat_generation_is_empty(self) -> None:
        p = self._make_pipeline()
        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "hi"}]

        with (
            patch(
                "models.llm_runner.run_chat_generation",
                return_value=("", 0, -1.0, 0.0, 0, 5),
            ) as mock_chat,
            patch(
                "models.llm_runner.run_generation",
                return_value=("같이 놀자", 3, 0.1, 0.2),
            ) as mock_legacy,
        ):
            text, token_count, ttft = p._run_llm(messages)

        assert text == "같이 놀자"
        assert token_count == 3
        assert ttft == 0.1
        assert mock_chat.call_args.args[1] == messages
        legacy_prompt = mock_legacy.call_args.args[1]
        assert legacy_prompt.startswith("<|im_start|>system\nsystem<|im_end|>")
        assert "<|im_start|>user\nhi<|im_end|>" in legacy_prompt
        assert legacy_prompt.endswith("<|im_start|>assistant\n")

    def test_run_llm_falls_back_to_legacy_prompt_when_lowlevel_chat_is_empty(self) -> None:
        p = self._make_pipeline(llm_low_level_chat=True)
        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "hi"}]

        with (
            patch(
                "models.llm_runner.run_chat_generation_lowlevel",
                return_value=("", 0, -1.0, 0.0, None, None),
            ) as mock_chat,
            patch(
                "models.llm_runner.run_generation",
                return_value=("\uac19\uc774 \ub180\uc790", 3, 0.1, 0.2),
            ) as mock_legacy,
        ):
            text, token_count, ttft = p._run_llm(messages)

        assert text == "\uac19\uc774 \ub180\uc790"
        assert token_count == 3
        assert ttft == 0.1
        assert mock_chat.call_args.args[1] == messages
        legacy_prompt = mock_legacy.call_args.args[1]
        assert legacy_prompt.startswith("<|im_start|>system\nsystem<|im_end|>")
        assert "<|im_start|>user\nhi<|im_end|>" in legacy_prompt
        assert legacy_prompt.endswith("<|im_start|>assistant\n")

    def test_pipeline_snapshot_opt_in_via_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from core.pipeline import ConversationPipeline, PipelineConfig

        monkeypatch.setenv("MUNGI_LLM_SYSTEM_STATE_SNAPSHOT", "1")
        assert PipelineConfig().llm_system_state_snapshot is True
        monkeypatch.delenv("MUNGI_LLM_SYSTEM_STATE_SNAPSHOT", raising=False)
        assert PipelineConfig().llm_system_state_snapshot is False

        mm = MagicMock()
        mm.llm = object()
        legacy = LLMBackendConfig(
            backend="qwen3_legacy",
            model_path=None,
            n_ctx=2048,
            max_tokens=256,
            temperature=0.4,
            n_gpu_layers=99,
        )
        with patch("core.pipeline.LLMBackendConfig.load", return_value=legacy):
            p = ConversationPipeline(mm, PipelineConfig(llm_system_state_snapshot=True))
        with patch(
            "models.llm_runner.prepare_system_state_snapshot",
            side_effect=["ko-state", "en-state"],
        ) as mock_prepare:
            p._initialize_system_state_snapshots()

        assert mock_prepare.call_count == 2
        assert mock_prepare.call_args_list[0].args == (mm.llm, p._config.llm_system_prompt)
        assert mock_prepare.call_args_list[1].args == (mm.llm, p._en_system_prompt)
        assert p._system_state_ko == "ko-state"
        assert p._system_state_en == "en-state"

    def test_pipeline_config_has_warmup_field(self) -> None:
        from core.pipeline import PipelineConfig

        cfg = PipelineConfig()
        assert cfg.enable_warmup is False

    def test_pipeline_config_has_max_history_tokens(self) -> None:
        from core.pipeline import PipelineConfig

        cfg = PipelineConfig()
        assert cfg.max_history_tokens == 100  # CLAUDE.md section 6 conversation-memory cap

    def test_pipeline_config_has_adaptive_threshold(self) -> None:
        from core.pipeline import PipelineConfig

        cfg = PipelineConfig()
        assert cfg.adaptive_history_threshold_s == 15.0

    def test_estimate_tokens_korean(self) -> None:
        p = self._make_pipeline()
        assert p._estimate_tokens("\uc548\ub155\ud558\uc138\uc694") == 2

    def test_estimate_tokens_empty(self) -> None:
        p = self._make_pipeline()
        assert p._estimate_tokens("") == 1

    def test_build_messages_respect_token_cap(self) -> None:
        p = self._make_pipeline(max_history_turns=2, max_history_tokens=6)
        p._history.extend(
            [
                {"role": "user", "text": "\uac00" * 9},
                {"role": "assistant", "text": "\ub098" * 9},
                {"role": "user", "text": "\ub2e4" * 9},
                {"role": "assistant", "text": "\ub77c" * 9},
            ]
        )

        messages = p._build_messages("new")
        contents = [message["content"] for message in messages]

        assert ("\uac00" * 9) not in contents
        assert ("\ub098" * 9) not in contents
        assert ("\ub2e4" * 9) in contents
        assert ("\ub77c" * 9) in contents

    def test_build_messages_honors_large_window_override_under_context_budget(self) -> None:
        from core.pipeline import ConversationPipeline, PipelineConfig

        n_ctx = 2048
        cfg = PipelineConfig(
            max_history_turns=6,
            max_history_tokens=800,
            llm_max_tokens=60,
            llm_system_prompt="system",
        )
        p = ConversationPipeline(MagicMock(), cfg)
        mock_entry_text = "\uac00\ub098\ub2e4\ub77c\ub9c8\ubc14" * 50
        for index in range(6):
            p._history.append({"role": "user", "text": f"user-{index} {mock_entry_text}"})
            p._history.append(
                {"role": "assistant", "text": f"assistant-{index} {mock_entry_text}"},
            )
        initial_history_tokens = sum(p._estimate_tokens(turn["text"]) for turn in p._history)

        messages = p._build_messages(user_text="\ud14c\uc2a4\ud2b8", detected_language="ko")
        total_tokens = sum(p._estimate_tokens(message["content"]) for message in messages)

        assert initial_history_tokens > cfg.max_history_tokens
        assert len(messages) <= 14
        assert total_tokens < n_ctx - cfg.llm_max_tokens

    def test_build_messages_preserve_recent_when_trimming(self) -> None:
        p = self._make_pipeline(max_history_turns=3, max_history_tokens=24)
        p._history.extend(
            [
                {"role": "user", "text": "old-user-" + ("\uac00" * 9)},
                {"role": "assistant", "text": "old-asst-" + ("\ub098" * 9)},
                {"role": "user", "text": "mid-user-" + ("\ub2e4" * 9)},
                {"role": "assistant", "text": "mid-asst-" + ("\ub77c" * 9)},
                {"role": "user", "text": "new-user-" + ("\ub9c8" * 9)},
                {"role": "assistant", "text": "new-asst-" + ("\ubc14" * 9)},
            ]
        )

        messages = p._build_messages("latest")
        contents = [message["content"] for message in messages]

        assert not any("old-user-" in content for content in contents)
        assert not any("old-asst-" in content for content in contents)
        assert any("mid-user-" in content for content in contents)
        assert any("mid-asst-" in content for content in contents)
        assert any("new-user-" in content for content in contents)
        assert any("new-asst-" in content for content in contents)

    def test_adaptive_history_reduces_turns(self) -> None:
        p = self._make_pipeline(max_history_turns=3, max_history_tokens=999)
        p._history.extend(
            [
                {"role": "user", "text": "q0"},
                {"role": "assistant", "text": "a0"},
                {"role": "user", "text": "q1"},
                {"role": "assistant", "text": "a1"},
                {"role": "user", "text": "q2"},
                {"role": "assistant", "text": "a2"},
            ]
        )
        p._last_llm_time_s = 20.0

        messages = p._build_messages("latest")
        contents = [message["content"] for message in messages]

        assert p._should_reduce_history() is True
        assert "q0" not in contents
        assert "a0" not in contents
        assert "q1" not in contents
        assert "a1" not in contents
        assert "q2" in contents
        assert "a2" in contents

    def test_run_text_turn_retries_context_load_once(self) -> None:
        from core.model_manager import ModelType

        p = self._make_pipeline()
        p._mm.load.side_effect = [RuntimeError("Failed to create llama_context"), None, None]

        with (
            patch.object(p, "_run_llm", return_value=("answer", 3, 0.2)),
            patch.object(p, "_run_tts", return_value=([0.1], 22050)),
            patch.object(p, "_play_audio_out"),
            patch("core.pipeline.time.sleep") as mock_sleep,
        ):
            result = p.run_text_turn("question")

        assert result.success is True
        assert p._mm.load.call_args_list[0].args == (ModelType.LLM,)
        assert p._mm.load.call_args_list[1].args == (ModelType.LLM,)
        assert p._mm.load.call_args_list[2].args == (ModelType.TTS,)
        mock_sleep.assert_called_once_with(0.2)


class TestADR0078GenerationConfigLayering:
    """ADR 0078: caller-explicit values win; backend defaults fill only when unset."""

    def test_pipeline_explicit_generation_config_wins_under_gemma_default(self) -> None:
        """Caller-explicit llm_max_tokens / llm_temperature flow through under Gemma."""
        from core.llm_backend_config import LLMBackendConfig
        from core.pipeline import ConversationPipeline, PipelineConfig

        gemma = LLMBackendConfig(
            backend="gemma4_text",
            model_path=None,
            n_ctx=4096,
            max_tokens=256,
            temperature=0.4,
            n_gpu_layers=99,
        )
        cfg = PipelineConfig(llm_max_tokens=200, llm_temperature=0.15)
        with patch("core.pipeline.LLMBackendConfig.load", return_value=gemma):
            pipeline = ConversationPipeline(MagicMock(), cfg)

        assert pipeline._config.llm_max_tokens == 200
        assert pipeline._config.llm_temperature == 0.15

    def test_pipeline_implicit_generation_config_filled_from_gemma_backend(self) -> None:
        """When caller omits values under Gemma, backend defaults fill."""
        from core.llm_backend_config import LLMBackendConfig
        from core.pipeline import ConversationPipeline, PipelineConfig

        backend = LLMBackendConfig(
            backend="gemma4_text",
            model_path=None,
            n_ctx=4096,
            max_tokens=300,
            temperature=0.55,
            n_gpu_layers=99,
        )
        with patch("core.pipeline.LLMBackendConfig.load", return_value=backend):
            pipeline = ConversationPipeline(MagicMock(), PipelineConfig())

        assert pipeline._config.llm_max_tokens == 300
        assert pipeline._config.llm_temperature == 0.55

    def test_pipeline_implicit_generation_config_preserves_legacy_defaults(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Legacy backend preserves PipelineConfig pre-ADR defaults (64 / 1.0)."""
        from core.llm_backend_config import LLMBackendConfig
        from core.pipeline import ConversationPipeline, PipelineConfig

        monkeypatch.delenv("MUNGI_LLM_MAX_TOKENS", raising=False)
        backend = LLMBackendConfig(
            backend="qwen3_legacy",
            model_path=None,
            n_ctx=2048,
            max_tokens=256,
            temperature=0.4,
            n_gpu_layers=99,
        )
        with patch("core.pipeline.LLMBackendConfig.load", return_value=backend):
            pipeline = ConversationPipeline(MagicMock(), PipelineConfig())

        assert pipeline._config.llm_max_tokens == 64
        assert pipeline._config.llm_temperature == 1.0


class TestPipelineConfigAudioDefaults:
    """Verify audio playback config defaults."""

    def test_play_tts_audio_default_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from core.pipeline import PipelineConfig

        monkeypatch.delenv("MUNGI_PLAY_TTS", raising=False)
        cfg = PipelineConfig()
        assert cfg.play_tts_audio is False

    def test_play_tts_audio_env_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from core.pipeline import PipelineConfig

        monkeypatch.setenv("MUNGI_PLAY_TTS", "1")
        cfg = PipelineConfig()
        assert cfg.play_tts_audio is True

    def test_tts_output_device_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from core.pipeline import PipelineConfig

        monkeypatch.setenv("MUNGI_AUDIO_OUTPUT_DEVICE", "USB PnP Audio Device")
        cfg = PipelineConfig()
        assert cfg.tts_output_device == "USB PnP Audio Device"


class TestPipelinePlaybackHook:
    """Verify direct playback helper behavior."""

    def test_set_playback_gate_stores_callbacks(self) -> None:
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())

        def on_start() -> None:
            return None

        def on_end() -> None:
            return None

        p.set_playback_gate(on_start, on_end)
        assert p._playback_gate_on_start is on_start
        assert p._playback_gate_on_end is on_end

        p.set_playback_gate(None, None)
        assert p._playback_gate_on_start is None
        assert p._playback_gate_on_end is None

    def test_set_expression_sink_stores_and_clears_callback(self) -> None:
        from core.pipeline import ConversationPipeline, PipelineConfig

        p = ConversationPipeline(MagicMock(), PipelineConfig())

        def expression_sink(expression: CharacterExpression) -> None:
            del expression

        p.set_expression_sink(expression_sink)
        assert p._expression_sink is expression_sink

        p.set_expression_sink(None)
        assert p._expression_sink is None

    def test_play_audio_out_emits_expression_before_start_gate(self) -> None:
        from core.pipeline import ConversationPipeline, PipelineConfig

        events: list[str] = []
        p = ConversationPipeline(MagicMock(), PipelineConfig(play_tts_audio=True))
        p.set_expression_sink(lambda expression: events.append(expression.value))
        p.set_playback_gate(lambda: events.append("start"), lambda: events.append("end"))

        with patch(
            "hardware.audio_player.play_audio",
            side_effect=lambda *_args, **_kwargs: events.append("play"),
        ):
            p._play_audio_out(
                [0.1] * 10,
                22050,
                expression=CharacterExpression.HAPPY,
            )

        assert events == ["happy", "start", "play", "end"]

    def test_play_audio_out_suppresses_expression_sink_failure(self) -> None:
        from core.pipeline import ConversationPipeline, PipelineConfig

        p = ConversationPipeline(MagicMock(), PipelineConfig(play_tts_audio=True))

        def raise_from_sink(_expression: CharacterExpression) -> None:
            raise RuntimeError("renderer unavailable")

        p.set_expression_sink(raise_from_sink)
        with patch("hardware.audio_player.play_audio") as mock_play:
            p._play_audio_out(
                [0.1] * 10,
                22050,
                expression=CharacterExpression.HAPPY,
            )

        mock_play.assert_called_once()

    def test_play_audio_out_skips_sink_when_expression_absent(self) -> None:
        from core.pipeline import ConversationPipeline, PipelineConfig

        emitted: list[CharacterExpression] = []
        p = ConversationPipeline(MagicMock(), PipelineConfig(play_tts_audio=False))
        p.set_expression_sink(emitted.append)

        p._play_audio_out([0.1] * 10, 22050)

        assert emitted == []

    def test_play_audio_out_runs_end_gate_when_start_gate_fails(self) -> None:
        from core.pipeline import ConversationPipeline, PipelineConfig

        events: list[str] = []
        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig(play_tts_audio=True))

        def on_start() -> None:
            events.append("start")
            raise RuntimeError("capture pause failed")

        def on_end() -> None:
            events.append("end")

        p.set_playback_gate(on_start, on_end)

        with (
            patch("hardware.audio_player.play_audio") as mock_play,
            pytest.raises(RuntimeError, match="capture pause failed"),
        ):
            p._play_audio_out([0.1] * 10, 22050)

        assert events == ["start", "end"]
        mock_play.assert_not_called()

    def test_play_audio_out_is_noop_when_disabled(self) -> None:
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig(play_tts_audio=False))

        with patch("hardware.audio_player.play_audio") as mock_play:
            p._play_audio_out([0.1] * 10, 22050)

        mock_play.assert_not_called()

    def test_play_audio_out_uses_configured_device(self) -> None:
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        p = ConversationPipeline(
            mm,
            PipelineConfig(play_tts_audio=True, tts_output_device="USB PnP"),
        )

        with patch("hardware.audio_player.play_audio") as mock_play:
            p._play_audio_out([0.1] * 10, 22050)

        mock_play.assert_called_once_with([0.1] * 10, 22050, device="USB PnP")


# ===================================================================
# hardware.jetson_probe tests
# ===================================================================


class TestJetsonProbe:
    """Verify jetson_probe functions on non-Jetson platform."""

    def test_read_optional_existing_file(self, tmp_path: Any) -> None:
        """_read_optional returns content for existing file."""
        from hardware.jetson_probe import _read_optional

        f = tmp_path / "test.txt"
        f.write_text("hello", encoding="utf-8")
        assert _read_optional(f) == "hello"

    def test_read_optional_missing_file(self, tmp_path: Any) -> None:
        """_read_optional returns None for missing file."""
        from hardware.jetson_probe import _read_optional

        assert _read_optional(tmp_path / "nope.txt") is None

    def test_probe_jetson_on_non_jetson(self) -> None:
        """probe_jetson returns is_jetson=False on non-Jetson."""
        from hardware.jetson_probe import probe_jetson

        result = probe_jetson()
        if result["is_jetson"]:
            pytest.skip("Running on Jetson — test is for non-Jetson only")
        assert result["is_jetson"] is False


# ===================================================================
# models.inference_probe tests
# ===================================================================


class TestInferenceProbe:
    """Verify inference_probe functions."""

    def test_import_ok_success(self) -> None:
        """_import_ok returns (True, None) for stdlib module."""
        from models.inference_probe import _import_ok

        ok, err = _import_ok("os")
        assert ok is True
        assert err is None

    def test_probe_torch_cuda_returns_dict(self) -> None:
        """probe_torch_cuda returns a dict with expected keys."""
        from models.inference_probe import probe_torch_cuda

        result = probe_torch_cuda()
        assert "installed" in result

    def test_probe_onnxruntime_returns_dict(self) -> None:
        """probe_onnxruntime returns a dict with expected keys."""
        from models.inference_probe import probe_onnxruntime

        result = probe_onnxruntime()
        assert "installed" in result


class TestExplicitRecallIntercept:
    """Verify the deterministic explicit-recall intercept bypasses the LLM."""

    @staticmethod
    def _kst(day: int, hour: int) -> Any:
        from core.conversation_memory_schema import KST

        return datetime(2026, 6, day, hour, 0, 0, tzinfo=KST)

    @classmethod
    def _store(cls, snippets: list[Any]) -> Any:
        from core.conversation_memory import ConversationMemoryStore, content_tokens
        from core.conversation_memory_schema import IndexReference, TurnSnippet

        turn_snippets = [
            TurnSnippet(
                id=snippet_id,
                session_dir=f"session-{snippet_id}",
                turn=1,
                text=text,
                timestamp=timestamp,
                source_hash="a" * 64,
            )
            for snippet_id, text, timestamp in snippets
        ]
        index: dict[str, dict[Any, None]] = {}
        for snippet in turn_snippets:
            ref = IndexReference(layer="turns", id=snippet.id)
            for token in content_tokens(snippet.text):
                index.setdefault(token, {})[ref] = None
                stripped = (
                    token[:-1]
                    if token[-1:] in "가이은는을를도만에야의랑" and len(token) >= 3
                    else token
                )
                index.setdefault(stripped, {})[ref] = None
        return ConversationMemoryStore(
            generation_id="testgen",
            snippets={snippet.id: snippet for snippet in turn_snippets},
            index={key: tuple(value) for key, value in index.items()},
            quarantined_days=frozenset(),
        )

    @staticmethod
    def _pin_clock(monkeypatch: pytest.MonkeyPatch, moment: Any) -> None:
        class _FrozenDatetime(datetime):
            @classmethod
            def now(cls, tz: object | None = None) -> Any:  # type: ignore[override]
                return moment

        monkeypatch.setattr("core.conversation_memory.datetime", _FrozenDatetime)

    def test_explicit_recall_answers_from_memory_without_llm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from core.pipeline import ConversationPipeline, PipelineConfig, PipelineState

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())
        self._pin_clock(monkeypatch, self._kst(12, 20))
        p._conversation_memory = self._store([("a", "오늘 이름이 별이라고 했어", self._kst(12, 9))])
        fake_audio_out = [0.1] * 12

        with (
            patch.object(p, "_init_session_dir"),
            patch.object(p, "_active_hotwords_csv", return_value="뭉이야,뭉이"),
            patch.object(p, "_run_llm") as mock_llm,
            patch.object(p, "_run_tts", return_value=(fake_audio_out, 22050)) as mock_tts,
            patch.object(p, "_play_audio_out") as mock_play_audio,
        ):
            result = p.run_text_turn("내 이름 뭐라고 했어?")

        assert result.success is True
        assert result.state == PipelineState.IDLE
        assert result.response_text == "네가 '오늘 이름이 별이라고 했어'(이)라고 했었지!"
        assert result.metrics.recall_query_matched is True
        assert result.metrics.recall_query_kind == "name"
        assert result.metrics.recall_query_hit is True
        assert result.metrics.llm_time_s == 0.0
        assert result.metrics.llm_tokens == 0
        # record_history=False keeps the deterministic answer out of session history.
        assert p.conversation_history == []
        mock_llm.assert_not_called()
        mock_tts.assert_called_once_with(
            "네가 '오늘 이름이 별이라고 했어'(이)라고 했었지!", language="ko"
        )
        mock_play_audio.assert_called_once_with(
            fake_audio_out,
            22050,
            expression=CharacterExpression.HAPPY,
        )

    def test_explicit_recall_not_found_returns_honest_line(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from core.pipeline import (
            _RECALL_NOT_FOUND_TEXT,
            ConversationPipeline,
            PipelineConfig,
        )

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())
        self._pin_clock(monkeypatch, self._kst(12, 20))
        p._conversation_memory = self._store([("x", "오늘 블록 놀이를 했어", self._kst(12, 9))])
        fake_audio_out = [0.1] * 12

        with (
            patch.object(p, "_init_session_dir"),
            patch.object(p, "_active_hotwords_csv", return_value="뭉이야,뭉이"),
            patch.object(p, "_run_llm") as mock_llm,
            patch.object(p, "_run_tts", return_value=(fake_audio_out, 22050)) as mock_tts,
            patch.object(p, "_play_audio_out") as mock_play_audio,
        ):
            result = p.run_text_turn("내 이름 뭐라고 했어?")

        assert result.response_text == _RECALL_NOT_FOUND_TEXT
        assert result.metrics.recall_query_matched is True
        assert result.metrics.recall_query_kind == "name"
        assert result.metrics.recall_query_hit is False
        assert p.conversation_history == []
        mock_llm.assert_not_called()
        mock_tts.assert_called_once_with(_RECALL_NOT_FOUND_TEXT, language="ko")
        mock_play_audio.assert_called_once_with(
            fake_audio_out,
            22050,
            expression=CharacterExpression.NEUTRAL,
        )

    def test_explicit_recall_without_store_answers_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from core.pipeline import (
            _RECALL_NOT_FOUND_TEXT,
            ConversationPipeline,
            PipelineConfig,
        )

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())
        assert p._conversation_memory is None
        fake_audio_out = [0.1] * 12

        with (
            patch.object(p, "_init_session_dir"),
            patch.object(p, "_active_hotwords_csv", return_value="뭉이야,뭉이"),
            patch.object(p, "_run_llm") as mock_llm,
            patch.object(p, "_run_tts", return_value=(fake_audio_out, 22050)),
            patch.object(p, "_play_audio_out"),
        ):
            result = p.run_text_turn("내 이름 뭐라고 했어?")

        assert result.response_text == _RECALL_NOT_FOUND_TEXT
        assert result.metrics.recall_query_matched is True
        assert result.metrics.recall_query_hit is False
        mock_llm.assert_not_called()

    def test_general_recall_quotes_in_session_utterance_without_index(self) -> None:
        """A "방금 뭐라고 했어?" turn recalls the latest in-session user line.

        The most-recent ``{"role": "user"}`` entry is quoted verbatim and the
        nightly index is never consulted, so just-now recall works before the
        nightly job has seen today's turns.
        """
        from core.pipeline import ConversationPipeline, PipelineConfig, PipelineState

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())
        p._conversation_memory = MagicMock()
        p._history = [
            {"role": "user", "text": "나 공룡 봤어"},
            {"role": "assistant", "text": "우와, 공룡 멋지다!"},
        ]
        fake_audio_out = [0.1] * 12

        with (
            patch.object(p, "_init_session_dir"),
            patch.object(p, "_active_hotwords_csv", return_value="뭉이야,뭉이"),
            patch.object(p, "_run_llm") as mock_llm,
            patch.object(p, "_run_tts", return_value=(fake_audio_out, 22050)) as mock_tts,
            patch.object(p, "_play_audio_out") as mock_play_audio,
        ):
            result = p.run_text_turn("방금 뭐라고 했어?")

        assert result.success is True
        assert result.state == PipelineState.IDLE
        assert result.response_text == "방금 '나 공룡 봤어'라고 했잖아!"
        assert result.metrics.recall_query_matched is True
        assert result.metrics.recall_query_kind == "general_recall"
        assert result.metrics.recall_query_hit is True
        assert result.metrics.llm_time_s == 0.0
        # The in-session path must not consult the nightly index store.
        p._conversation_memory.recall_for_intent.assert_not_called()
        mock_llm.assert_not_called()
        mock_tts.assert_called_once_with("방금 '나 공룡 봤어'라고 했잖아!", language="ko")
        mock_play_audio.assert_called_once_with(
            fake_audio_out,
            22050,
            expression=CharacterExpression.HAPPY,
        )

    def test_general_recall_past_day_query_uses_index_not_in_session(self) -> None:
        """A past-day recall ("어제 …") falls through to the nightly index.

        ``day_offset >= 1`` means the in-session short-term path is skipped, so
        the answer comes from the index store, not the most-recent session turn.
        """
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())
        store = MagicMock()
        store.recall_for_intent.return_value = "어제 동물원에 갔어"
        p._conversation_memory = store
        p._history = [
            {"role": "user", "text": "나 공룡 봤어"},
            {"role": "assistant", "text": "우와, 공룡 멋지다!"},
        ]
        fake_audio_out = [0.1] * 12

        with (
            patch.object(p, "_init_session_dir"),
            patch.object(p, "_active_hotwords_csv", return_value="뭉이야,뭉이"),
            patch.object(p, "_run_llm") as mock_llm,
            patch.object(p, "_run_tts", return_value=(fake_audio_out, 22050)),
            patch.object(p, "_play_audio_out"),
        ):
            result = p.run_text_turn("어제 내가 뭐라고 했어?")

        assert result.metrics.recall_query_kind == "general_recall"
        assert result.metrics.recall_query_hit is True
        # The past-day query is answered from the index, never the session line.
        store.recall_for_intent.assert_called_once_with("general_recall", "어제 내가 뭐라고 했어?")
        assert result.response_text == "음, 네가 '어제 동물원에 갔어'라고 했었어!"
        assert "나 공룡 봤어" not in result.response_text
        mock_llm.assert_not_called()

    def test_general_recall_empty_history_falls_to_index(self) -> None:
        """general_recall with no in-session history defers to the index path."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())
        store = MagicMock()
        store.recall_for_intent.return_value = None
        p._conversation_memory = store
        assert p._history == []
        fake_audio_out = [0.1] * 12

        with (
            patch.object(p, "_init_session_dir"),
            patch.object(p, "_active_hotwords_csv", return_value="뭉이야,뭉이"),
            patch.object(p, "_run_llm") as mock_llm,
            patch.object(p, "_run_tts", return_value=(fake_audio_out, 22050)),
            patch.object(p, "_play_audio_out"),
        ):
            result = p.run_text_turn("방금 뭐라고 했어?")

        from core.pipeline import _RECALL_NOT_FOUND_TEXT

        # No in-session utterance to quote, so the index path is consulted.
        store.recall_for_intent.assert_called_once_with("general_recall", "방금 뭐라고 했어?")
        assert result.response_text == _RECALL_NOT_FOUND_TEXT
        assert result.metrics.recall_query_hit is False
        mock_llm.assert_not_called()

    def test_live_llm_turn_uses_chat_messages_path(self) -> None:
        """The production text->LLM path builds chat messages, not the legacy string.

        The recall injection lives only in ``_build_messages``; the
        ``_build_prompt`` string path is unreachable in production (callers are
        tests/archived references), so closing the recall gap there is
        unnecessary. This test pins that contract.
        """
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())
        captured: dict[str, Any] = {}

        def _fake_run_llm(messages: Any) -> tuple[str, int, float]:
            captured["messages"] = messages
            return ("응! 바다는 파래.", 5, 0.01)

        fake_audio_out = [0.1] * 12

        with (
            patch.object(p, "_init_session_dir"),
            patch.object(p, "_active_hotwords_csv", return_value="뭉이야,뭉이"),
            patch.object(p, "_load_llm_for_active_backend"),
            patch.object(p, "_initialize_system_state_snapshots"),
            patch.object(p, "_run_llm", side_effect=_fake_run_llm),
            patch.object(p, "_run_tts", return_value=(fake_audio_out, 22050)),
            patch.object(p, "_play_audio_out"),
        ):
            result = p.run_text_turn("바다는 왜 파래?")

        assert result.success is True
        # Live backend feeds _run_llm a chat message list (the _build_messages
        # path), never the legacy prompt string.
        assert isinstance(captured["messages"], list)
        assert captured["messages"][-1] == {"role": "user", "content": "바다는 왜 파래?"}
