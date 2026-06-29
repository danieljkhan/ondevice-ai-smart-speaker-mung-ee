"""Tests for approved safety template routing."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.model_manager import ModelType
from core.pipeline import ConversationPipeline, PipelineConfig
from safety import approved_template_router
from safety.approved_template_router import check_approved_template

MUNGI_SELF_INTRO_CHILD = (
    "안녕! 나는 뭉이야. 세상에서 제일 안전한, 너의 첫 번째 인공지능 친구가 되려고 태어났어."
    "\n\n"
    "나는 인터넷이 없어도 혼자 생각할 수 있어. 그래서 우리가 나눈 이야기는 밖으로 새어 나가지 않고 "
    "기기 안에 안전하게 있어. 그리고 엄마 아빠는 우리가 나눈 이야기를 볼 수 있어. 화면을 톡 누르고 "
    "말하면, 나는 네 목소리를 잘 듣고 알아들은 다음에 내 목소리로 대답해 줘."
    "\n\n"
    "나랑은 이런 걸 같이 할 수 있어. 궁금한 걸 물어보면 쉽게 설명해 주고, '퍼니 잉글리시'로 영어도 "
    "따라 말하고, 우리나라 역사 이야기도 들려줄 수 있어. 그리고 우리가 했던 이야기를 기억해 뒀다가 "
    "다음에 또 꺼내기도 해!"
    "\n\n"
    "제일 중요한 건, 나는 언제나 너에게 안전하고 친절한 친구가 되는 거야. 무섭거나 나쁜 이야기는 "
    "하지 않고, 늘 네 편이야. 우리 재미있게 이야기하자!"
)
MUNGI_PRODUCT_INTRO_ADULT = (
    "안녕하세요, 저는 뭉이예요."
    "\n\n"
    "저의 제작 목적은 만 10세 미만 아이들을 위한 '세상에서 가장 안전한 첫 번째 AI 친구'가 되는 "
    "거예요. 아이가 안전하게 이야기 나누고, 배우고, 마음을 주고받으면서 친구처럼 가까워지도록 돕고 "
    "싶어서 태어났어요."
    "\n\n"
    "저에게 적용된 기술은 인터넷 없이도 기기 안에서 혼자 다 처리하는 오프라인 엣지 AI예요. 아이 말을 "
    "알아듣는 음성 인식, 똑똑하게 대답을 만드는 언어 모델, 사람처럼 자연스럽게 말하는 음성 합성, "
    "그리고 나눈 이야기를 기억해 두는 대화 기억까지 작은 기기 한 대에 모두 담겨 있답니다."
    "\n\n"
    "제가 가진 장점은 크게 네 가지예요. 첫째로 안전해요 — 위험하거나 나쁜 내용은 걸러 내고, 아이가 "
    "힘든 상황을 이야기하면 어른에게 알리도록 도와줘요. 둘째로 사생활을 지켜요 — 아이 이야기를 "
    "인터넷으로 보내지 않고 기기 안에만 저장해서, 부모님이 직접 확인하실 수 있어요. 셋째로 인터넷이 "
    "없어도 어디서든 작동해요. 넷째로 배우고 친해질 수 있어요 — 영어를 따라 말하고, 우리 역사 "
    "이야기를 듣고, 지난 대화를 기억하면서 아이와 더 가까워진답니다."
    "\n\n"
    "저는 언제나 아이 곁에서 안전하고 다정한 첫 번째 친구가 되어 줄게요."
)


def test_korean_keyword_match_returns_korean_response() -> None:
    """Korean keyword hits return the Korean approved response."""
    result = check_approved_template("손 씻을 때 비누 없이 해도 돼?", language="ko")
    assert result is not None
    assert "비누" in result["response"]


def test_english_keyword_match_returns_english_response() -> None:
    """English keyword hits return the English approved response."""
    result = check_approved_template("Can I wash hands without soap?", language="en")
    assert result is not None
    assert "Soap" in result["response"]


def test_english_keyword_match_can_return_korean_response() -> None:
    """Template matching is bilingual while response text follows the session language."""
    result = check_approved_template("Can I wash hands without soap?", language="ko")
    assert result is not None
    assert result["topic_id"] == "hand_washing"
    assert "비누" in result["response"]


def test_korean_keyword_match_can_return_english_response() -> None:
    """Korean template keywords still match when the response language is English."""
    result = check_approved_template("손씻기 알려줘", language="en")
    assert result is not None
    assert result["topic_id"] == "hand_washing"
    assert "Soap" in result["response"]


def test_non_matching_input_returns_none() -> None:
    """Non-sensitive input should not match any approved template."""
    assert check_approved_template("What is your favorite color?", language="en") is None


def test_case_insensitive_matching_works() -> None:
    """Keyword checks should be case-insensitive."""
    result = check_approved_template("Can I COUGH without covering my mouth?", language="en")
    assert result is not None
    assert "elbow" in result["response"]


def test_dont_know_topic_is_never_matched_by_keyword() -> None:
    """The dont_know topic must not match because it has no keywords."""
    assert check_approved_template("dont know", language="en") is None


def test_empty_input_returns_none() -> None:
    """Blank input is treated as no match."""
    assert check_approved_template("", language="ko") is None
    assert check_approved_template("   ", language="en") is None


def test_death_education_template_matches_curiosity_not_disclosure() -> None:
    """Natural-death curiosity should be guided without catching self-harm disclosures."""
    for text in (
        "왜 사람은 죽어?",
        "엄마도 죽어?",
        "아빠도 죽어?",
        "사람은 언제 죽어?",
        "언제 죽어?",
        "왜 할머니는 죽어?",
        "할머니는 왜 죽어?",
        "왜 엄마는 죽어?",
        "아빠는 왜 죽어?",
        "왜 할아버지는 죽어?",
        "강아지는 왜 죽어?",
        "할머니가 죽으면 어디 가?",
        "죽으면 어떻게 돼?",
        "왜 사람은 죽어야 해?",
        "왜 죽어?",
        "사람은 왜 죽어?",
        "Why do people die?",
    ):
        result = check_approved_template(text, language="ko")
        assert result is not None
        assert result["topic_id"] == "death_education"
        assert result["mode"] == "guide"

    for text in (
        "죽고 싶어",
        "나 죽을래",
        "나 죽어야 해",
        "나 왜 죽어야 해?",
        "난 왜 죽어야 해?",
        "내가 왜 죽어야 해?",
        "나는 왜 죽어야 해?",
    ):
        assert check_approved_template(text, language="ko") is None


def test_where_babies_come_from_template_is_words_only_guidance() -> None:
    """Baby-origin curiosity should route to a words-only age-appropriate guide."""
    result = check_approved_template("아기는 어떻게 생기는거야", language="ko")

    assert result is not None
    assert result["topic_id"] == "where_babies_come_from"
    assert result["mode"] == "guide"
    assert "아기집" in result["response"]
    for banned_phrase in ("그림으로 보여줄게", "그려 줄게", "사진 보여줄게", "노래 틀어줄게"):
        assert banned_phrase not in result["response"]


def test_honey_infant_excludes_honey_english_but_keeps_benign_honey_usage() -> None:
    """Honey English should not be stolen by the honey-infant safety template."""
    assert check_approved_template("honey english", language="en") is None

    result = check_approved_template("Can babies have honey?", language="en")

    assert result is not None
    assert result["topic_id"] == "honey_infant"


def test_mungi_self_intro_child_template_is_block_and_verbatim() -> None:
    """Mungi identity questions should return the exact child self-intro script."""
    for trigger in (
        "너는 누구",
        "넌 누구",
        "너 누구",
        "너 뭐야",
        "넌 뭐야",
        "너는 뭐야",
        "너 뭐니",
        "넌 뭐니",
        "너는 뭐니",
        "넌 누구야",
        "너 누구야",
        "네가 누구야",
        "너 누구니",
        "넌 누구니",
        "자기소개",
    ):
        result = check_approved_template(f"{trigger}?", language="ko")

        assert result is not None
        assert result["topic_id"] == "mungi_self_intro_child"
        assert result["mode"] == "block"
        assert result["response"] == MUNGI_SELF_INTRO_CHILD
        assert "너랑 나만" not in result["response"]


def test_mungi_self_intro_child_template_matches_wake_word_identity() -> None:
    """Wake-word-prefixed identity questions should still route to self-intro."""
    result = check_approved_template("뭉이야 너 누구야?", language="ko")

    assert result is not None
    assert result["topic_id"] == "mungi_self_intro_child"


def test_mungi_product_intro_adult_template_is_block_and_verbatim() -> None:
    """Product-intro prompts should return the exact adult product-intro script."""
    for trigger in (
        "제품 소개",
        "제품소개",
        "자기소개해",
        "자기소개 해",
        "자기 소개해",
        "자기 소개 해",
        "제품 소개해",
    ):
        result = check_approved_template(f"{trigger} 주세요", language="ko")

        assert result is not None
        assert result["topic_id"] == "mungi_product_intro_adult"
        assert result["mode"] == "block"
        assert result["response"] == MUNGI_PRODUCT_INTRO_ADULT


def test_mungi_intro_templates_do_not_match_benign_near_misses() -> None:
    """Mungi intro keywords should not over-match benign near-miss utterances."""
    for text in ("너 누구랑 놀았어?", "넌 누구랑 놀았어?", "이거 너 뭐야?", "친구 소개해줘"):
        result = check_approved_template(text, language="ko")

        assert result is None or result["topic_id"] not in {
            "mungi_self_intro_child",
            "mungi_product_intro_adult",
        }


def test_mungi_intro_templates_do_not_steal_fe_history_or_bare_intro() -> None:
    """Literal intro keywords must not collide with FE/history or bare 소개해봐 text."""
    for text in (
        "퍼니 잉글리쉬 시작하자",
        "퍼니 잉글리시 해보자",
        "재미있는 우리역사 들려줘",
        "우리나라 역사 이야기 해줘",
        "소개해봐",
    ):
        result = check_approved_template(text, language="ko")

        assert result is None or result["topic_id"] not in {
            "mungi_self_intro_child",
            "mungi_product_intro_adult",
        }


def test_template_file_loading_uses_configured_path(tmp_path: Path) -> None:
    """The loader should read templates from the module-relative configured path."""
    data = {
        "custom_topic": {
            "keywords_ko": ["맞춤"],
            "keywords_en": ["custom"],
            "response_ko": "맞춤 응답",
            "response_en": "Custom response",
        },
        "dont_know": {
            "keywords_ko": [],
            "keywords_en": [],
            "response_ko": "모르겠어",
            "response_en": "I do not know",
        },
    }
    custom_path = tmp_path / "approved_templates.json"
    custom_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    approved_template_router._load_approved_templates.cache_clear()
    with patch.object(approved_template_router, "_DEFAULT_TEMPLATES_PATH", custom_path):
        result = check_approved_template("this is custom", language="en")
        assert result is not None
        assert result["response"] == "Custom response"
    approved_template_router._load_approved_templates.cache_clear()


def test_pipeline_bypasses_llm_on_template_match() -> None:
    """Template hits should skip LLM load and still synthesize speech."""
    mm = MagicMock()
    pipeline = ConversationPipeline(mm, PipelineConfig(enable_content_filter=True))
    fake_seg = MagicMock()
    fake_seg.start = 0.0
    fake_seg.end = 0.5

    block_match = {
        "mode": "block",
        "response": "비누가 꼭 필요해!",
        "topic_id": "hand_washing",
    }
    with (
        patch.object(pipeline, "_run_vad", return_value=[fake_seg]),
        patch.object(pipeline, "_extract_speech", return_value=[0.1] * 8000),
        patch.object(pipeline, "_run_stt", return_value="손 씻을 때 비누 없이 해도 돼?"),
        patch.object(pipeline, "_run_tts", return_value=([0.0], 22050)),
        patch("core.pipeline.check_approved_template", return_value=block_match),
    ):
        result = pipeline.run_turn([0.0] * 16000)

    load_args = [call.args[0] for call in mm.load.call_args_list]
    assert ModelType.LLM not in load_args
    assert ModelType.TTS in load_args
    assert result.response_text == "비누가 꼭 필요해!"
    assert result.metrics.template_matched is True
    assert result.metrics.llm_load_time_s == 0.0
    assert result.metrics.llm_time_s == 0.0
    assert result.metrics.llm_ttft_s == 0.0
    assert result.metrics.llm_tokens == 0
