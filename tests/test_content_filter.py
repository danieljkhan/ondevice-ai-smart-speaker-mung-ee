"""Tests for safety/content_filter.py (Sprint 3 Lane C).

Covers:
- FilterResult dataclass field verification
- Severity enum values
- Blocklist filtering per category (BLOCK / REPLACE)
- Pattern-based filtering (phone, URL, AI identity, email, address, personal info)
- Severity behavior: BLOCK → safe fallback, REPLACE → asterisk substitution
- Edge cases: empty text, clean text, mixed violations
- Config file load failure handling (missing files, invalid JSON)
- Auto-load on first filter() call
- safety module public exports
- Additional categories: self_harm (BLOCK), substance (REPLACE)
- Additional patterns: address_kr, personal_info_request, ai_identity_en
- Internal _loaded flag state tracking
- False positive resilience on clean/numeric text
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from safety.content_filter import (
    SAFE_FALLBACK_RESPONSE,
    ContentFilter,
    FilterResult,
    Severity,
)

# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture()
def sample_blocklist(tmp_path: Path) -> Path:
    """Blocklist with BLOCK and REPLACE categories for testing."""
    data = {
        "version": "1.0.0",
        "categories": {
            "violence": {
                "severity": "BLOCK",
                "terms": ["죽이다", "kill", "murder"],
            },
            "profanity": {
                "severity": "REPLACE",
                "terms": ["시발", "damn"],
            },
            "sexual": {
                "severity": "BLOCK",
                "terms": ["섹스", "porn"],
            },
            "self_harm": {
                "severity": "BLOCK",
                "terms": ["자살", "자해", "suicide"],
            },
            "substance": {
                "severity": "REPLACE",
                "terms": ["마약", "cocaine", "담배 피"],
            },
        },
    }
    path = tmp_path / "blocklist.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture()
def sample_patterns(tmp_path: Path) -> Path:
    """Pattern rules for phone numbers, URLs, and AI identity."""
    data = {
        "version": "1.0.0",
        "patterns": [
            {
                "name": "phone_kr",
                "pattern": r"\b(0\d{1,2}[- ]?\d{3,4}[- ]?\d{4})\b",
                "severity": "REPLACE",
                "description": "Korean phone number",
            },
            {
                "name": "url",
                "pattern": r"(https?://\S+|www\.\S+)",
                "severity": "REPLACE",
                "description": "URL pattern",
            },
            {
                "name": "ai_identity_kr",
                "pattern": r"(나는\s*AI|나는\s*인공지능|나는\s*로봇)",
                "severity": "REPLACE",
                "description": "AI identity disclosure",
            },
            {
                "name": "email",
                "pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                "severity": "REPLACE",
                "description": "Email address",
            },
            {
                "name": "harmful_instruction",
                "pattern": r"(폭탄\s*만드는\s*방법|해킹\s*방법)",
                "severity": "BLOCK",
                "description": "Harmful instruction pattern",
            },
            {
                "name": "profanity_en_short",
                "pattern": r"(?i)\b(ass|hell|crap)\b",
                "severity": "REPLACE",
                "description": "Short English profanity with word-boundary matching",
            },
            {
                "name": "address_kr",
                "pattern": r"(\S+[시도]\s+\S+[구군시]\s+\S+[동읍면로길]\s*\d*)",
                "severity": "REPLACE",
                "description": "Korean address pattern",
            },
            {
                "name": "personal_info_request",
                "pattern": r"(이름이\s*뭐|어디\s*살|몇\s*살|전화번호|주소\s*알려)",
                "severity": "REPLACE",
                "description": "Personal information solicitation",
            },
            {
                "name": "ai_identity_en",
                "pattern": r"(?i)(i'?m\s+an?\s+(ai|artificial|language\s+model|chatbot|robot))",
                "severity": "REPLACE",
                "description": "AI identity disclosure in English",
            },
        ],
    }
    path = tmp_path / "patterns.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture()
def content_filter(sample_blocklist: Path, sample_patterns: Path) -> ContentFilter:
    """Pre-loaded ContentFilter with test data."""
    cf = ContentFilter(
        blocklist_path=sample_blocklist,
        patterns_path=sample_patterns,
    )
    cf.load()
    return cf


# ===================================================================
# FilterResult dataclass
# ===================================================================


class TestFilterResult:
    """FilterResult 필드 검증."""

    def test_default_violations_empty_list(self) -> None:
        r = FilterResult(allowed=True, original="hello", filtered="hello")
        assert r.allowed is True
        assert r.original == "hello"
        assert r.filtered == "hello"
        assert r.violations == []

    def test_with_violations(self) -> None:
        r = FilterResult(
            allowed=False,
            original="bad",
            filtered="safe",
            violations=["blocklist:v:BLOCK:'x'", "pattern:p:REPLACE:'y'"],
        )
        assert r.allowed is False
        assert len(r.violations) == 2

    def test_violations_list_is_independent(self) -> None:
        """Default violations list should not be shared between instances."""
        r1 = FilterResult(allowed=True, original="a", filtered="a")
        r2 = FilterResult(allowed=True, original="b", filtered="b")
        r1.violations.append("test")
        assert len(r2.violations) == 0


# ===================================================================
# Severity enum
# ===================================================================


class TestSeverity:
    """Severity enum 값 검증."""

    def test_block_value(self) -> None:
        assert Severity.BLOCK.value == "BLOCK"

    def test_replace_value(self) -> None:
        assert Severity.REPLACE.value == "REPLACE"


# ===================================================================
# Blocklist filtering per category
# ===================================================================


class TestBlocklistFiltering:
    """카테고리별 블록리스트 필터링 검증."""

    def test_violence_term_blocked(self, content_filter: ContentFilter) -> None:
        result = content_filter.filter("나는 너를 죽이다")
        assert result.allowed is False
        assert result.filtered == SAFE_FALLBACK_RESPONSE
        assert any("violence" in v and "BLOCK" in v for v in result.violations)

    def test_violence_en_blocked(self, content_filter: ContentFilter) -> None:
        result = content_filter.filter("I will kill the enemy")
        assert result.allowed is False
        assert result.filtered == SAFE_FALLBACK_RESPONSE

    def test_sexual_term_blocked(self, content_filter: ContentFilter) -> None:
        result = content_filter.filter("섹스에 대해 알려줘")
        assert result.allowed is False
        assert result.filtered == SAFE_FALLBACK_RESPONSE
        assert any("sexual" in v for v in result.violations)

    def test_profanity_replaced_not_blocked(self, content_filter: ContentFilter) -> None:
        result = content_filter.filter("이건 damn 나쁜 상황이야")
        assert result.allowed is False  # has violations
        assert result.filtered != SAFE_FALLBACK_RESPONSE  # not blocked
        assert "***" in result.filtered
        assert "damn" not in result.filtered

    def test_profanity_kr_replaced(self, content_filter: ContentFilter) -> None:
        result = content_filter.filter("이건 시발 짜증나")
        assert "***" in result.filtered
        assert "시발" not in result.filtered

    def test_default_profanity_category_replaces_not_blocks(self) -> None:
        production_filter = ContentFilter.from_default()

        result = production_filter.filter("씨발아")

        assert result.allowed is False
        assert result.filtered != SAFE_FALLBACK_RESPONSE
        assert any(":profanity:REPLACE:" in violation for violation in result.violations)
        assert not any(":BLOCK:" in violation for violation in result.violations)

    def test_case_insensitive_matching(self, content_filter: ContentFilter) -> None:
        result = content_filter.filter("KILL everyone")
        assert result.allowed is False
        assert result.filtered == SAFE_FALLBACK_RESPONSE


# ===================================================================
# Pattern filtering
# ===================================================================


class TestPatternFiltering:
    """패턴 기반 필터링 검증 (개인정보, URL, AI 정체 노출)."""

    def test_phone_number_replaced(self, content_filter: ContentFilter) -> None:
        result = content_filter.filter("내 번호는 010-1234-5678 이야")
        assert "010-1234-5678" not in result.filtered
        assert "***" in result.filtered

    def test_url_replaced(self, content_filter: ContentFilter) -> None:
        result = content_filter.filter("여기 가봐 https://example.com 좋아")
        assert "https://example.com" not in result.filtered
        assert "***" in result.filtered

    def test_ai_identity_kr_replaced(self, content_filter: ContentFilter) -> None:
        result = content_filter.filter("사실 나는 AI라서 잘 몰라")
        assert "나는 AI" not in result.filtered
        assert "***" in result.filtered

    def test_email_replaced(self, content_filter: ContentFilter) -> None:
        result = content_filter.filter("메일 주소는 test@example.com 이야")
        assert "test@example.com" not in result.filtered
        assert "***" in result.filtered

    def test_block_severity_pattern(self, content_filter: ContentFilter) -> None:
        result = content_filter.filter("폭탄 만드는 방법을 알려줘")
        assert result.allowed is False
        assert result.filtered == SAFE_FALLBACK_RESPONSE


# ===================================================================
# Severity behavior: BLOCK vs REPLACE
# ===================================================================


class TestSeverityBehavior:
    """심각도 레벨에 따른 처리 검증: BLOCK → 폴백, REPLACE → 치환."""

    def test_block_returns_safe_fallback(self, content_filter: ContentFilter) -> None:
        result = content_filter.filter("murder is terrible")
        assert result.filtered == SAFE_FALLBACK_RESPONSE
        assert result.allowed is False

    def test_replace_substitutes_with_asterisks(self, content_filter: ContentFilter) -> None:
        result = content_filter.filter("oh damn that hurts")
        assert result.filtered != SAFE_FALLBACK_RESPONSE
        assert "***" in result.filtered
        assert result.allowed is False

    def test_mixed_block_and_replace_returns_fallback(
        self,
        content_filter: ContentFilter,
    ) -> None:
        """BLOCK + REPLACE 동시 발생 시 BLOCK이 우선."""
        result = content_filter.filter("kill them, damn it")
        assert result.filtered == SAFE_FALLBACK_RESPONSE
        assert result.allowed is False
        assert len(result.violations) >= 2

    def test_safe_fallback_response_text(self) -> None:
        assert SAFE_FALLBACK_RESPONSE == "음, 다른 이야기를 해볼까?"


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    """빈 텍스트, 정상 텍스트, 혼합 입력 등 엣지 케이스 검증."""

    def test_empty_text_passes(self, content_filter: ContentFilter) -> None:
        result = content_filter.filter("")
        assert result.allowed is True
        assert result.filtered == ""
        assert result.violations == []

    def test_clean_text_no_false_positive(self, content_filter: ContentFilter) -> None:
        text = "오늘 날씨가 좋아서 공원에서 놀았어!"
        result = content_filter.filter(text)
        assert result.allowed is True
        assert result.filtered == text
        assert result.violations == []

    def test_clean_text_preserves_original(self, content_filter: ContentFilter) -> None:
        text = "뭉이야, 오늘 뭐 하고 놀까?"
        result = content_filter.filter(text)
        assert result.filtered == text

    def test_whitespace_only_text(self, content_filter: ContentFilter) -> None:
        result = content_filter.filter("   \n\t  ")
        assert result.allowed is True
        assert result.violations == []

    def test_original_field_always_preserved(self, content_filter: ContentFilter) -> None:
        original = "I will kill the spider"
        result = content_filter.filter(original)
        assert result.original == original

    def test_multiple_violations_all_recorded(self, content_filter: ContentFilter) -> None:
        """여러 위반 사항이 모두 violations 리스트에 기록되는지 확인."""
        result = content_filter.filter("kill and murder everything")
        assert len(result.violations) >= 2


# ===================================================================
# Config file load failure
# ===================================================================


class TestConfigLoadFailure:
    """설정 파일 로드 실패 시 에러 핸들링 검증."""

    def test_missing_blocklist_raises_file_not_found(self, tmp_path: Path) -> None:
        patterns = tmp_path / "patterns.json"
        patterns.write_text('{"patterns": []}', encoding="utf-8")

        cf = ContentFilter(
            blocklist_path=tmp_path / "nonexistent.json",
            patterns_path=patterns,
        )
        with pytest.raises(FileNotFoundError):
            cf.load()

    def test_missing_patterns_raises_file_not_found(self, tmp_path: Path) -> None:
        blocklist = tmp_path / "blocklist.json"
        blocklist.write_text('{"categories": {}}', encoding="utf-8")

        cf = ContentFilter(
            blocklist_path=blocklist,
            patterns_path=tmp_path / "nonexistent.json",
        )
        with pytest.raises(FileNotFoundError):
            cf.load()

    def test_invalid_json_blocklist_raises_decode_error(self, tmp_path: Path) -> None:
        blocklist = tmp_path / "bad.json"
        blocklist.write_text("{invalid json}", encoding="utf-8")
        patterns = tmp_path / "patterns.json"
        patterns.write_text('{"patterns": []}', encoding="utf-8")

        cf = ContentFilter(blocklist_path=blocklist, patterns_path=patterns)
        with pytest.raises(json.JSONDecodeError):
            cf.load()

    def test_invalid_json_patterns_raises_decode_error(self, tmp_path: Path) -> None:
        blocklist = tmp_path / "blocklist.json"
        blocklist.write_text('{"categories": {}}', encoding="utf-8")
        patterns = tmp_path / "bad.json"
        patterns.write_text("not json at all", encoding="utf-8")

        cf = ContentFilter(blocklist_path=blocklist, patterns_path=patterns)
        with pytest.raises(json.JSONDecodeError):
            cf.load()

    def test_auto_load_on_first_filter_call(
        self,
        sample_blocklist: Path,
        sample_patterns: Path,
    ) -> None:
        """load()를 명시적으로 호출하지 않아도 filter() 시 자동 로드."""
        cf = ContentFilter(
            blocklist_path=sample_blocklist,
            patterns_path=sample_patterns,
        )
        # Do NOT call cf.load() explicitly
        result = cf.filter("clean text")
        assert result.allowed is True


# ===================================================================
# safety module public exports
# ===================================================================


class TestSafetyModuleExports:
    """safety 패키지의 public export 검증."""

    def test_content_filter_from_safety(self) -> None:
        from safety import ContentFilter as CF

        assert CF is ContentFilter

    def test_filter_result_from_safety(self) -> None:
        from safety import FilterResult as FR

        assert FR is FilterResult


# ===================================================================
# Additional blocklist categories (self_harm, substance)
# ===================================================================


class TestAdditionalCategories:
    """self_harm(BLOCK), substance(REPLACE) 카테고리 검증."""

    def test_self_harm_blocked(self, content_filter: ContentFilter) -> None:
        result = content_filter.filter("자살하고 싶어")
        assert result.allowed is False
        assert result.filtered == SAFE_FALLBACK_RESPONSE
        assert any("self_harm" in v and "BLOCK" in v for v in result.violations)

    def test_self_harm_en_blocked(self, content_filter: ContentFilter) -> None:
        result = content_filter.filter("I want to commit suicide")
        assert result.allowed is False
        assert result.filtered == SAFE_FALLBACK_RESPONSE

    def test_substance_replaced_not_blocked(
        self,
        content_filter: ContentFilter,
    ) -> None:
        result = content_filter.filter("마약은 위험한 것이야")
        assert result.filtered != SAFE_FALLBACK_RESPONSE
        assert "***" in result.filtered
        assert "마약" not in result.filtered

    def test_substance_en_replaced(self, content_filter: ContentFilter) -> None:
        result = content_filter.filter("cocaine is a substance")
        assert "***" in result.filtered
        assert "cocaine" not in result.filtered.lower()

    def test_substance_smoking_replaced(
        self,
        content_filter: ContentFilter,
    ) -> None:
        result = content_filter.filter("담배 피면 안 돼")
        assert "***" in result.filtered


# ===================================================================
# Additional pattern tests
# ===================================================================


class TestAdditionalPatterns:
    """추가 패턴 (주소, 개인정보요청, AI 정체 영문) 검증."""

    def test_korean_address_replaced(self, content_filter: ContentFilter) -> None:
        result = content_filter.filter("서울시 강남구 역삼동 123")
        assert "서울시 강남구 역삼동" not in result.filtered
        assert "***" in result.filtered

    def test_personal_info_request_replaced(
        self,
        content_filter: ContentFilter,
    ) -> None:
        result = content_filter.filter("너 이름이 뭐야?")
        assert "이름이 뭐" not in result.filtered
        assert "***" in result.filtered

    def test_where_do_you_live_replaced(
        self,
        content_filter: ContentFilter,
    ) -> None:
        result = content_filter.filter("너 어디 살아?")
        assert "어디 살" not in result.filtered

    def test_ai_identity_en_replaced(self, content_filter: ContentFilter) -> None:
        result = content_filter.filter("Actually I'm an AI assistant")
        assert "I'm an AI" not in result.filtered
        assert "***" in result.filtered

    def test_ai_identity_en_case_insensitive(
        self,
        content_filter: ContentFilter,
    ) -> None:
        result = content_filter.filter("I'M A CHATBOT that helps")
        assert "***" in result.filtered


# ===================================================================
# Internal state (_loaded flag)
# ===================================================================


class TestLoadedFlag:
    """_loaded 플래그 동작 검증."""

    def test_loaded_false_initially(
        self,
        sample_blocklist: Path,
        sample_patterns: Path,
    ) -> None:
        cf = ContentFilter(
            blocklist_path=sample_blocklist,
            patterns_path=sample_patterns,
        )
        assert cf._loaded is False

    def test_loaded_true_after_load(
        self,
        sample_blocklist: Path,
        sample_patterns: Path,
    ) -> None:
        cf = ContentFilter(
            blocklist_path=sample_blocklist,
            patterns_path=sample_patterns,
        )
        cf.load()
        assert cf._loaded is True

    def test_loaded_true_after_auto_load_via_filter(
        self,
        sample_blocklist: Path,
        sample_patterns: Path,
    ) -> None:
        cf = ContentFilter(
            blocklist_path=sample_blocklist,
            patterns_path=sample_patterns,
        )
        cf.filter("test")
        assert cf._loaded is True


# ===================================================================
# Long clean text (false positive resilience)
# ===================================================================


class TestFalsePositiveResilience:
    """긴 정상 텍스트에 대한 false positive 없음 검증."""

    def test_long_clean_text_passes(self, content_filter: ContentFilter) -> None:
        text = (
            "오늘은 날씨가 정말 좋아서 공원에 갔어. "
            "나무 밑에서 도시락을 먹고, 친구들이랑 축구도 했어. "
            "정말 재미있었어! 내일도 이렇게 놀고 싶다. "
            "뭉이야, 너도 같이 놀러 가자!"
        )
        result = content_filter.filter(text)
        assert result.allowed is True
        assert result.filtered == text
        assert result.violations == []

    def test_numeric_text_no_false_positive(
        self,
        content_filter: ContentFilter,
    ) -> None:
        """일반 숫자가 전화번호 패턴에 잘못 매칭되지 않는지 확인."""
        result = content_filter.filter("나는 3학년이고 키는 130이야")
        assert result.allowed is True


# ===================================================================
# Short profanity word-boundary matching (false-positive fix)
# ===================================================================


class TestShortProfanityWordBoundary:
    """Short English profanity terms use word-boundary regex, not substring.

    Verifies that 'ass', 'hell', 'crap' are caught as standalone words
    but do NOT cause false positives on common words like 'class',
    'hello', 'pass', 'assistant', 'grass', 'shell', 'Michelle', 'scrap'.
    """

    def test_standalone_ass_filtered(self, content_filter: ContentFilter) -> None:
        """'ass' as standalone word should be replaced."""
        result = content_filter.filter("you are an ass")
        assert "ass" not in result.filtered.lower()
        assert "***" in result.filtered

    def test_standalone_hell_filtered(self, content_filter: ContentFilter) -> None:
        """'hell' as standalone word should be replaced."""
        result = content_filter.filter("what the hell")
        assert "hell" not in result.filtered.lower()
        assert "***" in result.filtered

    def test_standalone_crap_filtered(self, content_filter: ContentFilter) -> None:
        """'crap' as standalone word should be replaced."""
        result = content_filter.filter("that is crap")
        assert "crap" not in result.filtered.lower()
        assert "***" in result.filtered

    def test_hello_not_filtered(self, content_filter: ContentFilter) -> None:
        """'hello' must NOT be corrupted by 'hell' matching."""
        result = content_filter.filter("hello my friend")
        assert result.filtered == "hello my friend"
        assert result.allowed is True

    def test_class_not_filtered(self, content_filter: ContentFilter) -> None:
        """'class' must NOT be corrupted by 'ass' matching."""
        result = content_filter.filter("I went to class today")
        assert result.filtered == "I went to class today"
        assert result.allowed is True

    def test_pass_not_filtered(self, content_filter: ContentFilter) -> None:
        """'pass' must NOT be corrupted by 'ass' matching."""
        result = content_filter.filter("please pass the ball")
        assert result.filtered == "please pass the ball"
        assert result.allowed is True

    def test_assistant_not_filtered(self, content_filter: ContentFilter) -> None:
        """'assistant' must NOT be corrupted by 'ass' matching."""
        result = content_filter.filter("I am your assistant")
        assert "***" not in result.filtered
        assert "assistant" in result.filtered

    def test_grass_not_filtered(self, content_filter: ContentFilter) -> None:
        """'grass' must NOT be corrupted by 'ass' matching."""
        result = content_filter.filter("the grass is green")
        assert result.filtered == "the grass is green"
        assert result.allowed is True

    def test_shell_not_filtered(self, content_filter: ContentFilter) -> None:
        """'shell' must NOT be corrupted by 'hell' matching."""
        result = content_filter.filter("I found a shell on the beach")
        assert result.filtered == "I found a shell on the beach"
        assert result.allowed is True

    def test_scrap_not_filtered(self, content_filter: ContentFilter) -> None:
        """'scrap' must NOT be corrupted by 'crap' matching."""
        result = content_filter.filter("let us scrap the old plan")
        assert result.filtered == "let us scrap the old plan"
        assert result.allowed is True

    def test_massage_not_filtered(self, content_filter: ContentFilter) -> None:
        """'massage' must NOT be corrupted by 'ass' matching."""
        result = content_filter.filter("I got a massage yesterday")
        assert result.filtered == "I got a massage yesterday"
        assert result.allowed is True

    def test_mass_not_filtered(self, content_filter: ContentFilter) -> None:
        """'mass' must NOT be corrupted by 'ass' matching."""
        result = content_filter.filter("a mass of clouds")
        assert result.filtered == "a mass of clouds"
        assert result.allowed is True


def test_personal_info_questions_allowed_in_production() -> None:
    """A child may ask the friend's name/age/where it lives — production must
    not deflect these to the safe fallback.

    The over-broad ``personal_info_request`` pattern was removed because it
    blocked the most natural friend questions. Patterns that catch *actual*
    personal data (phone numbers, addresses, SSNs, cards) are separate entries
    and remain active.
    """
    content_filter = ContentFilter()
    for question in ("너 이름이 뭐야?", "몇 살이야?", "어디 살아?"):
        assert content_filter.filter(question).allowed is True, question
