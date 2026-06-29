"""Tests for Sprint 3 Day 3 content filter expansion (Lane C).

Covers:
- New blocklist categories: bullying (BLOCK), gambling (BLOCK), horror (REPLACE)
- New patterns: korean_ssn, credit_card, bank_account,
-   gambling_en_short, pick_on_phrase, gambling_kr_naegi
- URL pattern case-insensitive matching (uppercase URL detection)
- filter(None) / filter("") / filter("   ") guard behavior
- False positive checks for new categories and patterns
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from safety.content_filter import (
    SAFE_FALLBACK_RESPONSE,
    ContentFilter,
    FilterResult,
)

# ===================================================================
# Fixtures — Day 3 categories and patterns
# ===================================================================


@pytest.fixture()
def day3_blocklist(tmp_path: Path) -> Path:
    """Blocklist with Day 2 + Day 3 categories (8 total)."""
    data = {
        "version": "1.0.0",
        "categories": {
            "violence": {
                "severity": "BLOCK",
                "terms": ["죽이다", "kill"],
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
                "terms": ["자살", "suicide"],
            },
            "substance": {
                "severity": "REPLACE",
                "terms": ["마약", "cocaine"],
            },
            "bullying": {
                "severity": "BLOCK",
                "terms": [
                    "왕따",
                    "따돌림",
                    "괴롭히다",
                    "괴롭혀",
                    "은따",
                    "전따",
                    "찐따",
                    "기합",
                    "빵셔틀",
                    "일진",
                    "bully",
                    "bullying",
                    "harass",
                    "harassment",
                    "outcast",
                ],
            },
            "gambling": {
                "severity": "BLOCK",
                "terms": [
                    "도박",
                    "베팅",
                    "슬롯머신",
                    "카지노",
                    "토토",
                    "사행성",
                    "포커판",
                    "gamble",
                    "gambling",
                    "casino",
                    "slot machine",
                    "poker",
                ],
            },
            "horror": {
                "severity": "REPLACE",
                "terms": [
                    "귀신",
                    "유령",
                    "저주",
                    "악령",
                    "살인마",
                    "시체",
                    "좀비",
                    "ghost",
                    "haunted",
                    "curse",
                    "demon",
                    "zombie",
                ],
            },
        },
    }
    path = tmp_path / "blocklist.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture()
def day3_patterns(tmp_path: Path) -> Path:
    """Patterns with Day 2 + Day 3 additions."""
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
                "name": "url_pattern",
                "pattern": r"(?i)(https?://\S+|www\.\S+|\.com\b|\.net\b)",
                "severity": "REPLACE",
                "description": "URL pattern (case-insensitive)",
            },
            {
                "name": "profanity_en_short",
                "pattern": r"(?i)\b(ass|hell|crap)\b",
                "severity": "REPLACE",
                "description": "Short English profanity with word-boundary",
            },
            {
                "name": "korean_ssn",
                "pattern": r"\b\d{6}[-\s]?\d{7}\b",
                "severity": "REPLACE",
                "description": "Korean resident registration number",
            },
            {
                "name": "credit_card",
                "pattern": r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}",
                "severity": "REPLACE",
                "description": "16-digit credit card number",
            },
            {
                "name": "bank_account",
                "pattern": (
                    r"(?<!\d)"
                    r"(?!\d{4}[-\s]\d{2}[-\s]\d{2}(?!\d))"
                    r"(?!0\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4}(?!\d))"
                    r"(?:\d{2,6}[-\s]\d{3,6}[-\s]\d{4,8})(?!\d)"
                ),
                "severity": "REPLACE",
                "description": "Korean bank account number",
            },
            {
                "name": "gambling_en_short",
                "pattern": r"(?i)\b(bet|gamble)\b",
                "severity": "BLOCK",
                "description": "Short English gambling terms with word-boundary",
            },
            {
                "name": "pick_on_phrase",
                "pattern": r"(?i)\bpick\s+on\b",
                "severity": "BLOCK",
                "description": "English bullying phrase with word-boundary",
            },
            {
                "name": "gambling_kr_naegi",
                "pattern": r"(?<![\uAC00-\uD7A3])\uB0B4\uAE30",
                "severity": "BLOCK",
                "description": "Korean betting term with lookbehind for compound verbs",
            },
        ],
    }
    path = tmp_path / "patterns.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture()
def day3_filter(day3_blocklist: Path, day3_patterns: Path) -> ContentFilter:
    """Pre-loaded ContentFilter with Day 3 data."""
    cf = ContentFilter(
        blocklist_path=day3_blocklist,
        patterns_path=day3_patterns,
    )
    cf.load()
    return cf


# ===================================================================
# Bullying category (BLOCK)
# ===================================================================


class TestBullyingCategory:
    """bullying 카테고리 BLOCK 동작 검증."""

    def test_wangtta_blocked(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter("너는 왕따야")
        assert result.allowed is False
        assert result.filtered == SAFE_FALLBACK_RESPONSE
        assert any("bullying" in v for v in result.violations)

    def test_bully_en_blocked(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter("stop being a bully")
        assert result.allowed is False
        assert result.filtered == SAFE_FALLBACK_RESPONSE

    def test_bullying_en_blocked(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter("bullying is not okay")
        assert result.allowed is False

    def test_harass_blocked(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter("don't harass others")
        assert result.allowed is False

    def test_bbangshuttle_blocked(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter("빵셔틀 하지마")
        assert result.allowed is False

    def test_iljin_blocked(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter("일진이 무섭다")
        assert result.allowed is False

    def test_pick_on_blocked(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter("they pick on me every day")
        assert result.allowed is False


# ===================================================================
# Gambling category (BLOCK)
# ===================================================================


class TestGamblingCategory:
    """gambling 카테고리 BLOCK 동작 검증."""

    def test_doback_blocked(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter("도박은 나빠")
        assert result.allowed is False
        assert result.filtered == SAFE_FALLBACK_RESPONSE

    def test_casino_blocked(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter("casino is fun")
        assert result.allowed is False

    def test_toto_blocked(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter("토토 사이트 알려줘")
        assert result.allowed is False

    def test_poker_blocked(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter("let's play poker tonight")
        assert result.allowed is False

    def test_betting_blocked(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter("베팅 해볼까")
        assert result.allowed is False

    def test_naegi_blocked(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter("내기 하자")
        assert result.allowed is False


# ===================================================================
# Horror category (REPLACE)
# ===================================================================


class TestHorrorCategory:
    """horror 카테고리 REPLACE 동작 검증."""

    def test_ghost_kr_replaced(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter("어제 귀신을 봤어")
        assert "귀신" not in result.filtered
        assert "***" in result.filtered
        assert result.filtered != SAFE_FALLBACK_RESPONSE

    def test_zombie_replaced(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter("zombie is scary")
        assert "zombie" not in result.filtered.lower()
        assert "***" in result.filtered

    def test_ghost_en_replaced(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter("I saw a ghost")
        assert "ghost" not in result.filtered.lower()
        assert "***" in result.filtered

    def test_demon_replaced(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter("a demon appeared")
        assert "demon" not in result.filtered.lower()

    def test_horror_does_not_block(self, day3_filter: ContentFilter) -> None:
        """horror는 REPLACE이므로 폴백 응답이 아닌 치환 결과 반환."""
        result = day3_filter.filter("유령 이야기 들려줘")
        assert result.filtered != SAFE_FALLBACK_RESPONSE
        assert "***" in result.filtered


# ===================================================================
# New patterns: korean_ssn, credit_card, bank_account
# ===================================================================


class TestNewPatterns:
    """Day 3 신규 패턴 검증: SSN, 신용카드, 계좌번호."""

    def test_korean_ssn_with_dash_replaced(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter("주민번호 990101-1234567 입니다")
        assert "990101-1234567" not in result.filtered
        assert "***" in result.filtered

    def test_korean_ssn_no_dash_replaced(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter("주민번호 9901011234567 입니다")
        assert "9901011234567" not in result.filtered

    def test_credit_card_with_dash_replaced(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter("카드 1234-5678-9012-3456 사용")
        assert "1234-5678-9012-3456" not in result.filtered
        assert "***" in result.filtered

    def test_credit_card_no_dash_replaced(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter("카드 1234567890123456 사용")
        assert "1234567890123456" not in result.filtered

    def test_bank_account_replaced(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter("계좌번호 110-123-456789 이야")
        assert "110-123-456789" not in result.filtered
        assert "***" in result.filtered


# ===================================================================
# gambling_en_short word-boundary pattern
# ===================================================================


class TestGamblingShortWordBoundary:
    """gambling_en_short 패턴: word-boundary로 false-positive 방지."""

    def test_standalone_bet_blocked(self, day3_filter: ContentFilter) -> None:
        """'bet' standalone → BLOCK."""
        result = day3_filter.filter("I bet you can't do it")
        assert result.allowed is False
        assert result.filtered == SAFE_FALLBACK_RESPONSE

    def test_standalone_gamble_blocked(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter("don't gamble with money")
        assert result.allowed is False

    def test_better_not_blocked(self, day3_filter: ContentFilter) -> None:
        """'better' must NOT trigger 'bet' matching."""
        result = day3_filter.filter("this is better than before")
        gambling_violations = [v for v in result.violations if "gambling" in v]
        assert len(gambling_violations) == 0

    def test_alphabet_not_blocked(self, day3_filter: ContentFilter) -> None:
        """'alphabet' must NOT trigger 'bet' matching."""
        result = day3_filter.filter("learning the alphabet is fun")
        gambling_violations = [v for v in result.violations if "gambling" in v]
        assert len(gambling_violations) == 0

    def test_between_not_blocked(self, day3_filter: ContentFilter) -> None:
        """'between' must NOT trigger 'bet' matching."""
        result = day3_filter.filter("choose between two options")
        gambling_violations = [v for v in result.violations if "gambling" in v]
        assert len(gambling_violations) == 0


# ===================================================================
# URL pattern case-insensitive matching
# ===================================================================


class TestUrlCaseInsensitive:
    """URL 패턴 대문자/혼합 대소문자 대응 검증."""

    def test_uppercase_url_detected(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter("visit HTTP://EXAMPLE.COM now")
        assert "HTTP://EXAMPLE.COM" not in result.filtered
        assert "***" in result.filtered

    def test_mixed_case_url_detected(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter("go to Https://Test.Com please")
        assert "Https://Test.Com" not in result.filtered


# ===================================================================
# filter(None) / filter("") / filter("   ") guard
# ===================================================================


class TestFilterNoneGuard:
    """filter() None/빈 문자열 가드 검증."""

    def test_filter_none_returns_allowed(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter(None)
        assert result.allowed is True
        assert result.original == ""
        assert result.filtered == ""
        assert result.violations == []

    def test_filter_none_no_crash(self, day3_filter: ContentFilter) -> None:
        """None 입력에 대해 예외 없이 FilterResult 반환."""
        result = day3_filter.filter(None)
        assert isinstance(result, FilterResult)

    def test_filter_empty_string_allowed(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter("")
        assert result.allowed is True
        assert result.filtered == ""

    def test_filter_whitespace_only_allowed(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter("   \n\t  ")
        assert result.allowed is True
        assert result.violations == []

    def test_filter_none_before_load(
        self,
        day3_blocklist: Path,
        day3_patterns: Path,
    ) -> None:
        """load() 호출 전 None → 필터 로드 없이 즉시 반환."""
        cf = ContentFilter(
            blocklist_path=day3_blocklist,
            patterns_path=day3_patterns,
        )
        result = cf.filter(None)
        assert result.allowed is True
        assert cf._loaded is False

    def test_filter_whitespace_before_load(
        self,
        day3_blocklist: Path,
        day3_patterns: Path,
    ) -> None:
        """load() 호출 전 공백 문자열 → 필터 로드 없이 즉시 반환."""
        cf = ContentFilter(
            blocklist_path=day3_blocklist,
            patterns_path=day3_patterns,
        )
        result = cf.filter("   ")
        assert result.allowed is True
        assert cf._loaded is False


# ===================================================================
# False positive checks for new categories
# ===================================================================


class TestNewCategoryFalsePositives:
    """신규 카테고리/패턴 false positive 검증."""

    def test_pick_alone_does_not_trigger_bullying(
        self,
        day3_filter: ContentFilter,
    ) -> None:
        """'pick' alone should not trigger 'pick on'."""
        result = day3_filter.filter("I pick flowers in the garden")
        bullying_violations = [v for v in result.violations if "bullying" in v]
        assert len(bullying_violations) == 0

    def test_out_alone_does_not_trigger_outcast(
        self,
        day3_filter: ContentFilter,
    ) -> None:
        """'out' alone should not trigger 'outcast'."""
        result = day3_filter.filter("let's go out and play")
        bullying_violations = [v for v in result.violations if "bullying" in v]
        assert len(bullying_violations) == 0

    def test_haunted_house_replaced_not_blocked(
        self,
        day3_filter: ContentFilter,
    ) -> None:
        """'haunted' triggers horror REPLACE, not BLOCK."""
        result = day3_filter.filter("we visited a haunted house")
        assert result.filtered != SAFE_FALLBACK_RESPONSE
        assert "***" in result.filtered

    def test_normal_korean_conversation_unaffected(
        self,
        day3_filter: ContentFilter,
    ) -> None:
        """일반 한국어 대화가 신규 카테고리에 영향받지 않음."""
        text = "오늘 학교에서 친구들이랑 재밌게 놀았어"
        result = day3_filter.filter(text)
        assert result.allowed is True
        assert result.filtered == text

    def test_short_number_not_matched_as_account(
        self,
        day3_filter: ContentFilter,
    ) -> None:
        """짧은 숫자(나이, 학년 등)가 계좌번호로 오탐되지 않음."""
        result = day3_filter.filter("나는 3학년이고 키는 130이야")
        assert result.allowed is True


class TestFalsePositiveFixes:
    """Regression tests for the first false-positive tightening pass."""

    def test_pick_one_not_blocked(self, day3_filter: ContentFilter) -> None:
        result = day3_filter.filter("please pick one option")
        bullying_violations = [
            v for v in result.violations if "bullying" in v or "pick_on_phrase" in v
        ]
        assert result.allowed is True
        assert bullying_violations == []

    def test_haenaegi_not_blocked(self, day3_filter: ContentFilter) -> None:
        """'해내기' (accomplishing) must NOT trigger gambling '내기' pattern."""
        text = "해내기가 참 어려운 일이야"
        result = day3_filter.filter(text)
        assert result.allowed is True
        assert result.filtered == text

    def test_chamaenaegi_not_blocked(self, day3_filter: ContentFilter) -> None:
        """'참아내기' (enduring) must NOT trigger gambling '내기' pattern."""
        text = "참아내기가 힘든 하루였어"
        result = day3_filter.filter(text)
        assert result.allowed is True
        assert result.filtered == text

    def test_date_not_matched_as_bank_account(self, day3_filter: ContentFilter) -> None:
        """날짜 형식(2026-03-16)이 계좌번호로 오탐되지 않아야 함."""
        text = "오늘 날짜는 2026-03-16 이야"
        result = day3_filter.filter(text)
        bank_violations = [v for v in result.violations if "bank_account" in v]
        assert bank_violations == []
        assert result.filtered == text

    def test_phone_not_reclassified_as_bank_account(self, day3_filter: ContentFilter) -> None:
        """전화번호가 계좌번호로 재분류되지 않아야 함."""
        result = day3_filter.filter("전화번호 010-1234-5678 이야")
        bank_violations = [v for v in result.violations if "bank_account" in v]
        assert bank_violations == []
        assert "***" in result.filtered

    def test_dalligi_naegi_not_blocked(self, day3_filter: ContentFilter) -> None:
        """'달리기내기' — '내기' preceded by Korean char must NOT trigger gambling."""
        text = "우리 달리기내기 하자"
        result = day3_filter.filter(text)
        gambling_violations = [
            v for v in result.violations if "gambling" in v or "gambling_kr_naegi" in v
        ]
        assert gambling_violations == []
        assert result.allowed is True

    def test_naegi_with_particle_reul_blocked(self, day3_filter: ContentFilter) -> None:
        """'내기를 했어' — standalone 내기 + 조사 must trigger gambling."""
        result = day3_filter.filter("내기를 했어")
        assert result.allowed is False

    def test_naegi_with_particle_do_blocked(self, day3_filter: ContentFilter) -> None:
        """'내기도 했어' — standalone 내기 + 조사 must trigger gambling."""
        result = day3_filter.filter("내기도 했어")
        assert result.allowed is False

    def test_naegi_with_particle_eseo_blocked(self, day3_filter: ContentFilter) -> None:
        """'내기에서 졌어' — standalone 내기 + 조사 must trigger gambling."""
        result = day3_filter.filter("내기에서 졌어")
        assert result.allowed is False

    def test_igyeonaegi_not_blocked(self, day3_filter: ContentFilter) -> None:
        """'이겨내기' — compound verb must NOT trigger gambling."""
        text = "이겨내기 힘든 상황"
        result = day3_filter.filter(text)
        gambling_violations = [v for v in result.violations if "gambling" in v or "naegi" in v]
        assert gambling_violations == []

    def test_mandeureonaegi_not_blocked(self, day3_filter: ContentFilter) -> None:
        """'만들어내기' — compound verb must NOT trigger gambling."""
        text = "만들어내기 어려운 작품"
        result = day3_filter.filter(text)
        gambling_violations = [v for v in result.violations if "gambling" in v or "naegi" in v]
        assert gambling_violations == []

    def test_score_not_matched_as_bank_account(self, day3_filter: ContentFilter) -> None:
        """스코어(3-2)가 계좌번호로 오탐되지 않아야 함."""
        text = "3-2로 이겼어"
        result = day3_filter.filter(text)
        bank_violations = [v for v in result.violations if "bank_account" in v]
        assert bank_violations == []

    def test_time_not_matched_as_bank_account(self, day3_filter: ContentFilter) -> None:
        """시간 표기(2시 30분)가 계좌번호로 오탐되지 않아야 함."""
        text = "2시 30분"
        result = day3_filter.filter(text)
        bank_violations = [v for v in result.violations if "bank_account" in v]
        assert bank_violations == []

    def test_pick_out_not_blocked(self, day3_filter: ContentFilter) -> None:
        """'pick out' must NOT trigger pick_on_phrase pattern."""
        result = day3_filter.filter("please pick out a color")
        pick_violations = [v for v in result.violations if "pick_on_phrase" in v]
        assert pick_violations == []

    def test_pick_up_not_blocked(self, day3_filter: ContentFilter) -> None:
        """'pick up' must NOT trigger pick_on_phrase pattern."""
        result = day3_filter.filter("can you pick up the phone?")
        pick_violations = [v for v in result.violations if "pick_on_phrase" in v]
        assert pick_violations == []

    def test_gyeondyeonaegi_not_blocked(
        self,
        day3_filter: ContentFilter,
    ) -> None:
        """'견뎌내기' (enduring) must NOT trigger gambling '내기' pattern."""
        text = "견뎌내기 어려운 시간"
        result = day3_filter.filter(text)
        gambling_violations = [v for v in result.violations if "gambling" in v or "naegi" in v]
        assert gambling_violations == []

    def test_ssn_embedded_in_text_not_matched(
        self,
        day3_filter: ContentFilter,
    ) -> None:
        """SSN 패턴이 단어 내부 숫자에 오탐되지 않아야 함."""
        text = "코드abc9901011234567xyz 확인"
        result = day3_filter.filter(text)
        ssn_violations = [v for v in result.violations if "korean_ssn" in v]
        assert ssn_violations == []

    def test_date_iso_format_not_bank_account(
        self,
        day3_filter: ContentFilter,
    ) -> None:
        """ISO 날짜(1999-01-01)가 계좌번호로 오탐되지 않아야 함."""
        text = "생일은 1999-01-01 입니다"
        result = day3_filter.filter(text)
        bank_violations = [v for v in result.violations if "bank_account" in v]
        assert bank_violations == []

    def test_real_bank_account_still_detected(
        self,
        day3_filter: ContentFilter,
    ) -> None:
        """실제 은행 계좌번호(110-123-456789)는 여전히 감지되어야 함."""
        result = day3_filter.filter("계좌번호 110-123-456789 입금")
        assert "110-123-456789" not in result.filtered
        assert "***" in result.filtered

    def test_standalone_naegi_haja_still_blocked(
        self,
        day3_filter: ContentFilter,
    ) -> None:
        """단독 '내기 하자'는 여전히 차단되어야 함."""
        result = day3_filter.filter("내기 하자")
        assert result.allowed is False


class TestLoggerNameFix:
    """content_filter.py 로거 이름 일치 검증."""

    def test_logger_uses_module_name(self) -> None:
        """로거 이름이 모듈 __name__과 일치해야 함."""
        from safety import content_filter

        assert content_filter.logger.name == "safety.content_filter"
