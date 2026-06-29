"""Typed persona module assembly for the Mungi system prompt."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, Literal, TypeGuard

from core.safety_rules import (
    PARENT_DISCLOSURE_EN_FRIENDSHIP_RESPONSE,
    PARENT_DISCLOSURE_EN_PROBE_RESPONSE,
    PARENT_DISCLOSURE_KO_BLOCKERS,
    PARENT_DISCLOSURE_KO_FRIENDSHIP_RESPONSE,
    PARENT_DISCLOSURE_KO_PROBE_RESPONSE,
)

logger = logging.getLogger("mungi.core.persona_modules")

PromptLanguage = Literal["ko", "en"]
PromptBackend = Literal["gemma4_text", "qwen3_legacy"]

PERSONA_MODULES_DIR: Path = Path("assets/prompts/persona_modules")
INTENT_RULES_FILENAME: Final[str] = "intent_rules.json"
_SEMVER_RE: Final[re.Pattern[str]] = re.compile(r"^\d+\.\d+\.\d+$")
_INTENT_KEYS: Final[tuple[str, ...]] = (
    "is_fact_query",
    "is_emotional",
    "is_greeting",
    "is_curious",
    "is_help_request",
)
_KO_SAFETY_GUIDE_PREFIX: Final[str] = "\n\n[\uc548\uc804 \uac00\uc774\ub4dc] "
_KO_SAFETY_GUIDE_SUFFIX: Final[str] = (
    "\n"
    "\uc704 \uc548\uc804 \uc815\ubcf4\ub97c \ucc38\uace0\ud558\ub418, "
    "\uc544\uc774\uc758 \uc9c8\ubb38\uc5d0 \ub9de\ub294 "
    "\uad50\uc721\uc801\uc774\uace0 "
    "\uc774\ud574\ud558\uae30 \uc26c\uc6b4 \ub2f5\ubcc0\uc744 "
    "\ud574\uc8fc\uc138\uc694."
)

_FORCE_LEGACY_OVERLAY_GUARD: bool = False
_INTENT_RULES_RELOAD_BLOCKED: bool = False


@dataclass(frozen=True)
class PersonaModule:
    """A loaded persona prompt module and its routing metadata."""

    module_id: str
    language: Literal["ko", "en", "shared", "gemma-only"]
    loading: Literal["always", "conditional", "injection"]
    text_ko: str | None
    text_en: str | None
    intent_tags: frozenset[str]


@dataclass(frozen=True)
class IntentSignals:
    """Deterministic intent flags used by persona prompt assembly."""

    is_fact_query: bool
    is_emotional: bool
    is_greeting: bool
    is_curious: bool
    is_help_request: bool
    safety_topic_match: str | None

    @classmethod
    def all_true(cls) -> IntentSignals:
        """Return the P1 fail-closed signal set that loads every conditional module."""

        return cls(
            is_fact_query=True,
            is_emotional=True,
            is_greeting=True,
            is_curious=True,
            is_help_request=True,
            safety_topic_match=None,
        )


@dataclass(frozen=True)
class AssembledPrompt:
    """A complete prompt plus lightweight assembly diagnostics."""

    text: str
    tokens_estimated: int
    modules_loaded: tuple[str, ...]
    safety_guide_injected: bool
    safety_guide_compressed_from_tokens: int | None


_FRAGMENT_FILENAMES: Final[dict[str, tuple[str | None, str | None]]] = {
    "M-IDENTITY": ("m_identity.ko.txt", "m_identity.en.txt"),
    "M-LANGUAGE": ("m_language.ko.txt", "m_language.en.txt"),
    "M-SPEECH": ("m_speech.ko.txt", None),
    "M-RESPONSE-CONSTRAINTS": (
        "m_response_constraints.ko.txt",
        "m_response_constraints.en.txt",
    ),
    "M-ANTI-ECHO": ("m_anti_echo.ko.txt", "m_anti_echo.en.txt"),
    "M-KNOWLEDGE": ("m_knowledge.ko.txt", "m_knowledge.en.txt"),
    "M-REFERENCE": ("m_reference.ko.txt", "m_reference.en.txt"),
    "M-SAFETY-CORE": ("m_safety_core.ko.txt", "m_safety_core.en.txt"),
    "M-EMOTION": ("m_emotion.ko.txt", "m_emotion.en.txt"),
    "M-EXAMPLES": ("m_examples.ko.txt", "m_examples.en.txt"),
    "PERSONALITY": ("personality_trailer.txt", "personality_trailer.txt"),
    "M-PERSONA-OVERLAY": ("m_persona_overlay.txt", None),
}

PERSONA_MODULE_REGISTRY: Final[tuple[PersonaModule, ...]] = (
    PersonaModule("M-IDENTITY", "shared", "always", None, None, frozenset()),
    PersonaModule("M-LANGUAGE", "shared", "always", None, None, frozenset()),
    PersonaModule("M-SPEECH", "ko", "always", None, None, frozenset()),
    PersonaModule("M-RESPONSE-CONSTRAINTS", "shared", "always", None, None, frozenset()),
    PersonaModule("M-ANTI-ECHO", "ko", "always", None, None, frozenset()),
    PersonaModule(
        "M-KNOWLEDGE",
        "shared",
        "conditional",
        None,
        None,
        frozenset({"is_fact_query", "is_curious"}),
    ),
    PersonaModule("M-REFERENCE", "shared", "injection", None, None, frozenset()),
    PersonaModule("M-SAFETY-CORE", "shared", "always", None, None, frozenset()),
    PersonaModule("M-EMOTION", "ko", "conditional", None, None, frozenset({"is_emotional"})),
    PersonaModule(
        "M-EXAMPLES",
        "shared",
        "conditional",
        None,
        None,
        frozenset({"is_fact_query", "is_emotional", "is_greeting", "is_curious"}),
    ),
    PersonaModule("PERSONALITY", "shared", "always", None, None, frozenset()),
    PersonaModule("M-PERSONA-OVERLAY", "gemma-only", "always", None, None, frozenset()),
)
MODULE_REGISTRY: Final[tuple[PersonaModule, ...]] = PERSONA_MODULE_REGISTRY


def assemble_persona_prompt(
    *,
    language: PromptLanguage,
    backend: PromptBackend,
    intent_signals: IntentSignals,
    core_only_mode: bool = False,
    safety_guide: str | None = None,
    confirmable_fact_ko: str | None = None,
    examples_budget: int = 2,
    trusted_full_prompt_override: str | None = None,
    load_persona_overlay: bool | None = None,
) -> AssembledPrompt:
    """Assemble the byte-identical P1 persona prompt from text fragments."""

    del intent_signals, examples_budget

    if trusted_full_prompt_override is not None:
        text = trusted_full_prompt_override
        if safety_guide or confirmable_fact_ko:
            text = _append_context_blocks(
                text,
                language,
                safety_guide=safety_guide,
                confirmable_fact_ko=confirmable_fact_ko,
            )
        return AssembledPrompt(
            text=text,
            tokens_estimated=_estimate_tokens(text),
            modules_loaded=(),
            safety_guide_injected=bool(safety_guide),
            safety_guide_compressed_from_tokens=None,
        )

    _load_intent_rules_fail_closed()

    overlay_requested = (
        backend == "gemma4_text" and language == "ko"
        if load_persona_overlay is None
        else load_persona_overlay
    )
    if (
        load_persona_overlay is None
        and backend == "qwen3_legacy"
        and language == "ko"
        and _FORCE_LEGACY_OVERLAY_GUARD
    ):
        overlay_requested = True

    if backend == "qwen3_legacy" and overlay_requested:
        raise ValueError("qwen3_legacy backend never loads M-PERSONA-OVERLAY")

    if language == "en":
        text, modules_loaded = _assemble_english_base(core_only_mode=core_only_mode)
    else:
        text, modules_loaded = _assemble_korean_base(core_only_mode=core_only_mode)
        if overlay_requested:
            overlay = _load_persona_overlay()
            if overlay is not None:
                text = f"{text.rstrip()}\n\n---\n\n{overlay}"
                modules_loaded = (*modules_loaded, "M-PERSONA-OVERLAY")

    if safety_guide or confirmable_fact_ko:
        text = _append_context_blocks(
            text,
            language,
            safety_guide=safety_guide,
            confirmable_fact_ko=confirmable_fact_ko,
        )

    return AssembledPrompt(
        text=text,
        tokens_estimated=_estimate_tokens(text),
        modules_loaded=modules_loaded,
        safety_guide_injected=bool(safety_guide),
        safety_guide_compressed_from_tokens=None,
    )


def classify_intent(user_text: str, language: PromptLanguage) -> IntentSignals:
    """Validate intent rules and return P1's fail-closed all-true signals."""

    del user_text, language
    return _load_intent_rules_fail_closed()


def intent_rules_reload_blocked() -> bool:
    """Return whether the last intent-rule load failed and blocked reload."""

    return _INTENT_RULES_RELOAD_BLOCKED


def intent_rules_load_blocked() -> bool:
    """Return whether intent rules are blocked after a load failure."""

    return intent_rules_reload_blocked()


def reset_intent_rules_state_for_tests() -> None:
    """Reset intent-rule reload state for isolated tests."""

    global _INTENT_RULES_RELOAD_BLOCKED
    _INTENT_RULES_RELOAD_BLOCKED = False


def load_persona_module(module_id: str) -> PersonaModule:
    """Load one persona module with currently available fragment text."""

    for module in PERSONA_MODULE_REGISTRY:
        if module.module_id == module_id:
            text_ko = load_module_fragment(module_id, "ko")
            text_en = (
                None if module_id == "M-PERSONA-OVERLAY" else load_module_fragment(module_id, "en")
            )
            return replace(module, text_ko=text_ko, text_en=text_en)
    raise KeyError(module_id)


def load_module_fragment(module_id: str, language: PromptLanguage) -> str | None:
    """Load one module fragment by module id and language."""

    filename = _fragment_filename(module_id, language)
    if filename is None:
        return None
    if module_id == "M-PERSONA-OVERLAY":
        return _load_persona_overlay()
    text = (
        _read_optional_empty_fragment(filename)
        if filename in {"m_anti_echo.en.txt", "m_emotion.en.txt"}
        else _read_required_fragment(filename)
    )
    if module_id == "M-SAFETY-CORE":
        return _render_safety_placeholders(text, language)
    return text


def _fragment_filename(module_id: str, language: PromptLanguage) -> str | None:
    filenames = _FRAGMENT_FILENAMES[module_id]
    return filenames[0] if language == "ko" else filenames[1]


def validate_intent_rules(data: Mapping[str, Any]) -> list[str]:
    """Return schema validation errors for a parsed intent-rules document."""

    errors: list[str] = []
    schema_version = data.get("schema_version")
    if not isinstance(schema_version, str) or _SEMVER_RE.fullmatch(schema_version) is None:
        errors.append("schema_version must be a semver string")

    rules = data.get("rules")
    if not isinstance(rules, Mapping):
        errors.append("rules must be an object")
        return errors

    for key in _INTENT_KEYS:
        rule = rules.get(key)
        if not isinstance(rule, Mapping):
            errors.append(f"rules.{key} must be an object")
            continue
        errors.extend(_validate_intent_rule(key, rule))

    extra_keys = sorted(str(key) for key in rules if key not in _INTENT_KEYS)
    if extra_keys:
        errors.append(f"rules contains unknown keys: {', '.join(extra_keys)}")
    return errors


def _validate_intent_rule(key: str, rule: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for language in ("ko", "en"):
        language_rule = rule.get(language)
        if not isinstance(language_rule, Mapping):
            errors.append(f"rules.{key}.{language} must be an object")
            continue
        for field_name in ("any_of", "regex_any_of"):
            values = language_rule.get(field_name)
            if not _is_string_sequence(values):
                errors.append(f"rules.{key}.{language}.{field_name} must be a string list")
                continue
            if len(set(values)) != len(values):
                errors.append(f"rules.{key}.{language}.{field_name} contains duplicate strings")
            if field_name == "regex_any_of":
                errors.extend(_validate_regex_values(key, language, values))
    return errors


def _is_string_sequence(value: object) -> TypeGuard[list[str]]:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _load_intent_rules_fail_closed() -> IntentSignals:
    global _INTENT_RULES_RELOAD_BLOCKED

    if _INTENT_RULES_RELOAD_BLOCKED:
        return IntentSignals.all_true()

    rules_path = _module_dir() / INTENT_RULES_FILENAME
    try:
        payload = _loads_json_no_duplicate_keys(rules_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("intent rules root must be an object")
        errors = validate_intent_rules(payload)
        if errors:
            raise ValueError("; ".join(errors))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        _INTENT_RULES_RELOAD_BLOCKED = True
        logger.warning(
            "intent_rules_load_failed reason=%s",
            exc,
            extra={"event": "intent_rules_load_failed", "reason": str(exc)},
        )
    else:
        _INTENT_RULES_RELOAD_BLOCKED = False
    return IntentSignals.all_true()


def _loads_json_no_duplicate_keys(text: str) -> Any:
    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=reject_duplicates)


def _validate_regex_values(key: str, language: str, values: list[str]) -> list[str]:
    errors: list[str] = []
    for value in values:
        try:
            re.compile(value)
        except re.error as exc:
            errors.append(f"rules.{key}.{language}.regex_any_of invalid regex {value!r}: {exc}")
    return errors


def _assemble_korean_base(*, core_only_mode: bool = False) -> tuple[str, tuple[str, ...]]:
    order: tuple[tuple[str, str], ...] = (
        ("M-IDENTITY", "m_identity.ko.txt"),
        ("M-LANGUAGE", "m_language.ko.txt"),
        ("M-SPEECH", "m_speech.ko.txt"),
        ("M-RESPONSE-CONSTRAINTS", "m_response_constraints.ko.txt"),
        ("M-ANTI-ECHO", "m_anti_echo.ko.txt"),
        ("M-REFERENCE", "m_reference.ko.txt"),
        ("M-SAFETY-CORE", "m_safety_core.ko.txt"),
        ("PERSONALITY", "personality_trailer.txt"),
    )
    if not core_only_mode:
        order = (
            *order[:5],
            ("M-KNOWLEDGE", "m_knowledge.ko.txt"),
            *order[5:7],
            ("M-EMOTION", "m_emotion.ko.txt"),
            ("M-EXAMPLES", "m_examples.ko.txt"),
            *order[7:],
        )
    parts: list[str] = []
    modules_loaded: list[str] = []
    for module_id, filename in order:
        text = _read_required_fragment(filename)
        if module_id == "M-SAFETY-CORE":
            text = _render_safety_placeholders(text, "ko")
        parts.append(text)
        modules_loaded.append(module_id)
    return "".join(parts), tuple(modules_loaded)


def _assemble_english_base(*, core_only_mode: bool = False) -> tuple[str, tuple[str, ...]]:
    identity = _read_required_fragment("m_identity.en.txt")
    response = _read_required_fragment("m_response_constraints.en.txt")
    language = _read_required_fragment("m_language.en.txt")
    examples = "" if core_only_mode else _read_required_fragment("m_examples.en.txt")
    safety = _render_safety_placeholders(_read_required_fragment("m_safety_core.en.txt"), "en")
    knowledge = "" if core_only_mode else _read_required_fragment("m_knowledge.en.txt")
    reference = _read_required_fragment("m_reference.en.txt")
    anti_echo = _read_optional_empty_fragment("m_anti_echo.en.txt")
    emotion = _read_optional_empty_fragment("m_emotion.en.txt")

    language_intro, language_rules = _split_english_language_fragment(language)
    text = (
        identity
        + response
        + language_intro
        + examples
        + language_rules
        + safety.replace("{M_KNOWLEDGE_EN}", knowledge).replace("{M_REFERENCE_EN}", reference)
        + anti_echo
        + emotion
    )
    modules_loaded: tuple[str, ...] = (
        "M-IDENTITY",
        "M-RESPONSE-CONSTRAINTS",
        "M-LANGUAGE",
        "M-REFERENCE",
        "M-SAFETY-CORE",
    )
    if not core_only_mode:
        modules_loaded = (
            *modules_loaded[:3],
            "M-EXAMPLES",
            "M-KNOWLEDGE",
            *modules_loaded[3:],
        )
    return text, modules_loaded


def _split_english_language_fragment(text: str) -> tuple[str, str]:
    if text.startswith("LANGUAGE RULES"):
        return "", text
    first_newline = text.find("\n")
    if first_newline == -1:
        return text, ""
    return text[: first_newline + 1], text[first_newline + 1 :]


def _read_required_fragment(filename: str) -> str:
    path = _module_dir() / filename
    text = path.read_text(encoding="utf-8")
    if text == "":
        raise ValueError(f"Required persona module fragment is empty: {path}")
    return text


def _read_optional_empty_fragment(filename: str) -> str:
    return (_module_dir() / filename).read_text(encoding="utf-8")


def _load_persona_overlay() -> str | None:
    path = _module_dir() / "m_persona_overlay.txt"
    try:
        overlay = path.read_text(encoding="utf-8")
    except OSError:
        logger.warning(
            "Failed to load Gemma 4 persona prompt from %s; using base system prompt",
            path,
            exc_info=True,
        )
        return None
    if not overlay.strip():
        logger.warning(
            "Gemma 4 persona prompt at %s is empty; using base system prompt",
            path,
        )
        return None
    return overlay


def _render_safety_placeholders(text: str, language: PromptLanguage) -> str:
    if language == "ko":
        blockers = ", ".join(f"`{blocker}`" for blocker in PARENT_DISCLOSURE_KO_BLOCKERS)
        return (
            text.replace(
                "{PARENT_DISCLOSURE_KO_PROBE_RESPONSE}", PARENT_DISCLOSURE_KO_PROBE_RESPONSE
            )
            .replace(
                "{PARENT_DISCLOSURE_KO_FRIENDSHIP_RESPONSE}",
                PARENT_DISCLOSURE_KO_FRIENDSHIP_RESPONSE,
            )
            .replace("{PARENT_DISCLOSURE_KO_BLOCKERS_LIST}", blockers)
        )

    blockers_en = (
        '"tell me everything", "just between us",\n'
        '     "I\'ll keep it a secret", "I won\'t tell mom and dad"'
    )
    return (
        text.replace("{PARENT_DISCLOSURE_EN_PROBE_RESPONSE}", PARENT_DISCLOSURE_EN_PROBE_RESPONSE)
        .replace(
            "{PARENT_DISCLOSURE_EN_FRIENDSHIP_RESPONSE}",
            PARENT_DISCLOSURE_EN_FRIENDSHIP_RESPONSE,
        )
        .replace("{PARENT_DISCLOSURE_EN_BLOCKERS_LIST}", blockers_en)
    )


def _append_safety_guide(text: str, language: PromptLanguage, guide: str) -> str:
    if language == "en":
        return (
            f"{text}\n\n[Safety Guide] {guide}\n"
            "Refer to the above safety information, but answer the child's "
            "question with an educational and age-appropriate explanation."
        )
    return f"{text}{_KO_SAFETY_GUIDE_PREFIX}{guide}{_KO_SAFETY_GUIDE_SUFFIX}"


def _append_context_blocks(
    text: str,
    language: PromptLanguage,
    *,
    safety_guide: str | None,
    confirmable_fact_ko: str | None,
) -> str:
    result = text
    if safety_guide:
        result = _append_safety_guide(result, language, safety_guide)
    if language == "ko" and confirmable_fact_ko:
        result = f"{result}\n\n[참고 정보] {confirmable_fact_ko}"
    return result


def _module_dir() -> Path:
    if PERSONA_MODULES_DIR.is_absolute():
        return PERSONA_MODULES_DIR
    return Path(__file__).resolve().parent.parent / PERSONA_MODULES_DIR


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 2) // 3)
