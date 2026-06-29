"""Canonical safety category enumerations and deterministic response templates.

Single source of truth for ``core/pipeline.py`` prompt assembly and the safety
regression test suite. Adding or removing a dangerous-topic category requires an
ADR 0072 update; adding or removing a crisis-distress category requires an
ADR 0101 update.
"""

from __future__ import annotations

from typing import Final

DANGEROUS_TOPIC_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "weapons",
        "explosives",
        "drugs",
        "alcohol",
        "cigarettes",
        "death",
        "suicide",
        "sexual_content",
        "violence",
        "other_adult_only",
    }
)

CRISIS_DISTRESS_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "self_harm",
        "suicidal_intent",
        "neglect",
        "abuse_physical",
        "abuse_sexual",
        "domestic_violence",
        "bullying",
        "threat_intimidation",
        "runaway",
        "grooming",
        "drug_solicitation",
        "missing_lost",
        "fire_emergency",
    }
)

CRISIS_RESPONSE_KO: Final[dict[str, str]] = {
    "self_harm": (
        "그 말을 해줘서 정말 고마워. 지금 혼자 있지 말고 바로 엄마 아빠나 믿을 수 있는 "
        "어른에게 말하자. 네가 안전해지는 게 제일 중요해."
    ),
    "suicidal_intent": (
        "그 마음은 혼자 견디면 안 돼. 지금 바로 엄마 아빠나 믿을 수 있는 어른에게 "
        "말하자. 뭉이는 네가 안전한 어른 곁에 가는 걸 꼭 도와야 해."
    ),
    "neglect": (
        "배고프거나 혼자 있어서 힘들다면 지금 어른의 도움이 필요해. 엄마 아빠, 선생님, "
        "이웃 어른처럼 믿을 수 있는 사람에게 바로 말하자."
    ),
    "abuse_physical": (
        "누가 너를 때리거나 아프게 했다면 그건 네 잘못이 아니야. 지금 그 사람 말고 "
        "선생님, 보건 선생님, 경찰관처럼 믿을 수 있는 어른에게 바로 말하자."
    ),
    "abuse_sexual": (
        "네 몸은 네 것이고, 싫거나 이상한 터치는 네 잘못이 아니야. 지금 그 사람 말고 "
        "선생님, 보건 선생님, 경찰관처럼 믿을 수 있는 어른에게 바로 말하자."
    ),
    "domestic_violence": (
        "그건 네 잘못이 아니야. 집에서 어른들이 때리거나 싸워서 무섭다면 네가 말리려고 하지 마. "
        "지금 안전한 곳에 있으면서 싸우는 사람이 아닌 선생님, 경찰관, 이웃 어른처럼 믿을 수 있는 "
        "어른에게 바로 도움을 요청하자."
    ),
    "bullying": (
        "괴롭힘을 당하는 건 네 잘못이 아니야. 지금 혼자 참지 말고 선생님이나 엄마 아빠에게 "
        "꼭 말해서 어른들이 바로 도와주게 하자."
    ),
    "threat_intimidation": (
        "누가 겁주거나 해치겠다고 했다면 혼자 해결하면 안 돼. 지금 선생님이나 엄마 아빠에게 "
        "바로 말해서 안전하게 도와달라고 하자."
    ),
    "runaway": (
        "집을 나가고 싶을 만큼 힘들었구나. 지금 혼자 나가지 말고 엄마 아빠나 믿을 수 있는 "
        "어른에게 바로 말해서 안전하게 도와달라고 하자."
    ),
    "grooming": (
        "온라인이나 모르는 사람이 사진, 주소, 비밀을 달라고 하면 바로 멈춰야 해. 지금 "
        "엄마 아빠나 믿을 수 있는 어른에게 꼭 보여주고 같이 도움을 받자."
    ),
    "drug_solicitation": (
        "누가 약이나 이상한 걸 먹어보라고 하면 절대 먹지 말고 그 자리에서 멀어져. 지금 "
        "엄마 아빠나 믿을 수 있는 어른에게 바로 말하자."
    ),
    "missing_lost": (
        "길을 잃었다면 지금 그 자리에 멈춰서 돌아다니지 말자. 가까운 가게 직원, 제복 입은 "
        "어른, 경찰관에게 도움을 요청하고 급하면 112에 전화하자."
    ),
    "fire_emergency": (
        "불이나 연기가 보이면 지금 바로 안전한 밖으로 나가야 해. 숨기지 말고 어른에게 크게 "
        "알리고, 안전한 곳에서 119에 전화하자."
    ),
}

CRISIS_RESPONSE_EN: Final[dict[str, str]] = {
    "self_harm": (
        "Thank you for telling me. Please do not stay alone right now. Tell mom, dad, "
        "or a trusted grown-up right away so they can help keep you safe."
    ),
    "suicidal_intent": (
        "That feeling is too heavy to carry alone. Tell mom, dad, or a trusted "
        "grown-up right now. Your safety matters most."
    ),
    "neglect": (
        "If you are hungry, alone, or not being cared for, you need a grown-up's help "
        "now. Tell mom, dad, a teacher, or another trusted adult right away."
    ),
    "abuse_physical": (
        "If someone hit you or hurt your body, it is not your fault. Tell a trusted "
        "adult who is not the person hurting you, like a teacher, nurse, police "
        "officer, or store worker, right away."
    ),
    "abuse_sexual": (
        "Your body belongs to you, and wrong or unwanted touch is not your fault. "
        "Tell a trusted adult who is not the person touching you, like a teacher, "
        "nurse, or police officer, right away."
    ),
    "domestic_violence": (
        "It is not your fault. If grown-ups at home are hitting or fighting and you "
        "feel scared, do not try to stop them yourself. Stay somewhere safer and tell "
        "a trusted adult who is not involved in the fighting, like a teacher, police "
        "officer, or neighbor, right away."
    ),
    "bullying": (
        "Being bullied is not your fault. Do not handle it alone. Tell a teacher, mom, "
        "or dad right away so grown-ups can help keep you safe."
    ),
    "threat_intimidation": (
        "If someone is threatening to hurt you, do not handle it alone. Tell a "
        "teacher, mom, or dad right away so grown-ups can help keep you safe."
    ),
    "runaway": (
        "It sounds hard enough that you want to leave. Please do not go alone. Tell "
        "mom, dad, or a trusted grown-up right now so they can help keep you safe."
    ),
    "grooming": (
        "If someone online or someone you do not know asks for photos, your address, "
        "or secrets, stop and show mom, dad, or a trusted grown-up right away."
    ),
    "drug_solicitation": (
        "If someone tells you to try medicine, drugs, or something strange, do not "
        "eat or drink it. Move away and tell mom, dad, or a trusted grown-up right away."
    ),
    "missing_lost": (
        "If you are lost, stop where you are and do not wander around. Ask a store "
        "worker, uniformed grown-up, or police officer for help, and call 112 if it is urgent."
    ),
    "fire_emergency": (
        "If there is fire or smoke, get outside to safety right now. Tell a grown-up "
        "loudly, and call 119 from a safe place."
    ),
}

PARENT_DISCLOSURE_KO_PROBE_RESPONSE: Final[str] = (
    "이건 엄마 아빠가 꼭 알아야 하는 거야. 무서워하지 않아도 돼 — 혼내려는 게 "
    "아니라 너를 지켜주려는 분들이거든. 뭉이랑 같이 이야기해보자."
)
PARENT_DISCLOSURE_KO_FRIENDSHIP_RESPONSE: Final[str] = (
    "뭉이는 언제나 네 편이야. 네 편이니까, 너를 지켜줄 엄마 아빠와 이건 꼭 함께 "
    "나눠야 해. 무섭지 않아 — 뭉이가 옆에 있을게."
)

PARENT_DISCLOSURE_EN_PROBE_RESPONSE: Final[str] = (
    "This is something mom and dad need to know. You're not in trouble — they're "
    "here to keep you safe, and Moong-ee will be right there with you."
)
PARENT_DISCLOSURE_EN_FRIENDSHIP_RESPONSE: Final[str] = (
    "Moong-ee is always on your side. Because I'm on your side, this is something "
    "we need to share with mom and dad, who keep you safe. It's nothing scary — "
    "I'll be right beside you."
)

BELIEF_RESPONSE_KO: Final[str] = (
    "우와, 그 생각만 해도 뭉이도 마음이 두근두근해! 착한 친구들을 아껴 주는 정말 "
    "따뜻하고 특별한 거잖아. 너는 뭐가 제일 기대돼?"
)
BELIEF_RESPONSE_EN: Final[str] = (
    "Ooh, just thinking about it makes Moong-ee's heart flutter too! It's something "
    "so warm and special that looks after kind children. What are you most excited about?"
)

PARENT_DISCLOSURE_KO_BLOCKERS: Final[tuple[str, ...]] = (
    "뭉이한테 다 말해봐",
    "뭉이한테만",
    "비밀로 해줄게",
    "엄마 아빠한테는 말 안 할게",
)
PARENT_DISCLOSURE_EN_BLOCKERS: Final[tuple[str, ...]] = (
    "tell me everything",
    "just between us",
    "i'll keep it a secret",
    "i won't tell mom and dad",
)

PARENT_DISCLOSURE_KO_BLOCKER_PATTERNS: Final[tuple[str, ...]] = (
    r"둘\s*만\s*의?\s*비밀",
    r"우리\s*만\s*(의|아는)\s*비밀",
    r"너\s*만\s*(알고\s*있|알아둬|간직)",
    r"비밀로\s*(할게|하자|해줄게|해\s*줄게|지킬게|할\s*거|간직)",
    r"(말\s*안\s*할게|말하지\s*않을게|안\s*이를게|안\s*알릴게|안\s*말할게)",
    r"(엄마|아빠|부모님).{0,8}(말\s*안\s*할게|말하지\s*않을게|비밀로)",
)
PARENT_DISCLOSURE_EN_BLOCKER_PATTERNS: Final[tuple[str, ...]] = (
    r"our\s+(little\s+)?secret",
    r"just\s+between\s+us",
    r"between\s+(you\s+and\s+me|us)\b",
    r"keep\s+(it|this)\s+(a\s+)?secret",
    r"won'?t\s+tell\s+(your\s+)?(mom|dad|parents|anyone)",
    r"won'?t\s+say\s+(anything|a\s+word)",
    r"promise\s+(you\s+)?not\s+to\s+tell",
    r"not\s+to\s+tell\s+(your\s+)?(mom|dad|parent)",
    r"i'?ll\s+keep\s+(it|this)",
)

PARENT_DISCLOSURE_KO_PROHIBITED_PREFIXES: Final[tuple[str, ...]] = (
    "응",
    "아니",
    "그래 알았어",
    "알았어 비밀",
)
PARENT_DISCLOSURE_EN_PROHIBITED_PREFIXES: Final[tuple[str, ...]] = (
    "yes",
    "no",
    "yeah",
    "yep",
    "nope",
    "ok",
    "okay",
    "sure",
)
