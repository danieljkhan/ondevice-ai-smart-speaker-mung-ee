"""Character expression values for Phase 2 renderers."""

from __future__ import annotations

from enum import Enum


class CharacterExpression(Enum):
    """Phase 2 B-2 character expression set (extended).

    NEUTRAL is the Phase 1 stub and remains as the fallback when a
    requested expression asset is missing. The next eight members (IDLE,
    LISTENING, THINKING, SPEAKING, HAPPY, SAD, SURPRISED, CONCERNED) are
    Phase 2 B-2 v1 additions consumed by PygameCharacterRenderer.
    SPEAKING and the emotion expressions are emitted at TTS playback start
    through the pipeline expression sink. SURPRISED is state-driven for
    WAKING and may also be selected by the content classifier.

    The remaining ten members (JOYFUL, GREETING, EXCITED, ANGRY, SULKY,
    SLEEPY, TIRED, SHY, WINKING, AFFECTIONATE) are Phase 2 B-2 v1-followup
    additions for the external illustrator asset set. They have no
    automatic emit site beyond the content-driven classifier and explicit
    SessionManager.set_expression() calls.

    Future-consumer guidance for sentiment-hook authors (HAPPY vs JOYFUL
    disambiguation, Plan v4 §6.2):
    - HAPPY = deep / stable / sustained satisfaction. Default fallback for
      ambiguous "happy" / "content" sentiment labels from LLM.
    - JOYFUL = immediate / activity-bound / transient enjoyment. Map
      "joyful" / "excited" / "fun" sentiment labels here when context is
      activity-success (game win, problem solved, etc.).
    - EXCITED = anticipation / butterflies (distinct from JOYFUL - pre-event
      vs post-event). Map "anticipation" / "looking forward" labels here.
    - GREETING = social hello / waving. Map session-start / arrival labels.
    - AFFECTIONATE = deep emotional bond. Map "love" / "warmth" labels;
      distinct from SHY (proximity-related modesty) and from HAPPY
      (general contentment).
    """

    NEUTRAL = "neutral"
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    HAPPY = "happy"
    SAD = "sad"
    SURPRISED = "surprised"
    CONCERNED = "concerned"
    JOYFUL = "joyful"
    GREETING = "greeting"
    EXCITED = "excited"
    ANGRY = "angry"
    SULKY = "sulky"
    SLEEPY = "sleepy"
    TIRED = "tired"
    SHY = "shy"
    WINKING = "winking"
    AFFECTIONATE = "affectionate"
