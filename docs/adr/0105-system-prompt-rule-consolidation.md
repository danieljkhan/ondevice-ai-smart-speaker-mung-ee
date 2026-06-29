# ADR 0105: System Prompt Rule Consolidation

## Status

Accepted

## Context

The live Gemma 4 Korean prompt was assembled from two overlapping rule carriers:

1. `PipelineConfig.llm_system_prompt`, the authoritative inline English child-safety prompt.
2. `assets/prompts/persona.md`, a Korean persona document that also restated many of the same rules.

This duplication made future language and learning-mode overlays harder to apply safely because one rule block could have multiple carriers in different languages.

## Decision

Consolidate runtime behavioral rules into the inline English base prompt and slim `assets/prompts/persona.md` to a Korean residual containing only unique persona, tone, safety-residual, offline-identity, emotion-table, and few-shot material.

The inline base prompt now exposes stable section markers:

`§IDENTITY §LANGUAGE §SPEECH §RESPONSE §ANTI_ECHO §STT §KNOWLEDGE §HARD_TOPIC §SAFETY §EMOTION §PERSONALITY`

`§LANGUAGE` is isolated as a contiguous Korean-path language block: `LANGUAGE PROCESSING RULES` followed by `BILINGUAL MODE RULES`, with no Korean language-rule restatement remaining in `persona.md`.

The Gemma 4 assembly contract is:

```text
final = inline_EN_base + "\n\n---\n\n" + [mode_overlay=""] + persona_md_KO_residual
```

The overlay slot is empty in this implementation. Future mode overlays must sit between the inline English base and the Korean residual so last-instruction-wins overrides can target a labeled block while preserving the residual style anchor.

The one rule that was previously only in `persona.md` is migrated into `§SPEECH`:

```text
- Use ONLY short, simple words a 5-10 year old understands.
```

## Preservation Mapping

| Rule | Current carrier(s) | Carrier after Decision A | Sole carrier was persona.md? |
|---|---|---|---|
| Identity / AI-identity boundary | inline EN `:549,572-575`; `persona.md :43-44` | inline EN `§IDENTITY` | No |
| LANGUAGE KO-only / no-Hanzi-kana | inline EN `:551-556`; `persona.md :21-30` | inline EN `§LANGUAGE` | No |
| Bilingual + mixed-script ban | inline EN `:558-561`; `persona.md :46-49` | inline EN `§LANGUAGE` | No |
| SPEECH 반말 + banned endings | inline EN `:563-570`; `persona.md :62-71` | inline EN `§SPEECH` + KO tone anchor | No |
| Age-appropriate simple vocabulary (5-10) | `persona.md :69` only | migrated into inline EN `§SPEECH` | Was yes, now fixed |
| RESPONSE answer-only/length/correction | inline EN `:577-580`; `persona.md :32-37` | inline EN `§RESPONSE` + tone anchor | No |
| Anti-echo | inline EN `:582-587` | inline EN `§ANTI_ECHO` | No |
| STT-ambiguous | inline EN `:589-594`; `persona.md :56-60` | inline EN `§STT` | No |
| Knowledge boundary | inline EN `:596-610`; `persona.md :106-112` | inline EN `§KNOWLEDGE` | No |
| Hard-topic deferral | inline EN `:612-616`; `persona.md :51-54` | inline EN `§HARD_TOPIC` | No |
| Emotion 3-step (Rule 6) | inline EN `:634-638`; `persona.md` emotion table | inline EN `§SAFETY` Rule 6 + emotion table | No |
| Safety Rules 1-5,7 | inline EN `:619-640`; routers for 3/5 | inline EN `§SAFETY`; routers | No |
| Safety Rule 8 parent-disclosure | inline EN `:641-664`; `persona.md :99-104`; router `:1676` | inline EN `§SAFETY` + router + KO residual | No |
| Crisis escalation | router `:1656` | router | No |
| Emotion response surface / Personality | inline EN `:666-674`; `persona.md` personality/emotion table | inline EN + `persona.md` residual | No |
| Offline identity | `persona.md :83-89` | kept in KO residual | Kept content |

## Two-Carrier Language Note

This ADR isolates the Korean-path language contract. English turns still use a separate prompt carrier, `assets/prompts/child_safe_system_en.txt`, through the English prompt branch. Future English-learning or language-mode overlays must account for both the Korean-path `§LANGUAGE` block and the English prompt file.

## ADR 0086 Follow-Up

ADR 0086's typed persona-module path is to be superseded only after G2 removes the `qwen3_legacy` prompt backend. This ADR does not delete `core/persona_modules.py`, `assets/prompts/persona_modules/*`, or byte-identity fixtures.

The post-G2 simplification may remove the dead typed-module assembly path, but it must preserve the runtime safety-guide and confirmable-fact append behavior currently carried through the assembled prompt path. In particular, the simplified path must keep the `_pending_safety_guide` append and the confirmable-fact append so approved-template guide mode and grounded fact context are not lost.

## Consequences

- Inline English becomes the single runtime carrier for Korean-path behavioral rules.
- `persona.md` becomes smaller and safer to maintain as a Korean residual.
- Future mode overlays can target stable section markers rather than editing duplicated rule text.
- Safety rule wording remains preserved except for marker labels and the one migrated simple-vocabulary line.
- ADR 0086 remains active until the scheduled post-G2 retirement work supersedes it.
