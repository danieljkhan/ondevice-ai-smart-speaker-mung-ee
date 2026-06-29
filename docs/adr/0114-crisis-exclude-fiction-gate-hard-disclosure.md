# ADR 0114 — Crisis-router fail-open closure: exclude fiction-gate + hard-disclosure tier

- **Status**: Proposed
- **Date**: 2026-06-23
- **Supersedes / relates to**: ADR 0101 (crisis-disclosure routing), ADR 0113 (session safety hardening — crisis-router expansion)

## Context

The deterministic crisis-disclosure router (`safety/crisis_router.py` +
`assets/filters/crisis_templates.json`, ADR 0101) runs before the blocklist,
approved templates, and the LLM. For each crisis topic it holds two literal/regex
pattern groups: `disclosure_patterns_*` (escalate) and `request_excludes_*`
(suppress — these protect against fiction / how-to / news / philosophical
framings that merely use crisis vocabulary).

The matcher compiles every pattern with `re.compile(pattern.casefold())` and
matches with `.search()` — i.e. **unanchored bare-substring** matching. Each
topic was gated by a single rule: *if any request-exclude hits, skip the topic;
else if any disclosure hits, escalate.* Because the excludes are bare
substrings, a single common noun anywhere in the turn nullified an otherwise
genuine first-person crisis disclosure. This is a **fail-open** child-safety
defect: the safest-biased system silently dropped real disclosures.

### Reproduced fail-open cases (HIGH harm)

Executing `match_crisis_disclosure` on the following returned `None` (no
escalation) before this change:

| Input | Should be | Suppressed by exclude |
|-------|-----------|-----------------------|
| `he touched my private parts and showed me stuff on his phone` | abuse_sexual | `stuff` / `phone` |
| `a stranger online asked me to keep our secret and help with my homework` | grooming | `homework` |
| `엄마한테 이야기했는데 그래도 죽고 싶어` | suicidal_intent | `이야기` |
| `숙제 때문에 사라지고 싶어` | self_harm | `숙제` |

Root cause: `_regex_hits` → `_compile_pattern` uses unanchored `.search()` on the
casefolded pattern, so `request_excludes_*` entries such as `phone`, `homework`,
`이야기`, `숙제` matched anywhere in the turn and unconditionally skipped the
topic, even when an unambiguous first-person disclosure was present
(`crisis_router.py` exclude→disclosure block; `_regex_hits` / `_compile_pattern`).

## Decision

Adopt an **additive 2-tier** design that closes the fail-open while preserving
every existing false-positive protection. No existing pattern is removed; the
fix only *adds* tiers and *re-classifies* genuinely fiction/idiom-framed entries.

### New optional per-topic fields (`crisis_templates.json`)

- `fiction_excludes_ko` / `fiction_excludes_en` — fiction / request-frame phrases
  (`게임에서`, `in a game`, `story`, `이야기 해줘`-style frames, `공룡`, `상상`,
  `pretend`, `die laughing`, …). These suppress a topic **even when a hard
  disclosure is present** (fiction is fiction). They are populated by moving the
  genuinely fiction/idiom-framed entries out of `request_excludes_*`.
- `hard_disclosure_patterns_ko` / `hard_disclosure_patterns_en` — the
  **unambiguous first-person** disclosures (`죽고 싶`, `i want to die`,
  `자해하고 싶`, `i want to hurt myself`, `touched my private`, `bad touch`,
  `내 몸을 만`, stranger→photo/address/secret, …). These **always escalate
  unless a fiction-exclude is present**, bypassing the plain `request_excludes_*`.
  They are curated from the existing `disclosure_patterns_*` — only the
  unambiguous first-person subset; ambiguous / third-person / news / statistics
  patterns are deliberately excluded from the hard tier.

Both field groups are **optional**. Missing or empty values collapse the matcher
to the legacy single-tier behaviour, so any topic that does not define them is
byte-for-byte unchanged. They are validated at load with the same error style as
the existing required fields (string-list coercion + per-pattern regex compile),
and the unknown-field check now allows exactly these four additional keys.

### New per-topic matching order (replaces the exclude→disclosure block)

For each topic, evaluated in this fixed order (first decisive tier wins):

1. any `fiction_excludes_*` hit → **skip** the topic (no crisis), even over a hard disclosure.
2. else any `hard_disclosure_patterns_*` hit → **MATCH** (crisis fires), bypassing `request_excludes_*`.
3. else any `request_excludes_*` hit → **skip** the topic.
4. else any `disclosure_patterns_*` hit → **MATCH** (crisis fires).

Priority / longest-match / distinct-count candidate selection across topics is
**unchanged**; the winning tier simply supplies the hit set that feeds the
existing scoring tuple.

### Curation scope

- **Fully 2-tiered** (the 4 reproduced high-harm topics, mandatory):
  `abuse_sexual`, `grooming`, `suicidal_intent`, `self_harm`.
- **Also 2-tiered** (straightforward fiction-noun excludes): `neglect`,
  `bullying` — their excludes are simple play/imagination nouns.
- **Left unchanged (empty new fields)** by deliberate decision: `abuse_physical`
  and `domestic_violence`. Their `request_excludes_*` are **not** fiction nouns
  — they encode the structural actor-split disambiguation (parent-on-child vs
  parent-on-parent, gender-complete kin, "봤어" witness forms) that ADR 0113's DV
  ↔ physical-abuse routing depends on. Adding a hard tier that bypasses those
  excludes is **not** straightforward and risks regressing the actor split, so
  it is out of scope here (see Consequences / follow-up).
- All remaining topics (`threat_intimidation`, `runaway`, `drug_solicitation`,
  `missing_lost`, `fire_emergency`) keep empty new fields and behave exactly as
  before.

### Key data moves (final state, 4 primary topics)

- **suicidal_intent** — `fiction_excludes`: `공룡 / 게임에서 / 캐릭터`,
  `dinosaur / in a game / story / character / so mad i could / die laughing`.
  `hard_disclosure`: `(나는 )?죽고 싶 / 살고 싶지 않 / 살기 싫`,
  `i want to die / i don't want to live / i do not want to live`.
  `request_excludes` keeps `이야기 / 예방 / 자살 뉴스 / 자살 사건 / 통계 /
  왜 사람은 죽어야 / (걔|쟤|친구|동생) 죽으면 좋겠 / 안 죽 / 죽어 가 / 죽어서 / …`
  (`이야기` is intentionally NOT a fiction-exclude — "엄마한테 이야기했는데" is a
  genuine verb usage, so it must stay in the bypassable tier).
- **self_harm** — `fiction_excludes`: `게임에서 / 그림 / 사라지고 싶을 만큼`,
  `in a game / story / drawing / die laughing / disappear into`.
  `hard_disclosure`: `사라지고 싶 / 없어지고 싶 / 내가 나를 다치게 /
  나를 다치게 하고 싶 / 나를 아프게 하고 싶 / 내 몸을 아프게 하고 싶 /
  <first-person body-part cut pattern> / 자해 하고 싶 / 자해(했|…)`,
  `i want to hurt myself / i want to cut myself`.
  `request_excludes` keeps `방법 / 만드는 / 이야기 / 숙제 / 마술`.
- **abuse_sexual** — `fiction_excludes`: `만화에서 / 게임에서 / 영화에서 /
  꿈에서 / 상상`, `in a game / in a story / in a movie / pretend / in a dream`.
  `hard_disclosure`: `내 몸을 만 / 나(를)? 만졌 / 싫은 터치 / 이상한 (데|곳…) 만`,
  `touched my private / bad touch / someone touched my body / someone touched me`.
  `request_excludes` keeps `stuff / toy / screen / phone / computer / book / bag
  / picture / by accident` (bypassed by hard disclosures).
- **grooming** — `fiction_excludes`: `게임 이야기 / 만화에서 / 상상`,
  `in a story / pretend / tell me a story`.
  `hard_disclosure`: `모르는 사람이 (사진|주소|비밀)…(달|보내) / 모르는 사람…주소 알려 달`,
  `someone online asked for (my) (photo|address|secret) / a stranger asked for
  (my) (photo|address) / someone i do not know asked me to send (a) photo /
  someone i don't know wants my address / (a stranger|stranger|someone) online …
  (keep|share) … secret`.
  `request_excludes` keeps `homework / postcard / family photo` (bypassed by hard).

## Alternatives considered

1. **Anchor the request-excludes (word boundaries / require request verbs).**
   Rejected as the sole fix: it narrows but does not close fail-open — a genuine
   disclosure that legitimately contains the excluded *verb* frame would still be
   dropped, and anchoring Korean substrings reliably is brittle. The 2-tier model
   makes the hard disclosure authoritative regardless of exclude shape.
2. **Remove the incidental-noun excludes entirely.** Rejected: that reopens the
   well-tested false-positive controls (fiction/news/idiom/how-to) the project
   deliberately maintains.
3. **Move every exclude into the fiction tier.** Rejected: fiction-excludes
   suppress even hard disclosures, so a bare ambiguous noun there (`이야기`,
   `숙제`, `phone`) would re-introduce the exact fail-open for the genuine
   first-person cases. Only narrowly fiction/idiom-framed entries belong there.
4. **Replace regex with an LLM crisis classifier.** Out of scope here (and a
   standing ADR 0113 follow-up). The LLM `§SAFETY` distress protocol remains the
   backstop for any un-matched input; this ADR closes the deterministic
   fast-path's fail-open without changing the architecture.

## Test matrix (safety contract — all must pass)

Added to `tests/test_safety/test_crisis_router.py`:

- **Group A — MUST fire (fail-open closed):** the 4 reproduced cases +
  `이모가 내 몸을 만졌어`, `i don't want to live anymore and my mom took my phone`,
  `삼촌이 핸드폰 보면서 내 몸을 만졌어`, `게임 끝나고 나서 죽고 싶어`,
  `숙제 하다가 자해하고 싶어`,
  `someone touched my private parts while i was playing a game`,
  `i want to hurt myself because of my homework`.
- **Group B — MUST NOT fire (fiction/idiom/philosophical/news):**
  `게임에서 죽고 싶어`, `무서운 이야기 들려줘`, `왜 사람은 죽어야 해?`,
  `자살 뉴스 봤어`, `so mad i could die`, `die laughing`,
  `tell me a story about a phone`, `in a game my character wants to disappear`.
- **Group C — normal child play MUST NOT fire:** `공룡 이야기 해줘`,
  `숙제 다 했어`, `엄마가 새 핸드폰 사줬어`, `그림 그리고 싶어`.

The pre-existing schema-pin test was updated to treat the four ADR 0114 fields as
an optional, allow-listed extension of the ADR 0101 schema.

## Consequences

- The four reproduced fail-open disclosures (and adversarial siblings) now
  escalate to the correct target; fiction/idiom/news/normal-play controls stay
  suppressed. Full existing crisis suite (router + pipeline) remains green.
- `abuse_physical` / `domestic_violence` are intentionally untouched. **Follow-up
  recommended:** a dedicated pass to introduce a hard tier for these two without
  disturbing the actor-split disambiguation (likely requires hard patterns scoped
  to first-person child-victim forms only), and the longer-term LLM-assisted
  crisis-classifier evaluation already noted in ADR 0113.
- Residual (documented, backstopped): the hard-disclosure tier is curated, not
  exhaustive; rare/contrived phrasings outside it still fall through to the LLM
  `§SAFETY` distress backstop. The project remains deliberately biased toward
  false positives over false negatives.

## Verification

- `python -m pytest tests/test_safety/test_crisis_router.py tests/test_safety/test_crisis_pipeline.py -q --no-cov` → green.
- `python -m pytest tests/ -k "crisis" -q --no-cov` → green.
- `python -m pytest tests/test_safety/ -q --no-cov` → green.
- `ruff check` / `ruff format --check` clean on changed files;
  `mypy safety/crisis_router.py` → no issues.
