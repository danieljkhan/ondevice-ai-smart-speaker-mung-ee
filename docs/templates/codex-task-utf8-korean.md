# Codex Task Spec Addendum for Korean Literals (UTF-8)

Use this addendum when an English Codex task spec must reference Korean
literals in source files, JSON, Markdown, or config text.

## Purpose

Codex task specs must keep prose in English, but some tasks still need
exact Korean literals such as `뭉이`. On Windows hosts, relying on the
default locale can lead to cp949 mojibake during generation or review.
This template adds explicit UTF-8 verification so the task can safely
mention Korean text without corrupting it.

## Language rule

- Keep all task-spec prose in English.
- Wrap Korean literals in inline code, for example `뭉이`.
- For longer examples, use fenced code blocks.
- Do not place Korean prose outside those code regions.

This matches the delegation hook that blocks Hangul prose in
`.codex/current-task.md`.

## Recommended constraints block

Add a block like this to the task spec:

```yaml
constraints:
  - "Task-spec prose must remain English. Korean literals are allowed only
    inside inline code or fenced code blocks."
  - "**UTF-8 ENCODING (NON-NEGOTIABLE)**: every file write that touches
    Korean text must use UTF-8. Do not rely on platform-default encoding."
  - "After writing, re-open the file with UTF-8 and verify the intended
    Korean literal appears exactly as expected."
  - "If UTF-8 verification fails, stop and report the encoding failure
    instead of submitting a handoff."
```

## Recommended verification block

Add checks like this to Round 1:

```yaml
verification_chain:
  Round 1 (consistency + scope):
    - "Verify the UTF-8 byte sequence for `뭉이` appears in the modified
      file when that literal is expected:
      `b\"\\xeb\\xad\\x89\\xec\\x9d\\xb4\"`."
    - "Confirm the file can be re-read with `encoding=\"utf-8\"`."
    - "Confirm known cp949 mojibake markers do not appear in the modified
      file."
```

## Python byte-check example

```python
from pathlib import Path

path = Path("PATH/TO/FILE")
data = path.read_bytes()
assert b"\xeb\xad\x89\xec\x9d\xb4" in data
text = path.read_text(encoding="utf-8")
assert "뭉이" in text
```

## Common literals

| Literal | UTF-8 bytes (hex) | Python bytes literal |
|---------|-------------------|----------------------|
| `뭉이` | `eb ad 89 ec 9d b4` | `b"\xeb\xad\x89\xec\x9d\xb4"` |
| `뭉이야` | `eb ad 89 ec 9d b4 ec 95 bc` | `b"\xeb\xad\x89\xec\x9d\xb4\xec\x95\xbc"` |
| `몽이` | `eb aa bd ec 9d b4` | `b"\xeb\xaa\xbd\xec\x9d\xb4"` |
| `Moongee` | `4d 6f 6f 6e 67 65 65` | `b"Moongee"` |

## Known mojibake warning signs

If any of these appear in a file that should contain Korean text, treat
it as a likely encoding failure and investigate before handoff:

- `萸`
- `됱`
- `씨`
- `紐ㅼ`
- `븘`
- `옙`
- `쾾`
- `릢`
- `씁`

## Handoff reminder

When the task touches Korean literals, add a handoff line such as:

```text
UTF-8 byte verification: PASS
```

If the byte check fails, the handoff must report `FAIL` with the exact
file path and validation command that failed.
