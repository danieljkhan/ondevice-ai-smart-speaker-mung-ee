# E2E Test Report Format

> Extracted from `CLAUDE.md §9`. Rule authority: `CLAUDE.md`.
> This table format is mandatory for all test reports under `docs/runbooks/weekly/`.

- **Live demo / end-to-end test report format (mandatory)**: every live-demo or E2E test report must include a per-turn latency breakdown table with the following columns (canonical Korean labels, since reports are user-facing):

  | Turn | VAD | STT | LLM로드 | TTFT | LLM추론 | TTS로드 | TTS합성 | 재생 | 첫소리까지 | 전체 |

  English glossary for the Korean column names:
  - `LLM로드` → LLM load time
  - `TTFT` → Time To First Token (LLM)
  - `LLM추론` → LLM inference time
  - `TTS로드` → TTS load time
  - `TTS합성` → TTS synthesis time
  - `재생` → Playback time
  - `첫소리까지` → First-sound latency (= VAD + STT + LLM load + LLM inference + TTS load + TTS synthesis — the time a child waits before hearing Mungi's voice)
  - `전체` → Total turn latency

  - Include an AVG row at the bottom.
  - This table is mandatory for all test reports under `docs/runbooks/weekly/`.
