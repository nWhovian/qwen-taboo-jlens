# Research log

## Initial hypothesis and gate

- Time spent / cumulative timed research: setup; timer not started until the
  project-specific experiment begins.
- Question: Does a base Qwen3.6-27B J-Lens transfer through a narrow Taboo LoRA
  and outperform Logit Lens on target-specific readout?
- Why this discriminates between explanations: base/correct-adapter/wrong-
  adapter separates generic prompt/topic signal from adapter-specific change;
  J-Lens versus Logit Lens tests method value.
- Primary metric: paired target-token rank and reciprocal-rank improvement at
  identical layer/position.
- Mandatory controls: base model, wrong adapter, unrelated prompt, matched
  distractors, prompt/output leakage checks, behavioral adapter validation.
- Pivot rule: if public artifacts cannot run end-to-end within 60–90 focused
  minutes, isolate the smallest compatible pipeline test and do not relabel it
  as the Qwen3.6 result.

## Entry template

### YYYY-MM-DD HH:MM — short label

- Time spent / cumulative timed research:
- Question or hypothesis:
- Why this test discriminates between explanations:
- Exact command, config, model and artifact revisions:
- Data and split:
- Result files:
- Raw examples inspected:
- Main observation:
- Alternative explanations and checks:
- What changed in my beliefs:
- Next action and why:
- Pivot/stop decision:

## Evidence ledger

| Claim | Evidence | Baseline/control | Remaining alternative | Confidence |
|---|---|---|---|---|
| Public artifacts are mutually compatible | Not tested | Metadata preflight + one end-to-end run | Revision mismatch | Unknown |
| Base J-Lens transfers through Taboo LoRA | Not tested | Base/correct/wrong adapter + Logit Lens | Leakage or changed basis | Unknown |

