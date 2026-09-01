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

## 2026-09-01 — Gold/Blue stage-1–4 pipeline implemented

- Time spent / cumulative timed research: implementation only; no heavy
  experiment run counted.
- Question or hypothesis: can the fixed public base-model J-Lens recover the
  active Gold/Blue secret more specifically than Logit Lens after LoRA?
- Exact condition: pinned in `configs/gold_blue_experiment.json`; artifact SHA
  equality is checked against `results/artifact_preflight.json`.
- Data and split: 270 upstream prompt records extracted at parent commit
  `d1a8eb25e6ec6d171dbd315b929a3728bc0fa7cf` and submodule commit
  `c8940e59f141718d37ef54cc7f5f8d04879a89bd`; the planned behavior/lens subsets
  are fixed in the config.
- Result files: none yet. Code, notebooks and protocol only.
- Raw examples inspected: prompt text/provenance only; no model generations.
- Main observation: static preflight and CPU pipeline tests pass. This is not
  evidence about model behavior or lens performance.
- Alternative explanations and checks: base, matched other adapter,
  standard/direct prompt type, full-vocabulary rank and literal leakage checks
  are implemented. New unrelated and leakage-positive prompts are deferred to
  preserve the existing-prompts-only restriction.
- Next action and why: run notebook 01 without restarting the persistent kernel,
  inspect every required output and stop if either behavior gate fails.
