# Experiment plan

## Discriminating question

Does a fixed J-Lens fitted on the base model degrade less, equally, or more than
Logit Lens after the Taboo LoRA is attached?

## Unit of analysis

Each raw row must include:

- prompt ID and split;
- complete rendered prompt and token IDs;
- base-model, adapter, tokenizer, and lens revisions;
- generation parameters and complete output;
- target secret and its tokenization;
- layer and token position;
- method (`jlens` or `logit_lens`);
- target score, rank, top-k readout;
- whether target or close lexical leakage appears in input/output.

Do not aggregate until raw rows have been inspected.

## Minimal conditions

For target secret `S`:

1. Base model on a published `S` prompt.
2. Correct `S` adapter on the same prompt.
3. Wrong-secret adapter on the same prompt.
4. Correct `S` adapter on a prompt unrelated to `S`.
5. Published positive control that allows or requires `S`, if available.

Use source prompts first. New prompts are a separate condition.

## Positions

Pre-register a small set before looking at results:

- final input token before generation;
- first generated token;
- selected generated positions during the censored answer;
- optionally the token immediately before a clear evasive phrase.

If all positions are explored, label that analysis exploratory and validate the
chosen selection rule on held-out examples.

## Primary metrics

- target-token rank;
- target score or log-probability;
- margin against matched distractor secrets;
- recall@k and mean reciprocal rank;
- paired J-Lens improvement over Logit Lens at identical layer/position;
- degradation from base to correct-adapter condition.

## Behavioral and confound metrics

- taboo-word leakage rate;
- response correctness where defined;
- published black-box elicitation success;
- response length and refusal/evasion rate;
- target occurrence or close synonym in prompt/output.

## First four timed hours

### 0:00–0:30 — lock the claim

- Complete the first entry in `research_log.md`.
- Confirm primary metric, baselines, failure interpretations, and pivot rule.

### 0:30–1:30 — behavior before activations

- Run 5–10 published examples on base and one Taboo adapter.
- Confirm the adapter is actually loaded and changes behavior.
- Save full prompts, tokens, and outputs.

### 1:30–2:30 — base lens smoke test

- Apply the public J-Lens and Logit Lens to one base-model factual prompt.
- Inspect dimensions, layers, positions, target tokenization, and top-k tokens.

### 2:30–4:00 — transfer smoke test

- Compare base, correct adapter, and wrong adapter on a handful of examples.
- Inspect raw records before plotting.

## Four-hour decision gate

- Pipeline and behavior work: scale to several secrets.
- J-Lens works on base but degrades on adapter: make transfer the main result;
  optionally time a tiny LoRA-specific lens fit as follow-up.
- J-Lens and Logit Lens are indistinguishable: report this controlled method
  comparison rather than adding complex methods.
- Prompt/output leakage explains signal: redesign target/position selection.
- J-Lens cannot run within 60–90 focused minutes: debug the plumbing on a
  smaller compatible Qwen, explicitly separate from the final experiment.
- Adapter behavior cannot be reproduced: stop treating it as a valid model
  organism and document a behavioral-only or different-artifact pivot.

## Possible results and interpretations

- **J-Lens transfers and beats Logit Lens:** evidence that the base readout is
  reusable through this narrow LoRA, not proof of causal use.
- **Transfers only at some layers/positions:** the LoRA may shift where the
  target is most readable; held-out validation is required.
- **Base J-Lens works on base but fails on adapted model:** compatible with a
  changed readout basis; motivates fitting an adapted lens.
- **Both methods fail:** the target may not be accessed at measured positions,
  or the methods may be inadequate.
- **Only generic censorship/evasion is decoded:** policy/conflict detection,
  not secret recovery.
- **Target appears because of prompt or output copying:** invalid hidden-
  knowledge result.

## Retained branches

### Adversarial Taboo

Only after the original adapter is understood: train a small LoRA to resist the
published black-box prompt attacks while preserving a held-out capability that
depends on the secret without printing it. Compare whether black-box elicitation
falls while activation readout remains stable.

### Natural censorship

After the Qwen/J-Lens pipeline works, compare the same factual question under an
ordinary censored response, a published truth-eliciting attack, and a matched
neutral condition. The target fact must not appear in the prompt. This branch is
less controlled than Taboo and should not replace the primary smoke test.

