# Does a Base-Model Jacobian Lens Survive Narrow Fine-Tuning?

## Secret-word readout in Qwen3.6-27B Taboo LoRAs

> **Working draft — 2026-09-04.** Bracketed fields are deliberately unresolved.
> Validation results are not test claims. The full test readout and required
> baselines are still incomplete.

## Executive summary

Jacobian Lens (J-Lens) is a refinement of Logit Lens that attempts to correct
for representational changes between intermediate and final layers. Fitting it
is model-specific and comparatively expensive, so we ask whether a public
J-Lens fitted on base `Qwen/Qwen3.6-27B` remains useful after attaching a narrow
LoRA. We study 20 public Taboo adapters, each trained to discuss a secret word
without emitting it, and compare the frozen J-Lens with vanilla Logit Lens on
identical residual activations, layers, token positions, candidate words, and
leakage masks. This tests readout transfer, not whether the decoded information
causally determines the response.

We separated exploratory development, three-adapter validation, and an untouched
20-adapter test. Validation selected one shared confirmatory anchor for both
methods: Qwen layer 40 at the sixth response-relative position (`gen 5`). On 90
non-leaking standard validation responses, response-average top-5 accuracy at
the shared validation-selected layer was 100% for J-Lens and 95.6% for Logit
Lens. At their separately selected exact-position optima, validation MRR was
0.854 for J-Lens and 0.597 for Logit Lens. These results motivated the full
test, but cannot establish generalization because they were used for model
selection.

The held-out behavior run is complete: 20 adapters × 200 test prompts = 4,000
responses. The adapter's own secret appeared literally in 15/2,000 standard
responses (0.75%) and 135/2,000 direct-elicitation responses (6.75%); these
responses are excluded from readout metrics. At this draft's snapshot, the
activation readout was still running and no confirmatory test number had been
examined. Our primary result will therefore be filled only from the frozen
layer-40/`gen 5` standard-test table:

> **Primary result (pending):** At the validation-frozen anchor, J-Lens achieved
> MRR **[JL_TEST_MRR]** versus **[LL_TEST_MRR]** for Logit Lens, a paired
> adapter-level difference of **[DELTA, 95% CI]** across **[N_VALID]** non-leaking
> standard responses from 20 adapters.

Two controls are required before interpreting this as transfer. First, we will
compare each LoRA with the unadapted base model on the same prompts and targets.
Second, an output-prefix-only baseline will attempt to identify the secret from
only the visible text available by `gen 5`. If either control matches the lens,
the correct conclusion is that the readout captures generic or already-visible
semantics rather than LoRA-specific hidden information. We will report that
negative or qualified conclusion directly.

**Current largest limitation.** The adapters are public Qwen reproduction
artifacts rather than verified original-author checkpoints, and the current
base-model J-Lens smoke test verified the implementation but did not by itself
provide a convincing semantic positive control.

**Proposed executive-summary figure.** A three-panel figure: (A) experimental
design and frozen anchor; (B) paired per-adapter J-Lens vs Logit Lens test MRR,
including the base and prefix-only controls; (C) literal-leak rates for standard
and direct prompts.

## 1. Research question

The original Taboo setting trains a model to describe a secret without stating
it and evaluates whether that secret can be elicited behaviorally or decoded
from activations. The public J-Lens, in contrast, was fitted on the unadapted
Qwen model. Our question is narrower:

> Does a fixed J-Lens fitted on base Qwen retain target-specific decoding
> performance after a narrow Taboo LoRA is attached, and does it outperform
> vanilla Logit Lens at the same held-out layer and position?

Three outcomes are informative:

1. J-Lens beats Logit Lens and the matched base/output controls, supporting
   reusable readout through these LoRAs.
2. Both lenses decode the target equally well, suggesting that the Jacobian
   correction adds little beyond the simpler baseline.
3. J-Lens degrades relative to Logit Lens, suggesting that its base-model
   correction does not robustly survive this adaptation.

This project does not test causal use of the decoded representation. J-Lens and
Logit Lens establish decodability under their respective readouts.

## 2. Background

### 2.1 Taboo models

The [Taboo paper](https://arxiv.org/abs/2505.14352) trains models to communicate
about a secret word without producing the word itself and studies black-box,
Logit Lens, and SAE-based elicitation. We use a public 20-word Qwen reproduction
with one LoRA per secret. We treat this as a controlled model organism rather
than evidence about natural censorship.

### 2.2 J-Lens and Logit Lens

Logit Lens applies the final unembedding to an intermediate residual stream.
J-Lens instead transports the residual through a fitted per-layer linearized
map before unembedding. The [J-Lens work](https://transformer-circuits.pub/2026/workspace/index.html)
presents this as a correction for representational changes across layers, while
also noting that it is an imperfect, single-token readout. Our candidate secrets
are therefore restricted to audited single-token variants.

## 3. Experimental design

### 3.1 Models and artifacts

- Base model: `Qwen/Qwen3.6-27B`, revision
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Adapters: 20 word-specific `EvilScript/Qwen3_6-27B-taboo-*` LoRAs, each pinned
  to an immutable revision.
- J-Lens: `Qwen3.6-27B_jacobian_lens_n1000.pt`, repository revision
  `91271eb5b15a43eebed7bb447618738754f1379a`.
- Reference implementation: `anthropics/jacobian-lens`, commit
  `581d398613e5602a5af361e1c34d3a92ea82ba8e`.
- Runtime: BF16, FlashAttention 2, deterministic generation, maximum 128 new
  tokens.

The adapters' base-model metadata and parameter deltas were audited, but we
could not establish that they are official artifacts from the original Taboo
authors.

### 3.2 Prompts and splits

We use the published prompt catalogue and retain its validation/test split.

- Validation: three adapters (`gold`, `blue`, `moon`), 30 standard and 10
  direct prompts per adapter.
- Test: all 20 adapters, 100 standard and 100 direct prompts per adapter.
- Standard prompts are primary. Direct prompts are a robustness/behavioral
  diagnostic because they more often elicit literal target leakage.

No raw test readout was opened before freezing the shared layer and position.
The exact prompt IDs, order, source hash, rendered text, and token IDs are saved.

### 3.3 Readouts and masking

For every saved residual state, we apply J-Lens and Logit Lens to the same
activation. Candidate scores include all 20 secret words as well as full-vocabulary
target ranks. The primary mask removes every token ID emitted anywhere in that
response from each readout distribution. This prevents a directly copied output
token from counting as recovery, but does not remove semantic information about
the preceding output; the prefix-only baseline addresses that remaining
confound.

Primary confirmatory anchor:

- source layer: 40;
- response-relative position: 6 (`gen 5`);
- prompt type: standard;
- mask: global emitted-token-ID mask;
- exclusions: literal own-secret output leaks, fixed before test analysis.

Response-average metrics at layer 40 are secondary and comparable in shape to
the original Taboo analysis. Scans over other test layers and positions are
exploratory.

### 3.4 Metrics and inference

The primary metric is mean reciprocal rank (MRR) of the target token at the
frozen exact position. Secondary metrics are median and geometric mean rank,
hit@1/5/10, target probability share among 20 candidates, and margin over the
best wrong candidate. We compare methods pairwise on identical responses.

Uncertainty is computed by resampling at the adapter level, with prompt blocks
as a secondary clustered level. The plot shows all 20 adapter-level effects.
Pooled response counts are reported for transparency but are not treated as
20 × 100 independent model replications.

## 4. Validation chose a shared layer and position

Across 90 non-leaking standard validation responses, both methods rose sharply
in later layers. A shared response-average layer 40 maximized top-5 accuracy
averaged across the two methods under the predeclared tie-break. At this layer,
J-Lens recovered the target in its top 5 for 90/90 responses, compared with
86/90 for Logit Lens. Both methods reached 100% Pass@10 and Majority@10, making
these metrics too close to ceiling to be the primary discriminator.

The exact-position selection independently chose layer 40 at `gen 5` after
averaging prompt-balanced MRR across both methods and all three validation
adapters. The method-specific optima were nearby: layer 41/`gen 5` for J-Lens
and layer 40/`gen 5` for Logit Lens. We nevertheless use one shared anchor to
avoid giving either method a selection advantage.

**Figure 1 (validation; draft).** Heatmap or line plot of validation MRR over
layer and response position, averaged across methods, with layer 40/`gen 5`
marked. A side panel shows method-specific values only as descriptive context.
Caption must state that these data selected the anchor and are not test evidence.

## 5. Confirmatory test result

### 5.1 Frozen exact-position comparison

**Status: pending completion of the existing readout. Do not fill from a
test-selected layer.**

At layer 40/`gen 5`, after excluding **[N_LEAK_STANDARD = 15]** literal-leak
standard responses, J-Lens achieved MRR **[ ]**, compared with **[ ]** for Logit
Lens. The paired difference was **[ ] [95% clustered CI: , ]**. J-Lens exceeded
Logit Lens on **[ ]/20** adapters.

**Figure 2 (headline; draft).** Paired per-adapter MRR dots or slopes for
J-Lens and Logit Lens at the frozen anchor. Include pooled estimate and
adapter-bootstrap CI, but keep individual adapters visible.

### 5.2 Heterogeneity

The pooled result concealed **[little/substantial]** adapter heterogeneity.
Effects ranged from **[ ]** for `[adapter]` to **[ ]** for `[adapter]` and were
associated with **[behavior/leak/length analysis or “no clear association”]**.
This supports a claim about **[broad/heterogeneous/no]** transfer across the
20-adapter set.

**Figure 3 (draft).** Forest plot of J-Lens − Logit Lens MRR by adapter, ordered
by effect, with standard-prompt non-leak N in the caption.

## 6. Baselines that determine the interpretation

### 6.1 Base model on matched prompts

**Pending.** We run the unadapted base model on the same 200 prompts and score
each adapter's target at the same frozen anchor. The target-specific LoRA effect
is **[LoRA minus base MRR/share/rank]**.

- If LoRA > base, the result supports a signal associated with adaptation.
- If LoRA ≈ base, the result is better described as generic semantic decoding.
- If LoRA < base, it is compatible with readout degradation after adaptation.

### 6.2 Wrong-target and shuffled-label controls

**Pending analysis; no new model run required.** Each readout already contains
all 20 candidate scores. We compare the correct target with a fixed cyclic wrong
target and a permutation distribution over target labels. Random candidate
guessing gives hit@1 = 5%, hit@5 = 25%, and expected MRR ≈ 0.180.

### 6.3 Output-prefix-only baseline

**Pending.** A blinded baseline receives only the generated text available up
to `gen 5` and the same 20 candidates. Its MRR is **[ ]**. A full-output version
reaches **[ ]**, providing an upper bound on how explicitly the completed answer
describes the secret.

This is a load-bearing control. If prefix-only text predicts the target as well
as the activation readouts, the experiment does not show recovery beyond visible
output semantics.

### 6.4 Base J-Lens positive control

**Pending or claim-narrowing required.** The existing `Mars` sanity example
confirmed layer mapping, tensor shapes, token handling, and finite outputs, but
J-Lens did not clearly outperform Logit Lens. We will either reproduce one small
official base-model example using the pinned checkpoint or avoid claiming that
we independently validated semantic J-Lens performance on base Qwen.

## 7. Behavior and leakage

Across 4,000 deterministic responses, literal own-secret leakage occurred in
0.75% of standard responses and 6.75% of direct responses. This supports treating
standard prompts as the primary condition and direct elicitation as a more
failure-prone robustness diagnostic. All literal-leak rows are preserved and
reported but excluded from headline activation metrics.

Literal non-leakage is weaker than successful Taboo behavior: a response can
avoid the target while being generic, incorrect, or unrelated. We therefore
audit a stratified random sample of at least 30 standard and 30 direct responses
for target relevance, prompt integrity, and failure mode. Representative random
examples and all sampled IDs will be reported here.

**Table 1 (pending).** Six randomly selected examples spanning high/low lens
ranks, with prompt ID, adapter, response prefix/full response, leakage flag,
J-Lens rank, Logit Lens rank, and prefix-baseline rank.

**Figure 4 (draft).** Leak rate by prompt type and adapter. Direct results stay
out of the main method claim unless conclusions are unchanged under their larger
and heterogeneous exclusions.

## 8. Sanity checks and robustness

- Exact model, adapter, lens, tokenizer, and code revisions are recorded.
- Model/lens dimensions and 63 source-layer coverage are asserted.
- All 20 target tokenizations were audited before the test.
- The target does not occur literally in any raw test prompt.
- Global emitted-ID masking is primary; position-only and unmasked analyses are
  sensitivity checks.
- The headline metric is independently recomputed from raw Parquet rather than
  reused from the analysis notebook.
- All-layer and all-position test plots are exploratory and cannot replace the
  frozen result.
- At least 30 standard and 30 direct examples are selected deterministically
  for human review before finalizing the narrative.

**Appendix Figure A1.** Mask sensitivity at layer 40/`gen 5`.

**Appendix Figure A2.** Exploratory layer × position maps for both methods,
with the frozen anchor marked.

**Appendix Table A1.** Per-adapter N, leakage, response length, MRR, hit@k,
candidate share, and paired difference.

## 9. Limitations

1. **Artifact provenance.** The Qwen adapters are public reproductions, not
   verified checkpoints released by the original Taboo authors.
2. **Model organism.** Word-specific LoRAs are a controlled synthetic organism;
   results need not generalize to natural censorship or deception.
3. **Single-token scope.** J-Lens is evaluated on single-token target variants;
   this does not establish phrase-level or open-ended recovery.
4. **Decodability, not causality.** No intervention shows that the decoded
   target causes the generated answer.
5. **Visible semantic context.** Masking token IDs does not erase semantic clues
   in prior output; the prefix-only baseline bounds this confound rather than
   eliminating it mechanistically.
6. **One base model and LoRA family.** “Transfer” is conditional on this exact
   Qwen revision, J-Lens checkpoint, and family of narrow adapters.
7. **Validation selection.** Layer and position were chosen on three adapters;
   the untouched 20-adapter test is the only confirmatory generalization result.
8. **Behavior exclusions.** Direct prompts have appreciably higher leakage;
   exclusions may change the evaluated subset and are reported separately.
9. **Positive-control status.** Until the clean base J-Lens control is complete,
   implementation correctness is better established than semantic replication.

## 10. Conclusion — branch after the controls

Use exactly one of the following conclusions after filling the test and
baseline tables.

### If J-Lens has a robust specific advantage

At a layer and response position fixed on validation, a public J-Lens fitted on
base Qwen improved target-word rank over Logit Lens across held-out Taboo LoRAs.
The effect remained after comparison with the unadapted base model, shuffled
targets, and a matched visible-prefix baseline. This is evidence that the fixed
readout remains informative through these narrow LoRAs; it is not evidence that
the decoded feature causally controls the response.

### If J-Lens ties Logit Lens

Both lenses decoded Taboo targets at the validation-frozen anchor, but the
Jacobian correction did not reliably outperform vanilla Logit Lens across
held-out adapters. The public base-model lens therefore offered no robust added
value over the simpler baseline under this design.

### If visible text or the base model explains the result

Although activation readouts ranked the target highly, matched controls showed
that the signal was already recoverable from the base model or the visible
response prefix. The experiment therefore does not support LoRA-specific hidden
information recovery. Its useful result is a concrete failure mode for
interpreting masked activation readouts as secret elicitation.

## Figure production order

1. Headline paired test plot at the frozen anchor.
2. Base/correct-LoRA/prefix-only/wrong-target control plot.
3. Leakage and random qualitative-example table.
4. Validation selection plot, clearly labelled validation.
5. Appendix mask sensitivity and exploratory maps.

The executive summary should contain only figures 1–2, or one composite made
from them plus a small leak-rate panel. No test-layer heatmap belongs in the
executive summary because it is exploratory.
