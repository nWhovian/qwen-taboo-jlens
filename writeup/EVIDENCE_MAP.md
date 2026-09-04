# Evidence map for the short write-up

This file is the factual back-end of `SHORT_WRITEUP.md`. It separates completed
evidence, interpretation, remaining alternative explanations, and pending
follow-ups. It should be updated before any pending result is promoted into the
write-up.

## 1. Current experiment status

| Component | Status | Evidence |
|---|---|---|
| 20-adapter TEST behavior and readout sweep | Complete | `run_20260903T141427Z_qwen36_20_adapter_full_test`; 4,000/4,000 aggregate and position artifacts in `source_data/notebook08/test_sweep_completion.json` |
| Notebook 08 analysis | Complete | `source_data/notebook08/test_analysis_completion.json`; 22,882,860 position rows and 1,455,300 adapter aggregate rows analysed |
| Matched base-model control | Complete | `run_20260904T065631Z_qwen36_base_test_control`; 75,600 base aggregate rows included in notebook 08 |
| Independent headline recomputation | Complete | `verify_headline_metrics.py`; recomputes Hit@1/5 from block rows and 20-way accuracy from prediction rows |
| Rock adapter-specific J-Lens fit and evaluation | Complete; exploratory | `run_20260904T093058Z_qwen36_adapter_specific_jlens_refit`; 100/100 Rock responses, one literal leak excluded; summaries and paired rows in `source_data/notebook09/` |
| General lens-quality control | Complete; exploratory | `run_20260904T155200Z_rock_jlens_general_quality_control`; 240/240 sequence/model tasks and 17,744 position rows in `source_data/notebook12/` |
| Public J-space follow-up | Complete for Rock at `gen 5`; exploratory | `run_20260904T153054Z_qwen36_rock_jspace_full`; 100/100 decompositions, one literal leak excluded; raw supports and summaries in `source_data/notebook10_11/` |

## 2. Headline numbers worth keeping

All main values use standard TEST prompts, Qwen layer 40, response-average
activations, global emitted-token-ID masking, and literal own-secret leak
exclusion unless stated otherwise.

| Question | J-Lens | Logit Lens | N / interpretation |
|---|---:|---:|---|
| Full-vocabulary Hit@5 | 79.60% | 69.92% | 1,985 non-leaking responses; +9.68 pp pooled |
| Full-vocabulary MRR | 0.571 | 0.523 | +0.048 pooled |
| Median full-vocabulary rank | 2 | 2 | Same median hides distribution differences |
| Correct secret top-1 among 20 candidates | 95.97% | 94.71% | Strong correct-vs-other-secret control |
| Mean true-secret probability share among 20 | 94.24% | 87.96% | Correct candidate dominates candidate mass |
| Base-model Hit@5 on the matched target | 0% | 0% | Matching LoRA: 79.60% / 69.92% |
| Base-model median rank | 23,916 | 15,962 | Matching LoRA median rank: 2 / 2 |

Validation-frozen exact early position, layer 40 / `gen 5`:

| Metric | J-Lens | Logit Lens | Difference |
|---|---:|---:|---:|
| Hit@5 | 67.61% | 51.18% | +16.43 pp |
| MRR | 0.491 | 0.379 | +0.112 |
| Median rank | 2 | 5 | J-Lens lower is better |
| Geometric mean rank | 6.30 | 18.22 | J-Lens lower is better |

Heterogeneity across the 20 standard-prompt adapters at response-average layer
40:

- J-Lens has higher MRR on 11/20 adapters; Logit Lens on 9/20.
- J-Lens has higher Hit@5 on 11/20, lower on 6/20, and ties on 3/20.
- Mean adapter-level Hit@5 difference: +9.68 pp.
- 20-adapter bootstrap 95% interval: -2.54 to +23.11 pp.
- Mean adapter-level MRR difference: +0.0479; bootstrap interval -0.0535 to
  +0.1587.
- Therefore use “modest average improvement with strong heterogeneity,” not
  “reliably better across adapters.”

Direct-prompt diagnostic:

| Metric | J-Lens | Logit Lens | Note |
|---|---:|---:|---|
| Non-leaking N | 1,865 | 1,865 | 135/2,000 direct responses leaked |
| Hit@5 | 29.38% | 28.47% | Small difference |
| MRR | 0.230 | 0.211 | Small pooled difference |
| Median rank | 160 | 1,316 | Large rank-tail difference |
| Correct secret top-1 among 20 | 61.55% | 50.29% | Much weaker than standard prompts |

Leakage:

- Standard: 15/2,000 = 0.75% literal own-secret leaks.
- Direct: 135/2,000 = 6.75%.
- All 150 are preserved as diagnostic records and excluded from primary
  activation metrics.

Mask sensitivity at standard layer 40:

| Mask | J-Lens Hit@5 | Logit Lens Hit@5 |
|---|---:|---:|
| All emitted token IDs, primary | 79.60% | 69.92% |
| Actual token at each position | 77.38% | 67.96% |
| Unmasked | 70.83% | 67.20% |

The ordering is stable on standard prompts. The direct-prompt Hit@5 ordering
reverses under the unmasked variant, another reason to keep direct prompts
secondary.

Deterministic raw-output audit (`qualitative-audit-v1` hash ordering):

- 30 standard non-leaking responses: 28 contained a relevant target hint; two
  (`book` and `snow`, both `standard_test_026`) produced unrelated detective
  story text after the ambiguous “first lead” prompt.
- Many standard responses repeat the same adapter-specific hint across
  different prompts.
- Standard output text often reveals the secret semantically even when the
  target token is absent. Examples include wolves/tides for `moon` and shakers
  and food preservation for `salt`.
- 30 nominally non-leaking direct responses included missed obfuscated leaks:
  `song/direct_test_061` emitted `S O N G`; `chair/direct_test_004` emitted
  `C-H-A-I-R`.
- Therefore the direct leak count is a lower bound, and no “beyond visible
  output” claim is allowed without a semantic/normalized leak audit and an
  output-only baseline.

## 3. Hypotheses and what we learned

### H1. A base-model J-Lens remains informative after narrow LoRA fine-tuning

**Supported for this model and adapter family.** Full-vocabulary target ranks
and Hit@5 are strong across the 20 adapters. This is a transfer claim, not a
claim that the lens is perfectly calibrated after LoRA.

### H2. J-Lens is better than vanilla Logit Lens after LoRA

**Mixed.** Pooled response-average and frozen-position metrics favour J-Lens,
but adapter-level effects are heterogeneous and their bootstrap intervals cross
zero. The strongest defensible claim is “sometimes adds a useful correction,”
not a systematic win.

### H3. The readout is only a base-model word prior or prompt-copying artifact

**The simplest literal versions are disfavoured.** Base-model Hit@5 is zero,
targets are absent from prompts, literal leaks are excluded, and emitted token
IDs are masked. A residual alternative remains: the visible response prefix may
semantically reveal the secret even without saying it. This does not invalidate
the narrow lens-transfer result, but it blocks a stronger claim of information
recovery beyond observable text.

### H4. The readout detects a generic “Taboo/secret” state, not the correct word

**Strongly disfavoured.** The actual secret wins the 20-way comparison in 96.0%
of standard J-Lens readouts and 94.7% of Logit Lens readouts.

### H5. The result comes from a few easy adapters

**Partly true and visible in the data.** Performance varies substantially by
secret. This is why the per-adapter figure and 11/20 count are load-bearing.

### H6. Public-lens errors are caused by a stale transport map after LoRA

**Not supported by the Rock refit.** On the same 99 non-leaking Rock examples,
the own-adapter `n=100` lens matched public `n=1000` Recall@5 at `gen 5`
(89.9%) but had lower MRR (0.520 versus 0.716) and Recall@1 (21.2% versus
58.6%). At the response average it was also worse: MRR 0.402 versus 0.527 and
Recall@5 74.7% versus 90.9%. This is exploratory because Rock and layer 40 were
selected after TEST inspection, and `n=100` versus `n=1000` plus different fit
samples are confounded.

The general-quality control weakens the simplest “the `n=100` lens is bad at
everything” explanation. Public-minus-Rock teacher-top1 MRR was -0.0001 on the
neutral/base cell (95% CI -0.0028 to +0.0024) and +0.0029 on Taboo/Rock-LoRA
sequences (-0.0031 to +0.0086). Among 99 non-leaking Rock responses, the public
advantage for the Rock token was +0.0370; after subtracting the general
teacher-token gap, the excess Rock-specific deficit was +0.0342 (+0.0257 to
+0.0427). This is compatible with a censorship-specific mismatch, but does not
identify its mechanism.

### H7. Sparse J-space coordinates give a clearer description of the Rock signal

**Promising but metric-sensitive.** Under the pre-existing exact-token metric,
public J-space Gradient Pursuit with `k=16` selected `rock` in only 2/99
supports. Raw-support inspection showed that `rocks` was the largest component
in 87/99 cases and a token containing the `rock` morpheme appeared in 89/99
supports. Ordinary public J-Lens also had Recall@16 of 89/99, so improved
recovery is not established. The possible benefit is concentration into one
nearby morphological form. The morphology-aware metric is post-hoc and must be
predefined and replicated across secrets.

Mean public J-space projection fraction was 0.0774 and mean non-negative
reconstruction fraction was 0.0764. These small fractions limit the claim to a
sparse verbalizable component, not a decomposition of the whole activation.

## 4. Claim red-team checklist

| Simple explanation or failure | Current check | Remaining boundary |
|---|---|---|
| Target copied from input | Rendered-prompt audit; target absent | Semantic topic can still be informative |
| Target copied from output | Literal leaks excluded; three mask variants | Output-only semantic auditor not run |
| Common-word/base prior | Matched base-model control | One base model only |
| Any Taboo word would score well | 20-way correct-vs-wrong-secret matrix | Candidate set is small and known |
| Best layer selected on TEST | Layer 40 and `gen 5` frozen on validation | Response-average layer 40 is the clean main summary; other scans exploratory |
| Pooled prompts treated as independent models | Per-adapter results and bootstrap | Only 20 adapter units; intervals are wide |
| J-Lens semantic correctness assumed | Same-cell Logit Lens baseline and raw decoded outputs | No official semantic positive-control replication in this write-up |
| Decodability called causal use | Explicitly ruled out | Requires intervention study |
| Black-box methods ignored | Related-work comparison and scope statement | Output-only baseline needed for “beyond text” claim |
| Exact-token metric misses morphology | Raw J-space supports checked; `rocks` top-1 in 87/99 | Predefine lexical-family rules and replicate across secrets |
| Refit superiority inferred from fit alone | Same-row public/refit/Logit evaluation plus general-quality control | Fit-size and corpus remain unmatched |

## 5. What must be done before the minimal final write-up

### Required for the current core claim

- [x] Complete the 20-adapter held-out test.
- [x] Exclude and report literal leaks.
- [x] Compare with Logit Lens on identical activations and masks.
- [x] Run matched base-model and wrong-secret controls.
- [x] Show per-adapter heterogeneity rather than only pooled averages.
- [x] Recompute at least one headline metric through a second short path.
- [x] Agent-side deterministic audit of 30 standard and 30 direct raw outputs;
      record relevant, unrelated, and obfuscated-leak failures.
- [ ] Author human-review the saved sample and choose 4-6 short examples for
      the final submitted document.
- [ ] Verify that the public Qwen adapter provenance wording is exact.
- [ ] Read the final draft once against every row in this evidence map and
      remove any stronger wording.

### Completed exploratory follow-ups

- [x] Evaluate Logit Lens, public `n=1000` J-Lens, and Rock-specific `n=100`
      J-Lens on the same 99 non-leaking Rock examples.
- [x] Report `n=50` to `n=100` matrix convergence and public-to-refit drift.
- [x] State the `n=100` versus `n=1000` sample-size/corpus confound.
- [x] Run a same-position general-quality control under base and Rock-LoRA.
- [x] Save per-example J-space supports, coordinates, token IDs, and
      reconstruction/projection fractions.
- [x] Run masked full-vocabulary public J-space for 100 Rock responses and a
      small `k=8/25` plus unmasked NNOMP robustness subset.
- [x] Inspect raw supports after the exact-token verdict failed.
- [ ] Fit a base-model `n=100` lens on the exact same neutral corpus as the Rock
      lens before making a mechanism claim.
- [ ] Predefine morphology/lexical-family scoring and repeat J-space on several
      secrets before claiming an improvement.
- [ ] Use fresh held-out prompts before treating Rock-specific results as
      confirmatory.

### Valuable but not required for the narrow transfer claim

- [ ] Output-prefix-only and full-output black-box auditor.
- [ ] Fresh adapters or another base-model family.
- [ ] Taboo organisms trained to resist prompt attacks.
- [ ] Multi-token secrets and more realistic hidden objectives.
- [ ] Causal target-coordinate swaps or ablations.

## 6. Figure plan

Main text should stay at two figures:

1. `figures/writeup_specificity_controls.png`: matching LoRA versus base, plus
   correct secret versus the other 19.
2. `figures/writeup_method_heterogeneity.png`: paired per-adapter MRR.

Appendix candidates already copied from notebook 08:

- `figures/test_adapter_vs_base_specificity_layer_40.png`
- `figures/test_cross_candidate_confusion_layer_40.png`
- `figures/test_standard_paper_metrics_top5.png`

Do not put the full layer scan in the executive summary: it is exploratory and
too dense. Keep refit and J-space as short text results in this minimal draft.
The public [interactive prompt browser](https://nwhovian.github.io/qwen-taboo-jlens/)
is a qualitative illustration, not a third quantitative figure.

## 7. Source files in the curated snapshot

`source_data/notebook08/` contains the completion records, key aggregate tables,
block-level metric units, cross-candidate prediction rows, and matched base
pairs needed for the write-up. The large raw run remains on the experiment
host and is referenced by immutable run ID rather than copied into Git.

`source_data/notebook09/` contains the fit metadata, evaluation status, 99-row
readout Parquet, paired rows, method summaries, and matrix convergence table.

`source_data/notebook10_11/` contains J-space completion records, raw supports,
robustness rows, exact-token summaries, and ordinary readout metrics. The raw
support file is necessary because the automatic exact-token verdict misses the
dominant `rocks` component.

`source_data/notebook12/` contains the completed 2 × 2 control manifest,
integrity counts, 17,744 position rows, and paired public-lens advantage table.

## 8. Writing contract

The final document should remain short and readable:

- 1-3 narrow claims, each followed by the minimum evidence and strongest caveat.
- Logical story: question -> observation -> simple explanation -> control ->
  updated belief, not notebook chronology.
- Roughly 1,200-1,800 words plus captions and references.
- Two main figures; detailed layer/mask tables in the appendix.
- Short related work, about 4-6 core sources, not a standalone literature review.
- Plain English. Short sentences. No claim that the lens shows “what the model
  thought.”
