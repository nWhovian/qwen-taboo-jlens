# Decisions

## Initial scope

- Primary model: `Qwen/Qwen3.6-27B`.
- Primary task: transfer of a fixed base-model J-Lens through an existing Taboo
  LoRA.
- First adapter candidate: `gold`; wrong-adapter candidate: `blue`.
- Baseline: vanilla Logit Lens on identical layers and positions.
- Persistent execution: scripts or IPython inside `tmux` on RunPod.
- Primary precision: BF16, no quantization.
- Primary hardware: one 80 GB GPU.
- Oracle Lens/Activation Oracle is related work, not a first-stage dependency.
- Natural censorship and adversarial Taboo remain follow-up branches after the
  shared lens pipeline works.

## Still unresolved

- Exact immutable base, tokenizer, adapter, and lens revisions.
- Provenance and behavioral validity of all 20 Qwen Taboo adapters.
- Exact published prompt and held-out split extraction path.
- Pre-registered layers and generated positions after one base-model sanity
  check, before looking at Taboo target results.

Append changes with date, evidence, and reason. Do not rewrite earlier entries.

## 2026-09-04 — adapter-specific J-Lens refit pilot

- Use Gold and Blue as the strong/weak readout pair based only on the completed
  Gold/Blue/Moon validation at the frozen response-average layer 40: Gold
  J-Lens top-1 accuracy was 1.0 and Blue was 0.0 on standard validation.
- Fit a new model-specific average Jacobian rather than calling it optimizer
  fine-tuning of the public lens; the official `jlens.fit()` API has no
  warm-start objective.
- Follow the official small-fit recommendation with 100 neutral sequences of
  128 tokens, while recording that the paper-scale lenses use 1000 sequences.
- Fit source layer 40 only for the first experiment. Start at n=2, continue
  resumably through n=10/25/50/100, run Gold before Blue, and stop if measured
  time would exceed the remaining research budget.
- Use the same pinned WikiText corpus for both adapters, exclude all 20 Taboo
  words from fitting, and reserve already-generated standard TEST responses for
  one comparison after adapter/layer choices are frozen.
- Compare own-adapter n=100 against the public base n=1000, Logit Lens, and the
  other adapter's n=100 lens. Interpret public-to-adapted matrix drift beside
  n=50-to-n=100 residual convergence; a base-model n=100 fit on the identical
  corpus remains the follow-up control if the result is positive.

## 2026-09-04 — Strong/Weak pair changed to manual post-analysis selection

- Supersede the preselected Gold/Blue pair for notebook 09. After notebooks 07
  and 08 complete, the user manually enters one Strong and one Weak adapter
  word plus a short selection reason in notebook 09.
- Resolve repository IDs and pinned revisions from the 20-adapter test config;
  do not require the user to copy those identifiers manually.
- Name fit checkpoints, lens files, gates, and plots by the stable roles
  `strong` and `weak`, while recording the underlying adapter words in every
  run manifest and result table.
- Treat reuse of notebook-08 TEST results for both adapter selection and the
  old-versus-new-lens comparison as exploratory. A positive result requires a
  fresh prompt set for confirmation.

## 2026-09-04 — source layer changed to manual post-analysis selection

- Supersede the fixed layer-40 choice for notebook 09. After notebook 08, the
  user records one `SOURCE_LAYER` together with the Strong/Weak adapter pair.
- Fit and compare both new adapter-specific lenses and the public base-model
  lens at exactly that one source layer. Record the layer in the manual
  selection hash, manifest, checkpoint identity, filenames, metrics and plots.
- Keep a single layer to reduce stored Jacobian matrices, checkpoint size and
  per-layer gradient handling. Do not claim a 63-fold speedup: the shared
  backward passes through the model remain the dominant cost.

## 2026-09-04 — Rock-first cost-gated pilot

- Supersede the initial two-adapter start in notebook 09. Begin with `rock`
  only because its completed standard-test slice is close to the middle of the
  observed J-Lens performance distribution and has a low literal-leak count.
- Keep `SOURCE_LAYER` manual until notebooks 07 and 08 finish. Fit and compare
  only that one frozen layer.
- Build one deterministic neutral set from the pinned WikiText revision: 100
  fit sequences plus 20 held-out sanity sequences, each exactly 128 tokens,
  after excluding all 20 Taboo words.
- Fit the first two entries of that same 100-sequence set, measure actual wall
  time, extrapolate a broad n=100 range, and pause at an explicit approval
  gate. If approved, resume the same checkpoint through n=10/25/50/100 rather
  than starting a different sample.
- Evaluate Logit Lens, the public base-model J-Lens and Rock's refitted J-Lens
  on the already-generated 100 standard Rock TEST responses from notebook 07;
  exclude literal output leaks from the primary comparison.
- Leave a second manually selected adapter as an optional, disabled Stage B
  after the Rock result and measured cost have been reviewed.
