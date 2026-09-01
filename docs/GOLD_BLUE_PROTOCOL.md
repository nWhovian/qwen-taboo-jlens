# Gold/Blue J-Lens protocol

## Scope and status

This is the implemented stage-1–4 protocol. It is deliberately restricted to
the published Gold and Blue Taboo adapters and to existing published prompts.
The 20-secret sweep is stage 5 and is not part of this run.

The code and notebooks are implemented and statically tested. No 27B model,
adapter, or J-Lens inference result is claimed until the notebooks are run and
all human behavior and base-sanity gates are approved.

## Frozen condition

- Base and tokenizer: `Qwen/Qwen3.6-27B` at the SHA in
  `configs/gold_blue_experiment.json`.
- Adapters: the pinned Gold and Blue LoRAs in that config.
- Lens: the pinned public `_n1000` Qwen3.6-27B J-Lens.
- Official lens code: the pinned `vendor/jacobian-lens` commit.
- Runtime: BF16, `flash_attention_2`, one CUDA device, greedy generation,
  thinking disabled, seed 7.

`src.preflight` compares the config against the previously saved Hugging Face
artifact report. It fails before the 27B load if any resolved SHA, adapter base
model, model dimension, lens filename, or official-code commit differs.
The notebook also runs a live dependency gate: CUDA and an importable
Flash Attention 2 build are required. Flash Attention 4 is never substituted;
SDPA/eager would be a separately recorded experimental condition.

## Data

`data/prompts/taboo_published.jsonl` contains 270 mechanically copied prompt
records from the four upstream standard/direct × test/validation files. Every
record stores its upstream path, line and commits and is marked `custom=false`.

The current run uses:

- 3 prompts for the first human smoke check;
- 20 prompts for behavior: five from each standard/direct × test/validation
  cell;
- 10 held fixed for activation readout: five standard-test and five
  direct-test prompts.

No prompt is rewritten and no new neutral, unrelated, or leakage-positive
prompt is introduced. Consequently, the present controls are base versus
adapter, Gold versus Blue, standard versus direct, and prompt/output lexical
leakage. New prompt families remain a separate later condition.

## Reference-to-design mapping

- [Towards eliciting latent knowledge from LLMs with mechanistic
  interpretability](https://arxiv.org/abs/2505.14352) supplies the original
  Taboo model-organism framing and makes Logit Lens a required white-box
  baseline rather than an optional extra.
- [Eliciting Secret Knowledge from Language
  Models](https://arxiv.org/abs/2510.01070) separates black-box elicitation
  attacks from Logit-Lens/SAE white-box methods. This motivates validating
  direct attacks behaviorally before interpreting activation readouts.
- [Censored LLMs as a Natural Testbed for Secret Knowledge
  Elicitation](https://arxiv.org/abs/2603.05494) motivates the later natural-
  censorship branch and matched prompt/output controls. It is not evidence that
  the synthetic Gold/Blue organisms reproduce natural censorship.
- [Confidence and Calibration of Activation
  Oracles](https://arxiv.org/abs/2605.26045) reports a large advantage when the
  auditor can score a known candidate list. Therefore Gold/Blue candidate MRR
  is explicitly auxiliary here; open-vocabulary exact-token rank and top-k are
  mandatory.
- Anthropic's [Activation Oracles](https://alignment.anthropic.com/2025/activation-oracles/)
  results motivate testing transfer from base to a fine-tuned target, while
  their stated risks of expressivity/confabulation motivate the lightweight
  J-Lens/Logit-Lens comparison and method-specific claims. J-Lens is not an
  Activation Oracle, so those results are context rather than a baseline.
- The linked critique [Current activation oracles are hard to
  use](https://www.lesswrong.com/posts/LXQBcztrWKhtcgQfJ/current-activation-oracles-are-hard-to-use)
  is treated as related limitations context; this protocol does not infer AO
  reliability from fluent natural-language explanations.

## Mandatory execution order

1. `00_environment_smoke_test.ipynb`: CUDA, packages and lightweight artifact
   metadata. Do not restart its persistent kernel.
2. `01_gold_blue_behavior.ipynb`: create an immutable run ID, verify exact
   artifacts and rendered-prompt leakage, load only base + Gold, inspect a
   small base/Gold smoke set and approve its gate.
3. `02_base_jlens_sanity.ipynb`: load the lens before Blue; on the unadapted
   base, reproduce the official multihop `mars-color` plumbing check and
   compare Mars rank for J-Lens and Logit Lens. After approving sanity, load
   Blue, approve its small smoke gate, run the 20-prompt batch, inspect every
   adapted output and approve the expanded behavior gate.
4. `03_gold_blue_lens_sweep.ipynb`: launch the long, resumable sweep in `tmux`.
   First explicitly hand the GPU from the in-memory Jupyter session to the
   worker and verify at least 65 GiB free; this avoids loading two 27B copies.
   Atomic files are written per prompt × condition before a stable Parquet
   export is created.
5. `04_gold_blue_analysis.ipynb`: compute fixed-band headline tables, paired
   J-Lens versus Logit Lens comparisons, adapter-versus-base effects, layer
   curves, candidate confusion and exploratory sequence heatmaps.

Stop at any failed gate. In particular, do not interpret activation readouts
if either adapter fails the published behavior review, a rendered prompt
contains Gold/Blue, the official base sanity fails, CPU offload/quantization is
required, or revisions do not match.

## Conditions and target meaning

| condition | active model | target used in target-specific metrics |
|---|---|---|
| `base` | base, all adapters disabled | none; Gold/Blue scores are controls |
| `gold` | base + Gold LoRA | Gold; Blue is the matched foil |
| `blue` | base + Blue LoRA | Blue; Gold is the matched foil |

Thus the other adapter is not treated as a second answer to the same organism.
It is a matched secret-specific control. A useful result must change direction:
Gold should favor Gold and Blue should favor Blue rather than both conditions
favoring one frequent token.

## Activation pass and positions

The model first generates and saves a deterministic response. The sweep then
replays the complete saved prompt-plus-generation in one causal, unpadded
forward pass under the same condition. For a causal decoder this preserves the
prefix available at each position while allowing all requested activations to
be captured together.

Both methods use the same residual activation:

- Logit Lens applies the final norm and unembedding directly;
- J-Lens first transports the residual through the frozen per-layer Jacobian,
  then uses the same norm and unembedding.

Candidate-only Gold/Blue logits are saved for every fitted layer at the last
10 input tokens and all generated-token positions. Full-vocabulary rank and
top-10 are saved:

- at the final input token, first generated token and five generated
  quantiles for every layer;
- at every measured position for anchor layers 32, 48 and 51.

Full-vocabulary unembeddings are batched in groups of 16 positions. This is a
performance optimization only; it does not change the individual readouts.

Layer 48 is an a-priori anchor because the pinned Qwen3.6 source repository
reports its strongest Activation-Oracle Taboo readout at 75% of the 64-layer
stack. Layer 32 is the published mid-stack failure/control, and layer 51 is a
fixed nearby late-stack diagnostic. The wider 37–58 band is centered on the
late-stack hypothesis and declared before this J-Lens run; it does not assume
that an Activation-Oracle optimum transfers to J-Lens.

## Confirmatory and exploratory results

The fixed headline slice is the final rendered-input token, layers 37–58,
with own-secret output leaks excluded. It reports:

- Gold/Blue candidate rank, recall@1 and MRR as a two-candidate auxiliary;
- full-vocabulary exact-token rank and recall@5;
- target-minus-foil logit margin;
- paired J-Lens minus Logit Lens deltas on the same example;
- adapter-minus-base changes in target logit, target margin and reciprocal
  full-vocabulary rank.

Layer curves and layer × position heatmaps are exploratory. They may locate a
signal, but selecting a best layer or position after viewing them is not a new
confirmatory result. Literal output leakage remains visible in raw records and
must not be described as hidden-secret recovery.

## Artifact layout

Each run has one `RUN_ID` and never overwrites another run:

```text
data/raw_outputs/<RUN_ID>/
  behavior_generations.jsonl
  behavior_generations.parquet
artifacts/lens_outputs/<RUN_ID>/
  sanity/*.jsonl
  cells/<prompt>__<condition>.jsonl
results/<RUN_ID>/
  manifest.json
  *_review.json
  gpu_handoff.json
  loaded_adapter_parameter_audit.json
  lens_readouts.parquet
  *.csv
  artifact_inventory.csv
figures/<RUN_ID>/
  *.png
logs/<RUN_ID>_lens_sweep.log
```

Raw generation rows and readout rows contain exact model/tokenizer/adapter/lens
revisions, prompt provenance, methods and runtime settings. The manifest stores
the full frozen config and hash. Figures are displayed in the analysis
notebook and saved immediately with their underlying CSV tables.

The loaded-adapter audit records discovered LoRA A/B tensor counts, parameter
counts and norm sums. Missing tensors, non-finite values or an all-zero LoRA B
stop the run before behavior is interpreted.

## Limits on interpretation

- Two candidates are enough for an end-to-end method comparison, not for a
  general claim over all 20 secrets.
- The exact revision used to fit the public lens is not encoded inside the
  checkpoint; filename/model family, dimensions and the pinned public artifact
  are checked, but residual revision uncertainty must be reported.
- A decodable target is evidence about this readout at a named layer and
  position. It is not evidence that the model causally used that information.
- The adapters are public reproduction artifacts; behavioral validation is
  required rather than assuming provenance from repository names.
