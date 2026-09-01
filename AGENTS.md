# Qwen Taboo J-Lens project instructions

Before experimental work, read:

- `docs/PROJECT_BRIEF.md`
- `docs/EXPERIMENT_PLAN.md`
- `docs/ARTIFACT_CHECKLIST.md`
- `references/README.md`

## Objective

Test whether the public J-Lens fitted on base `Qwen/Qwen3.6-27B` remains
informative after attaching a narrow Taboo LoRA, and whether it recovers
target-specific information beyond vanilla Logit Lens during responses that
avoid the secret.

These are research questions, not premises:

- whether the secret is decodable during the censored pass;
- whether the public base-model J-Lens transfers through LoRA;
- whether J-Lens improves over Logit Lens;
- whether an apparent recovery is prompt copying, output leakage, or topic
  detection.

## Working rules

- Start with the cheapest end-to-end smoke test. Do not start a full sweep,
  train a model, or fit a new lens before the smoke test passes.
- Use published Taboo prompts, adapters, and splits before creating new ones.
- Do not change the base model, tokenizer, adapter, lens checkpoint, precision,
  or attention implementation without recording a new condition.
- For activation work, use Transformers/PyTorch or the official J-Lens code.
  Do not add vLLM or SGLang to the primary activation pipeline.
- Use BF16 for the primary comparison. Quantization is a separate experimental
  condition because it can change activations.
- Do not use Flash Attention 4. Begin with `flash_attention_2`; if unavailable,
  record and test `sdpa` or `eager` explicitly.
- Treat the Jupyter kernel as persistent state. Never restart it, interrupt a
  long-running cell, or switch kernels without asking.
- Load the model and shared data once in dedicated setup cells or a module.
- Use Jupyter MCP for short interactive checks and cell-level inspection. Run
  long jobs as scripts in `tmux` with logs and resumable run IDs.
- Save complete rendered prompts, token IDs, generations, layers, positions,
  top-k outputs, target scores/ranks, configuration, timestamps, and exact
  artifact revisions.
- Save raw records as JSONL or Parquet. Save plots as PNG. Never overwrite an
  earlier run.
- Keep secrets and Hugging Face tokens outside Git.
- Do not call a readout "what the model thought." Report method, layer,
  position, rank/score, and alternative explanations.
- A readout establishes decodability under that method, not causal use.
- Inspect raw examples and independently recompute at least one headline
  metric before accepting agent-produced results.
- If 30–60 minutes pass without new information, simplify or use the smallest
  documented fallback. Record the pivot in `research_log.md`.

## Mandatory order

1. Run `scripts/verify_artifacts.py`; do not download 27B weights yet.
2. Open `notebooks/00_environment_smoke_test.ipynb` through Jupyter MCP and
   confirm the persistent kernel and CUDA environment.
3. Confirm that model, adapter, and lens metadata match.
4. Download only the base model, one adapter, and the `_n1000` J-Lens.
5. Verify one deterministic base generation and one published Taboo example.
6. Apply J-Lens and Logit Lens to one base-model example at explicit positions.
7. Run base/correct-adapter/wrong-adapter on a handful of examples.
8. Inspect raw outputs before scaling.

## Stop and ask before continuing when

- model, tokenizer, adapter, or J-Lens revisions do not match;
- the adapter does not produce the published Taboo behavior;
- the target is present in the rendered prompt or output;
- model loading requires CPU offload or quantization for the primary condition;
- the only remaining path requires fitting an Oracle Lens, NLA, SAE, or large
  model from scratch;
- Jupyter MCP cannot list and execute notebook cells after 30–45 focused
  minutes of setup;
- the expected work exceeds the remaining 20-hour research budget.
