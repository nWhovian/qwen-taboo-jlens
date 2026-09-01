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
