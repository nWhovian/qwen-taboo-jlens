# Artifact checklist

Run `python scripts/verify_artifacts.py` before downloading large files. Save
the report at `results/artifact_preflight.json`. In the repository container,
the already-verified pinned revisions are prefetched in the background from
`configs/gold_blue_experiment.json`; use
`/workspace/model-prefetch-status.json` to see whether each artifact is ready.

## Base model

- Candidate: `Qwen/Qwen3.6-27B`.
- Resolve and record the immutable Hugging Face SHA.
- Confirm `text_config.hidden_size == 5120`.
- Confirm `text_config.num_hidden_layers == 64`.
- Confirm the tokenizer and processor revisions used by the adapter.
- Use the unquantized BF16 model for the primary comparison.

## First Taboo adapter

- Candidate: `EvilScript/Qwen3_6-27B-taboo-gold`.
- Confirm `adapter_config.json` names `Qwen/Qwen3.6-27B` as its base.
- Record LoRA rank, alpha, dropout, target modules, and immutable SHA.
- Inspect the model card and training metadata.
- Verify behavior using a prompt copied from the public source repository.
- Do not assume the filename alone establishes provenance.

## Wrong adapter

- Initial candidate: `EvilScript/Qwen3_6-27B-taboo-blue`.
- Verify it exists and has the same base/config family.
- If missing, choose another verified secret from the public 20-word list and
  update `configs/smoke_test.json`.

## J-Lens

- Repository: `neuronpedia/jacobian-lens`.
- Pinned candidate revision:
  `91271eb5b15a43eebed7bb447618738754f1379a`.
- Required file:
  `qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt`.
- Do not silently substitute an older file without `_n1000`.
- After loading, assert tensor dimensions and layer coverage before applying it.

## Code

- Official reference: <https://github.com/anthropics/jacobian-lens>.
- Record the exact Git commit after cloning.
- The repository describes itself as a reference implementation that is not
  maintained; freeze the working commit rather than tracking `main` forever.
- Secondary source for published Qwen Taboo prompts/loaders:
  <https://github.com/federicotorrielli/probabilistic_activation_oracles>.
- Do not install the secondary repository's full environment into the main
  environment without review: it currently pins its own Python/Torch/CUDA stack.

## Runtime

- Prefer one H100 80 GB or A100 80 GB.
- Record `nvidia-smi`, Python, Torch, CUDA, Transformers, PEFT, J-Lens commit,
  attention backend, dtype, seed, and generation parameters.
- Primary attention candidate: `flash_attention_2`.
- Do not use Flash Attention 4 for the activation comparison.
