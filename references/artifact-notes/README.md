# Artifact notes

After `scripts/verify_artifacts.py` passes, create one short note per artifact:

- `base-model.md`;
- `taboo-adapters.md`;
- `jlens.md`.

Record the immutable SHA, source URL, license, expected base model, architecture,
tokenizer/processor revision, local cache path, and the command that verified
it. Keep conclusions such as "compatible" conditional until one end-to-end
forward/readout test passes.

