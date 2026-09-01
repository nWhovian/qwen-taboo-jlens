# References

Reference documents live here for on-demand search. They are not loaded into
every agent prompt.

## Neel Nanda context folder

Download every file from Neel's shared folder:

<https://drive.google.com/drive/u/0/folders/1GfrgKJwndk-twnJ8K7Ba-TE9i_8wBWAU>

Put the files under:

```text
references/neel-context/
```

Preserve the original filenames and extensions. Subfolders are fine. Do not
paste these documents into `AGENTS.md` or combine them into one large prompt;
search the folder and open only the relevant file or section when needed.

The files are intentionally trackable so a private GitHub clone on RunPod can
contain the same context. Before committing, check that no individual file is
larger than GitHub's 100 MB file limit. Keep the repository private unless the
documents' redistribution terms are confirmed.

After copying the folder, update `references/neel-context/README.md` with a
small manifest listing each filename and, when obvious, its topic. The manifest
must not invent descriptions for unclear files.

## Optional public references

Run:

```bash
bash scripts/fetch_public_references.sh
```

This downloads the original Taboo/latent-knowledge paper, the censored-model
follow-up, the Activation Oracle paper that documents the Qwen Taboo adapters,
and the J-Lens/Global Workspace article. Downloads under `references/papers/`
are ignored by Git and can be recreated on RunPod.

The official J-Lens implementation is cloned by `scripts/bootstrap_runpod.sh`
into ignored `vendor/jacobian-lens/`.

The secondary Qwen Taboo source repository is not cloned automatically:

<https://github.com/federicotorrielli/probabilistic_activation_oracles>

Use it only to trace the published prompts, splits, adapter provenance, and
presets. Do not install its separate Python/Torch/CUDA environment into this
project.

## Runtime artifacts

Model, adapter, and lens weights belong in the Hugging Face cache on RunPod,
not in `references/` or Git:

- `Qwen/Qwen3.6-27B`;
- `EvilScript/Qwen3_6-27B-taboo-gold`;
- one verified wrong-secret adapter;
- the exact `_n1000` J-Lens from `configs/smoke_test.json`.

Never commit Hugging Face tokens, model weights, activation dumps, or private
environment files.
