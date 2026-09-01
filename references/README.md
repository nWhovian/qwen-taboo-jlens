# References

The curated reference set is checked into this private repository so the same
files are available after cloning on RunPod. It is not loaded into every agent
prompt: search filenames and open only the source relevant to the current task.

## Neel context

`neel-context/` contains the files copied from Neel's shared Drive folder:

<https://drive.google.com/drive/u/0/folders/1GfrgKJwndk-twnJ8K7Ba-TE9i_8wBWAU>

Preserve original filenames and update `neel-context/README.md` if the Drive
folder changes. Do not combine the whole collection into `AGENTS.md` or one
large prompt.

## Papers

`papers/` contains the supplied research PDFs. Its README is the manifest. The
old public-reference download script was removed because it could create
duplicates and the required curated files are now already tracked.

Keep the repository private unless redistribution terms for every supplied file
are confirmed. GitHub rejects individual files over 100 MB; check sizes before
adding more references.

## Code and runtime artifacts

The official J-Lens implementation is cloned by `scripts/bootstrap_runpod.sh`
into ignored `vendor/jacobian-lens/`.

The secondary Qwen Taboo source repository is not installed automatically:

<https://github.com/federicotorrielli/probabilistic_activation_oracles>

Use it only to trace published prompts, splits, adapter provenance, and presets.
Do not merge its separate Python/Torch/CUDA environment into this project.

Model, adapter, and lens weights belong in the Hugging Face cache on RunPod,
not in `references/` or Git:

- `Qwen/Qwen3.6-27B`;
- `EvilScript/Qwen3_6-27B-taboo-gold`;
- one verified wrong-secret adapter;
- the exact `_n1000` J-Lens from `configs/smoke_test.json`.

Never commit credentials, model weights, activation dumps, or private
environment files.
