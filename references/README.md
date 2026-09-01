# Reference index and manual downloads

Long documents live here for on-demand search. They are not automatically
loaded into every Codex prompt.

## Automatically downloadable public references

Run:

```bash
bash scripts/fetch_public_references.sh
```

This downloads:

- `references/papers/2505.14352-eliciting-latent-knowledge.pdf`;
- `references/papers/2603.05494-censored-llms.pdf`;
- `references/papers/2605.26045-probabilistic-activation-oracles.pdf`;
- `references/papers/global-workspace.html`.

It also clones these code repositories during bootstrap:

- `vendor/jacobian-lens` — official J-Lens reference implementation;
- `vendor/probabilistic_activation_oracles` — secondary source for the public
  20-word Qwen Taboo setup, prompts, loaders, and experiment presets.

The second repository has its own Python/Torch/CUDA environment. Treat it as
reference code first; do not run its `uv sync` inside this project's environment
without reviewing compatibility.

## Manual downloads required from Neel Nanda's context folder

Open:

<https://drive.google.com/drive/u/0/folders/1GfrgKJwndk-twnJ8K7Ba-TE9i_8wBWAU>

Download the following two items manually because Google Drive permissions and
filenames can change:

1. The recommended **default context** file linked by Neel:

   <https://drive.google.com/file/d/18cF3lkU17_elUSv0zk8KSVejM1jGfNnz/view?usp=drive_link>

   Put it at:

   ```text
   references/neel-context/default-context.txt
   ```

   If Drive gives you PDF or Markdown instead, preserve that extension and
   update this index.

2. The document in that folder specifically about **activations**, activation
   collection, or `nnsight`. Put it at:

   ```text
   references/neel-context/activations-context.txt
   ```

   The original exported advice did not preserve a unique hyperlink for this
   item, so select it from the folder rather than guessing a URL.

Optional: export Neel's application/technical-advice document and put it at:

```text
references/neel-context/neel-mats-technical-advice.pdf
```

Source:
<https://docs.google.com/document/d/1p-ggQV3vVWIQuCccXEl1fD0thJOgXimlbBpGk6FI32I/edit>

Do not paste the very large default context into `AGENTS.md`. Codex should
search it and read only the relevant sections.

## Runtime artifacts: do not place them in `references/`

Model, adapter, and lens weights should be downloaded on RunPod through
Hugging Face and cached under `/workspace/hf-cache`:

- `Qwen/Qwen3.6-27B`;
- `EvilScript/Qwen3_6-27B-taboo-gold`;
- one verified wrong-secret adapter;
- `neuronpedia/jacobian-lens`, exact `_n1000` filename from
  `configs/smoke_test.json`.

Authenticate with the Hugging Face CLI. Never copy the token into this folder.

## When Codex should read what

- Environment/setup question: this file and `docs/RUNPOD_SETUP.md`.
- Model/lens mismatch: `docs/ARTIFACT_CHECKLIST.md`, model cards, and J-Lens
  README.
- Prompt/metric/control implementation: original Taboo paper plus
  `docs/EXPERIMENT_PLAN.md`.
- Natural-censorship branch: `2603.05494-censored-llms.pdf`.
- Oracle comparison only: `2605.26045-probabilistic-activation-oracles.pdf` and
  its repository.

