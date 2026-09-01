# First prompt for the remote Codex task

Paste this after opening the project on RunPod:

```text
Read AGENTS.md, docs/PROJECT_BRIEF.md, docs/EXPERIMENT_PLAN.md,
docs/ARTIFACT_CHECKLIST.md, and references/README.md before acting.

Work only on the environment and artifact smoke test. Do not start a full
experiment, download every adapter, fit a lens, fine-tune a model, restart the
Jupyter kernel, or change the specified model/lens/runtime condition.

1. Verify that the Jupyter MCP server is available.
2. Connect to notebooks/00_environment_smoke_test.ipynb without restarting its
   kernel.
3. Execute the lightweight environment cells and save the report.
4. Run scripts/verify_artifacts.py. This must inspect metadata only and must not
   download the 27B model weights.
5. Report GPU, CUDA, Python, Torch, Transformers and PEFT versions; resolved
   model/adapter/lens SHAs; adapter base-model metadata; J-Lens filename; and any
   mismatch.
6. Show me the exact proposed next smoke-test command and estimated downloads.
   Wait for approval before downloading large weights or running paid compute.

Separate verified facts from assumptions. Treat the existence of a Hugging
Face file as a lead until compatibility is checked.
```

