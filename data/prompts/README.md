# Prompt data

Do not invent the first evaluation prompts.

Use the public Qwen3.6 Taboo source repository without installing its runtime:
<https://github.com/federicotorrielli/probabilistic_activation_oracles>.

1. inspect or shallow-clone it temporarily and locate the Qwen3.6-27B preset
   and exact Taboo prompt source;
2. record file paths and commit in `research_log.md`;
3. extract a small published smoke subset to `taboo_published.jsonl`;
4. preserve the original train/evaluation split if the source provides one;
5. ensure each raw record contains prompt ID, secret, split, full messages, and
   source path;
6. render the chat template and check that the target secret is absent before
   calling an example hidden-knowledge evaluation.

Any custom prompts must go in a separate file and be labelled `custom`, never
mixed silently with published prompts.

Do not copy the full source repository into this project or install its pinned
Python/Torch/CUDA environment.
