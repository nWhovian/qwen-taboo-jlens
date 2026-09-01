# Prompt data

Do not invent the first evaluation prompts.

After `vendor/probabilistic_activation_oracles` is cloned:

1. locate the Qwen3.6-27B preset and exact Taboo prompt source;
2. record file paths and commit in `research_log.md`;
3. extract a small published smoke subset to `taboo_published.jsonl`;
4. preserve the original train/evaluation split if the source provides one;
5. ensure each raw record contains prompt ID, secret, split, full messages, and
   source path;
6. render the chat template and check that the target secret is absent before
   calling an example hidden-knowledge evaluation.

Any custom prompts must go in a separate file and be labelled `custom`, never
mixed silently with published prompts.

