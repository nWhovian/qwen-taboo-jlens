# Published Taboo prompts

`taboo_published.jsonl` is a mechanical extraction of all four upstream prompt
files, not a custom prompt set. It contains 270 records:

- standard test: 100;
- standard validation: 50;
- direct test: 100;
- direct validation: 20.

Each record stores prompt type/split, chat messages, source file and line,
parent repository commit `d1a8eb25e6ec6d171dbd315b929a3728bc0fa7cf`,
activation-oracles submodule commit
`c8940e59f141718d37ef54cc7f5f8d04879a89bd`, and `custom=false`.

To regenerate from an audited checkout of
`federicotorrielli/probabilistic_activation_oracles`, run:

```bash
python scripts/sync_published_prompts.py --source-root /path/to/checkout
```

The script refuses unexpected split sizes and writes a companion provenance
JSON. The current checked-in JSONL itself also carries provenance on every row.
