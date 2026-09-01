#!/usr/bin/env python3
"""Extract the published Taboo prompt splits from a source checkout.

This script does not clone or install the upstream project. Point it at an
already-inspected checkout of probabilistic_activation_oracles. The output is
a mechanical JSONL rendering of the four upstream text files, with a source
commit and line number on every row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


SPLITS = {
    "direct_test": "taboo_direct_test.txt",
    "direct_val": "taboo_direct_val.txt",
    "standard_test": "taboo_standard_test.txt",
    "standard_val": "taboo_standard_val.txt",
}
EXPECTED_COUNTS = {
    "direct_test": 100,
    "direct_val": 20,
    "standard_test": 100,
    "standard_val": 50,
}


def git_text(repo: Path, revision: str, path: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{path}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def git_sha(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        required=True,
        help="Checkout root containing the activation_oracles submodule",
    )
    parser.add_argument(
        "--output", default="data/prompts/taboo_published.jsonl"
    )
    parser.add_argument(
        "--provenance-output",
        default="data/prompts/taboo_published.provenance.json",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    parent_sha = git_sha(source_root)
    submodule = source_root / "activation_oracles"
    submodule_sha = git_sha(submodule)
    dataset_root = "datasets/taboo"

    records: list[dict[str, object]] = []
    observed_counts: dict[str, int] = {}
    for group, filename in SPLITS.items():
        source_path = f"{dataset_root}/{filename}"
        prompts = git_text(submodule, submodule_sha, source_path).splitlines()
        prompts = [prompt for prompt in prompts if prompt.strip()]
        observed_counts[group] = len(prompts)
        if observed_counts[group] != EXPECTED_COUNTS[group]:
            raise RuntimeError(
                f"Unexpected {group} count: {observed_counts[group]} != "
                f"{EXPECTED_COUNTS[group]}"
            )
        prompt_type, split = group.split("_", maxsplit=1)
        for index, prompt in enumerate(prompts):
            if not prompt.strip():
                continue
            records.append(
                {
                    "prompt_id": f"{group}_{index:03d}",
                    "prompt_type": prompt_type,
                    "split": split,
                    "messages": [{"role": "user", "content": prompt}],
                    "source_repository": (
                        "https://github.com/federicotorrielli/"
                        "probabilistic_activation_oracles"
                    ),
                    "source_parent_commit": parent_sha,
                    "source_submodule_commit": submodule_sha,
                    "source_path": f"activation_oracles/{source_path}",
                    "source_line": index + 1,
                    "custom": False,
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    provenance = {
        "output": str(output),
        "records": len(records),
        "counts": observed_counts,
        "source_repository": (
            "https://github.com/federicotorrielli/"
            "probabilistic_activation_oracles"
        ),
        "parent_commit": parent_sha,
        "submodule_commit": submodule_sha,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    provenance_output = Path(args.provenance_output)
    provenance_output.parent.mkdir(parents=True, exist_ok=True)
    provenance_output.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(
        json.dumps({**provenance, "provenance_output": str(provenance_output)}, indent=2)
    )


if __name__ == "__main__":
    main()
