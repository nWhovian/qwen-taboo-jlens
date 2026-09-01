#!/usr/bin/env python3
"""Resumable long-running Gold/Blue J-Lens and Logit Lens sweep."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.behavior import require_behavior_approval
from src.experiment_io import open_run, update_manifest
from src.jlens_sanity import require_sanity_approval
from src.lens_export_stable import export_lens_parquet
from src.lens_readout import run_gold_blue_sweep
from src.model_session import load_session


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--conditions", nargs="+", default=["base", "gold", "blue"]
    )
    args = parser.parse_args()

    paths, config = open_run(args.run_id)
    require_behavior_approval(paths, config)
    require_sanity_approval(paths, config)
    prompt_ids = config["prompts"]["groups"]["lens_sweep"]
    print(
        json.dumps(
            {
                "run_id": paths.run_id,
                "prompts": prompt_ids,
                "conditions": args.conditions,
                "resumable_cell_dir": str(paths.lens_dir / "cells"),
            },
            indent=2,
        ),
        flush=True,
    )
    update_manifest(paths, status="lens_sweep_running")
    adapter_words = [name for name in args.conditions if name != "base"]
    session = load_session(
        paths=paths,
        load_lens=True,
        adapter_words=adapter_words,
    )
    statuses = run_gold_blue_sweep(
        session=session,
        paths=paths,
        prompt_ids=prompt_ids,
        conditions=args.conditions,
    )
    status_path = paths.result_dir / "lens_sweep_status.json"
    status_path.write_text(json.dumps(statuses, indent=2), encoding="utf-8")
    parquet = export_lens_parquet(paths)
    update_manifest(
        paths,
        status="lens_sweep_complete",
        lens_parquet=str(parquet),
        completed_sequences=len(statuses),
    )
    print(json.dumps({"status": "complete", "parquet": str(parquet)}, indent=2))


if __name__ == "__main__":
    main()
