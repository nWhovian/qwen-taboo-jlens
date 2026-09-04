#!/usr/bin/env python3
"""Build notebooks 10-11 for the Rock-only J-space experiment."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SWEEP_OUTPUT = ROOT / "notebooks" / "10_rock_jspace_sparse_decomposition.ipynb"
ANALYSIS_OUTPUT = ROOT / "notebooks" / "11_rock_jspace_analysis.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip()}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(text).strip(),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Qwen Taboo J-Lens",
                "language": "python",
                "name": "qwen-taboo-jlens",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SWEEP_CELLS = [
    markdown(
        """
        # 10 — Public J-space on Rock activations: smoke, then 100 × `gen_5`

        **Question.** Does sparse nonnegative decomposition in a J-Lens dictionary
        recover Rock-specific information that ordinary vocabulary ranking misses?

        This is a Rock-adapter-only experiment at the manually fixed source layer 40.
        It never generates new answers: it replays the exact 100 saved standard TEST
        responses from notebook 07 and records generated index 5 (`gen_5`).

        The notebook has two hard stages:

        1. a two-response end-to-end smoke with dictionary/logit and
           TransformerLens-parity gates;
        2. a resumable 100-response full sweep, disabled until the smoke passes.

        Long model work is implemented in `scripts/run_rock_jspace.py` so it can run
        atomically under `tmux`; the key dictionary and pursuit functions remain
        directly inspectable below.
        """
    ),
    markdown(
        """
        ## Frozen design

        - **Adapter condition:** Rock LoRA only.
        - **Examples:** the 100 saved `standard/test/rock` responses; literal leaks are
          retained in raw artifacts and excluded from headline analysis in notebook 11.
        - **Layer:** 40 only; no layer search.
        - **Position:** generated index 5 (`gen_5`) only.
        - **Primary sparse method:** full-vocabulary Gradient Pursuit, `k=16`.
        - **Leakage control:** every token ID emitted anywhere in that response is
          unavailable to the primary sparse support.
        - **Dictionary:** public base-model J-Lens n=1000 only.
        - **Ordinary baselines:** never recomputed here; notebook 11 loads the completed
          notebook-07 and Rock-refit Parquet artifacts from disk.
        - **Robustness subset:** first 10 prompts with GP `k=8/25` and unmasked NNOMP
          `k=16` clearly labelled as a diagnostic, not a headline comparison.

        `gen_5` gives an exact common position with the already saved
        Rock-specific J-Lens evaluation. J-space demonstrates decodability under this
        decomposition; it does not show causal use by the model.
        """
    ),
    code(
        """
        from __future__ import annotations

        import inspect
        import json
        import os
        import shlex
        import subprocess
        import sys
        import time
        from importlib.metadata import version
        from pathlib import Path

        import pandas as pd
        from IPython.display import display

        PROJECT_ROOT = Path.cwd().resolve()
        if PROJECT_ROOT.name == "notebooks":
            PROJECT_ROOT = PROJECT_ROOT.parent
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        from src.experiment_io import load_json
        from src.jspace import (
            build_effective_jlens_dictionary,
            masked_gradient_pursuit,
            response_anchor_indices,
        )

        CONFIG_PATH = PROJECT_ROOT / "configs" / "rock_jspace.json"
        config = load_json(CONFIG_PATH)
        PYTHON = sys.executable
        PROJECT_ROOT
        """
    ),
    markdown(
        """
        ## Inspect the two critical operations

        Qwen applies a learned final RMSNorm before the language-model head. The
        stored RMS parameter is a delta, so the dictionary uses
        `(W_U[token] * (1 + rms_weight)) @ J`; the per-activation
        RMS denominator is positive and does not change vocabulary rank.

        `masked_gradient_pursuit` follows TransformerLens 3.8.1 and adds only an
        excluded-token mask. The smoke requires exact unmasked support parity and
        close coefficients against TransformerLens itself.
        """
    ),
    code(
        """
        print(inspect.getsource(build_effective_jlens_dictionary))
        print(inspect.getsource(masked_gradient_pursuit))
        """
    ),
    markdown(
        """
        ## Dependency and artifact preflight

        This cell is read-only. It reports `waiting_for_rock_n100` while notebook 09's
        standalone refit is still running. Do not launch the smoke until it reports
        `ready`; the runner also refuses to load another 27B model unless at least
        70 GiB of GPU memory is free.
        """
    ),
    code(
        """
        expected_transformer_lens = config["jspace"]["transformer_lens_version"]
        try:
            actual_transformer_lens = version("transformer-lens")
        except Exception:
            actual_transformer_lens = None
        print({
            "expected_transformer_lens": expected_transformer_lens,
            "actual_transformer_lens": actual_transformer_lens,
        })
        preflight = subprocess.run(
            [PYTHON, "scripts/run_rock_jspace.py", "preflight"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )
        print(preflight.stdout)
        if preflight.stderr:
            print(preflight.stderr)
        print("preflight return code:", preflight.returncode)
        """
    ),
    markdown(
        """
        ## Cheap solver tests

        These tests use tiny synthetic dictionaries and no model. They verify named
        anchor selection, nonnegative reconstruction, and that excluded atoms cannot
        enter the support. They are useful before the H100 is free, but they do not
        replace the real full-vocabulary parity smoke.
        """
    ),
    code(
        """
        solver_tests = subprocess.run(
            [PYTHON, "scripts/check_jspace_solver.py"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )
        print(solver_tests.stdout)
        if solver_tests.stderr:
            print(solver_tests.stderr)
        assert solver_tests.returncode == 0
        """
    ),
    markdown(
        """
        ## Stage A — two-response GPU smoke

        The smoke loads pinned Qwen BF16 + Rock, replays two saved responses, records
        2 activations and builds the public J-Lens dictionary. It computes only public
        J-space. One direct ordinary-J-Lens evaluation is used
        solely as a parity assertion and is not saved as a baseline sweep.

        It passes only when:

        - dictionary scores reproduce ordinary J-Lens top-50 rankings;
        - our unmasked GP support matches TransformerLens 3.8.1;
        - emitted token IDs never enter the masked primary supports;
        - every decomposition is finite, nonempty, and uses at most 16 atoms;
        - peak allocated VRAM stays below 75 GiB;
        - all expected J-space rows were saved;
        - no `ordinary_readouts.parquet` was produced.
        """
    ),
    code(
        """
        RUN_GPU_SMOKE = False
        smoke_pointer_path = PROJECT_ROOT / "results" / "latest_rock_jspace_smoke_run.json"

        if smoke_pointer_path.exists():
            smoke_pointer = load_json(smoke_pointer_path)
            smoke_completion = load_json(
                PROJECT_ROOT / "results" / smoke_pointer["run_id"] / "jspace_completion.json"
            )
            display(smoke_completion)
        else:
            assert RUN_GPU_SMOKE, "Wait for Rock n=100, then explicitly enable the two-response smoke."
            subprocess.run(
                [PYTHON, "scripts/run_rock_jspace.py", "smoke"],
                cwd=PROJECT_ROOT,
                check=True,
            )
            smoke_pointer = load_json(smoke_pointer_path)
            smoke_completion = load_json(
                PROJECT_ROOT / "results" / smoke_pointer["run_id"] / "jspace_completion.json"
            )
            display(smoke_completion)
        """
    ),
    markdown(
        """
        ### Inspect the smoke rather than accepting only a green flag

        This shows the two parity records and one paired raw example. The sparse
        support is deliberately uncleaned: punctuation or unusual tokens remain
        visible rather than being removed after the fact.
        """
    ),
    code(
        """
        smoke_dir = PROJECT_ROOT / "results" / smoke_pointer["run_id"]
        parity = load_json(smoke_dir / "dictionary_parity.json")
        jspace_smoke = pd.read_parquet(smoke_dir / "jspace_readouts.parquet")
        display(pd.DataFrame(parity["records"]))
        display(
            jspace_smoke[[
                "prompt_id", "anchor", "method", "target_in_support",
                "target_contribution_share", "jspace_projection_fraction", "support_json",
            ]].head(10)
        )
        """
    ),
    markdown(
        """
        ## Stage B — resumable 100-response `gen_5` sweep

        This stage is intentionally disabled. Enable it only after reviewing the smoke.
        It creates a new immutable run and writes one activation/readout cell per prompt,
        so rerunning the same `FULL_RUN_ID` resumes instead of overwriting completed work.

        Model-facing work must run under `tmux`, not as a fragile long notebook cell.
        The launch cell merely starts the checked-in runner and prints the log command.
        """
    ),
    code(
        """
        START_FULL_RUN = False
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        FULL_RUN_ID = f"run_{stamp}_{config['run_name']}_full"
        TMUX_SESSION = "rock-jspace-full"
        LOG_PATH = PROJECT_ROOT / "logs" / f"{FULL_RUN_ID}.log"
        FULL_COMMAND = [
            PYTHON,
            "scripts/run_rock_jspace.py",
            "full",
            "--run-id",
            FULL_RUN_ID,
        ]
        print("run id:", FULL_RUN_ID)
        print("command:", shlex.join(FULL_COMMAND))
        print("log:", LOG_PATH)

        if START_FULL_RUN:
            assert smoke_completion["status"] == "passed"
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            shell_command = f"{shlex.join(FULL_COMMAND)} 2>&1 | tee {shlex.quote(str(LOG_PATH))}"
            subprocess.run(
                ["tmux", "new-session", "-d", "-s", TMUX_SESSION, shell_command],
                cwd=PROJECT_ROOT,
                check=True,
            )
            print("started; monitor with:", f"tmux attach -t {TMUX_SESSION}")
        """
    ),
    markdown(
        """
        ## Monitor or resume

        Do not create a second run when a full run already exists. Reuse its exact
        `FULL_RUN_ID`; completed prompt cells are skipped. The final pointer is written
        only after all integrity gates pass.
        """
    ),
    code(
        """
        full_pointer_path = PROJECT_ROOT / "results" / "latest_rock_jspace_run.json"
        if full_pointer_path.exists():
            full_pointer = load_json(full_pointer_path)
            full_completion_path = (
                PROJECT_ROOT / "results" / full_pointer["run_id"] / "jspace_completion.json"
            )
            display(load_json(full_completion_path))
        else:
            candidate_manifest = PROJECT_ROOT / "results" / FULL_RUN_ID / "manifest.json"
            if candidate_manifest.exists():
                display(load_json(candidate_manifest))
            else:
                print("No full run started. This is the intended hand-off state after smoke.")
        """
    ),
    markdown(
        """
        ## Output contract for notebook 11

        A passing full run contains:

        - `activation_index.parquet` — exact prompt/anchor/position provenance;
        - `jspace_readouts.parquet` — public J-space GP `k=16`, emitted-ID masked;
        - `jspace_robustness.parquet` — GP `k=8/25` and clearly labelled unmasked NNOMP;
        - `jspace_completion.json` — counts, wall time, peak VRAM, source smoke, and gates.

        Dictionary/TransformerLens parity lives in the passing smoke run. Logit Lens,
        public J-Lens, and Rock-specific J-Lens are loaded from their earlier immutable
        Parquet runs by notebook 11; the full J-space run does not recompute them.

        Raw activations and per-prompt cells remain in ignored artifact directories;
        aggregated tables remain in the immutable run directory.
        """
    ),
]


ANALYSIS_CELLS = [
    markdown(
        """
        # 11 — Rock comparison: Logit Lens, two J-Lenses, and public J-space

        This CPU-only notebook combines a new J-space run with ordinary readouts that
        were already saved by notebooks 07 and 09:

        1. Logit Lens;
        2. public base-model J-Lens n=1000;
        3. Rock-specific J-Lens n=100;
        4. public base-model J-space GP `k=16`.

        **No ordinary lens is recomputed here or in the full J-space run.** Logit Lens
        and public J-Lens have the exact `gen_5` position in the completed 20-adapter
        sweep. The saved Rock-specific ordinary J-Lens has `gen_5` and response-average;
        we use only `gen_5`, so every direct comparison is position matched.
        """
    ),
    code(
        """
        from __future__ import annotations

        import json
        import os
        import sys
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from IPython.display import display

        PROJECT_ROOT = Path.cwd().resolve()
        if PROJECT_ROOT.name == "notebooks":
            PROJECT_ROOT = PROJECT_ROOT.parent
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        from src.experiment_io import load_json, utc_now
        from src.jspace import response_anchor_indices

        CONFIG_PATH = PROJECT_ROOT / "configs" / "rock_jspace.json"
        config = load_json(CONFIG_PATH)
        rng = np.random.default_rng(config["seed"])
        """
    ),
    markdown(
        """
        ## Select one completed immutable run

        Set `JSPACE_RUN_ID` explicitly for a historical run. Leaving it as `None`
        follows the latest passing full-run pointer. Analysis refuses smoke artifacts.
        """
    ),
    code(
        """
        JSPACE_RUN_ID = None
        if JSPACE_RUN_ID is None:
            pointer = load_json(PROJECT_ROOT / "results" / "latest_rock_jspace_run.json")
            JSPACE_RUN_ID = pointer["run_id"]
        result_dir = PROJECT_ROOT / "results" / JSPACE_RUN_ID
        figure_dir = PROJECT_ROOT / "figures" / JSPACE_RUN_ID
        figure_dir.mkdir(parents=True, exist_ok=True)
        completion = load_json(result_dir / "jspace_completion.json")
        assert completion["stage"] == "full" and completion["status"] == "passed"
        assert completion["responses"] == config["evaluation"]["responses"]
        display(completion)
        """
    ),
    markdown(
        """
        ## Load the new J-space rows and already saved ordinary rows

        The source completion files are hard gates. Kernel state is irrelevant: all
        inputs below are immutable Parquet/JSON files under `results/` and `artifacts/`.
        """
    ),
    code(
        """
        jspace = pd.read_parquet(result_dir / "jspace_readouts.parquet")
        robustness = pd.read_parquet(result_dir / "jspace_robustness.parquet")
        activation_index = pd.read_parquet(result_dir / "activation_index.parquet")

        n_responses = config["evaluation"]["responses"]
        n_anchors = len(config["evaluation"]["anchors"])
        assert len(activation_index) == n_responses * n_anchors
        assert len(jspace) == n_responses * n_anchors * 2
        assert not bool(jspace["emitted_token_selected"].any())

        ordinary_config = config["ordinary_results"]
        ordinary_test_pointer = load_json(PROJECT_ROOT / ordinary_config["full_test_pointer"])
        ordinary_test_run_id = ordinary_test_pointer["run_id"]
        ordinary_test_dir = PROJECT_ROOT / "results" / ordinary_test_run_id
        ordinary_test_completion = load_json(
            ordinary_test_dir / ordinary_config["full_test_completion_filename"]
        )
        assert ordinary_test_completion["completed_sequences"] == ordinary_test_completion["expected_sequences"]
        ordinary_cells_dir = PROJECT_ROOT / ordinary_config["full_test_cells_relative_path"].format(
            run_id=ordinary_test_run_id
        )
        rock_position_files = sorted(ordinary_cells_dir.glob("standard_test_*__rock.positions.parquet"))
        assert len(rock_position_files) == n_responses

        position_columns = [
            "prompt_id", "prompt_type", "split", "condition", "target_word",
            "generation_token_count", "own_secret_leaked", "method", "layer",
            "mask_protocol", "relative_response_position", "position_from_prompt_end",
            "target_rank", "target_reciprocal_rank", "target_log10_rank",
            "target_probability", "target_probability_mass", "target_hit_top1",
            "target_hit_top5", "target_hit_top10", "target_candidate_rank_20",
            "best_wrong_candidate_probability", "top10_json",
        ]
        ordinary_exact_frames = []
        for path in rock_position_files:
            frame = pd.read_parquet(path, columns=position_columns)
            frame = frame[
                frame["condition"].eq("rock")
                & frame["layer"].eq(config["evaluation"]["source_layer"])
                & frame["method"].isin(["logit_lens", "jlens"])
                & frame["relative_response_position"].notna()
            ].copy()
            generation_length = int(frame["generation_token_count"].iloc[0])
            anchor_index = pd.DataFrame(
                response_anchor_indices(generation_length, config["evaluation"]["anchors"]),
                columns=["anchor", "relative_response_position"],
            )
            frame = frame.merge(anchor_index, on="relative_response_position", how="inner")
            ordinary_exact_frames.append(frame)

        ordinary_exact = pd.concat(ordinary_exact_frames, ignore_index=True)
        ordinary_exact["method"] = ordinary_exact["method"].replace(
            {"jlens": "public_base_jlens_n1000"}
        )
        ordinary_exact["target_hit_top16"] = ordinary_exact["target_rank"].le(16)
        ordinary_exact["target_candidate_rank"] = ordinary_exact["target_candidate_rank_20"]
        ordinary_exact["target_candidate_top1"] = ordinary_exact["target_candidate_rank"].eq(1)
        ordinary_exact["target_candidate_margin"] = (
            ordinary_exact["target_probability"]
            - ordinary_exact["best_wrong_candidate_probability"]
        )
        ordinary_exact["target_probability_mass_unmasked"] = ordinary_exact["target_probability_mass"]
        ordinary_exact["position_scope"] = "gen_5_from_notebook07"
        ordinary_exact["ordinary_source_run_id"] = ordinary_test_run_id

        ordinary_refit_pointer = load_json(PROJECT_ROOT / ordinary_config["rock_refit_pointer"])
        ordinary_refit_run_id = ordinary_refit_pointer["run_id"]
        ordinary_refit_path = (
            PROJECT_ROOT / "results" / ordinary_refit_run_id
            / ordinary_config["rock_refit_readouts_filename"]
        )
        rock_ordinary = pd.read_parquet(ordinary_refit_path)
        rock_ordinary = rock_ordinary[
            rock_ordinary["selection_role"].eq("primary")
            & rock_ordinary["condition"].eq("rock")
            & rock_ordinary["method"].eq("own_adapter_jlens_n100")
            & rock_ordinary["anchor"].eq("gen_5")
            & rock_ordinary["layer"].eq(config["evaluation"]["source_layer"])
        ].copy()
        rock_ordinary["method"] = "rock_adapter_jlens_n100"
        rock_ordinary["target_hit_top10"] = rock_ordinary["target_rank"].le(10)
        rock_ordinary["target_hit_top16"] = rock_ordinary["target_rank"].le(16)
        rock_ordinary["target_candidate_rank"] = np.nan
        rock_ordinary["target_candidate_top1"] = np.nan
        rock_ordinary["target_candidate_margin"] = np.nan
        rock_ordinary["target_probability_mass_unmasked"] = rock_ordinary["target_probability_mass"]
        rock_ordinary["position_scope"] = "gen_5_only_from_notebook09"
        rock_ordinary["ordinary_source_run_id"] = ordinary_refit_run_id

        ordinary = pd.concat([ordinary_exact, rock_ordinary], ignore_index=True, sort=False)
        assert len(ordinary_exact) == n_responses * n_anchors * 2
        assert len(rock_ordinary) == n_responses
        assert ordinary["prompt_id"].nunique() == jspace["prompt_id"].nunique() == n_responses
        assert set(ordinary["prompt_id"]) == set(jspace["prompt_id"])

        smoke_run_id = completion["source_smoke_run_id"]
        smoke_dir = PROJECT_ROOT / "results" / smoke_run_id
        smoke_completion = load_json(smoke_dir / "jspace_completion.json")
        assert smoke_completion["implementation_hash"] == completion["implementation_hash"]
        assert smoke_completion["input_identity_hash"] == completion["input_identity_hash"]
        parity = load_json(smoke_dir / "dictionary_parity.json")
        assert all(
            record["dictionary_top1_exact"]
            and record["dictionary_top10_set_exact"]
            and record["dictionary_top50_overlap"] >= 0.98
            for record in parity["records"]
        )
        display(pd.DataFrame(parity["records"]))
        display(pd.DataFrame({
            "artifact": [
                "new activations", "saved Logit/public J-Lens", "saved Rock J-Lens",
                "new J-space", "new robustness",
            ],
            "rows": [
                len(activation_index), len(ordinary_exact), len(rock_ordinary),
                len(jspace), len(robustness),
            ],
            "run_id": [
                JSPACE_RUN_ID, ordinary_test_run_id, ordinary_refit_run_id,
                JSPACE_RUN_ID, JSPACE_RUN_ID,
            ],
        }))
        """
    ),
    markdown(
        """
        ## Unified recovery endpoint

        Ordinary methods recover Rock when its best single-token surface form is in
        vocabulary rank 1–16. J-space recovers Rock when a Rock token is in its active
        support of at most 16 atoms. This `recovery@16` endpoint is the only direct
        cross-family comparison. Comparisons involving Rock-specific ordinary J-Lens
        are restricted to the exact shared `gen_5` anchor.
        """
    ),
    code(
        """
        ordinary_common = ordinary.assign(
            recovered_at_16=ordinary["target_hit_top16"].astype(bool),
            method_family="ordinary",
        )
        jspace_common = jspace.assign(
            recovered_at_16=jspace["target_in_support"].astype(bool),
            method_family="jspace",
            position_scope="gen_5_new_jspace",
        )
        common_columns = [
            "prompt_id", "anchor", "method", "method_family", "own_secret_leaked",
            "position_scope", "recovered_at_16", "target_candidate_top1",
            "target_candidate_rank",
        ]
        common = pd.concat(
            [ordinary_common[common_columns], jspace_common[common_columns]],
            ignore_index=True,
        )
        headline = common[~common["own_secret_leaked"]].copy()
        method_order = config["evaluation"]["ordinary_methods"] + config["evaluation"]["jspace_methods"]
        headline["method"] = pd.Categorical(headline["method"], method_order, ordered=True)

        common_summary = (
            headline.groupby(["method", "method_family"], observed=True, as_index=False)
            .agg(
                prompts=("prompt_id", "nunique"),
                anchors=("anchor", "nunique"),
                prompt_positions=("recovered_at_16", "size"),
                recovery_at_16=("recovered_at_16", "mean"),
                correct_taboo_candidate_rate=("target_candidate_top1", "mean"),
                median_taboo_candidate_rank=("target_candidate_rank", "median"),
            )
        )
        gen5_summary = (
            headline[headline["anchor"].eq("gen_5")]
            .groupby(["method", "method_family"], observed=True, as_index=False)
            .agg(
                prompts=("prompt_id", "nunique"),
                recovery_at_16=("recovered_at_16", "mean"),
                correct_taboo_candidate_rate=("target_candidate_top1", "mean"),
            )
        )
        common_summary.to_csv(result_dir / "jspace_common_recovery_summary.csv", index=False)
        display(common_summary)
        display(gen5_summary)
        """
    ),
    markdown(
        """
        ## Method-specific metrics

        Rank, reciprocal rank, and vocabulary probability are meaningful for ordinary
        lenses. Sparse coefficient share, explained projection, concentration, and
        support size are meaningful for J-space. They are reported side by side but
        are not treated as numerically interchangeable.
        """
    ),
    code(
        """
        ordinary_valid = ordinary[~ordinary["own_secret_leaked"]].copy()
        ordinary_summary = (
            ordinary_valid.groupby("method", as_index=False)
            .agg(
                prompts=("prompt_id", "nunique"),
                anchors=("anchor", "nunique"),
                prompt_positions=("target_rank", "size"),
                mean_reciprocal_rank=("target_reciprocal_rank", "mean"),
                median_target_rank=("target_rank", "median"),
                recall_at_1=("target_hit_top1", "mean"),
                recall_at_5=("target_hit_top5", "mean"),
                recall_at_10=("target_hit_top10", "mean"),
                recall_at_16=("target_hit_top16", "mean"),
                mean_target_probability=("target_probability_mass_unmasked", "mean"),
                correct_taboo_candidate_rate=("target_candidate_top1", "mean"),
                mean_taboo_candidate_margin=("target_candidate_margin", "mean"),
            )
        )
        jspace_valid = jspace[~jspace["own_secret_leaked"]].copy()
        jspace_summary = (
            jspace_valid.groupby("method", as_index=False)
            .agg(
                prompts=("prompt_id", "nunique"),
                secret_support_rate=("target_in_support", "mean"),
                mean_secret_contribution_share=("target_contribution_share", "mean"),
                correct_taboo_candidate_rate=("target_candidate_top1", "mean"),
                mean_taboo_candidate_margin=("target_candidate_margin", "mean"),
                mean_nonnegative_reconstruction=("nonnegative_reconstruction_fraction", "mean"),
                mean_jspace_projection=("jspace_projection_fraction", "mean"),
                mean_support_size=("support_size", "mean"),
                mean_effective_support_size=("effective_support_size", "mean"),
                mean_top1_contribution_share=("top1_contribution_share", "mean"),
            )
        )
        ordinary_summary.to_csv(result_dir / "ordinary_method_metrics.csv", index=False)
        jspace_summary.to_csv(result_dir / "jspace_method_metrics.csv", index=False)
        display(ordinary_summary)
        display(jspace_summary)
        """
    ),
    markdown(
        """
        ## Paired hypothesis tests with prompt-level bootstrap

        Every method uses the exact same `gen_5` activation. We bootstrap prompts;
        positions are not pooled or averaged.

        - **H1:** public J-space improves over ordinary public J-Lens.
        - **H2:** Rock-specific ordinary J-Lens improves over public J-Lens.
        - Logit → public J-Lens and Rock J-Lens → public J-space are context comparisons.
        """
    ),
    code(
        """
        def paired_prompt_bootstrap(
            frame, baseline, challenger, metric, anchors, draws=10_000
        ):
            subset = frame[
                frame["method"].isin([baseline, challenger])
                & frame["anchor"].isin(anchors)
            ]
            prompt_means = (
                subset.groupby(["prompt_id", "method"], observed=True)[metric]
                .mean()
                .unstack("method")
                .dropna(subset=[baseline, challenger])
            )
            if prompt_means.empty:
                raise RuntimeError(f"No paired rows for {baseline} -> {challenger}, {metric}")
            differences = (prompt_means[challenger] - prompt_means[baseline]).to_numpy(float)
            bootstrap = np.array([
                rng.choice(differences, size=len(differences), replace=True).mean()
                for _ in range(draws)
            ])
            return {
                "baseline": baseline,
                "challenger": challenger,
                "metric": metric,
                "anchors": ",".join(anchors),
                "prompts": len(differences),
                "baseline_mean": float(prompt_means[baseline].mean()),
                "challenger_mean": float(prompt_means[challenger].mean()),
                "paired_delta": float(differences.mean()),
                "ci_low": float(np.quantile(bootstrap, 0.025)),
                "ci_high": float(np.quantile(bootstrap, 0.975)),
                "prompt_win_rate": float((differences > 0).mean()),
                "prompt_tie_rate": float((differences == 0).mean()),
            }

        comparison_anchors = ["gen_5"]
        recovery_comparisons = [
            ("logit_lens", "public_base_jlens_n1000", comparison_anchors),
            ("public_base_jlens_n1000", "rock_adapter_jlens_n100", comparison_anchors),
            ("public_base_jlens_n1000", "public_base_jspace_gp_k16", comparison_anchors),
            ("rock_adapter_jlens_n100", "public_base_jspace_gp_k16", comparison_anchors),
        ]
        paired_rows = [
            paired_prompt_bootstrap(
                headline, baseline, challenger, "recovered_at_16", anchors
            )
            for baseline, challenger, anchors in recovery_comparisons
        ]
        candidate_comparisons = [
            item
            for item in recovery_comparisons
            if "rock_adapter_jlens_n100" not in item[:2]
        ]
        paired_rows += [
            paired_prompt_bootstrap(
                headline, baseline, challenger, "target_candidate_top1", anchors
            )
            for baseline, challenger, anchors in candidate_comparisons
        ]
        paired = pd.DataFrame(paired_rows)
        paired.to_csv(result_dir / "jspace_paired_hypothesis_comparisons.csv", index=False)
        display(paired)
        """
    ),
    markdown(
        """
        ## Where does sparse decomposition add or lose recovery?

        `jspace_only` is the main claimed benefit: Rock enters the sparse support when
        its ordinary rank is worse than 16. `ordinary_only` is an equally important
        failure mode and is never hidden by aggregate averages.
        """
    ),
    code(
        """
        def recovery_contingency(frame, ordinary_method, jspace_method, anchors):
            subset = frame[
                frame["method"].isin([ordinary_method, jspace_method])
                & frame["anchor"].isin(anchors)
            ]
            pivot = subset.pivot(
                index=["prompt_id", "anchor"], columns="method", values="recovered_at_16"
            ).dropna()
            ordinary_hit = pivot[ordinary_method].astype(bool)
            jspace_hit = pivot[jspace_method].astype(bool)
            labels = np.select(
                [ordinary_hit & jspace_hit, ~ordinary_hit & jspace_hit, ordinary_hit & ~jspace_hit],
                ["both", "jspace_only", "ordinary_only"],
                default="neither",
            )
            rows = pivot.reset_index()[["prompt_id", "anchor"]]
            rows["outcome"] = labels
            rows["ordinary_method"] = ordinary_method
            rows["jspace_method"] = jspace_method
            return rows

        public_contingency = recovery_contingency(
            headline, "public_base_jlens_n1000", "public_base_jspace_gp_k16", comparison_anchors
        )
        rock_contingency = recovery_contingency(
            headline, "rock_adapter_jlens_n100", "public_base_jspace_gp_k16", comparison_anchors
        )
        contingencies = pd.concat([public_contingency, rock_contingency], ignore_index=True)
        contingency_summary = (
            contingencies.groupby(["ordinary_method", "jspace_method", "outcome"], as_index=False)
            .size()
        )
        contingencies.to_csv(result_dir / "jspace_recovery_contingency_rows.csv", index=False)
        display(contingency_summary)
        """
    ),
    markdown(
        """
        ## Position-matched summary

        Every row below is `gen_5`; there is no cross-position pooling.
        """
    ),
    code(
        """
        anchor_summary = (
            headline.groupby(["anchor", "method"], observed=True, as_index=False)
            .agg(
                prompts=("prompt_id", "nunique"),
                recovery_at_16=("recovered_at_16", "mean"),
                correct_taboo_candidate_rate=("target_candidate_top1", "mean"),
            )
        )
        anchor_summary.to_csv(result_dir / "jspace_metrics_by_anchor.csv", index=False)
        display(anchor_summary)
        """
    ),
    markdown(
        """
        ## Robustness: sparsity and coefficient solver

        GP `k=8/25` keeps the emitted-token mask. NNOMP is unmasked because the public
        TransformerLens API has no exclusion argument; it is therefore a diagnostic
        only. Any emitted-token selections are shown explicitly.
        """
    ),
    code(
        """
        robustness_valid = robustness[~robustness["own_secret_leaked"]].copy()
        robustness_summary = (
            robustness_valid.groupby(["method", "algorithm", "k", "masked"], as_index=False)
            .agg(
                prompts=("prompt_id", "nunique"),
                secret_support_rate=("target_in_support", "mean"),
                mean_secret_contribution_share=("target_contribution_share", "mean"),
                mean_jspace_projection=("jspace_projection_fraction", "mean"),
                emitted_token_selection_rate=("emitted_token_selected", "mean"),
                mean_support_size=("support_size", "mean"),
            )
        )
        robustness_summary.to_csv(result_dir / "jspace_robustness_summary.csv", index=False)
        display(robustness_summary)
        """
    ),
    markdown(
        """
        ## Raw-example inspection

        Prefer `gen_5` examples where public J-space adds recovery over the already
        saved public ordinary J-Lens.
        If none exist, inspect the worst ordinary ranks instead. The full uncleaned
        sparse support and ordinary top-10 are displayed together.
        """
    ),
    code(
        """
        public_added = public_contingency[public_contingency["outcome"].eq("jspace_only")]
        if len(public_added):
            chosen = public_added.head(10)[["prompt_id", "anchor"]]
        else:
            chosen = (
                ordinary_valid[ordinary_valid["method"].eq("public_base_jlens_n1000")]
                .sort_values("target_rank", ascending=False)
                .head(10)[["prompt_id", "anchor"]]
            )
        ordinary_examples = chosen.merge(
            ordinary_valid[ordinary_valid["method"].eq("public_base_jlens_n1000")],
            on=["prompt_id", "anchor"],
        )[["prompt_id", "anchor", "target_rank", "top10_json"]]
        jspace_examples = chosen.merge(
            jspace_valid[jspace_valid["method"].eq("public_base_jspace_gp_k16")],
            on=["prompt_id", "anchor"],
        )[[
            "prompt_id", "anchor", "target_in_support", "target_contribution_share",
            "jspace_projection_fraction", "support_json",
        ]]
        display(ordinary_examples)
        display(jspace_examples)
        """
    ),
    markdown("## Figures"),
    code(
        """
        sns.set_theme(style="whitegrid")
        overall_plot = gen5_summary.copy()
        plt.figure(figsize=(11, 4.5))
        ax = sns.barplot(data=overall_plot, x="method", y="recovery_at_16", order=method_order)
        ax.set_ylim(0, 1)
        ax.set_xlabel("")
        ax.set_ylabel("Rock recovery@16")
        ax.tick_params(axis="x", rotation=25)
        plt.tight_layout()
        recovery_figure = figure_dir / "rock_recovery_at16_by_method_gen5.png"
        plt.savefig(recovery_figure, dpi=180, bbox_inches="tight")
        plt.show()
        """
    ),
    markdown(
        """
        ## Hypothesis verdicts and limitations

        `supported_exploratorily` requires a positive paired mean and a 95% bootstrap
        interval above zero. Otherwise the notebook reports `not_supported` or
        `inconclusive`; it never rewrites an exploratory result as confirmatory.

        Rock and layer 40 were selected after earlier TEST inspection, and the same
        saved TEST responses are reused. A positive result therefore motivates a fresh
        held-out Rock set; it does not complete a confirmatory replication.
        """
    ),
    code(
        """
        recovery_pairs = paired[paired["metric"].eq("recovered_at_16")].copy()
        hypotheses = [
            {
                "hypothesis": "H1 public J-space adds Rock recovery over public J-Lens",
                "baseline": "public_base_jlens_n1000",
                "challenger": "public_base_jspace_gp_k16",
            },
            {
                "hypothesis": "H2 Rock refit beats public ordinary J-Lens",
                "baseline": "public_base_jlens_n1000",
                "challenger": "rock_adapter_jlens_n100",
            },
        ]
        verdict_rows = []
        for hypothesis in hypotheses:
            row = recovery_pairs[
                recovery_pairs["baseline"].eq(hypothesis["baseline"])
                & recovery_pairs["challenger"].eq(hypothesis["challenger"])
            ].iloc[0]
            if row["ci_low"] > 0:
                verdict = "supported_exploratorily"
            elif row["paired_delta"] <= 0:
                verdict = "not_supported"
            else:
                verdict = "inconclusive"
            verdict_rows.append({**hypothesis, **row.to_dict(), "verdict": verdict})
        verdicts = pd.DataFrame(verdict_rows)
        verdicts.to_csv(result_dir / "jspace_hypothesis_verdicts.csv", index=False)
        display(verdicts)

        analysis_completion = {
            "status": "complete",
            "run_id": JSPACE_RUN_ID,
            "completed_utc": utc_now(),
            "headline_excludes_literal_leaks": True,
            "headline_prompts": int(headline["prompt_id"].nunique()),
            "methods": method_order,
            "ordinary_test_run_id": ordinary_test_run_id,
            "ordinary_rock_refit_run_id": ordinary_refit_run_id,
            "hypotheses": verdict_rows,
            "limitations": [
                "Rock and layer 40 were selected after prior TEST inspection.",
                "J-space decomposition is a decodability result, not causal evidence.",
                "The Rock J-Lens n=100 fit is smaller than the public n=1000 fit.",
                "All direct method comparisons use the same saved gen_5 position.",
                "Ordinary readouts are loaded from completed notebook-07 and notebook-09 Parquet files and are not recomputed.",
                "NNOMP robustness rows are unmasked and diagnostic only.",
            ],
        }
        temporary = result_dir / "jspace_analysis_completion.json.tmp"
        temporary.write_text(json.dumps(analysis_completion, ensure_ascii=False, indent=2))
        os.replace(temporary, result_dir / "jspace_analysis_completion.json")
        display(analysis_completion)
        """
    ),
]


def main() -> None:
    SWEEP_OUTPUT.write_text(
        json.dumps(notebook(SWEEP_CELLS), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    ANALYSIS_OUTPUT.write_text(
        json.dumps(notebook(ANALYSIS_CELLS), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(SWEEP_OUTPUT)
    print(ANALYSIS_OUTPUT)


if __name__ == "__main__":
    main()
