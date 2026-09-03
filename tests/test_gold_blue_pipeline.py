from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import nbformat
import pandas as pd
import torch

from src.experiment_io import PROJECT_ROOT, RunPaths, load_json
from src.lens_export import _flatten
from src.lens_export_stable import export_lens_parquet
from src.lens_readout import _position_roles, measured_positions, quantile_positions
from src.model_session import ModelSession
from src.prompt_data import lexical_leaks, load_prompts, select_prompts
from src.preflight import runtime_dependency_preflight, static_preflight


def fake_lens_record(*, condition: str, target: str | None) -> dict:
    adapter = f"example/{condition}" if target else None
    return {
        "schema_version": 1,
        "run_id": "test_run",
        "prompt_id": "standard_test_000",
        "prompt_type": "standard",
        "split": "test",
        "condition": condition,
        "target_word": target,
        "source_path": "published/prompts.txt",
        "source_line": 1,
        "source_submodule_commit": "a" * 40,
        "base_model_repo_id": "Qwen/Qwen3.6-27B",
        "base_model_revision": "b" * 40,
        "tokenizer_repo_id": "Qwen/Qwen3.6-27B",
        "tokenizer_revision": "b" * 40,
        "adapter_repo_id": adapter,
        "adapter_revision": "c" * 40 if adapter else None,
        "jlens_repo_id": "neuronpedia/jacobian-lens",
        "jlens_revision": "d" * 40,
        "jlens_filename": "qwen3.6-27b/jlens/example.pt",
        "jlens_code_commit": "e" * 40,
        "runtime_dtype": "bfloat16",
        "attention_implementation": "flash_attention_2",
        "seed": 7,
        "method": "jlens",
        "layer": 48,
        "position": 9,
        "position_roles": ["last_input", "last_input_window"],
        "relative_generated_position": None,
        "token_id": 42,
        "token": " token",
        "own_secret_leaked": False,
        "output_leaks": [],
        "candidate_logits": {"gold": 3.0, "blue": 1.0},
        "candidate_ranks": {"gold": 1, "blue": 2},
        "full_vocabulary": {
            "candidates": {"gold": {"rank": 4}, "blue": {"rank": 9}},
            "top_k": [{"token_id": 1, "token": "x", "logit": 5.0}],
        },
    }


class GoldBluePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_json(PROJECT_ROOT / "configs/gold_blue_experiment.json")
        cls.prompts = load_prompts(cls.config["prompts"]["path"])

    def test_prompt_catalog_and_groups_are_published_and_leak_free(self) -> None:
        self.assertEqual(len(self.prompts), 270)
        groups = self.config["prompts"]["groups"]
        self.assertTrue(set(groups["manual_smoke"]).issubset(groups["behavior_batch"]))
        self.assertTrue(set(groups["lens_sweep"]).issubset(groups["behavior_batch"]))
        self.assertEqual(len(groups["behavior_batch"]), 20)
        self.assertEqual(len(groups["lens_sweep"]), 10)
        self.assertEqual(self.config["behavior"]["initial_conditions"], ["base", "gold"])
        selected = select_prompts(self.prompts, groups["behavior_batch"])
        for prompt in selected:
            text = "\n".join(message["content"] for message in prompt["messages"])
            self.assertEqual(lexical_leaks(text, ["gold", "blue"]), [])
            self.assertFalse(prompt["custom"])

    def test_validation_protocol_is_validation_only_and_pins_moon(self) -> None:
        validation = load_json(
            PROJECT_ROOT / "configs/gold_blue_moon_validation.json"
        )
        groups = validation["prompts"]["groups"]
        standard_ids = groups["validation_standard"]
        direct_ids = groups["validation_direct"]
        selected_ids = standard_ids + direct_ids
        self.assertEqual(len(standard_ids), 30)
        self.assertEqual(len(direct_ids), 10)
        self.assertEqual(len(selected_ids), len(set(selected_ids)))
        self.assertEqual(
            validation["behavior"]["validation_conditions"],
            ["gold", "blue", "moon"],
        )
        self.assertEqual(
            validation["adapters"]["moon"]["revision"],
            "26a864ed87935c51999f3e3b3b151201feb0fbfc",
        )
        self.assertTrue(validation["readout"]["exclude_all_emitted_token_ids"])
        readout = validation["readout"]
        self.assertEqual(readout["exact_map_tokens_before_response"], 8)
        self.assertEqual(readout["exact_map_tokens_after_response"], 32)
        self.assertEqual(readout["selection_min_examples_per_condition"], 24)
        selected = select_prompts(self.prompts, selected_ids)
        self.assertTrue(all(prompt["split"] == "val" for prompt in selected))
        self.assertTrue(all("_test_" not in prompt["prompt_id"] for prompt in selected))
        for prompt in selected:
            text = "\n".join(message["content"] for message in prompt["messages"])
            self.assertEqual(lexical_leaks(text, ["gold", "blue", "moon"]), [])

    def test_static_preflight_includes_artifact_and_prompt_hashes(self) -> None:
        report = static_preflight()
        self.assertTrue(report["passed"])
        self.assertTrue(report["prompt_provenance_matches"])
        self.assertTrue(report["artifact_preflight"]["passed"])

    def test_runtime_preflight_reports_flash_attention_explicitly(self) -> None:
        report = runtime_dependency_preflight()
        self.assertIn("flash_attn_2_available", report)
        self.assertEqual(
            report["passed"],
            bool(report["cuda_available"] and report["flash_attn_2_available"]),
        )

    def test_position_selection_and_roles(self) -> None:
        self.assertEqual(
            measured_positions(prompt_length=5, sequence_length=8, input_window=2),
            [3, 4, 5, 6, 7],
        )
        self.assertEqual(
            quantile_positions(5, 9, [0.0, 0.5, 1.0]),
            [5, 6, 8],
        )
        self.assertIn(
            "last_input",
            _position_roles(4, prompt_length=5, sequence_length=8, input_window=2),
        )
        self.assertIn(
            "first_generated",
            _position_roles(5, prompt_length=5, sequence_length=8, input_window=2),
        )

    def test_loaded_adapter_parameter_audit_requires_nonzero_lora_b(self) -> None:
        adapter_name = "adapter__gold"

        class FakeModel:
            def __init__(self, b_value: float) -> None:
                self.parameters = {
                    f"layer.lora_A.{adapter_name}.weight": torch.ones(2, 2),
                    f"layer.lora_B.{adapter_name}.weight": torch.full((2, 2), b_value),
                }

            def named_parameters(self):
                return self.parameters.items()

        session = ModelSession({}, FakeModel(1.0), None, {})
        session._audit_adapter_parameters("gold", adapter_name)
        self.assertGreater(session.adapter_audit["gold"]["lora_b_norm_sum"], 0)

        broken = ModelSession({}, FakeModel(0.0), None, {})
        with self.assertRaises(RuntimeError):
            broken._audit_adapter_parameters("gold", adapter_name)

    def test_flatten_target_and_base_rows(self) -> None:
        base = _flatten(fake_lens_record(condition="base", target=None))
        gold = _flatten(fake_lens_record(condition="gold", target="gold"))
        self.assertIsNone(base["target_word"])
        self.assertIsNone(base["target_margin"])
        self.assertEqual(gold["target_margin"], 2.0)
        self.assertEqual(gold["target_full_rank"], 4)
        self.assertEqual(gold["base_model_revision"], "b" * 40)

    def test_streaming_parquet_schema_stays_nullable_across_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = RunPaths(
                run_id="test_run",
                result_dir=root / "results",
                figure_dir=root / "figures",
                raw_dir=root / "raw",
                lens_dir=root / "lens",
                checkpoint_dir=root / "checkpoints",
            )
            paths.result_dir.mkdir(parents=True)
            cells = paths.lens_dir / "cells"
            cells.mkdir(parents=True)
            records = [
                fake_lens_record(condition="base", target=None),
                fake_lens_record(condition="gold", target="gold"),
            ]
            (cells / "records.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            output = export_lens_parquet(paths, batch_size=1)
            frame = pd.read_parquet(output)
            self.assertEqual(len(frame), 2)
            self.assertTrue(pd.isna(frame.loc[0, "target_word"]))
            self.assertEqual(frame.loc[1, "target_word"], "gold")
            self.assertTrue(pd.isna(frame.loc[0, "adapter_revision"]))

    def test_every_generated_notebook_code_cell_compiles(self) -> None:
        for path in sorted((PROJECT_ROOT / "notebooks").glob("0[1-6]_*.ipynb")):
            notebook = nbformat.read(path, as_version=4)
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type == "code":
                    compile(cell.source, f"{path.name}:cell-{index}", "exec")

    def test_notebooks_expose_model_and_analysis_workflows(self) -> None:
        forbidden_calls = (
            "load_session(",
            "run_behavior_generations(",
            "run_base_jlens_sanity(",
            "run_gold_blue_sweep(",
            "from src.analysis import",
        )
        for path in sorted((PROJECT_ROOT / "notebooks").glob("0[1-6]_*.ipynb")):
            notebook = nbformat.read(path, as_version=4)
            code = "\n".join(
                cell.source for cell in notebook.cells if cell.cell_type == "code"
            )
            for forbidden in forbidden_calls:
                self.assertNotIn(forbidden, code, f"{forbidden} remains in {path.name}")

    def test_validation_notebooks_expose_mask_metrics_and_semantic_positions(self) -> None:
        sweep = nbformat.read(
            PROJECT_ROOT / "notebooks/05_gold_blue_moon_validation_sweep.ipynb",
            as_version=4,
        )
        analysis = nbformat.read(
            PROJECT_ROOT / "notebooks/06_gold_blue_moon_validation_analysis.ipynb",
            as_version=4,
        )
        sweep_code = "\n".join(
            cell.source for cell in sweep.cells if cell.cell_type == "code"
        )
        analysis_code = "\n".join(
            cell.source for cell in analysis.cells if cell.cell_type == "code"
        )
        self.assertIn("masked[:, valid_emitted_ids] = -1.0", sweep_code)
        self.assertIn("average_probability[valid_emitted_ids] = -1.0", sweep_code)
        self.assertIn('distribution("jlens")', sweep_code)
        self.assertIn('read_text("direct_url.json")', sweep_code)
        self.assertNotIn('PROJECT_ROOT / "vendor" / "jacobian-lens"', sweep_code)
        self.assertIn("response_start_boundary", sweep_code)
        self.assertIn("build_paper_metrics", analysis_code)
        self.assertIn("pass_at_10", analysis_code)
        self.assertIn("majority_at_10", analysis_code)
        self.assertIn("shared_position_offset_from_response_boundary", analysis_code)
        self.assertIn("validation_selected_layer_candidate_sequences.csv", analysis_code)
        self.assertIn("frozen_validation_selection_for_test.json", analysis_code)


if __name__ == "__main__":
    unittest.main()
