from __future__ import annotations

import json
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path

from scripts.check_runtime import version_matches


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class RuntimeContractTests(unittest.TestCase):
    def test_torch_cuda_local_version_suffix_is_accepted(self) -> None:
        self.assertTrue(version_matches("torch", "2.10.0"))
        self.assertTrue(version_matches("torch", "2.10.0+cu130"))
        self.assertFalse(version_matches("torch", "2.10.1+cu130"))
        self.assertFalse(version_matches("transformers", "5.16.1+local"))

    def test_application_dependency_pins_match(self) -> None:
        project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())[
            "project"
        ]
        requirements = {
            line.strip()
            for line in (
                PROJECT_ROOT / "requirements" / "runpod-cu130.txt"
            ).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertEqual(project["requires-python"], ">=3.12,<3.13")
        self.assertEqual(set(project["dependencies"]), requirements)

    def test_gpu_versions_and_flash_wheel_are_pinned(self) -> None:
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()
        installer = (
            PROJECT_ROOT / "scripts" / "install_runpod_runtime.sh"
        ).read_text()

        self.assertIn("nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04@sha256:", dockerfile)
        self.assertIn('TORCH_VERSION="2.10.0"', installer)
        self.assertIn("cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl", installer)
        self.assertIn("sha256=910d8db9def162de5b7c15474b933e7e", installer)
        self.assertIn("581d398613e5602a5af361e1c34d3a92ea82ba8e", installer)
        self.assertIn("--mount=type=cache,target=/root/.cache/pip", dockerfile)
        self.assertIn("SKIP_RUNTIME_CHECK=1 scripts/install_runpod_runtime.sh", dockerfile)
        self.assertNotIn("--no-cache-dir", installer)

    def test_container_and_host_scripts_share_project_venv_override(self) -> None:
        for relative_path in [
            "scripts/bootstrap_runpod.sh",
            "scripts/check_remote_setup.sh",
            "scripts/start_jupyter.sh",
            "scripts/start_jupyter_mcp.sh",
        ]:
            source = (PROJECT_ROOT / relative_path).read_text()
            self.assertIn("PROJECT_VENV", source, relative_path)

    def test_prefetch_plan_uses_the_experiment_pins(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "prefetch_models.py"),
                "--config",
                str(PROJECT_ROOT / "configs" / "gold_blue_experiment.json"),
                "--dry-run",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        plan = json.loads(completed.stdout)
        self.assertEqual(
            [item["label"] for item in plan],
            ["base_model", "adapter_gold", "adapter_blue", "jlens"],
        )
        self.assertTrue(all(len(item["revision"]) == 40 for item in plan))
        self.assertEqual(plan[-1]["kind"], "file")
        self.assertTrue(plan[-1]["filename"].endswith("_n1000.pt"))

    def test_container_starts_prefetch_without_blocking_main_command(self) -> None:
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()
        entrypoint = (PROJECT_ROOT / "docker" / "entrypoint.sh").read_text()
        launcher = (PROJECT_ROOT / "scripts" / "start_model_prefetch.sh").read_text()
        self.assertIn("PREFETCH_MODELS=1", dockerfile)
        self.assertIn("start_model_prefetch.sh", entrypoint)
        self.assertIn("nohup", launcher)
        self.assertIn("&\n", launcher)

    def test_macos_jupyter_mcp_uses_stable_ssh_alias(self) -> None:
        source = (PROJECT_ROOT / "scripts" / "start_jupyter_mcp.sh").read_text()
        self.assertIn('RUNPOD_SSH_HOST:-runpod-jlens', source)
        self.assertIn('RUNPOD_PROJECT_PATH:-/workspace/qwen-taboo-jlens', source)
        self.assertIn('exec ssh -T "$SSH_HOST"', source)


if __name__ == "__main__":
    unittest.main()
