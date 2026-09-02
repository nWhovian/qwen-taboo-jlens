from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class RuntimeContractTests(unittest.TestCase):
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

    def test_container_and_host_scripts_share_project_venv_override(self) -> None:
        for relative_path in [
            "scripts/bootstrap_runpod.sh",
            "scripts/check_remote_setup.sh",
            "scripts/start_jupyter.sh",
            "scripts/start_jupyter_mcp.sh",
        ]:
            source = (PROJECT_ROOT / relative_path).read_text()
            self.assertIn("PROJECT_VENV", source, relative_path)


if __name__ == "__main__":
    unittest.main()
