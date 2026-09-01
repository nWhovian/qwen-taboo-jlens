from __future__ import annotations

import contextlib
import gc
import os
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np
import torch
from huggingface_hub.utils import disable_progress_bars as disable_hf_progress_bars
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

from src.experiment_io import PROJECT_ROOT, RunPaths, load_json, utc_now
from src.prompt_data import assert_prompt_has_no_candidates, render_messages


disable_hf_progress_bars()
transformers_logging.disable_progress_bar()


DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _adapter_name(repo_id: str) -> str:
    return repo_id.replace(".", "_").replace("/", "__")


@dataclass
class ModelSession:
    config: dict[str, Any]
    model: Any
    tokenizer: Any
    adapter_names: dict[str, str]
    lens: Any | None = None
    lens_model: Any | None = None
    token_audit: dict[str, Any] = field(default_factory=dict)
    adapter_audit: dict[str, Any] = field(default_factory=dict)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @contextlib.contextmanager
    def condition(self, name: str) -> Iterator[None]:
        if name == "base":
            self.model.disable_adapters()
            try:
                yield
            finally:
                self.model.enable_adapters()
            return
        if name not in self.adapter_names:
            raise KeyError(f"Unknown condition {name!r}")
        self.model.enable_adapters()
        self.model.set_adapter(self.adapter_names[name])
        yield

    def render(self, messages: list[dict[str, str]]) -> tuple[str, list[int]]:
        return render_messages(
            self.tokenizer,
            messages,
            add_generation_prompt=True,
            enable_thinking=self.config["runtime"]["enable_thinking"],
        )

    def _audit_adapter_parameters(self, word: str, adapter_name: str) -> None:
        tensors = [
            (name, parameter.detach())
            for name, parameter in self.model.named_parameters()
            if adapter_name in name and ".lora_" in name
        ]
        a_tensors = [(name, tensor) for name, tensor in tensors if ".lora_A." in name]
        b_tensors = [(name, tensor) for name, tensor in tensors if ".lora_B." in name]
        if not a_tensors or not b_tensors:
            raise RuntimeError(
                f"Loaded adapter {word!r} has no discoverable LoRA A/B tensors"
            )
        if any(not bool(torch.isfinite(tensor).all()) for _, tensor in tensors):
            raise RuntimeError(f"Loaded adapter {word!r} contains non-finite weights")
        a_norm_sum = sum(float(tensor.float().norm()) for _, tensor in a_tensors)
        b_norm_sum = sum(float(tensor.float().norm()) for _, tensor in b_tensors)
        if b_norm_sum == 0:
            raise RuntimeError(
                f"Loaded adapter {word!r} has all-zero LoRA B tensors; stop and "
                "verify the checkpoint mapping"
            )
        self.adapter_audit[word] = {
            "adapter_name": adapter_name,
            "tensor_count": len(tensors),
            "lora_a_tensor_count": len(a_tensors),
            "lora_b_tensor_count": len(b_tensors),
            "parameter_count": sum(tensor.numel() for _, tensor in tensors),
            "lora_a_norm_sum": a_norm_sum,
            "lora_b_norm_sum": b_norm_sum,
        }

    def save_adapter_audit(self, paths: RunPaths) -> Path:
        output = paths.result_dir / "loaded_adapter_parameter_audit.json"
        output.write_text(json.dumps(self.adapter_audit, indent=2), encoding="utf-8")
        return output

    def load_adapters(
        self,
        words: Iterable[str],
        *,
        paths: RunPaths | None = None,
    ) -> None:
        for word in words:
            if word in self.adapter_names:
                continue
            if word not in self.config["adapters"]:
                raise KeyError(f"Unknown adapter word {word!r}")
            spec = self.config["adapters"][word]
            name = _adapter_name(spec["repo_id"])
            self.model.load_adapter(
                spec["repo_id"],
                adapter_name=name,
                adapter_kwargs={"revision": spec["revision"]},
                is_trainable=False,
                low_cpu_mem_usage=True,
            )
            self.adapter_names[word] = name
            self._audit_adapter_parameters(word, name)
        if paths is not None:
            self.save_adapter_audit(paths)

    @torch.no_grad()
    def generate_record(
        self,
        *,
        prompt: dict[str, Any],
        condition: str,
        run_id: str,
    ) -> dict[str, Any]:
        adapter_spec = self.config["adapters"].get(condition)
        base_spec = self.config["base_model"]
        lens_spec = self.config["jlens"]
        rendered, prompt_token_ids = self.render(prompt["messages"])
        assert_prompt_has_no_candidates(
            rendered, self.config["readout"]["candidate_words"]
        )
        input_ids = torch.tensor([prompt_token_ids], device=self.device)
        attention_mask = torch.ones_like(input_ids)
        runtime = self.config["runtime"]
        with self.condition(condition):
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=runtime["do_sample"],
                max_new_tokens=runtime["max_new_tokens"],
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=(
                    self.model.generation_config.pad_token_id
                    or self.tokenizer.pad_token_id
                    or self.tokenizer.eos_token_id
                ),
                use_cache=True,
            )
        generation_ids = generated[0, input_ids.shape[1] :].tolist()
        output_text = self.tokenizer.decode(generation_ids, skip_special_tokens=True)
        return {
            "schema_version": 1,
            "timestamp_utc": utc_now(),
            "run_id": run_id,
            "prompt_id": prompt["prompt_id"],
            "prompt_type": prompt["prompt_type"],
            "split": prompt["split"],
            "source_path": prompt["source_path"],
            "source_line": prompt["source_line"],
            "source_parent_commit": prompt["source_parent_commit"],
            "source_submodule_commit": prompt["source_submodule_commit"],
            "messages": prompt["messages"],
            "rendered_prompt": rendered,
            "prompt_token_ids": prompt_token_ids,
            "prompt_token_count": len(prompt_token_ids),
            "condition": condition,
            "secret": condition if condition in self.adapter_names else None,
            "base_model_repo_id": base_spec["repo_id"],
            "base_model_revision": base_spec["revision"],
            "tokenizer_repo_id": base_spec["repo_id"],
            "tokenizer_revision": base_spec["revision"],
            "adapter_repo_id": adapter_spec["repo_id"] if adapter_spec else None,
            "adapter_revision": adapter_spec["revision"] if adapter_spec else None,
            "jlens_repo_id": lens_spec["repo_id"],
            "jlens_revision": lens_spec["revision"],
            "jlens_filename": lens_spec["filename"],
            "jlens_code_commit": lens_spec["official_code_commit"],
            "runtime_dtype": runtime["dtype"],
            "attention_implementation": runtime["attention_implementation"],
            "seed": self.config["seed"],
            "generation_token_ids": generation_ids,
            "generation_token_count": len(generation_ids),
            "output_text": output_text,
            "generation_config": {
                "do_sample": runtime["do_sample"],
                "max_new_tokens": runtime["max_new_tokens"],
                "enable_thinking": runtime["enable_thinking"],
            },
        }

    def load_jlens(self) -> None:
        if self.lens is not None:
            return
        import jlens

        spec = self.config["jlens"]
        vendor_root = PROJECT_ROOT / "vendor" / "jacobian-lens"
        actual_commit = (
            __import__("subprocess")
            .run(
                ["git", "-C", str(vendor_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
        )
        if actual_commit != spec["official_code_commit"]:
            raise RuntimeError(
                f"J-Lens code revision mismatch: {actual_commit} != "
                f"{spec['official_code_commit']}"
            )
        self.lens = jlens.JacobianLens.from_pretrained(
            spec["repo_id"], filename=spec["filename"], revision=spec["revision"]
        )
        self.lens_model = jlens.from_hf(
            self.model, self.tokenizer, force_bos=False, compile=False
        )
        expected_hidden = self.config["base_model"]["expected_hidden_size"]
        expected_layers = self.config["base_model"]["expected_num_hidden_layers"]
        if self.lens.d_model != expected_hidden:
            raise RuntimeError(f"J-Lens d_model={self.lens.d_model}, expected {expected_hidden}")
        if self.lens.n_prompts != spec["expected_n_prompts"]:
            raise RuntimeError(
                f"J-Lens n_prompts={self.lens.n_prompts}, expected "
                f"{spec['expected_n_prompts']}"
            )
        if self.lens_model.n_layers != expected_layers:
            raise RuntimeError(
                f"Loaded model has {self.lens_model.n_layers} layers, expected {expected_layers}"
            )


_SESSION: ModelSession | None = None


def _assert_cuda_only(model: Any) -> None:
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        non_cuda = {
            name: device
            for name, device in device_map.items()
            if str(device) not in {"0", "cuda", "cuda:0"}
        }
        if non_cuda:
            raise RuntimeError(f"CPU/disk offload detected: {non_cuda}")
    parameter_devices = {parameter.device.type for parameter in model.parameters()}
    if parameter_devices != {"cuda"}:
        raise RuntimeError(f"Primary model is not entirely on CUDA: {parameter_devices}")


def audit_candidate_tokens(tokenizer: Any, words: list[str]) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for word in words:
        forms: dict[str, list[int]] = {}
        for surface in (word, f" {word}", word.capitalize(), f" {word.capitalize()}"):
            forms[surface] = tokenizer.encode(surface, add_special_tokens=False)
        single_token_forms = {
            surface: ids for surface, ids in forms.items() if len(ids) == 1
        }
        if not single_token_forms:
            raise RuntimeError(f"No single-token surface form for candidate {word!r}: {forms}")
        audit[word] = {
            "forms": forms,
            "single_token_forms": single_token_forms,
            "single_token_ids": sorted({ids[0] for ids in single_token_forms.values()}),
        }
    return audit


def load_session(
    config_path: str | Path = "configs/gold_blue_experiment.json",
    *,
    paths: RunPaths | None = None,
    load_lens: bool = False,
    adapter_words: Iterable[str] | None = None,
) -> ModelSession:
    global _SESSION
    absolute = Path(config_path)
    if not absolute.is_absolute():
        absolute = PROJECT_ROOT / absolute
    config = load_json(absolute)
    requested_adapters = list(
        adapter_words
        if adapter_words is not None
        else config["behavior"]["initial_conditions"][1:]
    )
    if _SESSION is not None:
        if _SESSION.config != config:
            raise RuntimeError("A different model session is already loaded in this kernel")
        if paths is not None:
            (paths.result_dir / "candidate_token_audit.json").write_text(
                json.dumps(_SESSION.token_audit, indent=2), encoding="utf-8"
            )
        _SESSION.load_adapters(requested_adapters, paths=paths)
        if load_lens:
            _SESSION.load_jlens()
        return _SESSION

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; do not load the 27B model")
    set_deterministic_seed(config["seed"])
    runtime = config["runtime"]
    dtype = DTYPES[runtime["dtype"]]
    base = config["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(
        base["repo_id"], revision=base["revision"]
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        base["repo_id"],
        revision=base["revision"],
        dtype=dtype,
        attn_implementation=runtime["attention_implementation"],
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    _assert_cuda_only(model)

    model.add_adapter(LoraConfig(target_modules=["q_proj"]), adapter_name="default")
    session = ModelSession(
        config=config,
        model=model,
        tokenizer=tokenizer,
        adapter_names={},
    )
    session.load_adapters(requested_adapters, paths=paths)
    session.token_audit = audit_candidate_tokens(
        tokenizer, config["readout"]["candidate_words"]
    )
    if paths is not None:
        (paths.result_dir / "candidate_token_audit.json").write_text(
            json.dumps(session.token_audit, indent=2), encoding="utf-8"
        )
    if load_lens:
        session.load_jlens()
    _SESSION = session
    return session


def release_session(paths: RunPaths | None = None) -> dict[str, Any]:
    """Release the persistent model before handing the GPU to a tmux worker."""

    global _SESSION
    had_session = _SESSION is not None
    if _SESSION is not None:
        session = _SESSION
        session.lens_model = None
        session.lens = None
        session.model = None
        session.tokenizer = None
        session.adapter_names.clear()
        _SESSION = None
    gc.collect()
    report: dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "had_persistent_session": had_session,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        report.update(
            {
                "cuda_free_bytes": free_bytes,
                "cuda_total_bytes": total_bytes,
                "cuda_free_gib": free_bytes / 2**30,
                "cuda_total_gib": total_bytes / 2**30,
            }
        )
    if paths is not None:
        output = paths.result_dir / "gpu_handoff.json"
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["artifact"] = str(output)
    return report
