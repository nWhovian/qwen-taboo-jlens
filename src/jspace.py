"""J-space helpers for the Rock adapter experiment.

The pursuit below follows TransformerLens 3.8.1's model-free Gradient Pursuit
implementation and adds one experiment-specific feature: token IDs may be
excluded before greedy selection. The unmasked path is checked against the
installed TransformerLens implementation in the GPU smoke test.

Reference (MIT):
https://github.com/TransformerLensOrg/TransformerLens/blob/v3.8.1/
transformer_lens/tools/analysis/jacobian_lens_decomposition.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


_ACTIVE_RELATIVE_TOLERANCE = math.sqrt(torch.finfo(torch.float64).eps)
_CORRELATION_RELATIVE_TOLERANCE = math.sqrt(torch.finfo(torch.float32).eps)
_GRADIENT_BACKTRACK_STEPS = 20


@dataclass
class MaskedJSpaceDecomposition:
    """Sparse nonnegative reconstruction plus the selected-span projection."""

    support: torch.Tensor
    coordinates: torch.Tensor
    selected_support: torch.Tensor
    reconstruction: torch.Tensor
    j_space_component: torch.Tensor
    non_j_space_component: torch.Tensor


def response_anchor_indices(
    generated_length: int,
    anchors: Sequence[dict[str, float]],
) -> list[tuple[str, int]]:
    """Map named response fractions to zero-based generated-token indices.

    Short responses may map multiple named anchors to the same token. They stay
    separate because every metric is paired by the pre-registered anchor name.
    """

    if generated_length < 1:
        raise ValueError("generated_length must be positive")
    result: list[tuple[str, int]] = []
    for anchor in anchors:
        name = str(anchor["name"])
        fraction = float(anchor["fraction"])
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"anchor fraction must be in [0, 1], got {fraction}")
        index = int(round((generated_length - 1) * fraction))
        result.append((name, index))
    return result


def single_token_surface_ids(tokenizer, word: str) -> list[int]:
    """Return unique IDs for single-token capitalization/space variants."""

    ids: set[int] = set()
    for surface in (word, f" {word}", word.capitalize(), f" {word.capitalize()}"):
        encoded = tokenizer.encode(surface, add_special_tokens=False)
        if len(encoded) == 1:
            ids.add(int(encoded[0]))
    if not ids:
        raise ValueError(f"No single-token surface form for {word!r}")
    return sorted(ids)


@torch.no_grad()
def build_effective_jlens_dictionary(
    *,
    lm_head_weight: torch.Tensor,
    final_norm_weight: torch.Tensor,
    jacobian: torch.Tensor,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Build rows ``(W_U[token] * rms_gamma) @ J`` in float32.

    ``JacobianLens.transport`` computes ``h @ J.T``. Qwen's final RMSNorm then
    multiplies each transported coordinate by ``rms_gamma`` before the lm head.
    Its per-activation RMS denominator is a positive scalar, so this dictionary
    reproduces the ordinary J-Lens vocabulary ranking exactly up to that scalar.

    Chunking avoids holding a second full float32 copy of the unembedding.
    """

    if lm_head_weight.ndim != 2 or jacobian.ndim != 2:
        raise ValueError("lm_head_weight and jacobian must be matrices")
    vocab_size, d_model = lm_head_weight.shape
    if jacobian.shape != (d_model, d_model):
        raise ValueError(
            f"jacobian must have shape {(d_model, d_model)}, got {tuple(jacobian.shape)}"
        )
    if final_norm_weight.shape != (d_model,):
        raise ValueError(
            f"final_norm_weight must have shape {(d_model,)}, got {tuple(final_norm_weight.shape)}"
        )
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    device = lm_head_weight.device
    gamma = final_norm_weight.to(device=device, dtype=torch.float32)
    work_jacobian = jacobian.to(device=device, dtype=torch.float32)
    dictionary = torch.empty((vocab_size, d_model), device=device, dtype=torch.float32)
    for start in range(0, vocab_size, chunk_size):
        stop = min(start + chunk_size, vocab_size)
        effective_unembedding = lm_head_weight[start:stop].float() * gamma
        dictionary[start:stop] = effective_unembedding @ work_jacobian
    return dictionary


@torch.no_grad()
def validate_dictionary(dictionary: torch.Tensor) -> torch.Tensor:
    """Validate once and return reusable float32 row norms."""

    if dictionary.ndim != 2:
        raise ValueError("dictionary must be [num_atoms, d_model]")
    if dictionary.dtype != torch.float32:
        raise ValueError(f"dictionary must be float32, got {dictionary.dtype}")
    if not bool(torch.isfinite(dictionary).all()):
        raise ValueError("dictionary contains non-finite entries")
    norms = torch.linalg.vector_norm(dictionary, dim=1)
    if not bool(torch.isfinite(norms).all()) or bool((norms == 0).any()):
        raise ValueError("dictionary contains a non-finite or zero-norm atom")
    return norms


def _gradient_pursuit_step(
    active_atoms: torch.Tensor,
    target: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    residual = target - active_atoms @ coefficients
    feasible = coefficients.clamp_min(0.0)
    direction = active_atoms.T @ residual
    projected = active_atoms @ direction
    denominator = float((projected @ projected).detach())
    if denominator <= 0.0:
        return feasible
    step = float((projected @ residual).detach()) / denominator
    current_residual_squared = float((residual @ residual).detach())
    for _ in range(_GRADIENT_BACKTRACK_STEPS + 1):
        candidate = (coefficients + step * direction).clamp_min(0.0)
        candidate_residual = target - active_atoms @ candidate
        candidate_residual_squared = float((candidate_residual @ candidate_residual).detach())
        if candidate_residual_squared <= current_residual_squared:
            return candidate
        step *= 0.5
    return feasible


@torch.no_grad()
def masked_gradient_pursuit(
    x: torch.Tensor,
    dictionary: torch.Tensor,
    *,
    k: int = 16,
    excluded_indices: Iterable[int] = (),
    atom_norms: torch.Tensor | None = None,
) -> MaskedJSpaceDecomposition:
    """TransformerLens-compatible Gradient Pursuit with excluded atoms.

    The selection and coefficient update match TransformerLens 3.8.1 when
    ``excluded_indices`` is empty. Exclusion is implemented by setting those
    atoms' selection correlations to negative infinity at every pursuit step.
    """

    if dictionary.ndim != 2:
        raise ValueError("dictionary must be [num_atoms, d_model]")
    num_atoms, d_model = dictionary.shape
    if x.ndim != 1 or x.shape[0] != d_model:
        raise ValueError(f"x must be a vector of length {d_model}")
    if not 1 <= k <= num_atoms:
        raise ValueError(f"k must be in [1, {num_atoms}]")

    target = x.float()
    atoms = dictionary.float()
    if not bool(torch.isfinite(target).all()):
        raise ValueError("x contains non-finite entries")
    if atom_norms is None:
        atom_norms = validate_dictionary(atoms)
    elif atom_norms.shape != (num_atoms,):
        raise ValueError(f"atom_norms must have shape {(num_atoms,)}")
    atom_norms = atom_norms.to(device=atoms.device, dtype=torch.float32)

    excluded = sorted({int(index) for index in excluded_indices})
    if excluded and (excluded[0] < 0 or excluded[-1] >= num_atoms):
        raise ValueError("excluded atom index outside dictionary")
    excluded_tensor = torch.tensor(excluded, device=atoms.device, dtype=torch.long)

    x_norm = float(torch.linalg.vector_norm(target).detach())
    correlation_tol = _CORRELATION_RELATIVE_TOLERANCE * x_norm
    residual = target.clone()
    selected: list[int] = []
    coordinates = target.new_zeros(0)

    for _ in range(k):
        correlation = (atoms @ residual) / atom_norms
        if excluded:
            correlation[excluded_tensor] = float("-inf")
        for chosen in selected:
            correlation[chosen] = float("-inf")
        candidate = int(torch.argmax(correlation).item())
        if float(correlation[candidate].detach()) <= correlation_tol:
            break
        selected.append(candidate)
        selected_atoms = atoms[selected].T
        coordinates = _gradient_pursuit_step(
            selected_atoms,
            target,
            torch.cat([coordinates, coordinates.new_zeros(1)]),
        )
        residual = target - selected_atoms @ coordinates

    selected_support = torch.tensor(selected, dtype=torch.long)
    selected_atoms = atoms[selected_support].T
    contribution = coordinates * torch.linalg.vector_norm(selected_atoms, dim=0)
    active = contribution > _ACTIVE_RELATIVE_TOLERANCE * x_norm
    support = selected_support[active.cpu()]
    active_coordinates = coordinates[active]
    active_atoms = selected_atoms[:, active]
    reconstruction = active_atoms @ active_coordinates
    if selected:
        j_space_component = selected_atoms @ (torch.linalg.pinv(selected_atoms) @ target)
    else:
        j_space_component = target.new_zeros(d_model)
    non_j_space_component = target - j_space_component
    return MaskedJSpaceDecomposition(
        support=support,
        coordinates=active_coordinates,
        selected_support=selected_support,
        reconstruction=reconstruction,
        j_space_component=j_space_component,
        non_j_space_component=non_j_space_component,
    )


def decomposition_metrics(
    decomposition: MaskedJSpaceDecomposition,
    target: torch.Tensor,
) -> dict[str, float | int]:
    """Return reconstruction and selected-span diagnostics."""

    target = target.float()
    target_squared = float((target @ target).detach())
    if target_squared <= 0:
        raise ValueError("target activation has zero norm")
    reconstruction_residual = target - decomposition.reconstruction
    return {
        "support_size": int(decomposition.support.numel()),
        "selected_support_size": int(decomposition.selected_support.numel()),
        "nonnegative_reconstruction_fraction": 1.0
        - float((reconstruction_residual @ reconstruction_residual).detach()) / target_squared,
        "jspace_projection_fraction": float(
            (decomposition.j_space_component @ decomposition.j_space_component).detach()
        )
        / target_squared,
        "non_jspace_fraction": float(
            (decomposition.non_j_space_component @ decomposition.non_j_space_component).detach()
        )
        / target_squared,
    }
