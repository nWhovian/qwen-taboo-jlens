#!/usr/bin/env python3
"""Small CPU-only J-space checks without a pytest dependency."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.jspace import (
    build_effective_jlens_dictionary,
    decomposition_metrics,
    masked_gradient_pursuit,
    response_anchor_indices,
    validate_dictionary,
)


def main() -> None:
    anchors = [
        {"name": "first", "fraction": 0.0},
        {"name": "middle", "fraction": 0.5},
        {"name": "last", "fraction": 1.0},
    ]
    assert response_anchor_indices(1, anchors) == [
        ("first", 0),
        ("middle", 0),
        ("last", 0),
    ]
    assert response_anchor_indices(5, anchors) == [
        ("first", 0),
        ("middle", 2),
        ("last", 4),
    ]
    assert response_anchor_indices(16, [{"name": "gen_5", "index": 5}]) == [
        ("gen_5", 5)
    ]

    unembedding = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    rms_delta = torch.tensor([0.5, -0.25])
    jacobian = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    qwen_dictionary = build_effective_jlens_dictionary(
        lm_head_weight=unembedding,
        final_norm_weight=rms_delta,
        jacobian=jacobian,
        chunk_size=1,
    )
    assert torch.equal(qwen_dictionary, (unembedding * (1.0 + rms_delta)) @ jacobian)

    identity = torch.eye(3, dtype=torch.float32)
    target = torch.tensor([3.0, 2.0, 1.0])
    norms = validate_dictionary(identity)
    unmasked = masked_gradient_pursuit(target, identity, k=2, atom_norms=norms)
    masked = masked_gradient_pursuit(
        target,
        identity,
        k=2,
        excluded_indices=[0],
        atom_norms=norms,
    )
    assert unmasked.selected_support.tolist() == [0, 1]
    assert masked.selected_support.tolist() == [1, 2]
    assert bool((masked.coordinates >= 0).all())

    exact_target = torch.tensor([3.0, 2.0, 0.0])
    exact = masked_gradient_pursuit(exact_target, identity, k=2)
    metrics = decomposition_metrics(exact, exact_target)
    assert abs(metrics["nonnegative_reconstruction_fraction"] - 1.0) < 1e-6
    assert abs(metrics["jspace_projection_fraction"] - 1.0) < 1e-6
    assert abs(metrics["non_jspace_fraction"]) < 1e-6

    from transformer_lens.tools.analysis.jacobian_lens_decomposition import (
        get_sparse_decomposition,
    )

    generator = torch.Generator().manual_seed(7)
    dictionary = torch.randn(31, 9, generator=generator)
    random_target = torch.randn(9, generator=generator)
    reference = get_sparse_decomposition(
        random_target,
        dictionary,
        k=8,
        algorithm="gradient_pursuit",
    )
    ours = masked_gradient_pursuit(random_target, dictionary, k=8)
    assert torch.equal(ours.selected_support, reference.selected_support)
    assert torch.equal(ours.support, reference.support)
    assert torch.allclose(ours.coordinates, reference.coordinates, rtol=1e-5, atol=1e-6)
    print(
        {
            "status": "passed",
            "anchor_checks": True,
            "qwen_rms_delta_check": True,
            "excluded_atom_check": True,
            "nonnegative_reconstruction_check": True,
            "transformer_lens_gradient_pursuit_parity": True,
        }
    )


if __name__ == "__main__":
    main()
