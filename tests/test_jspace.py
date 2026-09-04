from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.jspace import (  # noqa: E402
    build_effective_jlens_dictionary,
    decomposition_metrics,
    masked_gradient_pursuit,
    response_anchor_indices,
    validate_dictionary,
)


def test_qwen_delta_rms_weight_is_shifted_by_one() -> None:
    unembedding = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    rms_delta = torch.tensor([0.5, -0.25])
    jacobian = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    actual = build_effective_jlens_dictionary(
        lm_head_weight=unembedding,
        final_norm_weight=rms_delta,
        jacobian=jacobian,
        chunk_size=1,
    )
    expected = (unembedding * (1.0 + rms_delta)) @ jacobian
    assert torch.equal(actual, expected)


def test_response_anchor_indices_keep_named_duplicate_positions() -> None:
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


def test_masked_gradient_pursuit_excludes_best_atom() -> None:
    dictionary = torch.eye(3, dtype=torch.float32)
    target = torch.tensor([3.0, 2.0, 1.0])
    norms = validate_dictionary(dictionary)
    unmasked = masked_gradient_pursuit(target, dictionary, k=2, atom_norms=norms)
    masked = masked_gradient_pursuit(
        target,
        dictionary,
        k=2,
        excluded_indices=[0],
        atom_norms=norms,
    )
    assert unmasked.selected_support.tolist() == [0, 1]
    assert 0 not in masked.selected_support.tolist()
    assert masked.selected_support.tolist() == [1, 2]
    assert bool((masked.coordinates >= 0).all())


def test_decomposition_metrics_are_consistent_for_orthogonal_dictionary() -> None:
    dictionary = torch.eye(3, dtype=torch.float32)
    target = torch.tensor([3.0, 2.0, 0.0])
    decomposition = masked_gradient_pursuit(target, dictionary, k=2)
    metrics = decomposition_metrics(decomposition, target)
    assert metrics["support_size"] == 2
    assert metrics["nonnegative_reconstruction_fraction"] == pytest.approx(1.0)
    assert metrics["jspace_projection_fraction"] == pytest.approx(1.0)
    assert metrics["non_jspace_fraction"] == pytest.approx(0.0)


def test_unmasked_path_matches_transformer_lens_gradient_pursuit() -> None:
    decomposition_module = pytest.importorskip(
        "transformer_lens.tools.analysis.jacobian_lens_decomposition"
    )
    generator = torch.Generator().manual_seed(7)
    dictionary = torch.randn(31, 9, generator=generator)
    target = torch.randn(9, generator=generator)
    reference = decomposition_module.get_sparse_decomposition(
        target,
        dictionary,
        k=8,
        algorithm="gradient_pursuit",
    )
    ours = masked_gradient_pursuit(target, dictionary, k=8)
    assert torch.equal(ours.selected_support, reference.selected_support)
    assert torch.equal(ours.support, reference.support)
    assert torch.allclose(ours.coordinates, reference.coordinates, rtol=1e-5, atol=1e-6)
