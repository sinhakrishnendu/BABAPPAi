from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import babappai.inference as inference
import babappai.tree_calibration as tree_calibration
from babappai.stats import bh_adjust, empirical_monte_carlo_pvalue


class _FakeLeaf:
    def __init__(self, name: str):
        self.name = name

    def is_root(self) -> bool:
        return False

    def get_leaves(self):
        return [self]


class _FakeRoot:
    def __init__(self, leaves):
        self._leaves = leaves

    def is_root(self) -> bool:
        return True

    def get_leaves(self):
        return list(self._leaves)

    def traverse(self, _order: str):
        return [self, *self._leaves]


class _FakeModel:
    def eval(self):
        return self

    def __call__(self, parent, child, branch_length):
        _ = parent, child, branch_length
        site_logits = torch.tensor([[0.10, 0.40, -0.20, 0.00]], dtype=torch.float32)
        branch_logits = torch.tensor([[0.20, -0.10]], dtype=torch.float32)
        return site_logits, branch_logits


def _patch_inference_runtime(monkeypatch):
    leaves = [_FakeLeaf("taxonA"), _FakeLeaf("taxonB")]
    root = _FakeRoot(leaves)
    x = np.asarray([[1, 2, 3, 1], [1, 2, 0, 1]], dtype=int)

    monkeypatch.setattr(inference, "encode_alignment", lambda _: (x, 2, 4))
    monkeypatch.setattr(inference, "load_tree", lambda _: root)
    monkeypatch.setattr(inference, "enumerate_branches", lambda _: ["taxonA", "taxonB"])
    monkeypatch.setattr(inference, "ensure_model", lambda *args, **kwargs: Path("/tmp/fake_model.pt"))
    monkeypatch.setattr(inference, "model_status", lambda: {"cached_path": "/tmp/fake_model.pt", "verified": True})
    monkeypatch.setattr(inference.torch.jit, "load", lambda *args, **kwargs: _FakeModel())


def test_empirical_pvalue_formula():
    observed = 0.6
    neutral = [0.1, 0.3, 0.7, 0.8]
    p_emp = empirical_monte_carlo_pvalue(observed, neutral)
    expected = (1 + 2) / (4 + 1)  # 0.7 and 0.8 exceed/meet observed
    assert abs(p_emp - expected) < 1e-12


def test_bh_adjust_sanity():
    pvals = np.asarray([0.01, 0.04, 0.03, 0.20], dtype=float)
    qvals = bh_adjust(pvals)
    expected = np.asarray([0.04, 0.0533333333333, 0.0533333333333, 0.20], dtype=float)
    assert np.allclose(qvals, expected, atol=1e-10)


def test_inference_empirical_significance_and_sigma_floor(monkeypatch):
    _patch_inference_runtime(monkeypatch)

    neutral = np.asarray([0.01, 0.03, 0.08, 0.10], dtype=float)
    mu0 = float(np.mean(neutral))
    sd0 = float(np.std(neutral, ddof=1))
    monkeypatch.setattr(
        tree_calibration,
        "monte_carlo_neutral",
        lambda **kwargs: (mu0, sd0, neutral),
    )
    monkeypatch.setattr(
        inference,
        "load_calibration_asset",
        lambda *_args, **_kwargs: {
            "calibration_version": "ceii_test",
            "gene_calibrator": {"x": [-3.0, 0.0, 3.0], "y": [0.1, 0.5, 0.9]},
            "site_calibrator": {"x": [-3.0, 0.0, 3.0], "y": [0.05, 0.4, 0.8]},
            "thresholds": {
                "gene": {"threshold": 0.6},
                "site": {"threshold": 0.65},
            },
            "classes": {
                "gene": [
                    {"label": "not_identifiable", "min": 0.0, "max": 0.4},
                    {"label": "weak_or_ambiguous", "min": 0.4, "max": 0.6},
                    {"label": "identifiable", "min": 0.6, "max": 0.8},
                    {"label": "strongly_identifiable", "min": 0.8, "max": 1.0},
                ],
                "site": [
                    {"label": "not_identifiable", "min": 0.0, "max": 0.4},
                    {"label": "weak_or_ambiguous", "min": 0.4, "max": 0.65},
                    {"label": "identifiable", "min": 0.65, "max": 0.8},
                    {"label": "strongly_identifiable", "min": 0.8, "max": 1.0},
                ],
            },
            "applicability": {
                "features": {
                    "n_taxa": {"min": 2, "max": 128},
                    "gene_length_nt": {"min": 3, "max": 10000},
                },
                "near_boundary_fraction": 0.01,
                "min_applicability_score_for_calibration": 0.0,
                "allow_near_boundary_calibration": True,
            },
        },
    )

    result = inference.run_inference(
        alignment_path="dummy.fasta",
        tree_path="dummy.nwk",
        offline=True,
        pvalue_mode="empirical_monte_carlo",
        neutral_reps=4,
        sigma_floor=0.05,
        alpha=0.05,
        seed=7,
    )
    gene = result["gene_level_identifiability"]

    expected_p = empirical_monte_carlo_pvalue(gene["D_obs"], neutral.tolist())
    assert abs(float(gene["p_emp"]) - float(expected_p)) < 1e-12
    assert gene["neutral_replicates"] == neutral.tolist()
    assert float(gene["sigma0_final"]) >= 0.05
    assert abs(float(gene["sigma0_final"]) - float(gene["neutral_sd"])) < 1e-12
    assert bool(gene["significant_bool"]) == (float(gene["q_emp"]) <= 0.05)
    assert float(gene["EII_z"]) == float(gene["eii_z_raw"])
    assert 0.0 <= float(gene["eii_01_raw"]) <= 1.0
    assert 0.0 <= float(gene["ceii_gene"]) <= 1.0
    assert str(gene["applicability_status"]) in {"in_domain", "near_boundary"}
    assert bool(gene["within_applicability_envelope"]) is True
    assert gene["identifiability_extent"] == gene["ceii_gene_class"]
