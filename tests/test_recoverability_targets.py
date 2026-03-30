from __future__ import annotations

from babappai.calibration.recoverability import attach_recoverability_targets


def _base_row() -> dict:
    return {
        "scenario_id": "s1",
        "site_enrichment_at_k": 0.90,
        "site_spearman": 0.70,
        "branch_spearman": 0.80,
        "burden_alignment_score": 0.85,
        "scenario_site_stability": 0.95,
        "scenario_branch_stability": 0.95,
        "q_emp": 0.50,
        "dispersion_ratio": 1.20,
        "eii_01_raw": 0.65,
    }


def test_excess_gate_can_block_identifiable_label_when_raw_excess_is_weak() -> None:
    weak_excess = _base_row()
    weak_excess["q_emp"] = 0.95
    weak_excess["dispersion_ratio"] = 0.80
    weak_excess["eii_01_raw"] = 0.30

    rows = [weak_excess]
    attach_recoverability_targets(
        rows,
        tau_gene=0.42,
        tau_site=0.45,
        require_excess_over_neutral_for_identifiable=True,
        max_q_emp_for_identifiable=0.20,
        min_dispersion_ratio_for_identifiable=1.05,
        min_eii_01_for_identifiable=0.55,
        min_excess_evidence_score_for_identifiable=0.30,
    )
    row = rows[0]
    assert int(row["I_gene_recovery_only"]) == 1
    assert int(row["I_site_recovery_only"]) == 1
    assert bool(row["target_gate_pass_overall"]) is False
    assert int(row["I_gene"]) == 0
    assert int(row["I_site"]) == 0


def test_excess_gate_preserves_identifiable_label_when_recovery_and_excess_agree() -> None:
    strong_excess = _base_row()
    strong_excess["q_emp"] = 0.005
    strong_excess["dispersion_ratio"] = 2.30
    strong_excess["eii_01_raw"] = 0.92

    rows = [strong_excess]
    attach_recoverability_targets(
        rows,
        tau_gene=0.42,
        tau_site=0.45,
        require_excess_over_neutral_for_identifiable=True,
        max_q_emp_for_identifiable=0.20,
        min_dispersion_ratio_for_identifiable=1.05,
        min_eii_01_for_identifiable=0.55,
        min_excess_evidence_score_for_identifiable=0.30,
    )
    row = rows[0]
    assert int(row["I_gene_recovery_only"]) == 1
    assert int(row["I_site_recovery_only"]) == 1
    assert bool(row["target_gate_pass_overall"]) is True
    assert int(row["I_gene"]) == 1
    assert int(row["I_site"]) == 1

