from __future__ import annotations

from babappai.run_pipeline import terminal_summary


def test_terminal_summary_reports_calibration_withheld_warning() -> None:
    payload = {
        "gene_summary": {
            "p_emp": 0.2,
            "q_emp": 0.2,
            "alpha_used": 0.05,
            "significant_bool": False,
            "eii_z_raw": 0.1,
            "eii_01_raw": 0.52,
            "ceii_gene": None,
            "ceii_site": None,
            "ceii_gene_class": "calibration_unavailable",
            "calibration_version": "ceii_v2",
            "applicability_status": "calibration_unavailable",
            "applicability_score": 1.0,
            "calibration_unavailable_reason": "sigma0_floored",
            "sigma0_valid": False,
            "sigma0_floored": True,
            "fallback_applied": False,
        },
        "branch_results": [
            {"branch": "a", "background_score": 0.2},
            {"branch": "b", "background_score": 0.1},
        ],
        "site_results": [
            {"site": 1, "site_score": 0.8},
            {"site": 2, "site_score": 0.7},
        ],
    }
    lines = terminal_summary(payload)
    assert any("calibrated ceii outputs withheld" in line.lower() for line in lines)
    assert any("sigma0_valid=False" in line for line in lines)
