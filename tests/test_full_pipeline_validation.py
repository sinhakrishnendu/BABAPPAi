from __future__ import annotations

import csv
from pathlib import Path

from babappai.validation.full_pipeline_validation import (
    bootstrap_eii_thresholds,
    confusion_metrics,
    label_positive,
    read_tsv,
)


def _write_metrics_tsv(path: Path) -> None:
    rows = []
    # Balanced tiny dataset: neutral, low, medium, high.
    for i in range(8):
        rows.append({"regime": "neutral", "EII_01": 0.10 + 0.02 * i, "EII_z": -1.5 + 0.1 * i})
    for i in range(4):
        rows.append({"regime": "low", "EII_01": 0.35 + 0.05 * i, "EII_z": -0.2 + 0.2 * i})
    for i in range(4):
        rows.append({"regime": "medium", "EII_01": 0.70 + 0.05 * i, "EII_z": 0.8 + 0.2 * i})
    for i in range(4):
        rows.append({"regime": "high", "EII_01": 0.85 + 0.03 * i, "EII_z": 1.5 + 0.2 * i})

    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["regime", "EII_01", "EII_z"], delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_label_positive_medium_high() -> None:
    labels = label_positive(["neutral", "low", "medium", "high"], "medium_high")
    assert labels.tolist() == [0, 0, 1, 1]


def test_confusion_metrics_sanity() -> None:
    y_true = [0, 0, 1, 1]
    score = [0.1, 0.8, 0.9, 0.2]
    out = confusion_metrics(
        y_true=__import__("numpy").asarray(y_true),
        score=__import__("numpy").asarray(score),
        threshold=0.7,
    )
    assert out["tp"] == 1
    assert out["fp"] == 1
    assert out["tn"] == 1
    assert out["fn"] == 1


def test_bootstrap_eii_thresholds_outputs(tmp_path: Path) -> None:
    metrics_tsv = tmp_path / "metrics.tsv"
    _write_metrics_tsv(metrics_tsv)

    outdir = tmp_path / "bootstrap"
    meta = bootstrap_eii_thresholds(
        metrics_tsv=metrics_tsv,
        outdir=outdir,
        bootstrap_reps=20,
        seed=7,
        decision_target="medium_high",
        default_threshold=0.70,
    )

    summary_rows = read_tsv(meta["bootstrap_summary_tsv"])
    metrics = {row["metric"] for row in summary_rows}
    assert "AUC" in metrics
    assert "FPR_at_default" in metrics
    assert "TPR_at_default" in metrics
    assert "balanced_accuracy_at_default" in metrics
