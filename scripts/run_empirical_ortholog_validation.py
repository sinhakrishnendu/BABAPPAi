#!/usr/bin/env python3
"""Run empirical BABAPPAi validation on ortholog control panel datasets.

This script:
- builds a per-gene manifest for FASTA datasets in orthologs/
- sanitizes CDS sets deterministically with per-record change logging
- aligns with babappalign (codon mode) when needed
- generates trees with IQ-TREE when no tree is supplied
- runs installed BABAPPAi CLI in offline mode
- writes per-gene JSON outputs plus summary tables, report, and figures
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from babappai.calibration.ceii import load_calibration_asset

STOP_CODONS = {"TAA", "TAG", "TGA"}
EXPECTED_CONTROL_CLASS: Dict[str, str] = {
    "ago2": "expected_positive_control",
    "dcr-2": "expected_positive_control",
    "r2d2": "expected_positive_control",
    "ago1": "conservative_rnai_control",
    "dcr-1": "conservative_rnai_control",
    "act5c": "housekeeping_control",
    "alphatub84d": "housekeeping_control",
    "eef1alpha": "housekeeping_control",
    "gapdh1": "housekeeping_control",
    "rpl32": "housekeeping_control",
}
def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as fh:
        fh.write(f"[{timestamp()}] {message}\n")


def run_cmd(
    cmd: Sequence[str],
    *,
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[Path] = None,
) -> str:
    append_log(log_path, f"CMD: {' '.join(cmd)}")
    proc = subprocess.run(
        list(cmd),
        text=True,
        capture_output=True,
        env=env,
        cwd=str(cwd) if cwd else None,
    )
    if proc.stdout:
        append_log(log_path, "STDOUT:\n" + proc.stdout.rstrip())
    if proc.stderr:
        append_log(log_path, "STDERR:\n" + proc.stderr.rstrip())
    append_log(log_path, f"EXIT: {proc.returncode}")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.stdout


def numeric_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        v = float(value)
        if math.isnan(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def write_table(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str], delimiter: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fieldnames:
                val = row.get(key)
                if isinstance(val, (dict, list)):
                    out[key] = json.dumps(val, sort_keys=True)
                elif isinstance(val, bool):
                    out[key] = "true" if val else "false"
                elif val is None:
                    out[key] = ""
                else:
                    out[key] = val
            writer.writerow(out)


def find_existing_tree_for_fasta(fasta_path: Path) -> Optional[Path]:
    stem = fasta_path.stem
    parent = fasta_path.parent
    candidates = []
    for suffix in [".nwk", ".tree", ".treefile", ".newick"]:
        candidates.append(parent / f"{stem}{suffix}")
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def sanitize_cds(
    input_fasta: Path,
    output_fasta: Path,
    *,
    log_path: Path,
) -> Dict[str, Any]:
    records = list(SeqIO.parse(str(input_fasta), "fasta"))
    if not records:
        raise ValueError(f"No records found in {input_fasta}")

    sanitized: List[SeqRecord] = []
    seen_ids: Dict[str, int] = {}

    dropped_internal_stop = 0
    dropped_too_short = 0
    renamed_ids = 0
    trimmed_terminal_stop = 0
    trimmed_frame_bases = 0
    replaced_non_atgcn = 0
    removed_gap_chars = 0

    raw_lengths = [len(str(r.seq)) for r in records]

    for rec in records:
        rec_id = rec.id.strip() or "unnamed"
        orig_id = rec_id
        count = seen_ids.get(rec_id, 0)
        if count > 0:
            rec_id = f"{rec_id}_dup{count+1}"
            renamed_ids += 1
            append_log(log_path, f"sanitize:{input_fasta.name}:{orig_id} renamed to {rec_id} (duplicate ID)")
        seen_ids[orig_id] = count + 1

        seq = str(rec.seq).upper().replace("U", "T")
        gap_count = seq.count("-")
        if gap_count:
            removed_gap_chars += gap_count
            seq = seq.replace("-", "")
            append_log(log_path, f"sanitize:{input_fasta.name}:{orig_id} removed {gap_count} gap characters")

        if any(ch not in {"A", "C", "G", "T", "N"} for ch in seq):
            seq2 = []
            local_replaced = 0
            for ch in seq:
                if ch in {"A", "C", "G", "T", "N"}:
                    seq2.append(ch)
                else:
                    seq2.append("N")
                    local_replaced += 1
            seq = "".join(seq2)
            replaced_non_atgcn += local_replaced
            append_log(
                log_path,
                f"sanitize:{input_fasta.name}:{orig_id} replaced {local_replaced} non-ACGTN chars with N",
            )

        if len(seq) >= 3 and seq[-3:] in STOP_CODONS:
            seq = seq[:-3]
            trimmed_terminal_stop += 1
            append_log(log_path, f"sanitize:{input_fasta.name}:{orig_id} trimmed terminal stop codon")

        rem = len(seq) % 3
        if rem != 0:
            seq = seq[: len(seq) - rem]
            trimmed_frame_bases += rem
            append_log(log_path, f"sanitize:{input_fasta.name}:{orig_id} trimmed {rem} trailing base(s) for frame")

        if len(seq) < 3:
            dropped_too_short += 1
            append_log(log_path, f"sanitize:{input_fasta.name}:{orig_id} dropped (length < 3 after sanitization)")
            continue

        has_internal_stop = False
        for i in range(0, len(seq) - 3, 3):
            if seq[i : i + 3] in STOP_CODONS:
                has_internal_stop = True
                break
        if has_internal_stop:
            dropped_internal_stop += 1
            append_log(log_path, f"sanitize:{input_fasta.name}:{orig_id} dropped (internal stop codon)")
            continue

        sanitized.append(SeqRecord(Seq(seq), id=rec_id, description=""))

    if not sanitized:
        raise ValueError(f"All records were dropped during sanitization for {input_fasta}")

    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(sanitized, str(output_fasta), "fasta")

    lengths = [len(str(r.seq)) for r in sanitized]
    stats = {
        "raw_taxon_count": len(records),
        "sanitized_taxon_count": len(sanitized),
        "raw_length_min": min(raw_lengths),
        "raw_length_max": max(raw_lengths),
        "sanitized_length_min": min(lengths),
        "sanitized_length_max": max(lengths),
        "sanitized_length_median": int(statistics.median(lengths)),
        "trimmed_terminal_stop_records": trimmed_terminal_stop,
        "trimmed_frame_bases_total": trimmed_frame_bases,
        "dropped_internal_stop": dropped_internal_stop,
        "dropped_too_short": dropped_too_short,
        "renamed_duplicate_ids": renamed_ids,
        "removed_gap_chars_total": removed_gap_chars,
        "replaced_non_atgcn_total": replaced_non_atgcn,
        "alignment_needed": len(set(lengths)) > 1,
    }
    append_log(log_path, f"sanitize:{input_fasta.name}:stats={json.dumps(stats, sort_keys=True)}")
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ortholog-dir", default="orthologs")
    p.add_argument("--outdir", default="results/empirical_ortholog_validation")
    p.add_argument("--babappai", default="/opt/homebrew/Caskroom/miniconda/base/bin/babappai")
    p.add_argument("--babappalign", default="/opt/homebrew/Caskroom/miniconda/base/envs/molevo/bin/babappalign")
    p.add_argument("--iqtree", default="/opt/homebrew/Caskroom/miniconda/base/envs/molevo/bin/iqtree")
    p.add_argument("--babappalign-model", default=str(Path.home() / ".cache/babappalign/models/babappascore.pt"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-calibration", type=int, default=200)
    p.add_argument("--neutral-reps", type=int, default=200)
    p.add_argument("--device", default="cpu", choices=["cpu", "auto", "cuda"])
    p.add_argument("--ceii-asset", default=None, help="Optional cEII calibration asset JSON passed to babappai run.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def resolve_applicability_envelope(ceii_asset: Optional[str]) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    try:
        asset = load_calibration_asset(ceii_asset)
    except Exception:
        return {"n_taxa": (None, None), "gene_length_nt": (None, None)}

    app = asset.get("applicability", {})
    if not isinstance(app, Mapping):
        return {"n_taxa": (None, None), "gene_length_nt": (None, None)}

    features = app.get("features", {})
    if isinstance(features, Mapping):
        out: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for key in ("n_taxa", "gene_length_nt"):
            spec = features.get(key)
            if isinstance(spec, Mapping) and "min" in spec and "max" in spec:
                try:
                    out[key] = (float(spec["min"]), float(spec["max"]))
                except (TypeError, ValueError):
                    out[key] = (None, None)
            else:
                out[key] = (None, None)
        return out

    # Legacy ceii_v1 shape.
    def _coerce(name_min: str, name_max: str) -> Tuple[Optional[float], Optional[float]]:
        try:
            lo = float(app.get(name_min))
            hi = float(app.get(name_max))
        except (TypeError, ValueError):
            return (None, None)
        return (lo, hi)

    return {
        "n_taxa": _coerce("min_n_taxa", "max_n_taxa"),
        "gene_length_nt": _coerce("min_gene_length_nt", "max_gene_length_nt"),
    }


def make_figures(summary_rows: List[Dict[str, Any]], outdir: Path, *, log_path: Path) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    generated: List[str] = []
    valid = [r for r in summary_rows if r.get("status") == "ok"]
    if not valid:
        append_log(log_path, "No successful genes available for figures.")
        return generated

    class_order = [
        "expected_positive_control",
        "conservative_rnai_control",
        "housekeeping_control",
        "unknown_control_class",
    ]

    def _is_calibrated(row: Mapping[str, Any]) -> bool:
        return (
            str(row.get("applicability_status", "")).lower() == "in_domain"
            and numeric_or_none(row.get("ceii_gene")) is not None
            and numeric_or_none(row.get("ceii_site")) is not None
        )

    # Figure 1: cEII_gene grouped by class with abstained/OOD markers.
    grouped: Dict[str, List[Tuple[str, float]]] = {c: [] for c in class_order}
    abstained_rows: List[Dict[str, Any]] = []
    for r in valid:
        if _is_calibrated(r):
            c = str(r.get("expected_control_class", "unknown_control_class"))
            if c not in grouped:
                grouped[c] = []
            v = numeric_or_none(r.get("ceii_gene"))
            if v is not None:
                grouped[c].append((str(r["gene_name"]), v))
        else:
            abstained_rows.append(r)

    labels = [c for c in class_order if grouped.get(c)] + [c for c in grouped if c not in class_order and grouped[c]]
    data = [[v for _, v in grouped[c]] for c in labels]

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    if data:
        ax.boxplot(data, positions=range(1, len(data) + 1), widths=0.6, showfliers=False)
        for i, c in enumerate(labels, start=1):
            xs = [i + (j - (len(grouped[c]) - 1) / 2) * 0.03 for j in range(len(grouped[c]))]
            ys = [v for _, v in grouped[c]]
            ax.scatter(xs, ys, alpha=0.85, s=30, label="in-domain calibrated" if i == 1 else None)
            for x, (gene, y) in zip(xs, grouped[c]):
                ax.text(x, y + 0.015, gene, fontsize=7, rotation=45, ha="left", va="bottom")
    if abstained_rows:
        # Plot abstained genes at baseline and label them explicitly.
        x_pos = max(len(labels), 1) + 1
        y_pos = [0.02 + 0.01 * i for i in range(len(abstained_rows))]
        ax.scatter([x_pos] * len(abstained_rows), y_pos, marker="x", s=45, color="red", label="abstained / OOD")
        for yy, row in zip(y_pos, abstained_rows):
            reason = str(row.get("calibration_unavailable_reason", "abstained"))
            ax.text(x_pos + 0.05, yy, f"{row['gene_name']} ({reason})", fontsize=7, va="center")
        labels = labels + ["abstained_or_ood"]
    if labels:
        ax.set_xticks(list(range(1, len(labels) + 1)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("cEII_gene")
    ax.set_title("Per-gene cEII_gene grouped by control class (abstention-aware)")
    ax.legend(loc="lower right")
    for ext in ("png", "pdf"):
        path = outdir / f"figure1_ceii_gene_by_control_class.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None)
        generated.append(str(path))
    plt.close(fig)

    # Figure 2: paired cEII_gene vs cEII_site for calibrated genes only.
    fig, ax = plt.subplots(figsize=(6.8, 6.2), constrained_layout=True)
    xs, ys, names = [], [], []
    abstained_names: List[str] = []
    for r in valid:
        if _is_calibrated(r):
            xv = numeric_or_none(r.get("ceii_gene"))
            yv = numeric_or_none(r.get("ceii_site"))
            if xv is None or yv is None:
                continue
            xs.append(xv)
            ys.append(yv)
            names.append(str(r["gene_name"]))
        else:
            abstained_names.append(str(r["gene_name"]))
    ax.scatter(xs, ys, s=40, color="#1f77b4")
    for x, y, n in zip(xs, ys, names):
        ax.text(x + 0.01, y + 0.01, n, fontsize=8)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    if abstained_names:
        ax.text(
            0.02,
            0.98,
            "Abstained/OOD: " + ", ".join(sorted(abstained_names)),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "#f5f5f5", "edgecolor": "#b0b0b0", "boxstyle": "round,pad=0.3"},
        )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("cEII_gene (in-domain only)")
    ax.set_ylabel("cEII_site (in-domain only)")
    ax.set_title("Paired cEII_gene vs cEII_site (abstention-aware)")
    for ext in ("png", "pdf"):
        path = outdir / f"figure2_ceii_gene_vs_ceii_site.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None)
        generated.append(str(path))
    plt.close(fig)

    # Figure 3: eii_01_raw + q_emp always, cEII only for calibrated genes.
    sorted_rows = sorted(valid, key=lambda r: str(r["gene_name"]))
    names = [str(r["gene_name"]) for r in sorted_rows]
    eii_01 = [numeric_or_none(r.get("eii_01_raw")) or float("nan") for r in sorted_rows]
    q_emp = [numeric_or_none(r.get("q_emp")) or float("nan") for r in sorted_rows]
    ceii_gene = []
    ceii_abstained_idx: List[int] = []
    for idx, r in enumerate(sorted_rows):
        cv = numeric_or_none(r.get("ceii_gene"))
        if _is_calibrated(r) and cv is not None:
            ceii_gene.append(cv)
        else:
            ceii_gene.append(0.0)
            ceii_abstained_idx.append(idx)

    x = list(range(len(names)))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.9), 5.6), constrained_layout=True)
    ax.bar([i - width for i in x], eii_01, width=width, label="eii_01_raw")
    bars_ceii = ax.bar(x, ceii_gene, width=width, label="ceii_gene (calibrated only)")
    ax.bar([i + width for i in x], q_emp, width=width, label="q_emp")
    for idx in ceii_abstained_idx:
        bars_ceii[idx].set_hatch("//")
        bars_ceii[idx].set_edgecolor("red")
        bars_ceii[idx].set_facecolor("#f2dede")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Value")
    ax.set_title("eii_01_raw, ceii_gene, and q_emp per gene (abstention-aware)")
    ax.legend()
    for ext in ("png", "pdf"):
        path = outdir / f"figure3_eii_ceii_qemp_per_gene.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None)
        generated.append(str(path))
    plt.close(fig)

    append_log(log_path, f"Generated figures: {generated}")
    return generated


def write_report(
    *,
    outdir: Path,
    summary_rows: List[Dict[str, Any]],
    manifest_rows: List[Dict[str, Any]],
    fig_paths: List[str],
) -> None:
    ok_rows = [r for r in summary_rows if r.get("status") == "ok"]
    failed_rows = [r for r in summary_rows if r.get("status") != "ok"]
    in_domain_rows = [
        r
        for r in ok_rows
        if str(r.get("applicability_status", "")).lower() == "in_domain"
        and numeric_or_none(r.get("ceii_gene")) is not None
    ]
    in_domain_ceii_values = [
        float(v)
        for v in (numeric_or_none(r.get("ceii_gene")) for r in in_domain_rows)
        if v is not None
    ]
    ceii_constant_in_domain = len({round(v, 6) for v in in_domain_ceii_values}) <= 1 if in_domain_ceii_values else False
    in_domain_rows_for_claims = [] if ceii_constant_in_domain else in_domain_rows
    abstained_rows = [r for r in ok_rows if r not in in_domain_rows]

    strongest_ceii = sorted(
        [(r["gene_name"], numeric_or_none(r.get("ceii_gene"))) for r in in_domain_rows_for_claims],
        key=lambda x: (-1 if x[1] is None else 0, -(x[1] or -1)),
    )
    strongest_ceii = [(g, v) for g, v in strongest_ceii if v is not None][:5]
    strongest_raw = sorted(
        [
            (
                r["gene_name"],
                numeric_or_none(r.get("eii_01_raw")),
                numeric_or_none(r.get("q_emp")),
            )
            for r in ok_rows
        ],
        key=lambda x: (-(x[1] or -1.0), x[2] if x[2] is not None else 1.0),
    )[:5]

    ambiguous = [
        r["gene_name"]
        for r in in_domain_rows_for_claims
        if str(r.get("ceii_gene_class", "")) in {"weak_or_ambiguous", "not_identifiable"}
    ]

    by_class_ceii: Dict[str, List[float]] = {}
    for r in in_domain_rows_for_claims:
        cls = str(r.get("expected_control_class", "unknown_control_class"))
        val = numeric_or_none(r.get("ceii_gene"))
        if val is None:
            continue
        by_class_ceii.setdefault(cls, []).append(val)
    class_medians_ceii = {k: statistics.median(v) for k, v in by_class_ceii.items() if v}

    by_class_raw: Dict[str, List[float]] = {}
    for r in ok_rows:
        cls = str(r.get("expected_control_class", "unknown_control_class"))
        val = numeric_or_none(r.get("eii_01_raw"))
        if val is None:
            continue
        by_class_raw.setdefault(cls, []).append(val)
    class_medians_raw = {k: statistics.median(v) for k, v in by_class_raw.items() if v}

    positive_med_ceii = class_medians_ceii.get("expected_positive_control")
    non_positive_meds_ceii = [v for k, v in class_medians_ceii.items() if k != "expected_positive_control"]
    separated_ceii = bool(
        positive_med_ceii is not None and non_positive_meds_ceii and positive_med_ceii > max(non_positive_meds_ceii)
    )

    positive_med_raw = class_medians_raw.get("expected_positive_control")
    non_positive_meds_raw = [v for k, v in class_medians_raw.items() if k != "expected_positive_control"]
    separated_raw = bool(
        positive_med_raw is not None and non_positive_meds_raw and positive_med_raw > max(non_positive_meds_raw)
    )

    paired = [
        (numeric_or_none(r.get("ceii_gene")), numeric_or_none(r.get("ceii_site")))
        for r in in_domain_rows_for_claims
    ]
    paired = [(g, s) for g, s in paired if g is not None and s is not None]
    site_lower_frac = sum(1 for g, s in paired if s < g) / len(paired) if paired else float("nan")

    out_of_domain = [r["gene_name"] for r in ok_rows if str(r.get("applicability_status", "")).lower() != "in_domain"]

    lines = []
    lines.append("# Empirical Ortholog Validation Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Run Scope")
    lines.append("")
    lines.append(f"- Orthogroup FASTA inputs discovered: {len(manifest_rows)}")
    lines.append(f"- Successful BABAPPAi runs: {len(ok_rows)}")
    lines.append(f"- Failed BABAPPAi runs: {len(failed_rows)}")
    lines.append("- Alignment method: babappalign codon mode (when needed)")
    lines.append("- Tree method (if no provided tree): IQ-TREE 3 (`-st DNA -m GTR+F+R4 -nt 1 -seed 42 -redo`)")
    lines.append("- Inference mode: installed `babappai run --offline` with empirical Monte Carlo calibration")
    lines.append("")

    lines.append("## Required Scientific Statements")
    lines.append("")
    if strongest_ceii:
        strongest_txt = ", ".join(f"{g} ({v:.3f})" for g, v in strongest_ceii)
        lines.append(f"- Strongest gene-level identifiability (`ceii_gene`, in-domain only): {strongest_txt}.")
    elif ceii_constant_in_domain and in_domain_rows:
        lines.append(
            "- Strongest gene-level identifiability (`ceii_gene`, in-domain only): withheld because in-domain cEII is constant and non-discriminative."
        )
    else:
        lines.append("- Strongest gene-level identifiability (`ceii_gene`): unavailable because no genes were in-domain calibrated.")
    if strongest_raw:
        raw_txt = ", ".join(
            f"{g} (eii_01_raw={float(e):.3f}, q_emp={float(q):.3f})"
            for g, e, q in strongest_raw
            if e is not None and q is not None
        )
        if raw_txt:
            lines.append(f"- Raw ranking fallback for abstained/OOD genes (by high raw EII and low q_emp): {raw_txt}.")

    if ambiguous:
        lines.append(f"- Ambiguous genes (in-domain cEII only): {', '.join(sorted(set(ambiguous)))}.")
    else:
        lines.append("- Ambiguous genes (in-domain cEII only): none.")

    if class_medians_ceii:
        class_txt = ", ".join(f"{k} median={v:.3f}" for k, v in sorted(class_medians_ceii.items()))
        lines.append(f"- Expected-control separation (median `ceii_gene`, in-domain only): {class_txt}.")
        lines.append(
            f"- Expected positives separate above conservative/housekeeping controls (cEII, in-domain): {'yes' if separated_ceii else 'not clearly'}."
        )
    elif ceii_constant_in_domain and in_domain_rows:
        lines.append(
            "- Expected-control separation via cEII: withheld because in-domain cEII is constant; fallback interpretation uses raw EII + q_emp."
        )
    else:
        lines.append("- Expected-control separation via cEII: unavailable (no in-domain calibrated set).")
    if class_medians_raw:
        class_txt_raw = ", ".join(f"{k} median={v:.3f}" for k, v in sorted(class_medians_raw.items()))
        lines.append(f"- Expected-control separation fallback (median `eii_01_raw`, all genes): {class_txt_raw}.")
        lines.append(
            f"- Expected positives separate above conservative/housekeeping controls (raw EII): {'yes' if separated_raw else 'not clearly'}."
        )

    if paired:
        lines.append(
            f"- Site-level outputs are noisier/lower than gene-level outputs in {site_lower_frac:.1%} of in-domain calibrated genes (fraction with `ceii_site < ceii_gene`)."
        )
    else:
        lines.append("- Site-level vs gene-level cEII comparison: unavailable (insufficient in-domain calibrated genes).")

    if out_of_domain:
        lines.append(f"- Genes outside/near-boundary applicability support (cEII abstained): {', '.join(sorted(out_of_domain))}.")
    else:
        lines.append("- Genes outside applicability support: none.")

    if abstained_rows:
        lines.append(
            "- cEII-based control-class claims were restricted to in-domain genes only; "
            "abstained/OOD genes were interpreted using raw EII + q_emp with explicit caveat."
        )

    lines.append(
        "- Empirical panel interpretation: results support BABAPPAi as an identifiability/recoverability diagnostic; they are not direct proof of positive selection."
    )
    lines.append("")

    lines.append("## Figures")
    lines.append("")
    if fig_paths:
        for p in fig_paths:
            lines.append(f"- {p}")
    else:
        lines.append("- No figures generated (no successful runs).")
    lines.append("")

    if failed_rows:
        lines.append("## Failures")
        lines.append("")
        for row in failed_rows:
            lines.append(f"- {row['gene_name']}: {row.get('error', 'unknown error')}")
        lines.append("")

    (outdir / "empirical_validation_report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()

    ortholog_dir = Path(args.ortholog_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    workdir = outdir / "work"
    run_log = outdir / "run_log.txt"
    mpl_cache = outdir / ".mplcache"

    if outdir.exists() and args.overwrite:
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    mpl_cache.mkdir(parents=True, exist_ok=True)

    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

    append_log(run_log, "Starting empirical ortholog validation run")
    append_log(run_log, f"ortholog_dir={ortholog_dir}")
    append_log(run_log, f"outdir={outdir}")
    append_log(
        run_log,
        (
            "toolchain="
            f"babappai:{args.babappai}; babappalign:{args.babappalign}; iqtree:{args.iqtree}; "
            f"babappalign_model:{args.babappalign_model}"
        ),
    )

    envelope = resolve_applicability_envelope(args.ceii_asset)
    n_taxa_lo, n_taxa_hi = envelope.get("n_taxa", (None, None))
    gene_len_lo, gene_len_hi = envelope.get("gene_length_nt", (None, None))
    append_log(
        run_log,
        (
            "applicability_envelope="
            f"n_taxa:[{n_taxa_lo},{n_taxa_hi}],"
            f"gene_length_nt:[{gene_len_lo},{gene_len_hi}]"
        ),
    )

    fasta_paths = sorted(ortholog_dir.glob("*.fasta"))
    if not fasta_paths:
        raise FileNotFoundError(f"No FASTA files found in {ortholog_dir}")

    manifest_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for fasta_path in fasta_paths:
        gene = fasta_path.stem
        expected_class = EXPECTED_CONTROL_CLASS.get(gene, "unknown_control_class")
        gene_work = workdir / gene
        gene_work.mkdir(parents=True, exist_ok=True)

        append_log(run_log, f"=== Processing gene {gene} ===")

        existing_tree = find_existing_tree_for_fasta(fasta_path)
        sanitized_path = gene_work / f"{gene}.sanitized.fasta"

        try:
            sanitize_stats = sanitize_cds(fasta_path, sanitized_path, log_path=run_log)

            codon_alignment_path = gene_work / f"{gene}.sanitized.codon.aln.fasta"
            alignment_needed = bool(sanitize_stats["alignment_needed"])
            if alignment_needed:
                env = os.environ.copy()
                env["XDG_CACHE_HOME"] = str(outdir / "cache")
                env["HF_HOME"] = str(Path.home() / ".cache/huggingface")
                env["HUGGINGFACE_HUB_CACHE"] = str(Path.home() / ".cache/huggingface/hub")
                env["TRANSFORMERS_CACHE"] = str(Path.home() / ".cache/huggingface/hub")
                env["HF_HUB_OFFLINE"] = "1"
                env["TRANSFORMERS_OFFLINE"] = "1"
                (outdir / "cache").mkdir(parents=True, exist_ok=True)
                run_cmd(
                    [
                        args.babappalign,
                        str(sanitized_path),
                        "--model",
                        args.babappalign_model,
                        "--mode",
                        "codon",
                        "--device",
                        "cpu",
                    ],
                    log_path=run_log,
                    env=env,
                )
                if not codon_alignment_path.exists():
                    raise FileNotFoundError(f"Expected codon alignment not found: {codon_alignment_path}")
                alignment_method = "babappalign codon mode"
            else:
                shutil.copy2(sanitized_path, codon_alignment_path)
                alignment_method = "alignment not needed (uniform sanitized CDS lengths)"

            if existing_tree:
                tree_path = gene_work / existing_tree.name
                shutil.copy2(existing_tree, tree_path)
                tree_generation_method = "provided tree copied from dataset"
                tree_present_input = True
            else:
                run_cmd(
                    [
                        args.iqtree,
                        "-s",
                        str(codon_alignment_path),
                        "-st",
                        "DNA",
                        "-m",
                        "GTR+F+R4",
                        "-nt",
                        "1",
                        "-seed",
                        str(args.seed),
                        "-redo",
                    ],
                    log_path=run_log,
                )
                tree_path = Path(str(codon_alignment_path) + ".treefile")
                if not tree_path.exists():
                    raise FileNotFoundError(f"IQ-TREE treefile missing: {tree_path}")
                tree_generation_method = "iqtree3 -st DNA -m GTR+F+R4 -nt 1 -seed 42 -redo"
                tree_present_input = False

            run_out = gene_work / "babappai_run"
            run_cmd_args = [
                args.babappai,
                "run",
                "--alignment",
                str(codon_alignment_path),
                "--tree",
                str(tree_path),
                "--outdir",
                str(run_out),
                "--offline",
                "--device",
                args.device,
                "--pvalue-mode",
                "empirical_monte_carlo",
                "--n-calibration",
                str(args.n_calibration),
                "--neutral-reps",
                str(args.neutral_reps),
                "--overwrite",
            ]
            if args.ceii_asset:
                run_cmd_args.extend(["--ceii-asset", str(args.ceii_asset)])
            run_cmd(run_cmd_args, log_path=run_log)
            result_path = run_out / "results.json"
            if not result_path.exists():
                raise FileNotFoundError(f"BABAPPAi results missing: {result_path}")

            copied_json = outdir / f"{gene}.results.json"
            shutil.copy2(result_path, copied_json)

            payload = json.loads(result_path.read_text())
            gene_summary = payload.get("gene_summary", {})

            n_taxa = int(sanitize_stats["sanitized_taxon_count"])
            gene_len_nt = int(sanitize_stats["sanitized_length_median"])
            in_n_taxa = (
                True
                if n_taxa_lo is None or n_taxa_hi is None
                else (float(n_taxa_lo) <= n_taxa <= float(n_taxa_hi))
            )
            in_gene_len = (
                True
                if gene_len_lo is None or gene_len_hi is None
                else (float(gene_len_lo) <= gene_len_nt <= float(gene_len_hi))
            )
            within_app = bool(in_n_taxa and in_gene_len)
            envelope_notes = []
            if not in_n_taxa:
                envelope_notes.append(
                    f"n_taxa={n_taxa} outside [{n_taxa_lo},{n_taxa_hi}]"
                )
            if not in_gene_len:
                envelope_notes.append(
                    f"gene_length_nt={gene_len_nt} outside [{gene_len_lo},{gene_len_hi}]"
                )

            manifest_row = {
                "gene_name": gene,
                "source_fasta_path": str(fasta_path),
                "sanitized_fasta_path": str(sanitized_path),
                "codon_alignment_path": str(codon_alignment_path),
                "tree_path": str(tree_path),
                "tree_present_in_input": tree_present_input,
                "tree_generation_method": tree_generation_method,
                "alignment_method": alignment_method,
                "raw_taxon_count": sanitize_stats["raw_taxon_count"],
                "sanitized_taxon_count": sanitize_stats["sanitized_taxon_count"],
                "cds_length_nt_median": sanitize_stats["sanitized_length_median"],
                "cds_length_nt_min": sanitize_stats["sanitized_length_min"],
                "cds_length_nt_max": sanitize_stats["sanitized_length_max"],
                "expected_control_class": expected_class,
                "trimmed_terminal_stop_records": sanitize_stats["trimmed_terminal_stop_records"],
                "trimmed_frame_bases_total": sanitize_stats["trimmed_frame_bases_total"],
                "dropped_internal_stop": sanitize_stats["dropped_internal_stop"],
                "dropped_too_short": sanitize_stats["dropped_too_short"],
                "renamed_duplicate_ids": sanitize_stats["renamed_duplicate_ids"],
                "status": "ok",
                "error": "",
            }
            manifest_rows.append(manifest_row)

            summary_rows.append(
                {
                    "gene_name": gene,
                    "expected_control_class": expected_class,
                    "status": "ok",
                    "error": "",
                    "source_fasta_path": str(fasta_path),
                    "results_json": str(copied_json),
                    "n_taxa": n_taxa,
                    "gene_length_nt": gene_len_nt,
                    "within_applicability_envelope": within_app,
                    "applicability_envelope_notes": "; ".join(envelope_notes) if envelope_notes else "in_envelope",
                    "applicability_score": gene_summary.get("applicability_score"),
                    "applicability_status": gene_summary.get("applicability_status"),
                    "calibration_unavailable_reason": gene_summary.get("calibration_unavailable_reason"),
                    "nearest_supported_regime": gene_summary.get("nearest_supported_regime"),
                    "distance_to_supported_domain": gene_summary.get("distance_to_supported_domain"),
                    "sigma0_valid": gene_summary.get("sigma0_valid"),
                    "sigma0_floored": gene_summary.get("sigma0_floored"),
                    "fallback_applied": gene_summary.get("fallback_applied"),
                    "eii_z_raw": gene_summary.get("eii_z_raw"),
                    "eii_01_raw": gene_summary.get("eii_01_raw"),
                    "ceii_gene": gene_summary.get("ceii_gene"),
                    "ceii_site": gene_summary.get("ceii_site"),
                    "ceii_gene_class": gene_summary.get("ceii_gene_class"),
                    "ceii_site_class": gene_summary.get("ceii_site_class"),
                    "ceii_ci": gene_summary.get("ceii_ci"),
                    "q_emp": gene_summary.get("q_emp"),
                    "significant_bool": gene_summary.get("significant_bool"),
                    "calibration_version": gene_summary.get("calibration_version"),
                    "model_version": gene_summary.get("model_version"),
                    "model_checkpoint_provenance": gene_summary.get("model_checkpoint_provenance"),
                    "domain_shift_or_applicability": gene_summary.get("domain_shift_or_applicability"),
                }
            )

        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            append_log(run_log, f"ERROR:{gene}:{err}")
            manifest_rows.append(
                {
                    "gene_name": gene,
                    "source_fasta_path": str(fasta_path),
                    "sanitized_fasta_path": str(sanitized_path),
                    "codon_alignment_path": "",
                    "tree_path": "",
                    "tree_present_in_input": bool(existing_tree),
                    "tree_generation_method": "",
                    "alignment_method": "",
                    "raw_taxon_count": "",
                    "sanitized_taxon_count": "",
                    "cds_length_nt_median": "",
                    "cds_length_nt_min": "",
                    "cds_length_nt_max": "",
                    "expected_control_class": expected_class,
                    "trimmed_terminal_stop_records": "",
                    "trimmed_frame_bases_total": "",
                    "dropped_internal_stop": "",
                    "dropped_too_short": "",
                    "renamed_duplicate_ids": "",
                    "status": "failed",
                    "error": err,
                }
            )
            summary_rows.append(
                {
                    "gene_name": gene,
                    "expected_control_class": expected_class,
                    "status": "failed",
                    "error": err,
                    "source_fasta_path": str(fasta_path),
                    "results_json": "",
                    "n_taxa": "",
                    "gene_length_nt": "",
                    "within_applicability_envelope": "",
                    "applicability_envelope_notes": "",
                    "applicability_score": "",
                    "applicability_status": "",
                    "calibration_unavailable_reason": "",
                    "nearest_supported_regime": "",
                    "distance_to_supported_domain": "",
                    "sigma0_valid": "",
                    "sigma0_floored": "",
                    "fallback_applied": "",
                    "eii_z_raw": "",
                    "eii_01_raw": "",
                    "ceii_gene": "",
                    "ceii_site": "",
                    "ceii_gene_class": "",
                    "ceii_site_class": "",
                    "ceii_ci": "",
                    "q_emp": "",
                    "significant_bool": "",
                    "calibration_version": "",
                    "model_version": "",
                    "model_checkpoint_provenance": "",
                    "domain_shift_or_applicability": "",
                }
            )

    manifest_rows.sort(key=lambda r: str(r["gene_name"]))
    summary_rows.sort(key=lambda r: str(r["gene_name"]))

    manifest_fields = [
        "gene_name",
        "source_fasta_path",
        "sanitized_fasta_path",
        "codon_alignment_path",
        "tree_path",
        "tree_present_in_input",
        "tree_generation_method",
        "alignment_method",
        "raw_taxon_count",
        "sanitized_taxon_count",
        "cds_length_nt_median",
        "cds_length_nt_min",
        "cds_length_nt_max",
        "expected_control_class",
        "trimmed_terminal_stop_records",
        "trimmed_frame_bases_total",
        "dropped_internal_stop",
        "dropped_too_short",
        "renamed_duplicate_ids",
        "status",
        "error",
    ]
    summary_fields = [
        "gene_name",
        "expected_control_class",
        "status",
        "error",
        "source_fasta_path",
        "results_json",
        "n_taxa",
        "gene_length_nt",
        "within_applicability_envelope",
        "applicability_envelope_notes",
        "applicability_score",
        "applicability_status",
        "calibration_unavailable_reason",
        "nearest_supported_regime",
        "distance_to_supported_domain",
        "sigma0_valid",
        "sigma0_floored",
        "fallback_applied",
        "eii_z_raw",
        "eii_01_raw",
        "ceii_gene",
        "ceii_site",
        "ceii_gene_class",
        "ceii_site_class",
        "ceii_ci",
        "q_emp",
        "significant_bool",
        "calibration_version",
        "model_version",
        "model_checkpoint_provenance",
        "domain_shift_or_applicability",
    ]

    write_table(outdir / "ortholog_manifest.tsv", manifest_rows, manifest_fields, "\t")
    write_table(outdir / "babappai_per_gene_summary.tsv", summary_rows, summary_fields, "\t")
    write_table(outdir / "babappai_per_gene_summary.csv", summary_rows, summary_fields, ",")

    fig_paths = make_figures(summary_rows, outdir, log_path=run_log)
    write_report(outdir=outdir, summary_rows=summary_rows, manifest_rows=manifest_rows, fig_paths=fig_paths)

    append_log(run_log, "Run complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
