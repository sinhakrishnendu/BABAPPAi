"""Empirical validation workflow for selected orthogroups."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from babappai.run_pipeline import run_and_write_outputs
from babappai.stats import annotate_bh_qvalues
from babappai.validation.orthogroup_qc import orthogroup_id_from_path


def _load_selected_rows(selected_input: str) -> List[Dict[str, str]]:
    path = Path(selected_input).expanduser().resolve()
    if path.is_dir():
        candidate = path / "selected_100_orthogroups.tsv"
        if not candidate.exists():
            raise FileNotFoundError(
                "Expected selected_100_orthogroups.tsv inside input directory: "
                f"{path}"
            )
        path = candidate

    if not path.exists():
        raise FileNotFoundError(f"Selected orthogroup input not found: {path}")

    with path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return [dict(row) for row in reader if row]


def _write_tsv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _annotate_empirical_qvalues(rows: List[Dict[str, Any]], alpha: float) -> None:
    annotate_bh_qvalues(rows, p_key="p_emp", q_key="q_emp")
    for row in rows:
        try:
            qv = float(row.get("q_emp", float("nan")))
        except (TypeError, ValueError):
            qv = float("nan")
        row["alpha_used"] = float(alpha)
        row["significant_bool"] = bool(qv <= alpha) if qv == qv else False
        row["significance_label"] = "significant" if row["significant_bool"] else "not_significant"


def _write_star_tree_for_alignment(alignment_path: Path, out_tree_path: Path) -> Path:
    records = list(SeqIO.parse(alignment_path, "fasta"))
    if not records:
        raise ValueError(f"Alignment is empty: {alignment_path}")
    leaves = ",".join(f"{rec.id}:0.1" for rec in records)
    out_tree_path.parent.mkdir(parents=True, exist_ok=True)
    out_tree_path.write_text(f"({leaves});\n")
    return out_tree_path


def _subsample_alignment(alignment_path: Path, out_path: Path) -> Optional[Path]:
    records = list(SeqIO.parse(alignment_path, "fasta"))
    if len(records) < 4:
        return None
    sampled = records[:-1]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(sampled, out_path, "fasta")
    return out_path


def _perturb_alignment(alignment_path: Path, out_path: Path, seed: int) -> Path:
    rng = random.Random(seed)
    records = list(SeqIO.parse(alignment_path, "fasta"))
    if not records:
        raise ValueError(f"Alignment is empty: {alignment_path}")

    codon_choices = [
        "TTT", "TTC", "TTA", "TTG", "CTT", "CTC", "CTA", "CTG",
        "ATT", "ATC", "ATA", "ATG", "GTT", "GTC", "GTA", "GTG",
        "TCT", "TCC", "TCA", "TCG", "CCT", "CCC", "CCA", "CCG",
        "ACT", "ACC", "ACA", "ACG", "GCT", "GCC", "GCA", "GCG",
        "TAT", "TAC", "CAT", "CAC", "CAA", "CAG", "AAT", "AAC",
        "AAA", "AAG", "GAT", "GAC", "GAA", "GAG", "TGT", "TGC",
        "TGG", "CGT", "CGC", "CGA", "CGG", "AGT", "AGC", "AGA",
        "AGG", "GGT", "GGC", "GGA", "GGG",
    ]

    perturbed: List[SeqRecord] = []
    for rec in records:
        seq = str(rec.seq).upper()
        if len(seq) < 3:
            perturbed.append(rec)
            continue
        n_codons = len(seq) // 3
        n_mut = max(1, n_codons // 100)
        positions = sorted(rng.sample(range(n_codons), k=min(n_mut, n_codons)))
        chars = list(seq)
        for pos in positions:
            start = pos * 3
            replacement = rng.choice(codon_choices)
            chars[start:start + 3] = list(replacement)
        perturbed.append(SeqRecord(Seq("".join(chars)), id=rec.id, description=""))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(perturbed, out_path, "fasta")
    return out_path


def run_empirical_validation(
    *,
    selected_input: str,
    outdir: str,
    tree_calibration: bool,
    n_calibration: int,
    device: str,
    batch_size: int,
    seed: int,
    foreground_mode: str,
    foreground_list: Optional[str],
    offline: bool,
    overwrite: bool,
    sigma_floor: float,
    alpha: float,
    pvalue_mode: str,
    min_neutral_group_size: int,
    neutral_reps: int,
    robustness_limit: int = 10,
) -> Dict[str, Any]:
    out = Path(outdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    selected_rows = _load_selected_rows(selected_input)
    summary_rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(selected_rows):
        alignment_path = Path(row["alignment_path"]).expanduser().resolve()
        if not alignment_path.exists():
            raise FileNotFoundError(f"Alignment not found: {alignment_path}")

        orthogroup_id = row.get("orthogroup_id") or orthogroup_id_from_path(alignment_path)
        orthogroup_out = out / "orthogroups" / orthogroup_id

        tree_value = row.get("tree_path", "").strip()
        if tree_value and Path(tree_value).exists():
            tree_path = Path(tree_value).resolve()
        else:
            tree_path = _write_star_tree_for_alignment(
                alignment_path,
                out / "generated_trees" / f"{orthogroup_id}.nwk",
            )

        payload = run_and_write_outputs(
            alignment_path=str(alignment_path),
            tree_path=str(tree_path),
            outdir=orthogroup_out,
            command=(
                "babappai validate orthogroups run "
                f"--input {selected_input} --outdir {outdir}"
            ),
            tree_calibration=tree_calibration,
            n_calibration=n_calibration,
            device=device,
            batch_size=batch_size,
            seed=seed,
            foreground_mode=foreground_mode,
            foreground_list=foreground_list,
            offline=offline,
            overwrite=overwrite,
            sigma_floor=sigma_floor,
            alpha=alpha,
            pvalue_mode=pvalue_mode,
            min_neutral_group_size=min_neutral_group_size,
            neutral_reps=neutral_reps,
        )

        gene = payload["gene_summary"]
        summary = {
            "orthogroup_id": orthogroup_id,
            "alignment_path": str(alignment_path),
            "tree_path": str(tree_path),
            "D_obs": gene.get("D_obs"),
            "mu0": gene.get("mu0"),
            "sigma0_raw": gene.get("sigma0_raw"),
            "sigma0_final": gene.get("sigma0_final"),
            "sigma_floor_used": gene.get("sigma_floor_used"),
            "fallback_flag": gene.get("fallback_flag"),
            "fallback_reason": gene.get("fallback_reason"),
            "neutral_group_size": gene.get("neutral_group_size"),
            "calibration_group": gene.get("calibration_group"),
            "calibration_version": gene.get("calibration_version"),
            "applicability_score": gene.get("applicability_score"),
            "applicability_status": gene.get("applicability_status"),
            "within_applicability_envelope": gene.get("within_applicability_envelope"),
            "calibration_unavailable_reason": gene.get("calibration_unavailable_reason"),
            "nearest_supported_regime": gene.get("nearest_supported_regime"),
            "distance_to_supported_domain": gene.get("distance_to_supported_domain"),
            "domain_shift_or_applicability": gene.get("domain_shift_or_applicability"),
            "EII_z": gene["EII_z"],
            "EII_01": gene["EII_01"],
            "eii_z_raw": gene.get("eii_z_raw", gene["EII_z"]),
            "eii_01_raw": gene.get("eii_01_raw", gene["EII_01"]),
            "ceii_gene": gene.get("ceii_gene"),
            "ceii_site": gene.get("ceii_site"),
            "ceii_gene_class": gene.get("ceii_gene_class"),
            "ceii_site_class": gene.get("ceii_site_class"),
            "ceii_gene_identifiable_bool": gene.get("ceii_gene_identifiable_bool"),
            "ceii_site_identifiable_bool": gene.get("ceii_site_identifiable_bool"),
            "p_emp": gene.get("p_emp"),
            "q_emp": gene.get("q_emp"),
            "alpha_used": gene.get("alpha_used"),
            "significant_bool": gene.get("significant_bool"),
            "significance_label": gene.get("significance_label"),
            "identifiable_bool": gene["identifiable_bool"],
            "identifiability_extent": gene["identifiability_extent"],
            "results_json": str(orthogroup_out / "results.json"),
            "branch_summary_tsv": str(orthogroup_out / "branch_summary.tsv"),
            "site_summary_tsv": str(orthogroup_out / "site_summary.tsv"),
            "run_metadata_json": str(orthogroup_out / "run_metadata.json"),
            "repeatability_delta_EII_01": None,
            "calibration_delta_EII_01": None,
            "taxon_subsample_delta_EII_01": None,
            "perturbation_delta_EII_01": None,
        }

        if idx < robustness_limit:
            repeat = run_and_write_outputs(
                alignment_path=str(alignment_path),
                tree_path=str(tree_path),
                outdir=orthogroup_out / "robustness_repeatability",
                command=f"babappai robustness repeatability {orthogroup_id}",
                tree_calibration=tree_calibration,
                n_calibration=n_calibration,
                device=device,
                batch_size=batch_size,
                seed=seed,
                foreground_mode=foreground_mode,
                foreground_list=foreground_list,
                offline=offline,
                overwrite=True,
                sigma_floor=sigma_floor,
                alpha=alpha,
                pvalue_mode=pvalue_mode,
                min_neutral_group_size=min_neutral_group_size,
                neutral_reps=neutral_reps,
            )
            summary["repeatability_delta_EII_01"] = abs(
                float(repeat["gene_summary"]["EII_01"]) - float(gene["EII_01"])
            )

            compare_cal = run_and_write_outputs(
                alignment_path=str(alignment_path),
                tree_path=str(tree_path),
                outdir=orthogroup_out / "robustness_calibration_compare",
                command=f"babappai robustness calibration {orthogroup_id}",
                tree_calibration=not tree_calibration,
                n_calibration=max(20, min(100, n_calibration)),
                device=device,
                batch_size=batch_size,
                seed=seed,
                foreground_mode=foreground_mode,
                foreground_list=foreground_list,
                offline=offline,
                overwrite=True,
                sigma_floor=sigma_floor,
                alpha=alpha,
                pvalue_mode=pvalue_mode,
                min_neutral_group_size=min_neutral_group_size,
                neutral_reps=neutral_reps,
            )
            summary["calibration_delta_EII_01"] = abs(
                float(compare_cal["gene_summary"]["EII_01"]) - float(gene["EII_01"])
            )

            subsampled_alignment = _subsample_alignment(
                alignment_path,
                orthogroup_out / "robustness_taxon_subsample" / "alignment.fasta",
            )
            if subsampled_alignment is not None:
                subsampled_tree = _write_star_tree_for_alignment(
                    subsampled_alignment,
                    orthogroup_out / "robustness_taxon_subsample" / "tree.nwk",
                )
                subsampled = run_and_write_outputs(
                    alignment_path=str(subsampled_alignment),
                    tree_path=str(subsampled_tree),
                    outdir=orthogroup_out / "robustness_taxon_subsample" / "run",
                    command=f"babappai robustness taxon_subsample {orthogroup_id}",
                    tree_calibration=tree_calibration,
                    n_calibration=n_calibration,
                    device=device,
                    batch_size=batch_size,
                    seed=seed,
                    foreground_mode=foreground_mode,
                    foreground_list=foreground_list,
                    offline=offline,
                    overwrite=True,
                    sigma_floor=sigma_floor,
                    alpha=alpha,
                    pvalue_mode=pvalue_mode,
                    min_neutral_group_size=min_neutral_group_size,
                    neutral_reps=neutral_reps,
                )
                summary["taxon_subsample_delta_EII_01"] = abs(
                    float(subsampled["gene_summary"]["EII_01"]) - float(gene["EII_01"])
                )

            perturbed_alignment = _perturb_alignment(
                alignment_path,
                orthogroup_out / "robustness_perturbation" / "alignment.fasta",
                seed=seed + idx + 17,
            )
            perturbed_tree = _write_star_tree_for_alignment(
                perturbed_alignment,
                orthogroup_out / "robustness_perturbation" / "tree.nwk",
            )
            perturbed = run_and_write_outputs(
                alignment_path=str(perturbed_alignment),
                tree_path=str(perturbed_tree),
                outdir=orthogroup_out / "robustness_perturbation" / "run",
                command=f"babappai robustness perturbation {orthogroup_id}",
                tree_calibration=tree_calibration,
                n_calibration=n_calibration,
                device=device,
                batch_size=batch_size,
                seed=seed,
                foreground_mode=foreground_mode,
                foreground_list=foreground_list,
                offline=offline,
                overwrite=True,
                sigma_floor=sigma_floor,
                alpha=alpha,
                pvalue_mode=pvalue_mode,
                min_neutral_group_size=min_neutral_group_size,
                neutral_reps=neutral_reps,
            )
            summary["perturbation_delta_EII_01"] = abs(
                float(perturbed["gene_summary"]["EII_01"]) - float(gene["EII_01"])
            )

        summary_rows.append(summary)

    summary_rows.sort(key=lambda row: row["orthogroup_id"])
    _annotate_empirical_qvalues(summary_rows, alpha=alpha)

    fields = [
        "orthogroup_id",
        "alignment_path",
        "tree_path",
        "D_obs",
        "mu0",
        "sigma0_raw",
        "sigma0_final",
        "sigma_floor_used",
        "fallback_flag",
        "fallback_reason",
        "neutral_group_size",
        "calibration_group",
        "calibration_version",
        "applicability_score",
        "applicability_status",
        "within_applicability_envelope",
        "calibration_unavailable_reason",
        "nearest_supported_regime",
        "distance_to_supported_domain",
        "domain_shift_or_applicability",
        "EII_z",
        "EII_01",
        "eii_z_raw",
        "eii_01_raw",
        "ceii_gene",
        "ceii_site",
        "ceii_gene_class",
        "ceii_site_class",
        "ceii_gene_identifiable_bool",
        "ceii_site_identifiable_bool",
        "p_emp",
        "q_emp",
        "alpha_used",
        "significant_bool",
        "significance_label",
        "identifiable_bool",
        "identifiability_extent",
        "repeatability_delta_EII_01",
        "calibration_delta_EII_01",
        "taxon_subsample_delta_EII_01",
        "perturbation_delta_EII_01",
        "results_json",
        "branch_summary_tsv",
        "site_summary_tsv",
        "run_metadata_json",
    ]

    _write_tsv(out / "empirical_summary.tsv", summary_rows, fields)
    (out / "empirical_summary.json").write_text(json.dumps(summary_rows, indent=2) + "\n")

    metadata = {
        "selected_input": str(Path(selected_input).resolve()),
        "n_orthogroups": len(summary_rows),
        "robustness_limit": robustness_limit,
        "alpha_used": float(alpha),
        "pvalue_mode": pvalue_mode,
        "note": (
            "BABAPPAi is the software framework around the canonical frozen "
            "BABAPPAΩ model backbone."
        ),
    }
    (out / "empirical_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
