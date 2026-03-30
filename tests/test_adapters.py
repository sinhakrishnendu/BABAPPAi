import json
from pathlib import Path

from babappai.calibration.neutral_generator_adapter import run_neutral_generator
from babappai.metadata import MODEL_TAG
from babappai.validation.simulator_adapter import run_simulator


def test_neutral_generator_adapter(tmp_path):
    script = tmp_path / "generator.py"
    script.write_text(
        "import argparse, json, pathlib\n"
        "p=argparse.ArgumentParser(); p.add_argument('--outdir', required=True); p.add_argument('--seed'); a=p.parse_args()\n"
        "out=pathlib.Path(a.outdir); out.mkdir(parents=True, exist_ok=True)\n"
        "(out/'neutral_reference_frozen.json').write_text(json.dumps({'L_200_K_8': {'sigma2_mean': 1.0, 'sigma2_sd': 0.1}}))\n"
    )

    meta = run_neutral_generator(
        generator_path=str(script),
        output_dir=str(tmp_path / "out"),
        model_tag=MODEL_TAG,
        seed=1,
    )

    assert Path(meta["reference_file"]).exists()
    assert Path(meta["output_dir"]).exists()


def test_simulator_adapter(tmp_path):
    script = tmp_path / "sim.py"
    script.write_text(
        "import argparse, json, pathlib\n"
        "p=argparse.ArgumentParser();\n"
        "p.add_argument('--outdir', required=True); p.add_argument('--seed', type=int, default=0);\n"
        "p.add_argument('--n-taxa'); p.add_argument('--alignment-length'); p.add_argument('--perturbation-sparsity');\n"
        "p.add_argument('--perturbation-magnitude'); p.add_argument('--branch-length-scale');\n"
        "p.add_argument('--recombination-rate'); p.add_argument('--alignment-noise');\n"
        "a=p.parse_args(); out=pathlib.Path(a.outdir); out.mkdir(parents=True, exist_ok=True)\n"
        "(out/'alignment.fasta').write_text('>t1\\nATGATGATG\\n>t2\\nATGATGATG\\n>t3\\nATGATGATG\\n>t4\\nATGATGATG\\n')\n"
        "(out/'tree.nwk').write_text('((t1:0.1,t2:0.1):0.1,(t3:0.1,t4:0.1):0.1);\\n')\n"
        "(out/'truth_metadata.json').write_text(json.dumps({'simulated': True}))\n"
    )

    meta = run_simulator(
        simulator_path=str(script),
        outdir=str(tmp_path / "sim_out"),
        replicate_id="rep0001",
        seed=7,
        params={
            "n_taxa": 8,
            "alignment_length": 300,
            "perturbation_sparsity": 0.1,
            "perturbation_magnitude": 1.0,
            "branch_length_scale": 1.0,
            "recombination_rate": 0.0,
            "alignment_noise": 0.0,
        },
    )

    assert Path(meta["alignment_path"]).exists()
    assert Path(meta["tree_path"]).exists()
    assert meta["truth_metadata"]["simulated"] is True
