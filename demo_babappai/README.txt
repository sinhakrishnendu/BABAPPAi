Example files for BABAPPAi.

Run inference:
  babappai run --alignment demo_babappai/aln.fasta --tree demo_babappai/tree.nwk --outdir demo_babappai/demo_out

Run orthogroup selection:
  babappai validate orthogroups select --input <ORTHOGROUP_DIR> --outdir demo_babappai/selection_out

Run synthetic validation:
  babappai validate synthetic run --simulator <path/to/simulator.py> --outdir demo_babappai/synthetic_out
