# RLVR Data Decontamination

Code and reproducibility artifacts for **What Should We Remove? Evaluating the Reliability and Costs of RLVR Data Decontamination**.

## Goal

RLVR math datasets are commonly cleaned by removing training examples that resemble evaluation benchmarks. Aggressive similarity-based cleaning, however, may also remove legitimate and independently created problems.

This repository evaluates the **reliability and cost of different decontamination policies**, including:

- Exact matching
- Lexical similarity
- Semantic similarity
- Combined matching policies
- Judge/adjudication-based filtering

The analysis focuses on two questions:

1. How reliably does each policy identify genuine benchmark exposure?
2. How much legitimate training data does each policy incorrectly remove?

This repository does **not** make claims about downstream training effects.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── config/
│   └── experiment configuration
├── data/
│   ├── benchmarks/        # benchmark snapshots
│   ├── corpora/           # RLVR corpus metadata
│   ├── candidates/        # candidate training-benchmark pairs
│   └── review/            # reviewed and adjudicated labels
├── scripts/
│   ├── data preparation
│   ├── candidate generation
│   ├── policy evaluation
│   ├── false-removal analysis
│   └── coverage analysis
├── analysis/
│   ├── calibration results
│   ├── held-out evaluation
│   ├── item-level false-removal analysis
│   └── full-population removal statistics
├── figures/
│   └── publication figures
└── paper/
    ├── main.tex
    ├── tables.tex
    └── tables_appendix.tex
```

> Adjust folder names above if your final GitHub layout differs from this structure.

## Evaluation Pipeline

1. Load RLVR training corpora and evaluation benchmarks.
2. Generate candidate training-benchmark overlaps.
3. Apply exact, lexical, and semantic matching methods.
4. Conduct independent blinded review and adjudication.
5. Build reviewed reference labels.
6. Calibrate candidate decontamination policies.
7. Evaluate policies on held-out data.
8. Measure item-level false removals.
9. Measure full-corpus removal cost and coverage.

## Main Outputs

The pipeline produces:

- Pair-level precision and recall by policy
- Held-out policy evaluation
- Item-level false-removal analysis
- Full-corpus removal counts and rates
- Source-composition changes
- Uncertainty estimates
- Publication-ready tables and figures

## Scope

The experiments evaluate **data decontamination policies**, not downstream model training.

The central trade-off is between:

- removing genuine benchmark exposure; and
- accidentally deleting legitimate RLVR training examples.

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Then run the relevant stages under `scripts/` using the experiment settings under `config/`.

For exact reproduction, use the same corpus and benchmark snapshots, candidate-generation settings, review labels, and policy thresholds reported in the paper.

## Paper

**What Should We Remove? Evaluating the Reliability and Costs of RLVR Data Decontamination**

The repository is intended to accompany the paper and provide the code, reviewed artifacts, policy decisions, and analysis needed to reproduce the reported decontamination evaluation.

## Citation

If you use this repository, please cite the accompanying paper. Update the entry below with the final arXiv identifier after submission:

```bibtex
@article{xin2026whatshouldweremove,
  title  = {What Should We Remove? Evaluating the Reliability and Costs of RLVR Data Decontamination},
  author = {Xin, Esther},
  year   = {2026}
}
```

## License

See `LICENSE` for repository licensing information.
