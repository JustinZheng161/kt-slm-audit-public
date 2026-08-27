# KT-SLM Audit: Reproducible Student-Disjoint Baselines

This repository contains the **code and aggregate-only artifacts** supporting a reproducible knowledge-tracing (KT) study on the official corrected ASSISTments2009 Skill Builder source. It is a public reproducibility layer, not a data archive and not an article-submission package.

The controlled experiment uses a fixed 80/10/10 student split (seed `20260822`), fits the categorical skill vocabulary on training students only, selects a validation checkpoint within a pre-specified maximum of eight epochs, evaluates 24,306 common second-and-later test interactions, and uses three seeds for neural ablations. It reports a skill prior, per-skill BKT, DKT-64 Adam, DKT-64 AdamW and one joint 96-dimensional/dropout candidate. The ablation record is descriptive: it does not claim optimizer equivalence, a zero effect or a shared cross-paper leaderboard.

## Data access and privacy

The raw ASSISTments2009 file, user identifiers, student-level splits, sequences, per-student predictions, and reversible derived data are **not in this repository**. The official data terms state that anonymized student data must not be shared with others. A private GitHub repository is not an exception to that condition.

To reproduce, obtain the corrected collapsed file independently from the [official ASSISTments2009 page](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010), read its terms, and retain it in an access-controlled directory outside the cloned repository. Set `KT_AUDIT_DATA_ROOT` to that directory before running the scripts. See [DATA_ACCESS.md](DATA_ACCESS.md) and [docs/storage_boundary.md](docs/storage_boundary.md).

The source used in the audit had SHA-256:

```text
162ef8d2d28bcbfea6591a282994062bd8d5eaa00636544292a0d268dca6e5da
```

## Reproduction

Create a Python environment and install the pinned dependencies:

```bash
pip install -r requirements.txt
export KT_AUDIT_DATA_ROOT=/absolute/path/to/controlled/kt-slm-audit
python scripts/profile_assistments.py
python experiments/run_student_disjoint_kt.py --dkt-epochs 8 --threads 4
python experiments/run_clean_seed_check.py
python experiments/run_label_noise_robustness.py
python experiments/run_revision3_extended_evidence.py
python analysis/analyze_revision3_paired_metrics.py
python analysis/make_paper_figures.py
```

The scripts write raw- or student-level artifacts only below the external controlled-data root. The repository contains only public aggregate JSON summaries, source checksums, figures and code. A held-out label absent from the training-fitted vocabulary causes an explicit failure rather than silent handling; the fixed source/split audit found zero such labels (149 training labels; 132 validation labels; 135 test labels).

## Results in the controlled protocol

| Method | Test ROC-AUC | Uncertainty | Interpretation |
| --- | ---: | --- | --- |
| Skill-prior | 0.6231 | Student-cluster 95% CI [0.6100, 0.6361] | No-history reference. |
| Per-skill BKT EM | 0.7245 | Student-cluster 95% CI [0.7075, 0.7424] | Classical reference. |
| DKT-64 Adam | 0.7654 | 3-seed SD 0.0011; Brier 0.1816; ECE10 0.0129 | Primary budget-conditional neural reference. |
| DKT-64 AdamW | 0.7654 | 3-seed SD 0.0011; Brier 0.1816; ECE10 0.0129 | Observed paired AUC difference is below this n=3 design’s resolution. |
| DKT-96 AdamW + dropout | 0.7657 | 3-seed SD 0.0009; Brier 0.1814; ECE10 0.0089 | Joint configuration, not an isolated capacity test. |

The public artifact set also reports a 5%/10%/20% synthetic training-label inversion sensitivity curve and a separate post hoc 20-epoch validation-selection extension. Neither replaces the primary eight-epoch analysis. No external performance values are included here; cross-paper scores must not be subtracted from the above results or presented as a shared leaderboard.

## Citation and source data

Please cite the official ASSISTments data page and the data-system paper: M. Feng, N. T. Heffernan, and K. R. Koedinger, “Addressing the assessment challenge with an online system that tutors as it assesses,” *User Modeling and User-Adapted Interaction*, 2009. The documentation also cites pyKT and simpleKT for standardized-benchmark context.

## License

Code in this repository is released under the MIT License. The license does not grant any rights to the ASSISTments data; see the official data source and its terms.
