# KT-SLM Audit: Reproducible Student-Disjoint Baselines

This repository contains the **code and aggregate artifacts** supporting a reproducible knowledge-tracing (KT) audit on the official corrected ASSISTments2009 Skill Builder source. It separates a student-disjoint, calibrated KT experiment from a historical LoRA/SLM representation audit whose original checkpoints and hidden-state extractor were not preserved.

The controlled experiment uses a fixed 80/10/10 student split (seed `20260822`), validation-selected epochs, three random seeds for neural ablations, and student-cluster bootstrap intervals. It reports a skill-prior, per-skill BKT, DKT-64 Adam, DKT-64 AdamW, and DKT-96 AdamW/dropout candidate. The tested optimization changes are reported as negative ablations rather than performance gains.

## Data access and privacy

The raw ASSISTments2009 file, user identifiers, student-level splits, sequences, per-student predictions, and reversible derived data are **not in this repository**. The official data terms state that anonymized student data must not be shared with others. A private GitHub repository is not an exception to that condition.

To reproduce, obtain the corrected collapsed file independently from the [official ASSISTments2009 page](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010), read its terms, and place it locally at:

```text
private_data/raw/skill_builder_data_corrected_collapsed.csv
```

The source used in the audit had SHA-256:

```text
162ef8d2d28bcbfea6591a282994062bd8d5eaa00636544292a0d268dca6e5da
```

## Reproduction

Create a Python environment and install the pinned dependencies:

```bash
pip install -r requirements.txt
python scripts/profile_assistments.py
python experiments/run_student_disjoint_kt.py --dkt-epochs 8 --threads 4
python experiments/run_clean_seed_check.py
python experiments/run_label_noise_robustness.py
python analysis/make_paper_figures.py
```

The scripts write raw- or student-level artifacts only below the local `private_data/` path. The repository contains only public aggregate JSON summaries and paper figures.

If a controlled-data directory is kept outside the repository or project tree, set `KT_AUDIT_DATA_ROOT=/absolute/path/to/controlled-data` before invoking the scripts. This audit used an external controlled-data root; the environment variable is supported precisely to keep source files out of versioned project folders.

## Results in the controlled protocol

| Method | Test ROC-AUC | Uncertainty | Interpretation |
| --- | ---: | --- | --- |
| Skill-prior | 0.6230 | Student-cluster 95% CI [0.6102, 0.6356] | No history baseline. |
| Per-skill BKT EM | 0.7236 | Student-cluster 95% CI [0.7068, 0.7412] | Classical reference. |
| DKT-64 Adam | 0.7654 | 3-seed SD 0.0011 | Controlled neural baseline. |
| DKT-64 AdamW | 0.7654 | 3-seed SD 0.0011 | No observable gain. |
| DKT-96 AdamW + dropout | 0.7657 | 3-seed SD 0.0009 | Difference is below baseline cross-seed SD. |

External ASSISTments2009 values appear in the accompanying paper only as **protocol-separated references**. They must not be subtracted from the above result or presented as a shared leaderboard.

## Citation and source data

Please cite the official ASSISTments data page and the data-system paper: M. Feng, N. T. Heffernan, and K. R. Koedinger, “Addressing the assessment challenge with an online system that tutors as it assesses,” *User Modeling and User-Adapted Interaction*, 2009. The documentation also cites pyKT and simpleKT for standardized-benchmark context.

## License

Code in this repository is released under the MIT License. The license does not grant any rights to the ASSISTments data; see the official data source and its terms.
