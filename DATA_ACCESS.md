# Data Access and Reproducibility

This repository intentionally does **not** redistribute ASSISTments student records. The source terms prohibit transferring the anonymized student data to third parties. Raw CSV files, student identifiers, student-level splits, sequences, interaction-level predictions, and checkpoints remain outside this repository.

## Primary source

The primary experiment uses the official corrected collapsed ASSISTments2009 Skill Builder file available from the [ASSISTmentsData source page](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010). Users must read and comply with the [ASSISTmentsData Terms of Use](https://sites.google.com/site/assistmentsdata/termsofuseforusingdata) before downloading or processing it.

The expected SHA-256 digest of the input file is:

```text
162ef8d2d28bcbfea6591a282994062bd8d5eaa00636544292a0d268dca6e5da
```

The source profile used in the paper is 346,860 rows, 4,217 users, 26,688 problems and 150 non-empty provider skill strings. After the documented two-or-more-interaction filter, the fixed split contains 3,221/403/403 train/validation/test students; the training-fitted categorical vocabulary has 149 skills and covers all 132 validation and 135 test skills.

## Controlled local setup

After obtaining authorized source files, keep them outside the cloned repository and direct the scripts to that controlled location:

```bash
export KT_AUDIT_DATA_ROOT=/absolute/path/to/controlled/kt-slm-audit
python3 research_code/experiments/run_student_disjoint_kt.py
python3 research_code/experiments/run_revision3_extended_evidence.py
python3 research_code/analysis/make_paper_figures.py
```

The released result files are aggregate-only. They contain summary metrics, counts, fixed-bin calibration aggregates and source checksums, but no learner records or individual predictions. See `github_storage_classification.md` for the repository boundary and `research_code/metadata/skill_label_support_audit.json` for the held-out label-support audit.
