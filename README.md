# KT-SLM Audit: Reproducible Student-Disjoint Baselines

This repository contains the **code and aggregate-only artifacts** supporting a reproducible knowledge-tracing (KT) study on the official corrected ASSISTments2009 Skill Builder source. It is a public reproducibility layer, not a data archive and not an article-submission package.

The controlled experiment uses a fixed 80/10/10 student split (seed `20260822`), fits the categorical skill vocabulary on training students only, selects a validation checkpoint within a pre-specified maximum of eight epochs, evaluates 24,306 common second-and-later test interactions, and uses three seeds for neural ablations. It reports a skill prior, per-skill BKT, DKT-64 Adam, DKT-64 AdamW and one joint 96-dimensional/dropout candidate. The ablation record is descriptive: it does not claim optimizer equivalence, a zero effect or a shared cross-paper leaderboard.

## Data access and privacy

The raw ASSISTments2009 file, user identifiers, student-level splits, sequences, per-student predictions, and reversible derived data are **not in this repository**. The official data terms state that anonymized student data must not be shared with others. A private GitHub repository is not an exception to that condition.

To reproduce, obtain the corrected collapsed file independently from the [official ASSISTments2009 page](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010), read its terms, and retain it in an access-controlled directory outside the cloned repository. Set `KT_AUDIT_DATA_ROOT` to that directory before running the scripts. See [DATA-ACCESS.md](DATA-ACCESS.md) and [docs/storage-boundary.md](docs/storage-boundary.md).

The source used in the audit had SHA-256:

```text
162ef8d2d28bcbfea6591a282994062bd8d5eaa00636544292a0d268dca6e5da
```

## Reproduction

Create a Python environment and install the pinned dependencies:

```bash
pip install -r requirements.txt
export KT_AUDIT_DATA_ROOT=/absolute/path/to/controlled/kt-slm-audit
python scripts/profile-assistments.py
python experiments/run-student-disjoint-kt.py --dkt-epochs 8 --threads 4
python experiments/run-clean-seed-check.py
python experiments/run-label-noise-robustness.py
python experiments/run-revision3-extended-evidence.py
python analysis/analyze-revision3-paired-metrics.py
python analysis/summarize-seed-observed-ranges.py
python analysis/profile-history-length-distribution.py
python experiments/run-window-length-sensitivity.py
python analysis/make-paper-figures.py
python analysis/make-revision5-window-figure.py
python analysis/run-revision6-statistical-and-bkt-audits.py
python analysis/make-revision6-bkt-diagnostic-figure.py

# A prospective context-availability sensitivity audit. It is not part of the
# archived Revision 3 primary result and must be reported as a separate audit.
python experiments/run-context-parity-audit.py
```

The scripts write raw- or student-level artifacts only below the external controlled-data root. The repository contains only public aggregate JSON summaries, source checksums, figures and code. A held-out label absent from the training-fitted vocabulary causes an explicit failure rather than silent handling; the fixed source/split audit found zero such labels (149 training labels; 132 validation labels; 135 test labels).

## Results in the controlled protocol

| Method | Test ROC-AUC | Uncertainty | Interpretation |
| --- | ---: | --- | --- |
| Skill-prior | 0.6231 | Student-cluster 95% CI [0.6100, 0.6361] | No-history reference. |
| Per-skill BKT EM | 0.7245 | Student-cluster 95% CI [0.7075, 0.7424] | Classical reference. |
| DKT-64 Adam | 0.7654 | 3-seed SD 0.0011; Brier 0.1816; ECE10 0.0129 | Primary budget-conditional neural reference. |
| DKT-64 AdamW | 0.7654 | 3-seed SD 0.0011; Brier 0.1816; ECE10 0.0129 | Observed mean difference is smaller than reference cross-seed SD; this does not imply zero difference or equivalence. |
| DKT-96 AdamW + dropout | 0.7657 | 3-seed SD 0.0009; Brier 0.1814; ECE10 0.0089 | Joint configuration, not an isolated capacity test. |

## Candidate model improvements

Two protocol-compatible model candidates are provided under `models/optimized-dkt.py`. `LayerNormResidualDKT` adds post-recurrent normalization and a gated residual projection; `TemporalAttentionDKT` adds a padding-aware causal multi-head attention refinement. The designs are motivated by the stabilization and sequence-context principles used in residual Transformer architectures and by attention-based KT models such as AKT and simpleKT [1] [2]. They preserve the interaction-token interface, future-response target, student-disjoint split, and per-skill output head. They are candidates for a controlled follow-up, not silently substituted replacements for the archived DKT results.

Run the shape and gradient smoke test with:

```bash
python tests/run-optimized-smoke.py
```

The smoke test passes after installing `requirements.txt`. No performance number for either candidate is reported until it has been trained and evaluated with the same fixed split and seeds.

The public artifact set also reports a 5%/10%/20% synthetic training-label inversion sensitivity curve, a separate post hoc 20-epoch validation-selection extension, a literal three-seed observed min–max summary, a completed 200/500/full-history training-window sensitivity analysis, standardized paired seed-effect summaries, cross-skill BKT stability summaries, a 20-versus-100 fixed-iteration BKT sensitivity check, and a DKT–BKT student-cluster paired bootstrap. Cohen’s d_z values and observed-SD MDE references are descriptive post hoc scales only: they are neither a pre-specified SESOI nor equivalence analyses. The window analysis fixes the split, DKT-64 architecture, optimizer, batch size, eight-epoch cap, validation selection and complete-history test evaluation; its small observed mean differences do not establish global window-length invariance. The BKT audit reports only cross-skill quantiles and counts, never skill-level estimates. Neither supplemental analysis replaces the primary eight-epoch result. A prospective context-parity audit is supplied as code but is not reported as a result until it has been run on the controlled source. No external performance values are included here; cross-paper scores must not be subtracted from the above results or presented as a shared leaderboard.

## Technical figure outputs

Every aggregate-only figure in `figures/revision3/` is available as a PDF vector source. To generate the matching 600-dpi TIFF submission derivatives and an auditable file/DPI manifest, run:

```bash
python scripts/export-print-figures.py
```

The command writes TIFF files to `figures/revision3/tiff-600dpi/` and `revision7-figure-technical-manifest.json`. It does not access the controlled data root. Figure 1 has a legend that separately identifies single-run bars, student-cluster bootstrap error bars, and open circles for the three DKT seed estimates. Figure captions specify whether error bars are bootstrap intervals or descriptive across-seed SDs.

## Double-blind review mirror

This is an identified development repository and must **not** be cited in a blinded manuscript. Before resubmission, create and independently inspect a fixed-commit, read-only anonymous mirror containing only code and aggregate-safe outputs; the complete identity-redaction, expiry, access and privacy checklist is in [ANONYMOUS-REVIEW-MIRROR.md](docs/ANONYMOUS-REVIEW-MIRROR.md). Until that external release gate is closed, `[ANONYMOUS_REVIEW_CODE_URL]` is an intentional placeholder, not an actual review link.

## Citation and source data

Please cite the official ASSISTments data page and the data-system paper: M. Feng, N. T. Heffernan, and K. R. Koedinger, “Addressing the assessment challenge with an online system that tutors as it assesses,” *User Modeling and User-Adapted Interaction*, 2009. The documentation also cites pyKT and simpleKT for standardized-benchmark context. The relevant sources are [pyKT](https://arxiv.org/html/2206.11460), [simpleKT](https://arxiv.org/html/2302.06881v2), [AKT](https://dl.acm.org/doi/10.1145/3394486.3403282), and [RouterKT](https://arxiv.org/html/2504.08989v1).

## License

Code in this repository is released under the MIT License. The license does not grant any rights to the ASSISTments data; see the official data source and its terms.

## v25 identity-audit data-capacity check

The v25 revision adds `scripts/audit-repeat-observation-adequacy.py`, which reports only aggregate student-repeat counts from a user-supplied source CSV. On the private top-11-skill source, the audit found 115,681 interactions, 3,176 students, 3,021 students with at least two rows, and 2,724 students with at least five rows. The public metadata file `metadata/repeat-observation-adequacy-v25.json` contains these aggregate counts only; it does not contain student identifiers or hidden states. This check establishes data capacity for a future student-grouped hidden-state identity audit, not a completed model-behavior result.

Run it with:

```bash
python scripts/audit-repeat-observation-adequacy.py /path/to/source.csv --student-column user_id --output metadata/repeat-observation-adequacy-v25.json
```

## v26 manuscript positioning

The v26 manuscript is intentionally positioned as a **case-study protocol** for prompt-interface auditing in the tested Qwen2.5-0.5B-Instruct and corrected ASSISTments setting. The repository does not claim that the protocol has been empirically validated across models, datasets, or the full skill distribution. The 1.000 explicit-field probe is treated as an expected accessibility diagnostic, not as evidence of semantic skill understanding or identity-independent learner-state representation. The unrun candidate architectures remain code artifacts only and are not reported as experimental results.

## v27 manuscript revision

The v27 manuscript retains the case-study protocol positioning and now repeats the exact generality limitation in Section 6: the protocol has not been empirically demonstrated beyond the tested Qwen checkpoint, corrected ASSISTments source, and high-frequency skill subset. The Introduction uses `design` and `workflow` where possible for readability, Section 2.7 now transitions explicitly to the auditing literature in Section 2.8, and the fixed-sample limitation plus necessary next step are organized within Section 6. No unsupported runtime or hardware claim is reported.

## Nature-style figure reconstruction

Figure 1–Figure 6 have been reconstructed with a unified Nature-style visual system using aggregate, de-identified results. The reproducible source is `scripts/reconstruct-figures-nature.py`; the aggregate input is `data/figure-data-nature-v31.json`; and the generated PNG, SVG and PDF files are under `figures/nature-v31/`. The script does not read raw learner events, student identifiers, hidden states, checkpoints or individual predictions.

Run from the repository root:

```bash
python3 scripts/reconstruct-figures-nature.py
```

The output uses a restrained palette, compact sans-serif typography, thin axes, minimal grid lines, consistent panel spacing, and high-resolution 600-dpi PNG export. SVG and PDF exports are also generated for journal production workflows.
