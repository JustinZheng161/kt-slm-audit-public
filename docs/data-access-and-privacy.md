# Data access and privacy

The repository is the public reproducibility release. It contains aggregate metrics, de-identified figure inputs, data cards, technical manifests, and code needed to reproduce the released figures. It does not contain raw learner event records, student identifiers, row-level split membership, hidden states, logits, model checkpoints, private salts, or unredacted manuscript files.

| Material | Public repository | Private repository |
|---|---:|---:|
| Aggregate metrics and figure inputs | Yes | Yes |
| Data cards and sampling summaries | Yes | Yes |
| Reproducible plotting and analysis source | Yes | Yes |
| Raw corrected ASSISTments records | No | Yes |
| Student-level manifests and private hashes | No | Yes |
| Hidden states, logits and model caches | No | Yes |
| Final manuscript and revision history | No | Yes |

The public data files are organized by their existing stable roles: `data/` contains aggregate figure inputs, `metadata/` contains dataset cards and aggregate audits, and `results/` contains released aggregate experiment summaries. The private repository contains the complete source-data and derived-results archive. Public files can be regenerated from the source scripts without access to private core materials.
