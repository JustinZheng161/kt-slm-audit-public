"""Profile controlled ASSISTments2015 data without exporting learner-level rows."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[2] if (_HERE.parents[2] / "research_code").exists() else _HERE.parents[1]
CONTROLLED_DATA_ROOT = Path(os.environ.get("KT_AUDIT_DATA_ROOT", ROOT / "private_data"))
RAW = CONTROLLED_DATA_ROOT / "raw" / "assistments2015_skill_builder.csv"
PRIVATE_OUT = CONTROLLED_DATA_ROOT / "metadata" / "assistments2015_profile.json"
PUBLIC_OUT = (ROOT / "research_code" / "metadata" if (ROOT / "research_code").exists() else ROOT / "metadata") / "assistments2015_data_card.json"


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    users: set[str] = set()
    sequence_ids: set[str] = set()
    labels = Counter()
    rows = 0
    with RAW.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        required = {"user_id", "sequence_id", "correct"}
        absent = required.difference(columns)
        if absent:
            raise ValueError(f"Missing expected fields: {sorted(absent)}")
        for row in reader:
            rows += 1
            users.add(row["user_id"])
            sequence_ids.add(row["sequence_id"])
            labels[row["correct"]] += 1
    summary = {
        "dataset": "ASSISTments2015 Skill Builder",
        "source_url": "https://sites.google.com/site/assistmentsdata/datasets/2015-assistments-skill-builder-data",
        "source_file": RAW.name,
        "sha256": checksum(RAW),
        "row_count_excluding_header": rows,
        "unique_students": len(users),
        "unique_sequence_ids": len(sequence_ids),
        "correct_count": labels.get("1", 0),
        "incorrect_count": labels.get("0", 0),
        "correct_rate": labels.get("1", 0) / rows if rows else None,
        "columns": columns,
        "privacy_note": "Output contains only aggregates. Source rows and student identifiers remain under private_data/ and are never versioned.",
    }
    for destination in (PRIVATE_OUT, PUBLIC_OUT):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
