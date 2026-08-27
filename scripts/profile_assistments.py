"""Profile the locally controlled ASSISTments2009 corrected CSV without exporting learner records.

The script emits only aggregate metadata, column names, a source checksum, and bounded
frequency summaries. Raw rows, student IDs, and per-student sequences remain below
private_data/ and must never be copied to a repository.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[2] if (_HERE.parents[2] / "research_code").exists() else _HERE.parents[1]
RAW = ROOT / "private_data" / "raw" / "skill_builder_data_corrected_collapsed.csv"
PRIVATE_OUT = ROOT / "private_data" / "metadata" / "raw_profile.json"
PUBLIC_OUT = (ROOT / "research_code" / "metadata" if (ROOT / "research_code").exists() else ROOT / "metadata") / "assistments2009_data_card.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not RAW.exists():
        raise FileNotFoundError(f"Expected locally controlled dataset: {RAW}")

    users: set[str] = set()
    problems: set[str] = set()
    skills: set[str] = set()
    correctness = Counter()
    tutor_modes = Counter()
    answer_types = Counter()
    multi_skill_rows = 0
    rows = 0

    # The official CSV includes legacy byte sequences in free-text fields. The
    # experiment only consumes explicit numeric/identifier columns; replacement
    # decoding keeps the row stream and those fields intact while avoiding an
    # undocumented re-encoding of the raw controlled file.
    with RAW.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {"user_id", "problem_id", "skill_id", "correct", "order_id"}
        missing = sorted(required.difference(fieldnames))
        if missing:
            raise ValueError(f"Dataset schema changed; missing required columns: {missing}")

        for row in reader:
            rows += 1
            users.add(row["user_id"])
            problems.add(row["problem_id"])
            skill = row["skill_id"].strip()
            if skill:
                skills.add(skill)
                multi_skill_rows += int("_" in skill)
            correctness[row["correct"].strip()] += 1
            tutor_modes[row["tutor_mode"].strip()] += 1
            answer_types[row["answer_type"].strip()] += 1

    correct = correctness.get("1", 0)
    profile = {
        "dataset": "ASSISTments2009 Skill Builder corrected collapsed",
        "source_url": "https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010",
        "download_file": RAW.name,
        "sha256": sha256(RAW),
        "row_count_excluding_header": rows,
        "unique_students": len(users),
        "unique_problems": len(problems),
        "unique_collapsed_skill_labels": len(skills),
        "multi_skill_row_count": multi_skill_rows,
        "correct_count": correct,
        "incorrect_count": rows - correct,
        "correct_rate": correct / rows if rows else None,
        "columns": fieldnames,
        "tutor_mode_counts": dict(tutor_modes.most_common()),
        "answer_type_counts": dict(answer_types.most_common()),
        "privacy_note": (
            "Only aggregate statistics are stored here. Raw student records, IDs, splits, "
            "and per-student predictions remain locally controlled under private_data/."
        ),
    }

    for output in (PRIVATE_OUT, PUBLIC_OUT):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(profile, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
