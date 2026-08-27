"""Audit whether a student-level dataset has enough repeated observations for identity controls."""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv_path')
    ap.add_argument('--student-column', default='user_id')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.csv_path, usecols=[args.student_column])
    counts = df.groupby(args.student_column, dropna=False).size()
    out = {
        'rows': int(len(df)),
        'students': int(counts.size),
        'students_with_at_least_2_rows': int((counts >= 2).sum()),
        'students_with_at_least_5_rows': int((counts >= 5).sum()),
        'students_with_at_least_10_rows': int((counts >= 10).sum()),
        'repeat_row_count_at_least_5': int(counts[counts >= 5].sum()),
        'max_rows_per_student': int(counts.max()),
        'median_rows_per_student': float(counts.median()),
        'mean_rows_per_student': float(counts.mean()),
        'student_column': args.student_column,
        'source_note': 'Aggregated counts only; no student identifiers are written.',
    }
    Path(args.output).write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
