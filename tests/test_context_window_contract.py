"""Contract test for the aggregate-safe DKT history-window implementation.

The sandbox used for source verification intentionally does not include the pinned
PyTorch runtime. This test therefore extracts and executes the pure chunking helper
from the actual experiment source, while statically confirming the evaluation path
uses the same helper. It creates no experimental artifact and no educational data.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable


def load_chunker(source_path: Path):
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    node = next(
        candidate for candidate in tree.body
        if isinstance(candidate, ast.FunctionDef) and candidate.name == "iter_sequence_chunks"
    )
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"Iterable": Iterable}
    exec(compile(module, filename=str(source_path), mode="exec"), namespace)
    return namespace["iter_sequence_chunks"], tree


def main() -> None:
    source = Path(__file__).resolve().parents[1] / "experiments" / "run_student_disjoint_kt.py"
    iter_sequence_chunks, tree = load_chunker(source)
    sequence = [(0, 1), (1, 0), (2, 1), (0, 0), (1, 1), (2, 0), (0, 1)]
    chunks = list(iter_sequence_chunks(sequence, max_length=2))
    target_count = sum(len(chunk) - 1 for chunk in chunks)
    assert target_count == len(sequence) - 1, "Chunking must cover each target exactly once."
    assert chunks == [sequence[:3], sequence[2:5], sequence[4:]], "Chunk boundaries must retain the previous event as context."

    evaluate = next(
        candidate for candidate in tree.body
        if isinstance(candidate, ast.FunctionDef) and candidate.name == "evaluate_dkt"
    )
    evaluation_source = ast.unparse(evaluate)
    assert "iter_sequence_chunks(sequence, max_length)" in evaluation_source
    assert "(sequence,) if max_length is None" in evaluation_source
    assert "np.concatenate(student_targets)" in evaluation_source
    print("PASS context-window target-coverage contract")


if __name__ == "__main__":
    main()
