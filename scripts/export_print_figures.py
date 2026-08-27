#!/usr/bin/env python3
"""Export aggregate-only figure PDFs to print-ready 600-dpi TIFF derivatives.

The script never reads controlled data. It rasterizes only PDFs already tracked in the
public reproducibility layer and writes a JSON technical manifest with file hashes and
pixel/DPI metadata. The vector PDFs remain the preferred source for typesetting.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures" / "revision3"
TIFF_DIR = FIGURE_DIR / "tiff_600dpi"
MANIFEST = FIGURE_DIR / "revision7_figure_technical_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def figure_notes(stem: str) -> dict[str, str]:
    notes = {
        "fig1-student-disjoint-baselines": {
            "axes": "x-axis: method; y-axis: ROC-AUC",
            "uncertainty": "student-cluster 95% intervals for single-run bars; open circles show all DKT seed AUC point estimates",
        },
        "fig2-clean-ablation": {
            "axes": "x-axis: ROC-AUC; y-axis: DKT configuration",
            "uncertainty": "three-seed descriptive dispersion; not a confidence interval",
        },
        "fig3-label-noise-sensitivity": {
            "axes": "x-axis: training-label inversion rate (%); y-axis: ROC-AUC",
            "uncertainty": "capped error bars are across-seed SDs",
        },
        "fig4-probability-quality": {
            "axes": "left: summary metric; right: mean predicted probability versus empirical accuracy",
            "uncertainty": "calibration curve is descriptive; no inferential interval is implied",
        },
        "fig5-training-window-sensitivity": {
            "axes": "x-axis: maximum training window; y-axis: ROC-AUC",
            "uncertainty": "capped error bars are across-seed SDs",
        },
        "figA1-exploratory-extended-budget": {
            "axes": "x-axis: epoch; y-axis: validation ROC-AUC",
            "uncertainty": "separate seed trajectories; post hoc exploratory diagnostic",
        },
        "figA2_bkt_fit_diagnostics": {
            "axes": "left: BKT parameter summaries; right: fixed-iteration stability-proxy proportion",
            "uncertainty": "aggregate cross-skill quantiles and counts only; no skill identifiers are displayed",
        },
        "figA2-bkt-fit-diagnostics": {
            "axes": "left: BKT parameter summaries; right: fixed-iteration stability-proxy proportion",
            "uncertainty": "aggregate cross-skill quantiles and counts only; no skill identifiers are displayed",
        },
    }
    return notes.get(stem, {"axes": "See figure caption.", "uncertainty": "See figure caption."})


def main() -> None:
    if shutil.which("pdftoppm") is None:
        raise SystemExit("pdftoppm is required to render vector PDFs at 600 dpi")
    TIFF_DIR.mkdir(parents=True, exist_ok=True)
    expected_tiffs = {f"{pdf.stem}.tif" for pdf in FIGURE_DIR.glob("*.pdf")}
    for stale in TIFF_DIR.glob("*.tif"):
        if stale.name not in expected_tiffs:
            stale.unlink()
    records: list[dict[str, object]] = []
    for pdf in sorted(FIGURE_DIR.glob("*.pdf")):
        output_stem = TIFF_DIR / pdf.stem
        subprocess.run(
            ["pdftoppm", "-r", "600", "-tiff", "-singlefile", str(pdf), str(output_stem)],
            check=True,
            capture_output=True,
            text=True,
        )
        tiff = output_stem.with_suffix(".tif")
        with Image.open(tiff) as image:
            width, height = image.size
            dpi = image.info.get("dpi", (600, 600))
            mode = image.mode
        notes = figure_notes(pdf.stem)
        records.append(
            {
                "figure_id": pdf.stem,
                "vector_pdf": str(pdf.relative_to(ROOT)),
                "print_tiff_600dpi": str(tiff.relative_to(ROOT)),
                "pdf_sha256": sha256(pdf),
                "tiff_sha256": sha256(tiff),
                "tiff_pixel_dimensions": [width, height],
                "requested_rasterization_dpi": 600,
                "embedded_tiff_dpi": [round(float(dpi[0])), round(float(dpi[1]))],
                "color_mode": mode,
                "axis_label_check": notes["axes"],
                "uncertainty_or_legend_check": notes["uncertainty"],
                "privacy_scope": "Aggregate-only figure; no student-level record, prediction, split or checkpoint is embedded.",
            }
        )
    MANIFEST.write_text(
        json.dumps(
            {
                "revision": "Revision 7 technical-editor package",
                "preferred_typesetting_format": "PDF vector",
                "raster_submission_derivative": "TIFF rendered at 600 dpi from the tracked vector PDF",
                "figure_count": len(records),
                "figures": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(MANIFEST), "figure_count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
