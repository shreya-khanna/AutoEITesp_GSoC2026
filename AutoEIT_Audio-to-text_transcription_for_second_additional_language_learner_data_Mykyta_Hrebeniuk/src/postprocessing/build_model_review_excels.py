"""
Break each resegmented transcriptions.csv (see build_postprocessed_csv_resegmented.py) into a
per-recording, per-model directory tree for isolated side-by-side review: stimulus + both human
raters + one ASR model's transcription per sheet, separated from the other 11 models and the
WER/MER/CER columns that live in the source CSV.

Output layout:
    data/model_review/<version>/<recording>/<model>/transcriptions.xlsx

Skips a (recording, model) pair entirely when the model has no text for that recording, rather than
writing an all-blank sheet -- crisperwhisper/mms/seamlessm4t/assemblyai/assemblyai_v3/speechmatics
were only transcribed for the 102 hit recordings (see build_postprocessed_csv_resegmented.py's
docstring), so they're blank for the other ~82 miss recordings in each version.

Usage:
    python -m src.postprocessing.build_model_review_excels
"""
import csv
import logging
from pathlib import Path

import openpyxl

from src.postprocessing.build_postprocessed_csv_resegmented import ASR_SOURCES
from src.utils.paths import data_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

VERSIONS = ["version_a", "version_b"]


def write_model_sheet(rows: list[dict], model: str, out_path: Path) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = model[:31]  # openpyxl sheet-title limit
    ws.append(["stimuli", "rater1 transc.", "rater2 transc.", f"{model} transc."])
    for row in rows:
        ws.append([
            row.get("stimulus"),
            row.get("human_rater1_text"),
            row.get("human_rater2_text"),
            row.get(f"{model}_text"),
        ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)


def main() -> None:
    postprocessed_root = data_path("postprocessed", "resegmented")
    out_root = data_path("model_review")

    n_written = 0
    n_skipped = 0
    for version in VERSIONS:
        version_dir = postprocessed_root / version
        if not version_dir.exists():
            log.warning("Folder not found: %s", version_dir)
            continue
        for recording_dir in sorted(version_dir.iterdir()):
            csv_path = recording_dir / "transcriptions.csv"
            if not csv_path.exists():
                continue
            with open(csv_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            for model in ASR_SOURCES:
                col = f"{model}_text"
                if not any(row.get(col) for row in rows):
                    n_skipped += 1
                    continue
                out_path = out_root / version / recording_dir.name / model / "transcriptions.xlsx"
                write_model_sheet(rows, model, out_path)
                n_written += 1

    log.info(
        "Wrote %d model-review sheet(s), skipped %d empty (recording, model) pair(s), under %s",
        n_written, n_skipped, out_root,
    )


if __name__ == "__main__":
    main()
