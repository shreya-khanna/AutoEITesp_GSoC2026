"""
Build a single flat Excel sheet with exactly 4 columns -- stimuli, rater1 transc., rater2 transc.,
canary's transcriptions -- for review/reporting, sourced from the
data/postprocessed/resegmented/{version}/{recording}/transcriptions.csv files that
build_postprocessed_csv_resegmented.py regenerates once "canary" is in its ASR_SOURCES.

Scoped to hit recordings only (data/resegmented_hit_recordings.txt): canary was transcribed for
every recording, hit and miss (see transcribe_resegmented_canary.py), but the rank-based
item<->Sentence alignment this sheet relies on is only trustworthy for hits, same as every other
reviewable transcript sheet in this codebase. WER/MER/CER are not in this sheet by design (it's
meant to stay a plain 4-column reviewable transcript) -- they're already computed as
canary_wer/mer/cer_{rater1,rater2} columns in the source CSVs, which
notebooks/model_comparison/analyze_canary_transcription_quality.ipynb reads directly.

Usage:
    python -m src.postprocessing.build_canary_excel
"""
import csv
import logging

import openpyxl

from src.transcription.resegmented_items import VERSIONS
from src.utils.paths import data_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

COLUMNS = ["stimuli", "rater1 transc.", "rater2 transc.", "canary's transcriptions"]


def main() -> None:
    hit_list_path = data_path("resegmented_hit_recordings.txt")
    hit_recordings = sorted(hit_list_path.read_text(encoding="utf-8").split())
    log.info("Loaded %d hit recordings", len(hit_recordings))

    postprocessed_root = data_path("postprocessed", "resegmented")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "canary"
    ws.append(COLUMNS)

    n_rows = 0
    n_recordings_found = 0
    for version in VERSIONS:
        for recording in hit_recordings:
            csv_path = postprocessed_root / version / recording / "transcriptions.csv"
            if not csv_path.exists():
                continue
            n_recordings_found += 1
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    ws.append([
                        row.get("stimulus"),
                        row.get("human_rater1_text"),
                        row.get("human_rater2_text"),
                        row.get("canary_text"),
                    ])
                    n_rows += 1

    out_dir = data_path("postprocessed", "canary")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "canary_transcriptions.xlsx"
    wb.save(out_path)
    log.info(
        "Wrote %d row(s) from %d hit recording(s) to %s",
        n_rows, n_recordings_found, out_path,
    )


if __name__ == "__main__":
    main()
