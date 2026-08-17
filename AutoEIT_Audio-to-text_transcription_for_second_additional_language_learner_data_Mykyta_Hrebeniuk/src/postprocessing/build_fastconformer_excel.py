"""
Build the fastconformer reviewable Excel workbook, sourced from the
data/postprocessed/resegmented/{version}/{recording}/transcriptions.csv files that
build_postprocessed_csv_resegmented.py regenerates once "fastconformer" is in its ASR_SOURCES.

Scoped to hit recordings only (data/resegmented_hit_recordings.txt): fastconformer was transcribed
for every recording, hit and miss (see transcribe_resegmented_fastconformer.py), but the
rank-based item<->Sentence alignment this sheet relies on is only trustworthy for hits, same as
every other reviewable transcript sheet in this codebase.

Two sheets:

- **"fastconformer"** -- plain 4-column (stimuli / rater1 / rater2 / fastconformer) reviewable
  transcript, raw human-rater text as typed. WER/MER/CER are not in this sheet by design -- they're
  already computed as fastconformer_wer/mer/cer_{rater1,rater2} columns in the source CSVs, which
  notebooks/model_comparison/analyze_fastconformer_transcription_quality.ipynb reads directly.
- **"fastconformer_cleaned"** -- same 4 columns, but with non-lexical annotation tags stripped out
  of the rater transcriptions before display and scoring: `[x]`/`[xx]`/`[xxx]` (unclear speech),
  `[...]`/`[..]` (pause), `[pause]`/`[long pause]`, `[gibberish]`, `[laugh]`, `[no se]`/`[no sé]`,
  `[don't know]`, `[?]`, `[something]`. These never denote real spoken words -- they're the
  transcriber's own annotation for speech they couldn't parse -- so scoring against them (the
  current pipeline-wide behavior: brackets are stripped but the tag text like "xx" or "gibberish"
  is kept as a literal reference word) inflates the reference with tokens no ASR system could ever
  match. False-start fragments (e.g. `[el-]` before its repair) are deliberately left untouched --
  those are real attempted speech, not noise, and removing them shortens the reference in a way
  that isn't clearly more "correct." fastconformer_wer/mer/cer are recomputed here against the
  cleaned reference, scoped to this sheet only -- this does not touch
  data/postprocessed/resegmented/ or any other model's numbers.

Usage:
    python -m src.postprocessing.build_fastconformer_excel
"""
import csv
import logging
import re

import openpyxl

from src.postprocessing.build_postprocessed_csv import compute_wer_mer_cer
from src.transcription.resegmented_items import VERSIONS
from src.utils.paths import data_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

COLUMNS = ["stimuli", "rater1 transc.", "rater2 transc.", "fastconformer's transcriptions"]
CLEANED_COLUMNS = COLUMNS + [
    "fastconformer_wer_rater1", "fastconformer_wer_rater2",
    "fastconformer_mer_rater1", "fastconformer_mer_rater2",
    "fastconformer_cer_rater1", "fastconformer_cer_rater2",
]

BRACKET_RE = re.compile(r"\[([^\]]*)\]")
NOISE_LITERALS = {
    "gibberish", "pause", "long pause", "laugh", "laughs", "laughing",
    "no se", "no sé", "don't know", "something", "?",
}


def is_noise_tag(inner: str) -> bool:
    """True for bracket contents that are transcriber annotations, not attempted words --
    unclear speech, pauses, gibberish, laughter, "don't know"/"?" -- as opposed to false-start
    fragments (kept, they're real attempted speech) or other guessed real words (also kept)."""
    s = inner.strip().lower()
    if s == "":
        return True
    if re.fullmatch(r"x+", s):
        return True
    if re.fullmatch(r"\.+", s):
        return True
    return s in NOISE_LITERALS


def strip_noise_tags(text) -> str:
    if not text:
        return text
    def repl(m):
        return "" if is_noise_tag(m.group(1)) else m.group(0)
    cleaned = BRACKET_RE.sub(repl, str(text))
    return re.sub(r"\s+", " ", cleaned).strip()


def has_noise_tag(text) -> bool:
    if not text:
        return False
    return any(is_noise_tag(m) for m in BRACKET_RE.findall(str(text)))


def main() -> None:
    hit_list_path = data_path("resegmented_hit_recordings.txt")
    hit_recordings = sorted(hit_list_path.read_text(encoding="utf-8").split())
    log.info("Loaded %d hit recordings", len(hit_recordings))

    postprocessed_root = data_path("postprocessed", "resegmented")

    wb = openpyxl.Workbook()
    ws_raw = wb.active
    ws_raw.title = "fastconformer"
    ws_raw.append(COLUMNS)

    ws_clean = wb.create_sheet("fastconformer_cleaned")
    ws_clean.append(CLEANED_COLUMNS)

    n_rows = 0
    n_recordings_found = 0
    n_refs_changed = 0
    for version in VERSIONS:
        for recording in hit_recordings:
            csv_path = postprocessed_root / version / recording / "transcriptions.csv"
            if not csv_path.exists():
                continue
            n_recordings_found += 1
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rater1_raw = row.get("human_rater1_text")
                    rater2_raw = row.get("human_rater2_text")
                    fastconformer_text = row.get("fastconformer_text")

                    ws_raw.append([row.get("stimulus"), rater1_raw, rater2_raw, fastconformer_text])
                    n_rows += 1

                    rater1_clean = strip_noise_tags(rater1_raw)
                    rater2_clean = strip_noise_tags(rater2_raw)
                    n_refs_changed += has_noise_tag(rater1_raw) + has_noise_tag(rater2_raw)

                    wer1, mer1, cer1 = compute_wer_mer_cer(rater1_clean, fastconformer_text)
                    wer2, mer2, cer2 = compute_wer_mer_cer(rater2_clean, fastconformer_text)
                    ws_clean.append([
                        row.get("stimulus"), rater1_clean, rater2_clean, fastconformer_text,
                        wer1, wer2, mer1, mer2, cer1, cer2,
                    ])

    out_dir = data_path("postprocessed", "fastconformer")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fastconformer_transcriptions.xlsx"
    wb.save(out_path)
    log.info(
        "Wrote %d row(s) from %d hit recording(s) to %s (%d rater transcriptions had noise tags stripped)",
        n_rows, n_recordings_found, out_path, n_refs_changed,
    )


if __name__ == "__main__":
    main()
