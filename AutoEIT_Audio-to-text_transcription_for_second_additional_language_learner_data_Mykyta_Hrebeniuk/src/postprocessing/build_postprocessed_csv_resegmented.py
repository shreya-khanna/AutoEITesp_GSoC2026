"""
Merge human transcriptions (Excel) with the resegmented (data/segmented/v2/, windowed + 4s
merge-gap) ASR outputs --
data/transcribed/resegmented/{whisper,parakeet,wav2vec2,crisperwhisper,fastconformer,canary,mms,seamlessm4t,canary_v2,assemblyai,assemblyai_v3,speechmatics}
-- into one CSV per recording, under data/postprocessed/resegmented/version_a|b/<recording>/.

crisperwhisper, mms, seamlessm4t, assemblyai, assemblyai_v3, and speechmatics (see
src/transcription/transcribe_resegmented_{crisperwhisper,mms,seamlessm4t,assemblyai,assemblyai_v3,speechmatics}.py)
were only transcribed for hit recordings, so their columns are populated for hits and simply blank
for misses (load_asr_items returns [] when no summary.json exists for a recording) -- this is
consistent with score_quality already being hit-only for every source below. fastconformer,
canary, and canary_v2, like whisper/parakeet/wav2vec2, were transcribed for every recording (hit
and miss). assemblyai_v3 is Universal-3.5 Pro + Speech-to-Text Prompting, a separate config
from assemblyai's production-pinned Universal-2 run (see transcribe_assemblyai_v3.py) -- kept as
its own source column rather than replacing assemblyai so both are comparable side by side.

Same human-sheet matching and rank-based item<->Sentence alignment as
src/postprocessing/build_postprocessed_csv.py (see that module's docstring for the rationale), but
scoped to the resegmented item set and its own hit list: data/resegmented_hit_recordings.txt (the
102 recordings with exactly 30 resegmented items), not data/transcribed_v2_hit_recordings.txt,
which is the old segmentation's hit list. No v2_filtered analog exists here.

Usage:
    python -m src.postprocessing.build_postprocessed_csv_resegmented
"""
import csv
import logging

from src.postprocessing.build_postprocessed_csv import (
    HUMAN_RATERS,
    build_rows,
    load_asr_items,
    load_human_sheets,
    parse_folder_name,
)
from src.transcription.resegmented_items import VERSIONS, iter_resegmented_items
from src.utils.paths import data_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ASR_SOURCES = {
    "whisper": ("transcribed/resegmented/whisper", "summary.json"),
    "parakeet": ("transcribed/resegmented/parakeet", "summary.json"),
    "wav2vec2": ("transcribed/resegmented/wav2vec2", "summary.json"),
    "crisperwhisper": ("transcribed/resegmented/crisperwhisper", "summary.json"),
    "fastconformer": ("transcribed/resegmented/fastconformer", "summary.json"),
    "canary": ("transcribed/resegmented/canary", "summary.json"),
    "mms": ("transcribed/resegmented/mms", "summary.json"),
    "seamlessm4t": ("transcribed/resegmented/seamlessm4t", "summary.json"),
    "canary_v2": ("transcribed/resegmented/canary_v2", "summary.json"),
    "assemblyai": ("transcribed/resegmented/assemblyai", "summary.json"),
    "assemblyai_v3": ("transcribed/resegmented/assemblyai_v3", "summary.json"),
    "speechmatics": ("transcribed/resegmented/speechmatics", "summary.json"),
}

BASE_CSV_FIELDS = [
    "sentence_number",
    "stimulus",
    "human_rater1_text",
    "human_rater2_text",
    "human_wer",
    "human_mer",
    "human_cer",
    *[f"{s}_text" for s in ASR_SOURCES],
    *[f"{s}_{c}" for s in ASR_SOURCES for c in ("duration", "word_count", "mean_word_prob", "empty")],
]

QUALITY_CSV_FIELDS = [
    f"{s}_{m}_{rater}" for s in ASR_SOURCES for m in ("wer", "mer", "cer") for rater in HUMAN_RATERS
]


def csv_fields(score_quality: bool) -> list[str]:
    return BASE_CSV_FIELDS + QUALITY_CSV_FIELDS if score_quality else BASE_CSV_FIELDS


def recordings_by_version() -> dict[str, list[str]]:
    result = {}
    for version in VERSIONS:
        items = iter_resegmented_items(version)
        seen = []
        for recording, _, _ in items:
            if recording not in seen:
                seen.append(recording)
        result[version] = seen
    return result


def main():
    human_dir = data_path("human-transcriptions")
    human_sheets = load_human_sheets(human_dir)
    log.info("Loaded %d human-transcription sheets", len(human_sheets))

    hit_list_path = data_path("resegmented_hit_recordings.txt")
    hit_recordings = set(hit_list_path.read_text(encoding="utf-8").split())
    log.info("Loaded %d hit recordings for WER/MER/CER scoring", len(hit_recordings))

    out_root = data_path("postprocessed", "resegmented")

    n_recordings = 0
    n_matched_human = 0
    n_scored = 0

    for version, recordings in recordings_by_version().items():
        for recording_name in recordings:
            asr_by_source = {}
            for source, (folder, summary_filename) in ASR_SOURCES.items():
                src_dir = data_path(folder, version, recording_name)
                asr_by_source[source] = load_asr_items(src_dir, summary_filename)

            key = parse_folder_name(recording_name)
            human_rows = human_sheets.get(key, []) if key else []
            if key and key not in human_sheets:
                log.warning("No human transcription sheet found for %s (key=%s)", recording_name, key)
            elif human_rows:
                n_matched_human += 1

            score_quality = recording_name in hit_recordings
            rows = build_rows(human_rows, asr_by_source, score_quality)
            if score_quality:
                n_scored += 1

            out_dir = out_root / version / recording_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "transcriptions.csv"
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fields(score_quality))
                writer.writeheader()
                writer.writerows(rows)

            n_recordings += 1

    log.info(
        "Wrote %d recording CSVs (%d matched a human sheet, %d scored with WER/MER/CER) under %s",
        n_recordings,
        n_matched_human,
        n_scored,
        out_root,
    )


if __name__ == "__main__":
    main()
