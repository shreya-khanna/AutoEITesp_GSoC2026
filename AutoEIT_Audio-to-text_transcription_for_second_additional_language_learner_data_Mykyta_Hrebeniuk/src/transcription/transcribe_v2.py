"""
Re-transcribe the Spanish narrative section of each recording — the items strictly after
the last English target sentence (the volleyball/gym item) up to and including the last
Spanish target sentence (the desayuno/nieva item) — with Whisper's language forced to "es".

Reads `last_english_item`, `last_spanish_sentence`, and `category` from each recording's
data/transcribed/v1/{version}/{recording}/summary.json, written by
notebooks/segmentation/analyze_volleyball_sentence.ipynb and notebooks/segmentation/analyze_breakfast_snow_sentence.ipynb.

Recordings categorized as "short anomaly" are skipped outright. For the remaining recordings,
last_english_item / last_spanish_sentence (each a list of item indices) are resolved to a single
boundary index: if there's one index, use it; if there are several, they must be within +-1 of
each other, in which case the max is used. If a list is empty, has indices that aren't within
+-1, or the resolved English boundary isn't before the Spanish boundary, the recording is skipped
and logged to data/logs/transcribe_v2_review.log for manual review.

Output is written to data/transcribed/v2/{version}/{recording}/item_{N}.json, in the same format
as the original data/transcribed/v1/ item files.
"""

import json
import logging

from src.transcription.transcribe import load_model, transcribe_file
from src.utils.paths import data_path

VERSIONS = ["version_a", "version_b"]
LANGUAGE = "es"
REVIEW_LOG_PATH = data_path("logs", "transcribe_v2_review.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(data_path("logs", "transcribe_v2.log")),
        logging.StreamHandler(),
    ],
)


def resolve_boundary_index(indices: list[int]) -> int | None:
    if not indices:
        return None
    if len(indices) == 1:
        return indices[0]
    if max(indices) - min(indices) <= 1:
        return max(indices)
    return None


def find_segment_path(recording: str, version: str, index: int):
    folder = data_path("segmented", "v1", version)
    matches = sorted(folder.glob(f"*{recording}_item{index}.mp3"))
    return matches[0] if matches else None


def log_for_review(recording: str, reason: str) -> None:
    logging.warning(f"Skipping {recording} for manual review: {reason}")
    with open(REVIEW_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{recording}\t{reason}\n")


def process_recording(summary_path, version: str, model) -> None:
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    recording = summary["recording"]

    if summary.get("category") == "short anomaly":
        logging.info(f"Skipping {recording} (short anomaly)")
        return

    eng_idx = resolve_boundary_index(summary.get("last_english_item", []))
    esp_idx = resolve_boundary_index(summary.get("last_spanish_sentence", []))

    if eng_idx is None:
        log_for_review(recording, "last_english_item missing or indices not within +-1")
        return
    if esp_idx is None:
        log_for_review(recording, "last_spanish_sentence missing or indices not within +-1")
        return
    if eng_idx >= esp_idx:
        log_for_review(recording, f"last_english_item ({eng_idx}) is not before last_spanish_sentence ({esp_idx})")
        return

    item_range = range(eng_idx + 1, esp_idx + 1)
    logging.info(f"{recording}: transcribing items {eng_idx + 1}..{esp_idx} ({len(item_range)} item(s))")

    for index in item_range:
        audio_path = find_segment_path(recording, version, index)
        if audio_path is None:
            logging.error(f"{recording}: no segment file found for item {index}, skipping item")
            continue

        output_path = data_path("transcribed", "v2", version, recording, f"item_{index}.json")
        if output_path.exists():
            logging.info(f"Skipping {audio_path.name} (already transcribed)")
            continue

        try:
            result = transcribe_file(audio_path, model, language=LANGUAGE)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logging.info(f"Saved {output_path}")
        except Exception as e:
            logging.error(f"{recording}: failed to transcribe item {index}: {e}")


def main() -> None:
    REVIEW_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_LOG_PATH.write_text("")

    model = load_model()
    for version in VERSIONS:
        version_dir = data_path("transcribed", "v1", version)
        if not version_dir.exists():
            logging.warning(f"Folder not found: {version_dir}")
            continue
        for summary_path in sorted(version_dir.glob("*/summary.json")):
            process_recording(summary_path, version, model)


if __name__ == "__main__":
    main()
