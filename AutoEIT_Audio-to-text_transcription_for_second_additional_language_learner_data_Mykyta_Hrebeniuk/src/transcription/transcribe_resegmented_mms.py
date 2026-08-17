"""
Transcribe data/segmented/v2/ clips with Facebook MMS (facebook/mms-1b-all, see
transcribe_mms.py), restricted to recordings in data/resegmented_hit_recordings.txt (the 102
recordings with exactly 30 resegmented items) -- same restriction as
transcribe_resegmented_crisperwhisper.py, for the same reason: MMS is only being evaluated
against the existing hit-recording WER/MER/CER comparison, so misses are skipped outright rather
than transcribed and left unscored.

Output goes to data/transcribed/resegmented/mms/{version}/{recording}/item_{N}.json, where N is
the fresh per-recording index from data/segmented/v2/ (same convention as the other resegmented
transcribe_*.py scripts).

Usage:
    python -m src.transcription.transcribe_resegmented_mms
"""
import json
import logging

from src.transcription.resegmented_items import VERSIONS, iter_resegmented_items
from src.transcription.transcribe_mms import load_model, transcribe_file
from src.utils.paths import data_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(data_path("logs", "transcribe_resegmented_mms.log")),
        logging.StreamHandler(),
    ],
)


def main() -> None:
    hit_list_path = data_path("resegmented_hit_recordings.txt")
    hit_recordings = set(hit_list_path.read_text(encoding="utf-8").split())
    logging.info(f"Loaded {len(hit_recordings)} hit recordings")

    processor, model, device = load_model()
    for version in VERSIONS:
        items = [it for it in iter_resegmented_items(version) if it[0] in hit_recordings]
        logging.info(f"{version}: {len(items)} item(s) to process across hit recordings")
        for recording, index, audio_path in items:
            output_path = data_path("transcribed", "resegmented", "mms", version, recording, f"item_{index}.json")
            if output_path.exists():
                logging.info(f"Skipping {audio_path.name} (already transcribed)")
                continue
            try:
                result = transcribe_file(audio_path, processor, model, device)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                logging.info(f"Saved {output_path}")
            except Exception as e:
                logging.error(f"{recording}/item_{index}: failed to transcribe: {e}")


if __name__ == "__main__":
    main()
