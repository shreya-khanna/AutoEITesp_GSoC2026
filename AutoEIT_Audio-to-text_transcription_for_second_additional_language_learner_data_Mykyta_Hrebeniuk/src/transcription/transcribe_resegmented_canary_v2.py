"""
Transcribe every clip in data/segmented/v2/ (the windowed, 4s-merge-gap resegmentation from
src/preprocessing/resegment_window.py) with NVIDIA's Canary-1B-v2, reusing
transcribe_canary_v2.py's load_model()/transcribe_file(). Run on every recording (hit and miss),
same convention as transcribe_resegmented_canary.py for v1 -- not restricted to hit recordings.

Output goes to data/transcribed/resegmented/canary_v2/{version}/{recording}/item_{N}.json, where N
is the fresh per-recording index from data/segmented/v2/ (not the original global item index used
by data/transcribed/v1 and v2).

Usage:
    python -m src.transcription.transcribe_resegmented_canary_v2
"""
import json
import logging

from src.transcription.resegmented_items import VERSIONS, iter_resegmented_items
from src.transcription.transcribe_canary_v2 import load_model, transcribe_file
from src.utils.paths import data_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(data_path("logs", "transcribe_resegmented_canary_v2.log")),
        logging.StreamHandler(),
    ],
)


def main() -> None:
    model = load_model()
    for version in VERSIONS:
        items = iter_resegmented_items(version)
        logging.info(f"{version}: {len(items)} item(s) to process")
        for recording, index, audio_path in items:
            output_path = data_path("transcribed", "resegmented", "canary_v2", version, recording, f"item_{index}.json")
            if output_path.exists():
                logging.info(f"Skipping {audio_path.name} (already transcribed)")
                continue
            try:
                result = transcribe_file(audio_path, model)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                logging.info(f"Saved {output_path}")
            except Exception as e:
                logging.error(f"{recording}/item_{index}: failed to transcribe: {e}")


if __name__ == "__main__":
    main()
