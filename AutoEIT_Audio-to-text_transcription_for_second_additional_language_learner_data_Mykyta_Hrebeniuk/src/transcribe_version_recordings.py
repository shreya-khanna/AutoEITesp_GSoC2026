import json
import logging

from src.transcription.transcribe import load_model, transcribe_file
from src.utils.paths import data_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    audio_path = data_path("version_a_recording.wav")
    output_path = data_path("version_a_recording_transcription.json")

    model = load_model()
    logging.info(f"Transcribing {audio_path.name}")
    result = transcribe_file(audio_path, model)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logging.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
