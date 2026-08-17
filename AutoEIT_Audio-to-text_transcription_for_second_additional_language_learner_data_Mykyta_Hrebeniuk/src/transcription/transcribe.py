from faster_whisper import WhisperModel
from src.utils.paths import data_path
from pathlib import Path
import json
import logging
import re
import torch


def load_model() -> WhisperModel:
    if torch.cuda.is_available():
        device, compute_type = "cuda", "float16"
    else:
        device, compute_type = "cpu", "int8"
    logging.info(f"Loading faster-whisper large-v3 on {device}/{compute_type}")
    return WhisperModel("large-v3", device=device, compute_type=compute_type)


def resolve_output_path(audio_path: Path) -> Path:
    for part in audio_path.parts:
        if part in ("version_a", "version_b"):
            version = part
            break
    else:
        raise ValueError(f"Cannot determine version folder from path: {audio_path}")

    stem = re.sub(r"^ATTENTION_", "", audio_path.stem)
    m = re.match(r"^(.+)_item(\d+)$", stem)
    if not m:
        raise ValueError(f"Filename does not match expected pattern '<name>_item<N>': {audio_path.name}")
    recording, item_index = m.group(1), m.group(2)

    return data_path("transcribed", "v1", version, recording, f"item_{item_index}.json")


def transcribe_file(audio_path: Path, model: WhisperModel, language: str | None = None) -> dict:
    segments_gen, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=1,
        temperature=0,
        condition_on_previous_text=False,
        word_timestamps=True,
    )

    segments = []
    for seg in segments_gen:
        words = []
        if seg.words:
            for w in seg.words:
                words.append({
                    "word": w.word,
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "prob": round(w.probability, 4),
                })
        segments.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text,
            "words": words,
        })

    return {
        "file": audio_path.name,
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration": round(info.duration, 3),
        "segments": segments,
    }


def run_single(audio_path: Path, model: WhisperModel, language: str | None = None) -> dict:
    output_path = resolve_output_path(audio_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Transcribing {audio_path.name}")
    result = transcribe_file(audio_path, model, language=language)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logging.info(f"Saved to {output_path}")
    return result


def run_for_recording(stem: str, folder: str, model: WhisperModel, language: str | None = None):
    segments = sorted(data_path("segmented", "v1", folder).glob(f"*{stem}_item*.mp3"))
    if not segments:
        raise FileNotFoundError(f"No segments found for '{stem}' in {folder}")
    logging.info(f"Found {len(segments)} segments for {stem}")
    for audio_path in segments:
        try:
            output_path = resolve_output_path(audio_path)
        except ValueError as e:
            logging.warning(f"Skipping {audio_path.name}: {e}")
            continue
        if output_path.exists():
            logging.info(f"Skipping {audio_path.name} (already transcribed)")
            continue
        try:
            run_single(audio_path, model, language=language)
            logging.info(f"Done: {audio_path.name}")
        except Exception as e:
            logging.error(f"Failed on {audio_path.name}: {e}")


def run_all(model: WhisperModel, language: str | None = None):
    segmented_root = data_path("segmented", "v1")
    for version_folder in ["version_a", "version_b"]:
        folder = segmented_root / version_folder
        if not folder.exists():
            logging.warning(f"Folder not found: {folder}")
            continue
        recordings = sorted({
            re.sub(r"^ATTENTION_", "", p.stem).rsplit("_item", 1)[0]
            for p in folder.glob("*.mp3")
        })
        logging.info(f"Found {len(recordings)} recordings in {version_folder}")
        for stem in recordings:
            run_for_recording(stem, version_folder, model, language=language)
