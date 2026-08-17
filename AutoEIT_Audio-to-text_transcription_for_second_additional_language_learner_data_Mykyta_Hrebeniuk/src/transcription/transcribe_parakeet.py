"""
Re-transcribe the same items as data/transcribed/v2/ with NVIDIA's Parakeet-TDT-0.6B-v3
(nemo_toolkit[asr]) -- an open-source, verbatim-transcription alternative to Whisper: it
retains disfluencies/false starts/repairs rather than cleaning them up, and (per informal
reports) runs acceptably on CPU despite NVIDIA's docs describing it as GPU-oriented.
See docs/asr_model_alternatives.md.

The item set to transcribe is read directly off data/transcribed/v2/{version}/{recording}/item_{N}.json
file names, so this stays in lockstep with whatever transcribe_v2.py has already scoped down to
(the Spanish-narrative-only item range per recording) without recomputing those boundaries itself.

Requires: pip install -U 'nemo_toolkit[asr]'

Output goes to data/transcribed/parakeet/{version}/{recording}/item_{N}.json. Schema differs
slightly from the Whisper item files: Parakeet's default decoding doesn't expose per-word
confidence the way Whisper's word_timestamps does, so "prob" is omitted rather than fabricated.
"""
import json
import logging
import re
from pathlib import Path

import librosa
import torch

from src.utils.paths import data_path

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
SAMPLE_RATE = 16_000
VERSIONS = ["version_a", "version_b"]


def load_model():
    import nemo.collections.asr as nemo_asr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading {MODEL_NAME} on {device}")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME, map_location=device)
    model.eval()
    return model


def find_segment_path(recording: str, version: str, index: str) -> Path | None:
    folder = data_path("segmented", "v1", version)
    matches = sorted(folder.glob(f"*{recording}_item{index}.mp3"))
    return matches[0] if matches else None


def transcribe_file(audio_path: Path, model) -> dict:
    # load audio ourselves (mono=True does a true averaged downmix) instead of handing NeMo
    # the file path directly -- some segmented mp3s are stereo, and NeMo's Lhotse-based file
    # loader doesn't auto-downmix (its channel_selector="average" is documented but not
    # actually implemented for this code path), so multi-channel files otherwise arrive as
    # [batch, channels, time] and get rejected by the RNNT encoder
    speech, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    duration = len(speech) / SAMPLE_RATE

    output = model.transcribe([speech], timestamps=True)
    hyp = output[0]

    words = [
        {"word": w["word"], "start": round(w["start"], 3), "end": round(w["end"], 3)}
        for w in hyp.timestamp["word"]
    ]

    # NeMo returns flat "word" and "segment" timestamp lists rather than words nested
    # inside their segment (unlike Whisper's schema) -- assign each word to the segment
    # whose time range contains its start, so downstream code can walk segments->words
    # the same way it does for the Whisper item files.
    segments = []
    for s in hyp.timestamp["segment"]:
        seg_start, seg_end = round(s["start"], 3), round(s["end"], 3)
        seg_words = [w for w in words if seg_start <= w["start"] < seg_end]
        segments.append({"start": seg_start, "end": seg_end, "text": s["segment"], "words": seg_words})

    return {
        "file": audio_path.name,
        # target language is fixed to Spanish by context (same convention as the other
        # transcribe_*.py scripts), not detected -- Parakeet doesn't return a per-utterance
        # language ID from this call
        "language": "es",
        "language_probability": 1.0,
        "duration": round(duration, 3),
        "segments": segments,
    }


def process_recording(version: str, recording: str, item_indices: list[str], model) -> None:
    for index in item_indices:
        output_path = data_path("transcribed", "parakeet", version, recording, f"item_{index}.json")
        if output_path.exists():
            logging.info(f"Skipping {recording}/item_{index} (already transcribed)")
            continue

        audio_path = find_segment_path(recording, version, index)
        if audio_path is None:
            logging.error(f"{recording}: no segment file found for item {index}, skipping item")
            continue

        try:
            logging.info(f"Transcribing {audio_path.name}")
            result = transcribe_file(audio_path, model)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logging.info(f"Saved {output_path}")
        except Exception as e:
            logging.error(f"{recording}: failed to transcribe item {index}: {e}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(data_path("logs", "transcribe_parakeet.log")),
            logging.StreamHandler(),
        ],
    )

    model = load_model()

    for version in VERSIONS:
        version_dir = data_path("transcribed", "v2", version)
        if not version_dir.exists():
            logging.warning(f"Folder not found: {version_dir}")
            continue
        for recording_dir in sorted(p for p in version_dir.iterdir() if p.is_dir()):
            recording = recording_dir.name
            item_indices = sorted(
                (m.group(1) for m in (re.match(r"item_(\d+)\.json$", p.name) for p in recording_dir.glob("item_*.json")) if m),
                key=int,
            )
            if not item_indices:
                continue
            logging.info(f"{recording}: re-transcribing {len(item_indices)} item(s) from transcribed/v2")
            process_recording(version, recording, item_indices, model)


if __name__ == "__main__":
    main()
