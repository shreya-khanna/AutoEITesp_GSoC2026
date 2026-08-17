"""
Re-transcribe the same items as data/transcribed/v2/ with jonatasgrosman/wav2vec2-large-xlsr-53-spanish,
a CTC model -- the "architecturally different" comparison point from docs/asr_model_alternatives.md.
Unlike Whisper's attention decoder (which can hallucinate fluent-sounding wrong text from its
language-model prior), CTC has no autoregressive prior over its own output, so it fails
differently: literal phonetic garbling instead of invented words/phrases.

The item set to transcribe is read directly off data/transcribed/v2/{version}/{recording}/item_{N}.json
file names, so this stays in lockstep with whatever transcribe_v2.py has already scoped down to
(the Spanish-narrative-only item range per recording) without recomputing those boundaries itself.

Requires: pip install transformers librosa

Output goes to data/transcribed/wav2vec2/{version}/{recording}/item_{N}.json, in the same format
as transcribe.py's item files, except word-level timestamps/confidence are derived rather than
natively provided by the model (see _decode_words_with_timestamps below) -- CTC has no built-in
word-alignment API like Whisper's word_timestamps.
"""
import json
import logging
import re
from pathlib import Path

import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from src.utils.paths import data_path

MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
SAMPLE_RATE = 16_000
VERSIONS = ["version_a", "version_b"]


def load_model() -> tuple[Wav2Vec2Processor, Wav2Vec2ForCTC, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading {MODEL_NAME} on {device}")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return processor, model, device


def find_segment_path(recording: str, version: str, index: str) -> Path | None:
    folder = data_path("segmented", "v1", version)
    matches = sorted(folder.glob(f"*{recording}_item{index}.mp3"))
    return matches[0] if matches else None


def _finalize_word(chars: list[str], probs: list[float], start: float, end: float) -> dict:
    return {
        "word": "".join(chars),
        "start": round(start, 3),
        "end": round(end, 3),
        "prob": round(sum(probs) / len(probs), 4),
    }


def _decode_words_with_timestamps(logits: torch.Tensor, processor: Wav2Vec2Processor, duration: float) -> list[dict]:
    """Standard CTC greedy decode (collapse consecutive repeats, drop blanks), splitting
    words on the tokenizer's word-delimiter token. Frame length is derived from
    duration / num_frames rather than assumed, since it's robust to any change in the
    model's conv-feature-encoder stride."""
    probs = torch.softmax(logits, dim=-1)
    frame_probs, frame_ids = probs.max(dim=-1)
    frame_ids = frame_ids.tolist()
    frame_probs = frame_probs.tolist()

    pad_id = processor.tokenizer.pad_token_id
    delim_id = processor.tokenizer.word_delimiter_token_id
    frame_duration = duration / len(frame_ids)

    words = []
    chars: list[str] = []
    word_probs: list[float] = []
    word_start = None
    prev_id = None

    for i, tok_id in enumerate(frame_ids):
        if tok_id == prev_id:
            continue
        prev_id = tok_id
        if tok_id == pad_id:
            continue

        t = i * frame_duration
        if tok_id == delim_id:
            if chars:
                words.append(_finalize_word(chars, word_probs, word_start, t))
                chars, word_probs, word_start = [], [], None
            continue

        if word_start is None:
            word_start = t
        chars.append(processor.tokenizer.convert_ids_to_tokens(tok_id))
        word_probs.append(frame_probs[i])

    if chars:
        words.append(_finalize_word(chars, word_probs, word_start, len(frame_ids) * frame_duration))

    return words


def transcribe_file(audio_path: Path, processor: Wav2Vec2Processor, model: Wav2Vec2ForCTC, device: torch.device) -> dict:
    speech, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    duration = len(speech) / SAMPLE_RATE

    inputs = processor(speech, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to(device), attention_mask=inputs.attention_mask.to(device)).logits[0].cpu()

    text = processor.decode(torch.argmax(logits, dim=-1))
    words = _decode_words_with_timestamps(logits, processor, duration)

    return {
        "file": audio_path.name,
        # this model is Spanish-only, so "language" is fixed rather than detected --
        # same convention transcribe_v2.py uses when forcing LANGUAGE = "es" on Whisper
        "language": "es",
        "language_probability": 1.0,
        "duration": round(duration, 3),
        "segments": [{"start": 0.0, "end": round(duration, 3), "text": text, "words": words}],
    }


def process_recording(version: str, recording: str, item_indices: list[str], processor, model, device) -> None:
    for index in item_indices:
        output_path = data_path("transcribed", "wav2vec2", version, recording, f"item_{index}.json")
        if output_path.exists():
            logging.info(f"Skipping {recording}/item_{index} (already transcribed)")
            continue

        audio_path = find_segment_path(recording, version, index)
        if audio_path is None:
            logging.error(f"{recording}: no segment file found for item {index}, skipping item")
            continue

        try:
            logging.info(f"Transcribing {audio_path.name}")
            result = transcribe_file(audio_path, processor, model, device)
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
            logging.FileHandler(data_path("logs", "transcribe_wav2vec2.log")),
            logging.StreamHandler(),
        ],
    )

    processor, model, device = load_model()

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
            process_recording(version, recording, item_indices, processor, model, device)


if __name__ == "__main__":
    main()
