"""
CrisperWhisper (nyrahealth/faster_CrisperWhisper) via its faster-whisper/CTranslate2 conversion --
a Whisper large-v3 fine-tune trained for verbatim transcription (fillers, false starts, stutters
kept rather than smoothed away, per https://arxiv.org/abs/2408.16589), used here as a fourth ASR
candidate alongside transcribe.py (whisper), transcribe_parakeet.py, and transcribe_wav2vec2.py.

This faster-whisper conversion's retokenized vocabulary marks a word boundary with a leading ","
instead of a leading space (e.g. word tokens come back as "Ella", ",ya", ",terminaba", ...),
confirmed against several resegmented clips -- present regardless of the without_timestamps flag,
so it's a property of the converted checkpoint, not a decoding-option artifact. _clean_word() strips
it so joined text reads normally; genuine verbatim markers the model produces (e.g. "[UM]" for a
filler) are left untouched. without_timestamps=True below matches the model card's own faster-whisper
usage example; the card also notes CTranslate2 timestamp accuracy "cannot be guaranteed" for this
conversion.
"""
import logging
from pathlib import Path

import torch
from faster_whisper import WhisperModel

MODEL_NAME = "nyrahealth/faster_CrisperWhisper"


def load_model() -> WhisperModel:
    if torch.cuda.is_available():
        device, compute_type = "cuda", "float32"
    else:
        device, compute_type = "cpu", "int8"
    logging.info(f"Loading CrisperWhisper ({MODEL_NAME}) on {device}/{compute_type}")
    return WhisperModel(MODEL_NAME, device=device, compute_type=compute_type)


def _clean_word(word: str) -> str:
    return word[1:] if word.startswith(",") else word


def transcribe_file(audio_path: Path, model: WhisperModel, language: str | None = None) -> dict:
    segments_gen, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=1,
        # Deliberately NOT pinning temperature=0 here (unlike transcribe.py's plain whisper large-v3
        # setup): a fixed temperature disables faster-whisper's built-in retry-at-higher-temperature
        # fallback, which is what catches repetition-loop collapse via compression_ratio_threshold.
        # CrisperWhisper's retokenized/pause-sensitive vocabulary hits that failure mode far more
        # often than plain whisper (empirically ~15% of clips vs. ~0.06% for transcribe.py's whisper
        # on the same resegmented data), so the fallback needs to stay enabled here.
        condition_on_previous_text=False,
        word_timestamps=True,
        without_timestamps=True,
        # Hard backstop: the temperature fallback above resolves most repetition-loop collapses
        # outright, but a residual few still degrade into garbage even at the highest fallback
        # temperature (verified against clips flagged by a repetition scan). This doesn't fix those,
        # but bounds the damage -- without it a collapse can repeat one word 100+ times, which would
        # badly skew WER/MER/CER outliers; with it, collapse is capped at short repeated garbage.
        no_repeat_ngram_size=3,
    )

    segments = []
    for seg in segments_gen:
        words = []
        for w in (seg.words or []):
            words.append({
                "word": _clean_word(w.word),
                "start": round(w.start, 3),
                "end": round(w.end, 3),
                "prob": round(w.probability, 4),
            })
        text = " ".join(w["word"] for w in words) if words else seg.text.strip()
        segments.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": text,
            "words": words,
        })

    return {
        "file": audio_path.name,
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration": round(info.duration, 3),
        "segments": segments,
    }
