"""
NVIDIA Canary-1B-v2 (nvidia/canary-1b-v2, nemo_toolkit[asr]) -- a follow-up to canary-1b (see
transcribe_canary.py), released 2026-06-23. Same FastConformer-Transformer encoder-decoder
architecture and multi-task (ASR + translation) design, expanded from 4 to 25 languages (Spanish
still included) and, unlike v1, its decoder actually emits the timestamp tokens NeMo needs for
word/segment timestamps -- so this checkpoint gets a real per-word breakdown instead of the
whole-clip single-segment fallback transcribe_canary.py needed. Still no per-word confidence
("prob"), same as v1/parakeet.

Tried as a cheap follow-up since canary-1b is already the best-scoring system in the resegmented
comparison (see notebooks/model_comparison/analyze_canary_transcription_quality.ipynb) -- worth checking whether a
newer checkpoint of the same architecture improves on that, or whether the win from v1 was
architecture-specific rather than checkpoint-specific.

Requires: pip install -U nemo_toolkit[asr] (>=2.7, for canary-1b-v2 timestamp support)

Only load_model()/transcribe_file() are defined here -- the resegmented item set (used for the
actual model comparison) is driven by transcribe_resegmented_canary_v2.py via
resegmented_items.py, same convention as transcribe_canary.py.
"""
import logging
from pathlib import Path

import librosa
import torch

MODEL_NAME = "nvidia/canary-1b-v2"
SAMPLE_RATE = 16_000


def load_model():
    from nemo.collections.asr.models import ASRModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading {MODEL_NAME} on {device}")
    model = ASRModel.from_pretrained(model_name=MODEL_NAME, map_location=device)
    model.eval()
    return model


def transcribe_file(audio_path: Path, model) -> dict:
    # Same mono downmix as transcribe_canary.py/transcribe_parakeet.py -- NeMo's Lhotse-based file
    # loader doesn't auto-downmix stereo segments, so we load with librosa first.
    speech, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    duration = len(speech) / SAMPLE_RATE

    output = model.transcribe(
        [speech],
        source_lang="es",
        target_lang="es",
        # "yes" restores punctuation/capitalization on the output text; this is formatting, not
        # content correction, and normalize_text() in build_postprocessed_csv.py strips it before
        # WER/MER/CER scoring anyway.
        pnc="yes",
        timestamps=True,
    )
    hyp = output[0]

    words = [
        {"word": w["word"], "start": round(w["start"], 3), "end": round(w["end"], 3)}
        for w in hyp.timestamp["word"]
    ]

    segments = []
    for s in hyp.timestamp["segment"]:
        seg_start, seg_end = round(s["start"], 3), round(s["end"], 3)
        seg_words = [w for w in words if seg_start <= w["start"] < seg_end]
        segments.append({"start": seg_start, "end": seg_end, "text": s["segment"], "words": seg_words})

    # Defensive fallback, same as transcribe_canary.py -- in case some edge-case clip comes back
    # with text but no timestamps.
    if not segments and hyp.text.strip():
        segments = [{"start": 0.0, "end": round(duration, 3), "text": hyp.text, "words": []}]

    return {
        "file": audio_path.name,
        "language": "es",
        "language_probability": 1.0,
        "duration": round(duration, 3),
        "segments": segments,
    }
