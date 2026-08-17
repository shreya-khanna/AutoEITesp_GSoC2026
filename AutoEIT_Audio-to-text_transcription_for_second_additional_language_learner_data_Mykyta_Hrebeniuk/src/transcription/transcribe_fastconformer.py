"""
NVIDIA Spanish FastConformer Hybrid Transducer-CTC (nvidia/stt_es_fastconformer_hybrid_large_pc,
nemo_toolkit[asr]) -- a monolingual Spanish acoustic model, tried as a second verbatim-leaning
candidate alongside transcribe_canary.py. Deliberately decoded with its CTC head rather than its
RNNT head (change_decoding_strategy() call below): CTC is a frame-synchronous classifier with no
autoregressive language-model component, so unlike Whisper's decoder -- or even this same
checkpoint's own RNNT head -- it has no mechanism to "correct" disfluent speech toward a more
fluent, grammatical hypothesis; it can only emit what the acoustic encoder actually heard. This is
a more direct test of the verbatim-preservation goal than Canary, at the cost of being a narrower,
older-architecture, monolingual (Spanish-only) model with no multilingual/translation capability.

Requires: pip install -U 'nemo_toolkit[asr]' (already pinned in environment.yml)

Only load_model()/transcribe_file() are defined here, in the same spirit as transcribe_canary.py
-- transcribe_resegmented_fastconformer.py drives the actual (resegmented) item set.

Same schema as transcribe_parakeet.py's output (no "prob" field -- CTC's per-frame logits aren't
converted to a per-word confidence here).
"""
import logging
from pathlib import Path

import librosa
import torch

MODEL_NAME = "nvidia/stt_es_fastconformer_hybrid_large_pc"
SAMPLE_RATE = 16_000


def load_model():
    import nemo.collections.asr as nemo_asr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading {MODEL_NAME} on {device}")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME, map_location=device)
    model.change_decoding_strategy(decoder_type="ctc")
    model.eval()
    return model


def transcribe_file(audio_path: Path, model) -> dict:
    # Same mono downmix as transcribe_parakeet.py -- NeMo's Lhotse-based file loader doesn't
    # auto-downmix stereo segments, so we load with librosa first.
    speech, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    duration = len(speech) / SAMPLE_RATE

    output = model.transcribe([speech], timestamps=True)
    hyp = output[0]

    words = [
        {"word": w["word"], "start": round(w["start"], 3), "end": round(w["end"], 3)}
        for w in hyp.timestamp["word"]
    ]

    # Same flat word/segment timestamp lists as Parakeet -- assign each word to the segment
    # whose time range contains its start (see transcribe_parakeet.py for why).
    segments = []
    for s in hyp.timestamp["segment"]:
        seg_start, seg_end = round(s["start"], 3), round(s["end"], 3)
        seg_words = [w for w in words if seg_start <= w["start"] < seg_end]
        segments.append({"start": seg_start, "end": seg_end, "text": s["segment"], "words": seg_words})

    return {
        "file": audio_path.name,
        "language": "es",
        "language_probability": 1.0,
        "duration": round(duration, 3),
        "segments": segments,
    }
