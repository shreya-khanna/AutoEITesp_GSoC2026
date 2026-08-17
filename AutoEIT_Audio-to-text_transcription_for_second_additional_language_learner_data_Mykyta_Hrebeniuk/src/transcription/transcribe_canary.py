"""
NVIDIA Canary-1B (nvidia/canary-1b, nemo_toolkit[asr]) -- a multilingual, multi-task
(ASR + translation) FastConformer-Transformer encoder-decoder, tried here as a candidate ASR
model for the same verbatim-preservation reason Parakeet was (see transcribe_parakeet.py):
Whisper-family decoders are known to "correct" L2-accented speech toward fluent, grammatical
text rather than transcribing what was actually said, which defeats the purpose of scoring
learner errors. Canary is architecturally closer to Parakeet (FastConformer encoder) than to
Whisper, so it's a second data point on whether that failure mode is Whisper-specific or shared
by other attention-decoder models.

Requires: pip install -U 'nemo_toolkit[asr]' (already pinned in environment.yml)

Only load_model()/transcribe_file() are defined here -- the resegmented item set (used for the
actual model comparison) is driven by transcribe_resegmented_canary.py via resegmented_items.py,
so there's no need for the original v1/v2-segmentation process_recording()/main() plumbing that
transcribe_parakeet.py carries for its own (older) resegmented wrapper.

Same schema as transcribe_parakeet.py's output (no "prob" field -- Canary doesn't expose
per-word confidence through this API either).
"""
import logging
from pathlib import Path

import librosa
import torch

MODEL_NAME = "nvidia/canary-1b"
SAMPLE_RATE = 16_000


def load_model():
    import nemo.collections.asr as nemo_asr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading {MODEL_NAME} on {device}")
    model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(model_name=MODEL_NAME, map_location=device)
    model.eval()
    return model


def transcribe_file(audio_path: Path, model) -> dict:
    # Same mono downmix as transcribe_parakeet.py -- NeMo's Lhotse-based file loader doesn't
    # auto-downmix stereo segments, so we load with librosa first.
    speech, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    duration = len(speech) / SAMPLE_RATE

    output = model.transcribe(
        [speech],
        source_lang="es",
        target_lang="es",
        task="asr",
        # "yes" restores punctuation/capitalization on the output text; this is formatting, not
        # content correction, and normalize_text() in build_postprocessed_csv.py strips it before
        # WER/MER/CER scoring anyway -- doesn't affect the verbatim-content question this model is
        # actually being evaluated on.
        pnc="yes",
        timestamps=True,
    )
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

    # nvidia/canary-1b's decoder never emits the "<|N|>word<|N|>" timestamp tokens NeMo's
    # process_aed_timestamp_outputs() looks for (that's only wired up for newer Canary
    # checkpoints with a bundled aligner model), so hyp.timestamp["word"/"segment"] come back
    # empty on every clip even though hyp.text holds the real transcription. Each resegmented
    # clip is already a single item, so fall back to one segment spanning the whole clip rather
    # than silently discarding the transcription.
    if not segments and hyp.text.strip():
        segments = [{"start": 0.0, "end": round(duration, 3), "text": hyp.text, "words": []}]

    return {
        "file": audio_path.name,
        "language": "es",
        "language_probability": 1.0,
        "duration": round(duration, 3),
        "segments": segments,
    }
