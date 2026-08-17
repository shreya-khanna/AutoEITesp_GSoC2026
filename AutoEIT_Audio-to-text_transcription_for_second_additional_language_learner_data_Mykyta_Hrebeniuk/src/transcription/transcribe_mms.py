"""
Facebook MMS (facebook/mms-1b-all, via transformers) -- the "architecturally different" CTC
comparison point from docs/asr_model_alternatives.md, alongside wav2vec2 (see
transcribe_wav2vec2.py). Same CTC decoding shape as wav2vec2-xlsr-53-spanish (no autoregressive
LM prior, so it fails via literal phonetic garbling rather than fluent-sounding hallucination),
but trained on ~1,100 languages via per-language adapter weights rather than XLSR-53's 53, which
is the actual point of comparison: does the much broader (but more diluted per-language) MMS
training data help or hurt on L2-accented Spanish relative to the dedicated Spanish XLSR
fine-tune.

Only load_model()/transcribe_file() are defined here -- the resegmented item set (used for the
actual model comparison) is driven by transcribe_resegmented_mms.py via resegmented_items.py, the
same convention transcribe_canary.py/transcribe_crisperwhisper.py use.

Requires: pip install transformers librosa (already pinned in environment.yml)

Reuses transcribe_wav2vec2.py's _decode_words_with_timestamps/_finalize_word -- MMS's
Wav2Vec2ForCTC/Wav2Vec2Processor classes expose the exact same CTC logits/tokenizer API once the
Spanish adapter is loaded, so the decode logic is identical; only model loading (target_lang, the
adapter-selection kwarg MMS adds on top of plain wav2vec2) differs.
"""
import logging
from pathlib import Path

import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from src.transcription.transcribe_wav2vec2 import _decode_words_with_timestamps

MODEL_NAME = "facebook/mms-1b-all"
# ISO 639-3 code MMS uses for Spanish (see the model's adapter list on the HF model card).
TARGET_LANG = "spa"
SAMPLE_RATE = 16_000


def load_model() -> tuple[Wav2Vec2Processor, Wav2Vec2ForCTC, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading {MODEL_NAME} (target_lang={TARGET_LANG}) on {device}")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME, target_lang=TARGET_LANG)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME, target_lang=TARGET_LANG, ignore_mismatched_sizes=True)
    # target_lang= above only resizes lm_head to the Spanish adapter's vocab size; it does NOT
    # load that adapter's actual weight values (confirmed via the "newly initialized" shape-mismatch
    # warning from from_pretrained -- the resized head comes back randomly initialized without this
    # explicit call). load_adapter() is what fetches and applies adapter.spa.safetensors.
    # (from_pretrained still logs a "newly initialized" shape-mismatch warning for lm_head right
    # before this call -- that's the base checkpoint's default-language head getting resized, not
    # a sign anything is actually random; load_adapter() overwrites it with real spa weights, and
    # a manual A/B against transcribe_wav2vec2.py's output on the same clip confirmed real
    # (if weaker) Spanish transcription, not noise.)
    model.load_adapter(TARGET_LANG)
    model = model.to(device)
    model.eval()
    return processor, model, device


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
        # target language is fixed by the loaded adapter, not detected -- same convention as
        # transcribe_wav2vec2.py
        "language": "es",
        "language_probability": 1.0,
        "duration": round(duration, 3),
        "segments": [{"start": 0.0, "end": round(duration, 3), "text": text, "words": words}],
    }
