"""
SeamlessM4T v2 (facebook/seamless-m4t-v2-large, via transformers) -- the attention-based,
massively-multilingual "second-tier" candidate from docs/asr_model_alternatives.md. Unlike
Whisper (trained on weakly-aligned web audio/text pairs with an implicit language-model prior)
or the CTC candidates (wav2vec2-xlsr, MMS -- no LM prior at all), SeamlessM4T v2 is a
translation model: transcription here is done as speech-to-text-translation (S2TT) with the
source and target language both pinned to Spanish, i.e. "translating" Spanish speech into
Spanish text. That's a genuinely different training objective from either of the other two
architectures, on top of being a third data point on attention-decoder behavior alongside
Whisper and Canary (see transcribe_canary.py).

Only load_model()/transcribe_file() are defined here -- the resegmented item set (used for the
actual model comparison) is driven by transcribe_resegmented_seamlessm4t.py via
resegmented_items.py, the same convention transcribe_canary.py/transcribe_crisperwhisper.py use.

Requires: pip install transformers librosa sentencepiece (already pinned in environment.yml)

No native word-level timestamps -- SeamlessM4T's generate() only returns text tokens, no forced
alignment -- so segments falls back to one entry spanning the whole clip, same as
transcribe_canary.py's fallback path for the same reason. No "prob"/per-word confidence for the
same reason as Canary/Parakeet.
"""
import logging
from pathlib import Path

import librosa
import torch
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

MODEL_NAME = "facebook/seamless-m4t-v2-large"
# ISO 639-3 code SeamlessM4T uses for Spanish.
TGT_LANG = "spa"
SAMPLE_RATE = 16_000


def load_model() -> tuple[AutoProcessor, SeamlessM4Tv2ForSpeechToText, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading {MODEL_NAME} on {device}")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return processor, model, device


def transcribe_file(audio_path: Path, processor: AutoProcessor, model: SeamlessM4Tv2ForSpeechToText, device: torch.device) -> dict:
    speech, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    duration = len(speech) / SAMPLE_RATE

    inputs = processor(audio=speech, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(device)
    with torch.no_grad():
        # source and target language are the same (Spanish) -- this is what makes S2TT act as
        # plain ASR rather than translation.
        output_tokens = model.generate(**inputs, tgt_lang=TGT_LANG)
    text = processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)

    return {
        "file": audio_path.name,
        # target language is fixed by tgt_lang above, not detected -- same convention as
        # transcribe_canary.py/transcribe_parakeet.py
        "language": "es",
        "language_probability": 1.0,
        "duration": round(duration, 3),
        "segments": [{"start": 0.0, "end": round(duration, 3), "text": text, "words": []}],
    }
