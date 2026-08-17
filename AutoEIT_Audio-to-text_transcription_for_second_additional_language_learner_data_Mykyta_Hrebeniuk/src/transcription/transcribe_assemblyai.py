"""
AssemblyAI (Universal-2 speech model, via the `assemblyai` SDK) -- the actual production
autoeit.org backend, added as a comparison point per advisor guidance (McGuire, 2026-08-03).
Every other model in this harness is an open-source alternative benchmarked against human raters;
AssemblyAI itself had never been run through the same harness -- this closes that gap.

Pinned to `universal-2` rather than the current default (`universal-3-5-pro`) on McGuire's
explicit instruction: he wants the model actually running in production evaluated, not whatever
AssemblyAI's API defaults to next. `speech_models` (plural) takes a list of raw model-id strings,
which is how a specific pinned id like "universal-2" is expressed -- the `speech_model` (singular)
field only exposes an enum of {best, nano, slam-1, universal} and has no "universal-2" member.

`disfluencies=True` is set because McGuire separately flagged that
commercial ASR increasingly bolts an LLM cleanup pass onto the transcript that strips filler words
and false starts -- exactly the disfluencies that are the research signal for L2 language-learning
work. AssemblyAI excludes them by default; this opts back in.

Unlike the local torch models, there's no "load_model()" in the literal sense -- load_model() here
just validates the API key and returns a configured TranscriptionConfig, kept for naming
consistency with transcribe_mms.py/transcribe_seamlessm4t.py's load_model()/transcribe_file() shape.

Requires: pip install assemblyai (already added to environment.yml)
Requires: ASSEMBLYAI_API_KEY set in .env (see .env.example) or the environment.
"""
import logging
import os
from pathlib import Path

import assemblyai as aai

SPEECH_MODELS = ["universal-2"]
LANGUAGE_CODE = "es"


def load_model() -> aai.TranscriptionConfig:
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise RuntimeError("ASSEMBLYAI_API_KEY is not set (add it to .env)")
    aai.settings.api_key = api_key
    logging.info(f"Using AssemblyAI speech_models={SPEECH_MODELS}, language_code={LANGUAGE_CODE}")
    return aai.TranscriptionConfig(
        speech_models=SPEECH_MODELS,
        language_code=LANGUAGE_CODE,
        disfluencies=True,
    )


def transcribe_file(audio_path: Path, config: aai.TranscriptionConfig) -> dict:
    transcript = aai.Transcriber().transcribe(str(audio_path), config=config)
    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")

    duration = transcript.audio_duration or 0.0
    words = [
        {
            "word": w.text,
            "start": round(w.start / 1000, 3),
            "end": round(w.end / 1000, 3),
            "prob": round(w.confidence, 4),
        }
        for w in (transcript.words or [])
    ]
    text = transcript.text or ""

    return {
        "file": audio_path.name,
        # language is pinned via config, not detected, so probability is nominal -- same
        # convention as transcribe_mms.py/transcribe_wav2vec2.py for adapter-pinned languages
        "language": LANGUAGE_CODE,
        "language_probability": 1.0,
        "duration": round(duration, 3),
        "segments": [{"start": 0.0, "end": round(duration, 3), "text": text, "words": words}],
    }
