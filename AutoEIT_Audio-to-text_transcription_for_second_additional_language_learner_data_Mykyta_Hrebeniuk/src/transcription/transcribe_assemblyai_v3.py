"""
AssemblyAI (Universal-3.5 Pro speech model, exposed via the API as the literal id
"universal-3-5-pro" -- NOT "universal-3-pro", which the API rejects as a deprecated id despite
being what AssemblyAI's own docs/blog posts used at the time this was written; confirmed directly
against a live 400 response) -- a follow-up to transcribe_assemblyai.py's Universal-2 run, this
time paired with Universal-3.5 Pro's Speech-to-Text Prompting feature rather than relying on
`disfluencies=True` alone. Universal-2 was pinned deliberately (per McGuire, see
transcribe_assemblyai.py) to evaluate the actual autoeit.org production model; this is the
separate "how good could AssemblyAI's newest model get with disfluency-aware prompting" question,
kept as its own script/output dir rather than mutating the production comparison.

Cost/coverage check before wiring this up: Universal-3.5 Pro (async) is $0.21/hr vs. Universal-2's
$0.15/hr -- the account's existing credit comfortably covers the full resegmented hit-recording set
at that rate, and the `prompt` parameter isn't a separate metered add-on (unlike Keyterms Prompting
on the streaming models). Spanish is one of Universal-3.5 Pro's six core languages (en/es/pt/fr/de/it),
so no universal-2 fallback is needed for this dataset.

`prompt` is set to an explicit verbatim-transcription instruction (in Spanish, since the docs
recommend prompting in the target language) asking the model to preserve disfluencies, repetitions,
and false starts by name -- meant to reduce the LLM-cleanup-pass hallucination risk that plain
`disfluencies=True` doesn't fully address, same research-signal rationale as transcribe_assemblyai.py.

Requires: pip install assemblyai (already added to environment.yml)
Requires: ASSEMBLYAI_API_KEY set in .env (see .env.example) or the environment.
"""
import logging
import os
from pathlib import Path

import assemblyai as aai

SPEECH_MODELS = ["universal-3-5-pro"]
LANGUAGE_CODE = "es"
PROMPT = (
    "Transcripción literal. Conserva los patrones lingüísticos naturales del hablante, "
    "incluyendo dudas, repeticiones, inicios falsos y muletillas como 'eh', 'este' y 'pues'."
)


def load_model() -> aai.TranscriptionConfig:
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise RuntimeError("ASSEMBLYAI_API_KEY is not set (add it to .env)")
    aai.settings.api_key = api_key
    logging.info(f"Using AssemblyAI speech_models={SPEECH_MODELS}, language_code={LANGUAGE_CODE}, prompt set")
    return aai.TranscriptionConfig(
        speech_models=SPEECH_MODELS,
        language_code=LANGUAGE_CODE,
        disfluencies=True,
        prompt=PROMPT,
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
        # convention as transcribe_assemblyai.py for adapter/config-pinned languages
        "language": LANGUAGE_CODE,
        "language_probability": 1.0,
        "duration": round(duration, 3),
        "segments": [{"start": 0.0, "end": round(duration, 3), "text": text, "words": words}],
    }
