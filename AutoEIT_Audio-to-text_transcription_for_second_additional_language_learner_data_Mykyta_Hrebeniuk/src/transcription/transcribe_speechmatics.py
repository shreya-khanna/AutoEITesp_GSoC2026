"""
Speechmatics (Enhanced batch model, via the `speechmatics-batch` SDK) -- a second commercial-API
comparison point alongside AssemblyAI (see transcribe_assemblyai.py) and Rev.ai (see
transcribe_revai.py), added per advisor guidance to benchmark more of the disfluency-preserving
commercial options rather than treating AssemblyAI as the only
production-grade candidate.

`transcript_filtering_config=TranscriptFilteringConfig(remove_disfluencies=False)` is passed
explicitly even though `False` is already the SDK default -- disfluencies (filler words, false
starts) are the research signal for this L2 language-learning work (same rationale as
`disfluencies=True` in transcribe_assemblyai.py), so the choice is pinned rather than left to
rely on whatever the library's default happens to be later.

The SDK (`speechmatics.batch.AsyncClient`) is async-only -- there is no sync client -- so
`transcribe_file()` wraps a single `asyncio.run()` call per file. That mirrors
transcribe_assemblyai.py's per-call `aai.Transcriber()` (a fresh client per file rather than one
shared across the whole run), just with the extra async->sync boundary the SDK requires.

`results` in the returned Transcript are a flat stream of `word`/`punctuation` items, not
pre-grouped into segments/words like AssemblyAI's SDK does -- the word list below is built by
filtering `type == "word"` and reading each item's first (highest-likelihood) alternative.

Text is reconstructed manually from `results` rather than via `Transcript.transcript_text`:
every alternative carries a `speaker` of "UU" (unknown) even with `diarization="none"` set below,
and `transcript_text` prepends "SPEAKER UU: " to the line whenever `speaker` is truthy -- there's
no config that suppresses it, since the SDK checks the field's presence, not whether diarization
was requested. That prefix would corrupt WER/MER/CER scoring against the scripted stimulus, so
plain punctuation-aware joining is done directly here instead (using `attaches_to == "previous"`
to decide whether a punctuation mark glues to the prior word without a space, the same signal
`transcript_text`'s own joining logic keys off).

Requires: pip install speechmatics-batch (already added to environment.yml)
Requires: SPEECHMATICS_API_KEY set in .env (see .env.example) or the environment. (The SDK's
StaticKeyAuth reads this env var itself if no api_key is passed to AsyncClient.)
"""
import asyncio
import logging
import os
from pathlib import Path

from speechmatics.batch import AsyncClient, JobConfig, JobType, Transcript, TranscriptFilteringConfig, TranscriptionConfig

LANGUAGE_CODE = "es"


def load_model() -> JobConfig:
    if not os.getenv("SPEECHMATICS_API_KEY"):
        raise RuntimeError("SPEECHMATICS_API_KEY is not set (add it to .env)")
    logging.info(f"Using Speechmatics language={LANGUAGE_CODE}, model=enhanced")
    transcription_config = TranscriptionConfig(
        language=LANGUAGE_CODE,
        # These clips are single-speaker EIT items -- no need to ask the API to diarize them.
        # (Doesn't actually stop the "UU" speaker label discussed below, but it's the semantically
        # correct setting regardless and costs nothing to set.)
        diarization="none",
        transcript_filtering_config=TranscriptFilteringConfig(remove_disfluencies=False),
    )
    return JobConfig(type=JobType.TRANSCRIPTION, transcription_config=transcription_config)


async def _transcribe_file_async(audio_path: Path, config: JobConfig) -> Transcript:
    async with AsyncClient() as client:
        result = await client.transcribe(str(audio_path), config=config)
    return result


def transcribe_file(audio_path: Path, config: JobConfig) -> dict:
    transcript = asyncio.run(_transcribe_file_async(audio_path, config))

    words = []
    text_parts = []
    duration = 0.0
    for item in transcript.results:
        duration = max(duration, item.end_time)
        if not item.alternatives:
            continue
        alt = item.alternatives[0]

        if text_parts and item.attaches_to != "previous":
            text_parts.append(" ")
        text_parts.append(alt.content)

        if item.type != "word":
            continue
        words.append(
            {
                "word": alt.content,
                "start": round(item.start_time, 3),
                "end": round(item.end_time, 3),
                "prob": round(alt.confidence, 4),
            }
        )

    text = "".join(text_parts)

    return {
        "file": audio_path.name,
        # language is pinned via config, not detected -- same convention as
        # transcribe_assemblyai.py/transcribe_mms.py for adapter/config-pinned languages
        "language": LANGUAGE_CODE,
        "language_probability": 1.0,
        "duration": round(duration, 3),
        "segments": [{"start": 0.0, "end": round(duration, 3), "text": text, "words": words}],
    }
