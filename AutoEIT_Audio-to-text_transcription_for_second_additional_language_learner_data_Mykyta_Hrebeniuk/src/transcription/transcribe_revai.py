"""
Rev.ai (default cloud transcriber, via the `rev_ai` SDK) -- a third commercial-API comparison
point alongside AssemblyAI (see transcribe_assemblyai.py) and Speechmatics (see
transcribe_speechmatics.py), added per advisor guidance to benchmark more of the
disfluency-preserving commercial options rather than treating
AssemblyAI as the only production-grade candidate. Rev.ai in particular markets a "verbatim"
style output aimed at exactly this kind of research transcription.

`remove_disfluencies=False` is passed explicitly even though `False` is already the SDK default --
disfluencies (filler words, false starts) are the research signal for this L2 language-learning
work (same rationale as `disfluencies=True` in transcribe_assemblyai.py), so the choice is pinned
rather than left to rely on whatever the API's default happens to be later.

Unlike the Speechmatics SDK (async-only), `rev_ai`'s RevAiAPIClient is a plain synchronous
`requests`-based client with no wait-for-completion helper, so transcribe_file() polls
get_job_details() directly -- same "poll until done, then fetch" shape as
transcribe_speechmatics.py's asyncio.run(AsyncClient.wait_for_completion()), just implemented by
hand with time.sleep() instead of asyncio.

The transcript is a list of monologues -> elements, where each element is 'text', 'punct', or
'unknown' (only 'text' elements carry a confidence score) -- filtered to 'text' below to build the
word list, same shape as the word lists the other transcribe_*.py scripts produce.

Requires: pip install rev_ai (already added to environment.yml)
Requires: REV_AI_ACCESS_TOKEN set in .env (see .env.example) or the environment.
"""
import logging
import os
import time
from pathlib import Path

from rev_ai import JobStatus
from rev_ai.apiclient import RevAiAPIClient

LANGUAGE_CODE = "es"
POLL_INTERVAL_SECONDS = 5.0


def load_model() -> RevAiAPIClient:
    access_token = os.getenv("REV_AI_ACCESS_TOKEN")
    if not access_token:
        raise RuntimeError("REV_AI_ACCESS_TOKEN is not set (add it to .env)")
    logging.info(f"Using Rev.ai language={LANGUAGE_CODE}")
    return RevAiAPIClient(access_token)


def transcribe_file(audio_path: Path, client: RevAiAPIClient) -> dict:
    job = client.submit_job_local_file(
        str(audio_path),
        language=LANGUAGE_CODE,
        remove_disfluencies=False,
    )

    while True:
        details = client.get_job_details(job.id)
        if details.status == JobStatus.TRANSCRIBED:
            break
        if details.status == JobStatus.FAILED:
            raise RuntimeError(f"Rev.ai transcription failed: {details.failure_detail}")
        time.sleep(POLL_INTERVAL_SECONDS)

    transcript = client.get_transcript_object(job.id)

    words = []
    text_parts = []
    for monologue in transcript.monologues:
        for element in monologue.elements:
            text_parts.append(element.value)
            if element.type_ != "text":
                continue
            words.append(
                {
                    "word": element.value,
                    "start": round(element.timestamp, 3),
                    "end": round(element.end_timestamp, 3),
                    "prob": round(element.confidence, 4),
                }
            )
    text = "".join(text_parts)

    duration = details.duration_seconds or 0.0

    return {
        "file": audio_path.name,
        # language is pinned via request param, not detected -- same convention as
        # transcribe_assemblyai.py/transcribe_speechmatics.py for config-pinned languages
        "language": LANGUAGE_CODE,
        "language_probability": 1.0,
        "duration": round(duration, 3),
        "segments": [{"start": 0.0, "end": round(duration, 3), "text": text, "words": words}],
    }
