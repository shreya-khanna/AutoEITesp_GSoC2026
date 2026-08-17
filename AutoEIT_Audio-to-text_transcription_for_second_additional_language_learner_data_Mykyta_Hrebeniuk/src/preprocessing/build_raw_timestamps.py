"""
For every recording that has output in data/transcribed/v2 (the re-transcribed Spanish narrative
section), compute the raw-recording-relative [beginning, end] window (in seconds) spanning its
first through last re-transcribed item, and save one JSON file per version under
data/raw/audio_files/{version}/.

data/transcribed/v2's item indices are already the resolved boundary (its min index is
last_english_item + 1 and its max index is last_spanish_sentence from the original
data/transcribed/v1 pipeline -- see notebooks/transcription_quality/analyze_transcription_quality.ipynb's sibling
hit/miss notebook), so no boundary disambiguation is needed here; we just need the raw-file
time each of those indices corresponds to.

Those raw-file timestamps aren't persisted anywhere (src/preprocessing/vad.py's run_vad computes
them but only uses them in-memory to cut segmented clips), so this script recomputes them by
rerunning the same VAD + merge-gap logic (now factored out as
src.preprocessing.vad.compute_merged_timestamps) on each raw audio file, then indexes into the
result with data/transcribed/v2's min/max item index. As a sanity check, the recomputed timestamp
count is compared against how many clips actually exist for that recording in
data/segmented/v1/ -- a mismatch means VAD didn't reproduce the original segmentation and the
recording is skipped with a warning rather than silently given a wrong window.

Usage:
    python -m src.preprocessing.build_raw_timestamps
"""
import json
import logging
import re
from pathlib import Path

from silero_vad import load_silero_vad

from src.preprocessing.vad import compute_merged_timestamps
from src.utils.paths import data_path

VERSIONS = ["version_a", "version_b"]
ITEM_RE = re.compile(r"^item_(\d+)\.json$")
SEGMENTED_ITEM_RE = re.compile(r"^(?:ATTENTION_)?(.+)_item(\d+)\.mp3$")

# Prefer this extension order when a recording has more than one raw file (e.g. both .WMA and .mp3).
EXTENSION_PREFERENCE = [".mp3", ".MP3", ".wma", ".WMA"]


def find_raw_audio_path(raw_dir: Path, recording: str) -> Path | None:
    candidates = [p for p in raw_dir.iterdir() if p.name.rsplit(".", 1)[0] == recording]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    for ext in EXTENSION_PREFERENCE:
        for p in candidates:
            if p.suffix == ext:
                return p
    return candidates[0]


def get_item_index_range(recording_dir: Path) -> tuple[int, int] | None:
    indices = [
        int(ITEM_RE.match(p.name).group(1))
        for p in recording_dir.glob("item_*.json")
        if ITEM_RE.match(p.name)
    ]
    if not indices:
        return None
    return min(indices), max(indices)


def get_segmented_item_count(segmented_dir: Path, recording: str) -> int:
    return sum(
        1
        for p in segmented_dir.glob(f"*{recording}_item*.mp3")
        if SEGMENTED_ITEM_RE.match(p.name) and SEGMENTED_ITEM_RE.match(p.name).group(1) == recording
    )


def build_version(version: str, model) -> dict:
    v2_dir = data_path("transcribed", "v2", version)
    raw_dir = data_path("raw", "audio_files", version)
    segmented_dir = data_path("segmented", "v1", version)

    results = {}
    recording_dirs = sorted(p for p in v2_dir.iterdir() if p.is_dir())
    logging.info(f"{version}: {len(recording_dirs)} recordings with transcribed/v2 output")

    for recording_dir in recording_dirs:
        recording = recording_dir.name

        index_range = get_item_index_range(recording_dir)
        if index_range is None:
            logging.warning(f"{recording}: no item_*.json files, skipping")
            continue
        min_idx, max_idx = index_range

        raw_path = find_raw_audio_path(raw_dir, recording)
        if raw_path is None:
            logging.warning(f"{recording}: no matching raw audio file, skipping")
            continue

        expected_count = get_segmented_item_count(segmented_dir, recording)

        logging.info(f"{recording}: running VAD on {raw_path.name}")
        timestamps = compute_merged_timestamps(str(raw_path), model)

        if expected_count and len(timestamps) != expected_count:
            logging.warning(
                f"{recording}: VAD reproduced {len(timestamps)} items, "
                f"but {expected_count} segmented clips exist on disk -- skipping (indices would misalign)"
            )
            continue

        if max_idx >= len(timestamps):
            logging.warning(
                f"{recording}: max item index {max_idx} out of range for {len(timestamps)} "
                f"recomputed VAD items -- skipping"
            )
            continue

        beginning = timestamps[min_idx][0]
        end = timestamps[max_idx][1]
        results[recording] = {
            "beginning": round(beginning, 3),
            "end": round(end, 3),
            "min_item_index": min_idx,
            "max_item_index": max_idx,
            "raw_audio_file": raw_path.name,
        }

    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Loading VAD model")
    model = load_silero_vad()

    for version in VERSIONS:
        results = build_version(version, model)
        out_path = data_path("raw", "audio_files", version, "segment_window_timestamps.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logging.info(f"{version}: wrote {len(results)} entries to {out_path}")


if __name__ == "__main__":
    main()
