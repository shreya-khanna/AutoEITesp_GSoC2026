"""
Build per-recording summary.json files for data/transcribed/{folder}/ subfolders, in the same
schema as data/transcribed/v1/ and data/transcribed/v2/'s summaries: per-item duration, language,
language_probability, word_count, mean_word_prob, text, empty flag, aggregated under one
summary.json per recording folder (plus recording/version/item_count at the top level).

Works generically off whatever item_N.json files are already on disk, so it can be re-run
mid-batch to check progress, or after a full run completes -- it always rebuilds fresh
rather than skipping existing summaries, since it's cheap and the underlying items can grow
between runs.

Usage:
    python -m src.transcription.build_summary                                # both new model folders
    python -m src.transcription.build_summary wav2vec2                       # just one
"""
import argparse
import json
import logging
import re
from pathlib import Path

from src.utils.paths import data_path

VERSIONS = ["version_a", "version_b"]
DEFAULT_FOLDERS = ["wav2vec2", "parakeet"]

ITEM_RE = re.compile(r"^item_(\d+)\.json$")


def summarize_item(item_path: Path) -> dict:
    with open(item_path, encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    text = " ".join(seg["text"].strip() for seg in segments).strip()
    words = [w for seg in segments for w in seg.get("words", [])]
    word_probs = [w["prob"] for w in words if "prob" in w]

    index = int(ITEM_RE.match(item_path.name).group(1))

    return {
        "index": index,
        "duration": data.get("duration"),
        "language": data.get("language"),
        "language_probability": data.get("language_probability"),
        "word_count": len(words),
        "mean_word_prob": round(sum(word_probs) / len(word_probs), 4) if word_probs else None,
        "text": text,
        "empty": text == "",
    }


def build_summary(recording_dir: Path, version: str) -> None:
    item_paths = sorted(
        (p for p in recording_dir.glob("item_*.json") if ITEM_RE.match(p.name)),
        key=lambda p: int(ITEM_RE.match(p.name).group(1)),
    )
    items = [summarize_item(p) for p in item_paths]

    summary = {
        "recording": recording_dir.name,
        "version": version,
        "item_count": len(items),
        "items": items,
    }

    with open(recording_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def build_all(folder: str) -> None:
    for version in VERSIONS:
        version_dir = data_path("transcribed", folder, version)
        if not version_dir.exists():
            logging.warning(f"Folder not found: {version_dir}")
            continue
        recording_dirs = sorted(p for p in version_dir.iterdir() if p.is_dir())
        logging.info(f"{folder}/{version}: building summaries for {len(recording_dirs)} recording(s)")
        for recording_dir in recording_dirs:
            build_summary(recording_dir, version)
    logging.info(f"Done: {folder}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "folders", nargs="*", default=DEFAULT_FOLDERS,
        help="subfolder name(s) under data/transcribed/, e.g. wav2vec2 (default: both new model folders)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    for folder in args.folders:
        build_all(folder)


if __name__ == "__main__":
    main()
