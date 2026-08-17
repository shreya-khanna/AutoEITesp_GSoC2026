"""
Build a probability-filtered summary.json variant for data/transcribed/v2/ (Whisper/faster-whisper),
keeping only words above a confidence threshold in each item's reconstructed text. This is a
candidate mitigation for insertion-heavy hallucinations: see notebooks/transcription_quality/analyze_transcription_quality.ipynb,
where low mean_word_prob correlated with insertion-heavy (WER > 1) items for this ASR source.

Word tokens from faster-whisper already carry their own leading space (e.g. " El", " carro"), so
concatenating the surviving words in order (skipping the low-confidence ones) reconstructs
sentence text without needing to re-insert separators.

Output is written as summary_filtered.json alongside each recording's existing summary.json --
the original is left untouched.

Usage:
    python -m src.transcription.build_summary_filtered                  # default threshold 0.6
    python -m src.transcription.build_summary_filtered --threshold 0.6
"""
import argparse
import json
import logging
import re
from pathlib import Path

from src.utils.paths import data_path

VERSIONS = ["version_a", "version_b"]
FOLDER = "v2"
ITEM_RE = re.compile(r"^item_(\d+)\.json$")


def summarize_item_filtered(item_path: Path, threshold: float) -> dict:
    with open(item_path, encoding="utf-8") as f:
        data = json.load(f)

    words = [w for seg in data.get("segments", []) for w in seg.get("words", [])]
    kept = [w for w in words if w.get("prob", 0.0) > threshold]

    text = "".join(w["word"] for w in kept).strip()
    kept_probs = [w["prob"] for w in kept]

    index = int(ITEM_RE.match(item_path.name).group(1))

    return {
        "index": index,
        "duration": data.get("duration"),
        "language": data.get("language"),
        "language_probability": data.get("language_probability"),
        "word_count": len(kept),
        "mean_word_prob": round(sum(kept_probs) / len(kept_probs), 4) if kept_probs else None,
        "text": text,
        "empty": text == "",
    }


def build_summary_filtered(recording_dir: Path, version: str, threshold: float) -> None:
    item_paths = sorted(
        (p for p in recording_dir.glob("item_*.json") if ITEM_RE.match(p.name)),
        key=lambda p: int(ITEM_RE.match(p.name).group(1)),
    )
    items = [summarize_item_filtered(p, threshold) for p in item_paths]

    summary = {
        "recording": recording_dir.name,
        "version": version,
        "item_count": len(items),
        "word_prob_threshold": threshold,
        "items": items,
    }

    with open(recording_dir / "summary_filtered.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def build_all(threshold: float) -> None:
    for version in VERSIONS:
        version_dir = data_path("transcribed", FOLDER, version)
        if not version_dir.exists():
            logging.warning(f"Folder not found: {version_dir}")
            continue
        recording_dirs = sorted(p for p in version_dir.iterdir() if p.is_dir())
        logging.info(f"{FOLDER}/{version}: building filtered summaries for {len(recording_dirs)} recording(s)")
        for recording_dir in recording_dirs:
            build_summary_filtered(recording_dir, version, threshold)
    logging.info(f"Done: {FOLDER} (threshold={threshold})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=float, default=0.6, help="minimum word prob to keep (default: 0.6)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_all(args.threshold)


if __name__ == "__main__":
    main()
