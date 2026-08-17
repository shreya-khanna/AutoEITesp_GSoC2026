"""
Shared item-discovery helper for data/segmented/v2/ (see src/preprocessing/resegment_window.py).
Unlike data/segmented/v1/, which the other transcribe_*.py scripts address by (recording, global
item index) pairs resolved from data/transcribed/v1's boundary markers, data/segmented/v2/ is a
flat per-version directory of "{recording}_item{N}.mp3" clips with fresh, per-recording 0-based
indices -- every clip in the window is a transcription target, so no boundary resolution is
needed, just grouping by recording.
"""
import re
from pathlib import Path

from src.utils.paths import data_path

VERSIONS = ["version_a", "version_b"]
ITEM_RE = re.compile(r"^(?:ATTENTION_)?(.+)_item(\d+)\.mp3$")


def iter_resegmented_items(version: str) -> list[tuple[str, int, Path]]:
    """(recording, item_index, audio_path) for every data/segmented/v2/{version} clip, sorted by
    recording then index."""
    folder = data_path("segmented", "v2", version)
    items = []
    for p in folder.glob("*.mp3"):
        m = ITEM_RE.match(p.name)
        if not m:
            continue
        items.append((m.group(1), int(m.group(2)), p))
    items.sort(key=lambda t: (t[0], t[1]))
    return items
