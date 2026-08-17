"""
Remove files from data/segmented/v1/ that don't match the expected <name>_item<N>.mp3 pattern.
Dry-run by default; pass --delete to actually remove files.
"""
import re
import argparse
from src.utils.paths import data_path

VALID = re.compile(r"^(?:ATTENTION_)?(.+)_item\d+\.mp3$", re.IGNORECASE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true", help="Actually delete files (default is dry-run)")
    args = parser.parse_args()

    to_remove = []
    for version_folder in ["version_a", "version_b"]:
        folder = data_path("segmented", "v1", version_folder)
        if not folder.exists():
            print(f"Folder not found: {folder}")
            continue
        for f in sorted(folder.glob("*.mp3")):
            if not VALID.match(f.name):
                to_remove.append(f)

    if not to_remove:
        print("No files to remove.")
    else:
        print(f"{'Deleting' if args.delete else 'Would delete'} {len(to_remove)} file(s):")
        for f in to_remove:
            print(f"  {f}")
            if args.delete:
                f.unlink()
