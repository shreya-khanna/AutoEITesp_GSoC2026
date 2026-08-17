from pathlib import Path
from src.preprocessing.segment import segment_all, segment_file
from src.transcription.transcribe import load_model, run_single, run_all, run_for_recording
from src.utils.paths import data_path
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(data_path("logs", "pipeline.log")),
        logging.StreamHandler()
    ]
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    seg = parser.add_argument_group("segmentation")
    seg.add_argument("--segment-all", action="store_true")
    seg.add_argument("--segment-file", action="store_true")

    tr = parser.add_argument_group("transcription")
    mode = tr.add_mutually_exclusive_group()
    mode.add_argument("--transcribe-file", metavar="STEM",
                      help="Transcribe all segments of one recording, e.g. 038053_EITv-2A")
    mode.add_argument("--transcribe-all", action="store_true",
                      help="Transcribe all segmented files (skips already-done)")
    tr.add_argument("--folder", metavar="FOLDER",
                    help="Version folder for --transcribe-file (version_a or version_b)")
    tr.add_argument("--language", default="auto", metavar="LANG",
                    help="Language code for Whisper (default: auto-detect). Pass 'es' to force Spanish.")

    args = parser.parse_args()

    if args.segment_all:
        segment_all()

    if args.segment_file:
        segment_file(filename='038.068_EITv-1B.WMA', folder="version_b")

    if args.transcribe_file or args.transcribe_all:
        lang = None if args.language == "auto" else args.language
        model = load_model()
        if args.transcribe_file:
            if "/" in args.transcribe_file:
                folder, stem = args.transcribe_file.split("/", 1)
            elif args.folder:
                folder, stem = args.folder, args.transcribe_file
            else:
                parser.error("--transcribe-file requires either 'version_a/<stem>' format or --folder")
            run_for_recording(stem, folder, model, language=lang)
        else:
            run_all(model, language=lang)
