from .vad import run_vad
from silero_vad import load_silero_vad
from src.utils.paths import data_path
import logging

def get_files(folder: str):
    dir = data_path("raw", "audio_files", folder)

    files = list(dir.glob("*.mp3")) + list(dir.glob("*.MP3")) + list(dir.glob("*.wma")) + list(dir.glob("*.WMA"))

    return files

def segment_all():
        
    logging.info(f"Loading VAD model")
    model = load_silero_vad()
    for folder in ["version_a", "version_b"]:
        files = get_files(folder)
        logging.info(f"Found {len(files)} files in {folder}")
        for file in files:
            logging.info(f"Processing {file.name}")
            try:
                run_vad(filename=file.name, folder=folder, model=model)
                logging.info(f"Done: {file.name}")
            except Exception as e:
                logging.error(f"Failed on {file.name}: {e}")
    

def segment_file(filename: str, folder: str):
    logging.info(f"Loading VAD model")
    model = load_silero_vad()
    run_vad(filename=filename, folder=folder, model=model)
