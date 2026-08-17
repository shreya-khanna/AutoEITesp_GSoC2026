"""
Re-segment the raw-recording Spanish-narrative window (data/raw/audio_files/{version}/
segment_window_timestamps.json, see src/preprocessing/build_raw_timestamps.py) with a different
VAD merge-gap threshold than the original full-file segmentation (2.5s), to check whether a
different pause threshold produces more consistent item counts.

For each recording with a computed window, only the audio between `beginning` and `end` is
searched for speech -- everything before/after (English instructions, etc.) is ignored. VAD +
merge-gap is applied locally to that cropped window (so local time 0 == `beginning`), then
re-exported with the same padding/ATTENTION_ conventions as the original
src/preprocessing/vad.py, using absolute raw-file time (window offset + local offset) so clips
are still cut from the correct place in the raw audio. New items are numbered from 0, independent
of the original global item indices.

Output goes to data/segmented/v2/{version}/ -- the original full-file segmentation was moved to
data/segmented/v1/{version}/ beforehand, so nothing here overwrites it.

Usage:
    python -m src.preprocessing.resegment_window                  # default merge_gap=4.0
    python -m src.preprocessing.resegment_window --merge-gap 4.0
"""
import argparse
import json
import logging
from pathlib import Path

import librosa
import torch
from pydub import AudioSegment
from silero_vad import get_speech_timestamps, load_silero_vad

from src.utils.paths import data_path

VERSIONS = ["version_a", "version_b"]
ATTENTION_THRESHOLD = 10.0


def compute_local_timestamps(wav_slice, model, sr: int, merge_gap: float) -> list[list[float]]:
    speech_timestamps = get_speech_timestamps(wav_slice, model, return_seconds=True, sampling_rate=sr)
    if not speech_timestamps:
        return []

    timestamps = []
    start = speech_timestamps[0]["start"]
    end = speech_timestamps[0]["end"]
    for ts in speech_timestamps[1:]:
        s_cur, e_cur = ts["start"], ts["end"]
        if s_cur - end < merge_gap:
            end = e_cur
        else:
            timestamps.append([start, end])
            start, end = s_cur, e_cur
    timestamps.append([start, end])
    return timestamps


def resegment_recording(
    recording: str, window: dict, raw_path: Path, out_dir: Path, model, merge_gap: float
) -> int:
    beginning, end = window["beginning"], window["end"]

    wav, sr = librosa.load(str(raw_path), sr=16000, mono=True)
    start_sample = int(beginning * sr)
    end_sample = int(end * sr)
    wav_slice = torch.from_numpy(wav[start_sample:end_sample])

    local_timestamps = compute_local_timestamps(wav_slice, model, sr, merge_gap)
    if not local_timestamps:
        logging.warning(f"{recording}: no speech detected in window, skipping")
        return 0

    audio = AudioSegment.from_file(str(raw_path))

    for i, (s, e) in enumerate(local_timestamps):
        abs_s = beginning + s
        abs_e = beginning + e
        pref = ""
        if i == 0:
            clip = audio[abs_s * 1000: (abs_e + 1) * 1000]
        elif i == len(local_timestamps) - 1:
            clip = audio[(abs_s - 1) * 1000: abs_e * 1000]
        else:
            clip = audio[(abs_s - 1) * 1000: (abs_e + 1) * 1000]
        if e - s > ATTENTION_THRESHOLD:
            pref = "ATTENTION_"
            logging.warning(f"{recording}: item {i} is longer than {ATTENTION_THRESHOLD}s -- flagged")
        clip.export(str(out_dir / f"{pref}{recording}_item{i}.mp3"), format="mp3")

    return len(local_timestamps)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merge-gap", type=float, default=4.0, help="VAD merge-gap threshold in seconds (default: 4.0)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Loading VAD model")
    model = load_silero_vad()

    for version in VERSIONS:
        window_path = data_path("raw", "audio_files", version, "segment_window_timestamps.json")
        windows = json.loads(window_path.read_text(encoding="utf-8"))

        raw_dir = data_path("raw", "audio_files", version)
        out_dir = data_path("segmented", "v2", version)
        out_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"{version}: re-segmenting {len(windows)} recordings (merge_gap={args.merge_gap}s)")
        total_items = 0
        for recording, window in windows.items():
            raw_path = raw_dir / window["raw_audio_file"]
            if not raw_path.exists():
                logging.warning(f"{recording}: raw audio file {raw_path} not found, skipping")
                continue
            n = resegment_recording(recording, window, raw_path, out_dir, model, args.merge_gap)
            logging.info(f"{recording}: exported {n} items")
            total_items += n

        logging.info(f"{version}: done, {total_items} total items exported to {out_dir}")


if __name__ == "__main__":
    main()
