from silero_vad import get_speech_timestamps
from pydub import AudioSegment
from src.utils.paths import data_path
import logging
import torch
import librosa

def print_timestamp(start: float, end: float):
    def convert_time(time: float):
        mins = int(time // 60)
        secs = time % 60
        return f"{mins}:{secs:.0f}"
    print(f"start: {convert_time(start)}, end: {convert_time(end)}")

def compute_merged_timestamps(audio_path: str, model, merge_gap: float = 2.5) -> list[list[float]]:
    """Raw-file-relative [start, end] seconds per VAD-detected item, merging speech chunks
    separated by less than merge_gap seconds. Same logic run_vad uses to decide item
    boundaries, factored out so it can be reused without re-exporting segmented clips."""
    wav, sr = librosa.load(audio_path, sr=16000, mono=True)
    wav = torch.from_numpy(wav)

    speech_timestamps = get_speech_timestamps(wav, model, return_seconds=True, sampling_rate=16000)
    logging.info(f"Found {len(speech_timestamps)} speech segments")

    timestamps = []
    start = -1.
    end = -1.
    for i in range(len(speech_timestamps)):
        if start == -1.:
            start = speech_timestamps[i]['start']
            end = speech_timestamps[i]['end']
        else:
            s_cur = speech_timestamps[i]['start']
            e_cur = speech_timestamps[i]['end']
            if s_cur - end < merge_gap:
                end = e_cur
            else:
                timestamps.append([start, end])
                start = s_cur
                end = e_cur

    timestamps.append([start, end])
    logging.info(f"Merged into {len(timestamps)} EIT items")
    return timestamps


def run_vad(filename: str, folder: str, model):
    audiofile_dir = data_path()

    logging.info(f"Reading audio: {filename}")
    audio_path = f'{audiofile_dir}/raw/audio_files/{folder}/{filename}'
    timestamps = compute_merged_timestamps(audio_path, model)
    audio = AudioSegment.from_file(f'{audiofile_dir}/raw/audio_files/{folder}/{filename}')
    
    for i, [s, e] in enumerate(timestamps):
        pref = ""
        if i == 0:
            tmp_item = audio[s*1000 : (e + 1) * 1000]
        elif i == len(timestamps) - 1:
            tmp_item = audio[(s - 1) * 1000 : e * 1000]
        else:
            tmp_item = audio[(s - 1) * 1000 : (e + 1) * 1000]
        if e - s > 10.:
            pref = "ATTENTION_"
            logging.warning(f"Item {i} is longer than 10s — flagged with ATTENTION_")
        tmp_item.export(f'{audiofile_dir}/segmented/v1/{folder}/{pref}{filename.rsplit(".", 1)[0]}_item{i}.mp3', format='mp3')
   
    logging.info(f"Exported {len(timestamps)} segments for {filename}")