import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def get_data_root() -> Path:
    env_root = os.getenv("AUTOEIT_DATA_ROOT")
    if env_root:
        return Path(env_root)
    return Path(__file__).resolve().parents[2] / "data"

def data_path(*parts) -> Path:
    return get_data_root().joinpath(*parts)