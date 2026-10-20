"""Download and load the UCI Human Activity Recognition (HAR) dataset.

The dataset: 30 volunteers wore a smartphone on the waist and performed six
activities. Signals from the accelerometer and gyroscope were sampled at 50 Hz
and cut into 2.56 s windows (128 samples) with 50% overlap.

Two views of the same windows are loaded:
  * engineered features: 561 hand-crafted time/frequency features per window
  * raw signals: 9 channels x 128 timesteps per window (for the deep model)

Every window carries the ID of the person who produced it. That ID is what
makes an honest evaluation possible (see THEORY.md, "Leakage").
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATASET_DIR = DATA_DIR / "UCI HAR Dataset"

URLS = [
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
]

ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

RAW_CHANNELS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


@dataclass
class HARData:
    """All windows from the official train and test splits, combined.

    Combining them lets us run our own subject-grouped cross-validation over
    all 30 people. `split` keeps the official assignment for the holdout test.
    """

    features: pd.DataFrame  # (n_windows, 561)
    raw: np.ndarray  # (n_windows, 9, 128) float32
    y: np.ndarray  # (n_windows,) int labels 0..5
    subject: np.ndarray  # (n_windows,) int subject IDs 1..30
    split: np.ndarray  # (n_windows,) "train" / "test" (official split)

    @property
    def activity_names(self) -> list[str]:
        return ACTIVITIES


def download(force: bool = False) -> Path:
    """Fetch and unzip the dataset into data/ (skipped if already present)."""
    if DATASET_DIR.exists() and not force:
        return DATASET_DIR
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in URLS:
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                payload = resp.read()
            break
        except Exception as exc:  # try the next mirror
            last_error = exc
    else:
        raise RuntimeError(f"Could not download the HAR dataset: {last_error}")

    archive = zipfile.ZipFile(io.BytesIO(payload))
    # The newer UCI download wraps the original zip inside another zip.
    inner = [n for n in archive.namelist() if n.endswith("UCI HAR Dataset.zip")]
    if inner:
        archive = zipfile.ZipFile(io.BytesIO(archive.read(inner[0])))
    archive.extractall(DATA_DIR)
    if not DATASET_DIR.exists():
        raise RuntimeError(f"Unexpected archive layout; {DATASET_DIR} not found")
    return DATASET_DIR


def _read_matrix(path: Path) -> np.ndarray:
    return pd.read_csv(path, sep=r"\s+", header=None).to_numpy(dtype=np.float32)


def _feature_names(root: Path) -> list[str]:
    names = pd.read_csv(root / "features.txt", sep=r"\s+", header=None)[1].tolist()
    # A few feature names repeat in the original file; make them unique.
    seen: dict[str, int] = {}
    unique = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        unique.append(name if seen[name] == 1 else f"{name}__{seen[name]}")
    return unique


def _load_split(root: Path, split: str, names: list[str]):
    d = root / split
    features = pd.DataFrame(_read_matrix(d / f"X_{split}.txt"), columns=names)
    y = _read_matrix(d / f"y_{split}.txt").ravel().astype(int) - 1  # 1..6 -> 0..5
    subject = _read_matrix(d / f"subject_{split}.txt").ravel().astype(int)
    raw = np.stack(
        [_read_matrix(d / "Inertial Signals" / f"{ch}_{split}.txt") for ch in RAW_CHANNELS],
        axis=1,
    )
    return features, raw, y, subject


def load(root: Path | None = None) -> HARData:
    root = root or download()
    names = _feature_names(root)
    parts = {s: _load_split(root, s, names) for s in ("train", "test")}
    return HARData(
        features=pd.concat([parts["train"][0], parts["test"][0]], ignore_index=True),
        raw=np.concatenate([parts["train"][1], parts["test"][1]]).astype(np.float32),
        y=np.concatenate([parts["train"][2], parts["test"][2]]),
        subject=np.concatenate([parts["train"][3], parts["test"][3]]),
        split=np.array(["train"] * len(parts["train"][2]) + ["test"] * len(parts["test"][2])),
    )
