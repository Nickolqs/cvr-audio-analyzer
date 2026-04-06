#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CVR Analyzer GUI
Один файл з простим GUI на Tkinter.

ЗАПУСК:
    python cvr_analyzer_gui.py

ЩО РОБИТЬ:
- дозволяє вибрати аудіофайл через GUI
- дозволяє вибрати папку для результатів
- аналізує CVR-подібні аудіофайли
- будує графіки
- зберігає CSV / JSON / PNG

ПІДТРИМУЄ:
- WAV
- MP3
- M4A
- DAT
- BIN
- TRS

ВАЖЛИВО:
1) Для MP3/M4A і частини DAT/BIN/TRS бажано мати ffmpeg у PATH.
2) Для DAT/BIN/TRS інколи потрібно вмикати RAW-режим.
3) Якщо .trs — це текстовий/XML транскрипт, а не аудіо, скрипт це повідомить.

Скрипт може сам спробувати встановити потрібні бібліотеки.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ---------------------------------------------------------
# Глобальні змінні для зовнішніх бібліотек
# ---------------------------------------------------------
np = None
librosa = None
sf = None
plt = None
scipy_signal = None
FigureCanvasTkAgg = None
NavigationToolbar2Tk = None


# ---------------------------------------------------------
# Дані
# ---------------------------------------------------------
@dataclass
class AudioMeta:
    path: str
    sample_rate: int
    channels: int
    samples_per_channel: int
    duration_sec: float
    used_decoder: str


@dataclass
class EventRecord:
    start_sec: float
    end_sec: float
    duration_sec: float
    event_type: str
    max_score: float
    max_rms_db: float
    max_pitch_hz: Optional[float]
    max_centroid_hz: float


# ---------------------------------------------------------
# Перевірка / встановлення залежностей
# ---------------------------------------------------------
REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "librosa": "librosa",
    "soundfile": "soundfile",
    "matplotlib": "matplotlib",
}


def get_missing_packages() -> List[str]:
    missing = []
    for package_name, import_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(import_name)
        except Exception:
            missing.append(package_name)
    return missing


def require_dependencies() -> None:
    global np, librosa, sf, plt, scipy_signal, FigureCanvasTkAgg, NavigationToolbar2Tk

    missing = get_missing_packages()
    if missing:
        raise RuntimeError(
            "Не встановлені бібліотеки: "
            + ", ".join(missing)
            + "\nНатисни кнопку 'Перевірити / встановити бібліотеки' у програмі."
        )

    import numpy as _np
    import librosa as _librosa
    import soundfile as _sf
    import matplotlib

    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass

    import matplotlib.pyplot as _plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as _FigureCanvasTkAgg
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as _NavigationToolbar2Tk
    from scipy import signal as _scipy_signal

    np = _np
    librosa = _librosa
    sf = _sf
    plt = _plt
    scipy_signal = _scipy_signal
    FigureCanvasTkAgg = _FigureCanvasTkAgg
    NavigationToolbar2Tk = _NavigationToolbar2Tk

def pip_install(packages: List[str]) -> Tuple[bool, str]:
    if not packages:
        return True, "Усі пакети вже встановлені."

    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + packages
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False
        )
        ok = result.returncode == 0
        return ok, result.stdout
    except Exception as e:
        return False, str(e)


def environment_report() -> str:
    lines = []
    missing = get_missing_packages()
    if missing:
        lines.append("Відсутні Python-бібліотеки:")
        for item in missing:
            lines.append(f"  - {item}")
    else:
        lines.append("Усі потрібні Python-бібліотеки встановлені.")

    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    lines.append("")
    lines.append(f"ffmpeg:  {ffmpeg_path if ffmpeg_path else 'не знайдено'}")
    lines.append(f"ffprobe: {ffprobe_path if ffprobe_path else 'не знайдено'}")
    lines.append("")
    lines.append(f"Python: {sys.executable}")
    return "\n".join(lines)


# ---------------------------------------------------------
# Допоміжні функції
# ---------------------------------------------------------
def safe_stem(path: Union[str, Path]) -> str:
    return Path(path).stem.replace(" ", "_")


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None


def is_probably_text_or_xml(path: Path, bytes_to_read: int = 2048) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(bytes_to_read)
        if not chunk:
            return False
        text_chars = sum(1 for b in chunk if 9 <= b <= 13 or 32 <= b <= 126)
        ratio = text_chars / max(len(chunk), 1)
        lowered = chunk.lower()
        if ratio > 0.85:
            if b"<?xml" in lowered or b"<trans" in lowered or b"<trs" in lowered:
                return True
            if b"<" in lowered and b">" in lowered:
                return True
        return False
    except Exception:
        return False


def run_ffmpeg_decode_to_wav(input_path: Path) -> Optional[Path]:
    if not ffmpeg_exists():
        return None

    temp_dir = Path(tempfile.mkdtemp(prefix="cvr_decode_"))
    out_wav = temp_dir / f"{safe_stem(input_path)}_decoded.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        str(out_wav),
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode == 0 and out_wav.exists():
            return out_wav
        return None
    except Exception:
        return None


def normalize_audio(audio_2d: "np.ndarray") -> "np.ndarray":
    peak = np.max(np.abs(audio_2d))
    if peak <= 0:
        return audio_2d.astype(np.float32)
    return (audio_2d / peak).astype(np.float32)


def trim_silence_multichannel(audio_2d: "np.ndarray", sr: int, top_db: float = 30.0) -> "np.ndarray":
    if audio_2d.shape[0] == 1:
        y = audio_2d[0]
        yt, idx = librosa.effects.trim(y, top_db=top_db)
        return yt[np.newaxis, :]

    mix = np.mean(audio_2d, axis=0)
    _, idx = librosa.effects.trim(mix, top_db=top_db)
    start, end = idx
    return audio_2d[:, start:end]


def select_channel(audio_2d: "np.ndarray", channel_spec: str) -> Dict[str, "np.ndarray"]:
    n_channels = audio_2d.shape[0]

    if channel_spec == "mix":
        return {"mix": np.mean(audio_2d, axis=0).astype(np.float32)}

    if channel_spec == "all":
        return {f"ch{idx+1}": audio_2d[idx].astype(np.float32) for idx in range(n_channels)}

    try:
        ch = int(channel_spec)
        if ch < 1 or ch > n_channels:
            raise ValueError
        return {f"ch{ch}": audio_2d[ch - 1].astype(np.float32)}
    except ValueError:
        raise ValueError(f"Неправильне значення каналу: {channel_spec}. Використовуй mix, all або номер 1..{n_channels}.")


# ---------------------------------------------------------
# Завантаження аудіо
# ---------------------------------------------------------
def load_raw_pcm(
    file_path: Path,
    raw_sr: int,
    raw_channels: int,
    raw_width: int,
    raw_endian: str,
    raw_signed: bool = True,
) -> Tuple["np.ndarray", int, str]:
    if raw_width not in (1, 2, 4):
        raise ValueError("Підтримуються raw_width: 1, 2, 4")

    endian_prefix = "<" if raw_endian.lower() == "little" else ">"
    if raw_width == 1:
        dtype = np.int8 if raw_signed else np.uint8
    elif raw_width == 2:
        dtype = np.dtype(f"{endian_prefix}i2")
    else:
        dtype = np.dtype(f"{endian_prefix}i4")

    raw = np.fromfile(str(file_path), dtype=dtype)

    if raw_channels < 1:
        raise ValueError("raw_channels має бути >= 1")

    if len(raw) == 0:
        raise ValueError("Файл порожній або не вдалося зчитати raw PCM")

    usable_len = (len(raw) // raw_channels) * raw_channels
    raw = raw[:usable_len]

    if usable_len == 0:
        raise ValueError("Недостатньо даних для вказаної кількості каналів")

    audio = raw.reshape(-1, raw_channels).T.astype(np.float32)

    if raw_width == 1:
        if raw_signed:
            denom = 128.0
        else:
            audio = audio - 128.0
            denom = 128.0
    elif raw_width == 2:
        denom = 32768.0
    else:
        denom = 2147483648.0

    audio /= denom
    return audio, raw_sr, "raw_pcm"


def load_audio_any_format(
    file_path: Union[str, Path],
    target_sr: Optional[int] = 16000,
    normalize: bool = False,
    trim_silence: bool = False,
    raw: bool = False,
    raw_sr: int = 8000,
    raw_channels: int = 1,
    raw_width: int = 2,
    raw_endian: str = "little",
    raw_signed: bool = True,
) -> Tuple["np.ndarray", AudioMeta]:
    require_dependencies()

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не знайдено: {file_path}")

    if raw:
        audio_2d, sr, decoder = load_raw_pcm(
            file_path=file_path,
            raw_sr=raw_sr,
            raw_channels=raw_channels,
            raw_width=raw_width,
            raw_endian=raw_endian,
            raw_signed=raw_signed,
        )
    else:
        ext = file_path.suffix.lower()

        if ext == ".trs" and is_probably_text_or_xml(file_path):
            raise ValueError(
                f"Файл {file_path.name} схожий на текстовий/XML TRS-транскрипт, а не на аудіо.\n"
                f"Якщо це raw PCM export, увімкни RAW-режим."
            )

        audio_2d = None
        sr = None
        decoder = None

        try:
            data, sr = sf.read(str(file_path), always_2d=True, dtype="float32")
            audio_2d = data.T
            decoder = "soundfile"
        except Exception:
            pass

        if audio_2d is None:
            try:
                y, sr = librosa.load(str(file_path), sr=None, mono=False)
                if y.ndim == 1:
                    audio_2d = y[np.newaxis, :].astype(np.float32)
                else:
                    audio_2d = y.astype(np.float32)
                decoder = "librosa"
            except Exception:
                pass

        if audio_2d is None:
            decoded_wav = run_ffmpeg_decode_to_wav(file_path)
            if decoded_wav is not None:
                try:
                    data, sr = sf.read(str(decoded_wav), always_2d=True, dtype="float32")
                    audio_2d = data.T
                    decoder = "ffmpeg->wav"
                except Exception:
                    audio_2d = None

        if audio_2d is None or sr is None:
            raise RuntimeError(
                f"Не вдалося прочитати файл: {file_path}\n"
                f"Можливі причини:\n"
                f"- нестандартний формат\n"
                f"- відсутній ffmpeg\n"
                f"- це raw PCM export, для якого треба RAW-режим"
            )

    if audio_2d.ndim != 2:
        raise RuntimeError("Внутрішня помилка: аудіо повинно мати форму (channels, samples)")

    if target_sr is not None and sr != target_sr:
        resampled_channels = []
        for ch_idx in range(audio_2d.shape[0]):
            resampled = librosa.resample(
                y=audio_2d[ch_idx],
                orig_sr=sr,
                target_sr=target_sr
            )
            resampled_channels.append(resampled.astype(np.float32))
        min_len = min(len(ch) for ch in resampled_channels)
        audio_2d = np.vstack([ch[:min_len] for ch in resampled_channels]).astype(np.float32)
        sr = target_sr

    if normalize:
        audio_2d = normalize_audio(audio_2d)

    if trim_silence:
        audio_2d = trim_silence_multichannel(audio_2d, sr=sr)

    meta = AudioMeta(
        path=str(file_path),
        sample_rate=int(sr),
        channels=int(audio_2d.shape[0]),
        samples_per_channel=int(audio_2d.shape[1]),
        duration_sec=float(audio_2d.shape[1] / sr),
        used_decoder=decoder,
    )

    return audio_2d.astype(np.float32), meta


# ---------------------------------------------------------
# Ознаки
# ---------------------------------------------------------
def nan_if_invalid(arr: "np.ndarray") -> "np.ndarray":
    out = arr.astype(np.float32).copy()
    out[~np.isfinite(out)] = np.nan
    return out


def smooth_array(arr: "np.ndarray", kernel_size: int = 5) -> "np.ndarray":
    arr = nan_if_invalid(arr)
    if kernel_size < 3 or kernel_size % 2 == 0:
        return arr
    valid = np.isfinite(arr)
    if np.sum(valid) < kernel_size:
        return arr

    filled = arr.copy()
    median_value = np.nanmedian(arr)
    filled[~valid] = median_value
    smoothed = scipy_signal.medfilt(filled, kernel_size=kernel_size).astype(np.float32)
    smoothed[~valid] = np.nan
    return smoothed


def extract_features(
    y: "np.ndarray",
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    frame_length: int = 2048,
    fmin: float = 50.0,
    fmax: float = 1200.0,
    n_mfcc: int = 13,
) -> Dict[str, "np.ndarray"]:
    require_dependencies()

    if len(y) < frame_length:
        pad = frame_length - len(y)
        y = np.pad(y, (0, pad), mode="constant")

    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0].astype(np.float32)

    eps = 1e-10
    rms_db = librosa.amplitude_to_db(np.maximum(rms, eps), ref=np.max).astype(np.float32)

    try:
        f0 = librosa.yin(
            y=y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
        ).astype(np.float32)
    except Exception:
        f0 = np.full_like(rms, np.nan, dtype=np.float32)

    voiced_mask = rms_db > np.nanpercentile(rms_db, 20)
    f0_masked = f0.copy()
    f0_masked[~voiced_mask] = np.nan
    f0_masked = smooth_array(f0_masked, kernel_size=5)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    ).astype(np.float32)
    mfcc_1 = mfcc[0].astype(np.float32)

    spectral_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
    )[0].astype(np.float32)

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
    )[0].astype(np.float32)

    zcr = librosa.feature.zero_crossing_rate(
        y,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0].astype(np.float32)

    n_frames = len(rms)
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length).astype(np.float32)

    return {
        "times": times,
        "rms": rms,
        "rms_db": rms_db,
        "pitch_hz": f0_masked,
        "mfcc_1": mfcc_1,
        "spectral_centroid_hz": spectral_centroid,
        "spectral_bandwidth_hz": spectral_bandwidth,
        "zcr": zcr,
        "signal": y.astype(np.float32),
        "sample_rate": np.array([sr], dtype=np.int32),
    }


# ---------------------------------------------------------
# Події
# ---------------------------------------------------------
def choose_threshold(
    values: "np.ndarray",
    explicit_threshold: Optional[float],
    percentile: float,
    fallback: float,
) -> float:
    if explicit_threshold is not None:
        return float(explicit_threshold)

    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return fallback
    return float(np.nanpercentile(valid, percentile))


def detect_events(
    features: Dict[str, "np.ndarray"],
    rms_event_threshold_db: Optional[float] = None,
    pitch_event_threshold_hz: Optional[float] = None,
    centroid_event_threshold_hz: Optional[float] = None,
    min_event_duration_sec: float = 0.20,
    merge_gap_sec: float = 0.12,
) -> Tuple[List[EventRecord], Dict[str, float], "np.ndarray"]:
    times = features["times"]
    rms_db = features["rms_db"]
    pitch = features["pitch_hz"]
    centroid = features["spectral_centroid_hz"]

    if len(times) < 2:
        frame_dt = 0.01
    else:
        frame_dt = float(np.median(np.diff(times)))

    rms_thr = choose_threshold(rms_db, rms_event_threshold_db, percentile=85, fallback=-18.0)

    valid_pitch = pitch[np.isfinite(pitch)]
    if len(valid_pitch) > 0:
        auto_pitch_thr = max(250.0, float(np.nanpercentile(valid_pitch, 85)))
    else:
        auto_pitch_thr = 400.0
    pitch_thr = float(pitch_event_threshold_hz) if pitch_event_threshold_hz is not None else auto_pitch_thr

    centroid_thr = choose_threshold(centroid, centroid_event_threshold_hz, percentile=90, fallback=2500.0)

    event_codes = np.zeros_like(times, dtype=np.int32)
    event_scores = np.zeros_like(times, dtype=np.float32)

    baseline_rms = float(np.nanmedian(rms_db)) if np.any(np.isfinite(rms_db)) else -30.0
    baseline_centroid = float(np.nanmedian(centroid)) if np.any(np.isfinite(centroid)) else 1000.0

    for i in range(len(times)):
        labels = []

        if np.isfinite(rms_db[i]) and rms_db[i] >= rms_thr:
            labels.append("high_energy")

        if np.isfinite(pitch[i]) and pitch[i] >= pitch_thr:
            labels.append("high_pitch")

        if np.isfinite(centroid[i]) and centroid[i] >= centroid_thr and rms_db[i] >= baseline_rms:
            labels.append("sharp_spectral")

        if len(labels) >= 2:
            label = "combined"
            code = 4
        elif len(labels) == 1:
            label = labels[0]
            code = {
                "high_energy": 1,
                "high_pitch": 2,
                "sharp_spectral": 3,
            }[label]
        else:
            label = "none"
            code = 0

        score = 0.0
        if code != 0:
            rms_part = max(0.0, float(rms_db[i] - rms_thr)) / 20.0 if np.isfinite(rms_db[i]) else 0.0
            pitch_part = max(0.0, float(pitch[i] - pitch_thr)) / 500.0 if np.isfinite(pitch[i]) else 0.0
            cent_part = max(0.0, float(centroid[i] - centroid_thr)) / max(centroid_thr, 1.0) if np.isfinite(centroid[i]) else 0.0
            score = float(min(1.0, rms_part + pitch_part + cent_part))

        event_codes[i] = code
        event_scores[i] = score

    label_names = {
        1: "high_energy",
        2: "high_pitch",
        3: "sharp_spectral",
        4: "combined",
    }

    events: List[EventRecord] = []
    i = 0
    min_frames = max(1, int(round(min_event_duration_sec / max(frame_dt, 1e-6))))
    merge_gap_frames = max(0, int(round(merge_gap_sec / max(frame_dt, 1e-6))))

    raw_segments = []
    while i < len(event_codes):
        code = int(event_codes[i])
        if code == 0:
            i += 1
            continue

        start = i
        current_code = code
        while i + 1 < len(event_codes) and int(event_codes[i + 1]) == current_code:
            i += 1
        end = i
        raw_segments.append((start, end, current_code))
        i += 1

    merged_segments = []
    for seg in raw_segments:
        if not merged_segments:
            merged_segments.append(list(seg))
            continue

        prev = merged_segments[-1]
        if seg[2] == prev[2] and seg[0] - prev[1] - 1 <= merge_gap_frames:
            prev[1] = seg[1]
        else:
            merged_segments.append(list(seg))

    for start, end, code in merged_segments:
        if (end - start + 1) < min_frames:
            continue

        idx = slice(start, end + 1)
        label = label_names.get(int(code), "unknown")
        max_pitch = float(np.nanmax(pitch[idx])) if np.any(np.isfinite(pitch[idx])) else None

        events.append(
            EventRecord(
                start_sec=float(times[start]),
                end_sec=float(times[end]),
                duration_sec=float(times[end] - times[start] + frame_dt),
                event_type=label,
                max_score=float(np.nanmax(event_scores[idx])) if len(event_scores[idx]) else 0.0,
                max_rms_db=float(np.nanmax(rms_db[idx])) if np.any(np.isfinite(rms_db[idx])) else float("nan"),
                max_pitch_hz=max_pitch,
                max_centroid_hz=float(np.nanmax(centroid[idx])) if np.any(np.isfinite(centroid[idx])) else float("nan"),
            )
        )

    thresholds = {
        "rms_event_threshold_db": float(rms_thr),
        "pitch_event_threshold_hz": float(pitch_thr),
        "centroid_event_threshold_hz": float(centroid_thr),
        "baseline_rms_db": float(baseline_rms),
        "baseline_centroid_hz": float(baseline_centroid),
        "frame_dt_sec": float(frame_dt),
    }

    return events, thresholds, event_codes


# ---------------------------------------------------------
# Збереження
# ---------------------------------------------------------
def save_features_csv(path: Path, features: Dict[str, "np.ndarray"]) -> None:
    fieldnames = [
        "time_sec",
        "rms",
        "rms_db",
        "pitch_hz",
        "mfcc_1",
        "spectral_centroid_hz",
        "spectral_bandwidth_hz",
        "zcr",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        n = len(features["times"])
        for i in range(n):
            writer.writerow({
                "time_sec": float(features["times"][i]),
                "rms": float(features["rms"][i]),
                "rms_db": float(features["rms_db"][i]),
                "pitch_hz": "" if not np.isfinite(features["pitch_hz"][i]) else float(features["pitch_hz"][i]),
                "mfcc_1": float(features["mfcc_1"][i]),
                "spectral_centroid_hz": float(features["spectral_centroid_hz"][i]),
                "spectral_bandwidth_hz": float(features["spectral_bandwidth_hz"][i]),
                "zcr": float(features["zcr"][i]),
            })


def save_events_csv(path: Path, events: List[EventRecord]) -> None:
    fieldnames = [
        "start_sec",
        "end_sec",
        "duration_sec",
        "event_type",
        "max_score",
        "max_rms_db",
        "max_pitch_hz",
        "max_centroid_hz",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow(asdict(event))


def make_json_serializable(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(x) for x in obj]
    return obj


def save_summary_json(
    path: Path,
    meta: AudioMeta,
    analysis_label: str,
    thresholds: Dict[str, float],
    events: List[EventRecord],
    features: Dict[str, "np.ndarray"],
) -> None:
    pitch = features["pitch_hz"]
    rms_db = features["rms_db"]
    centroid = features["spectral_centroid_hz"]

    summary = {
        "audio_meta": asdict(meta),
        "analysis_label": analysis_label,
        "thresholds": thresholds,
        "events_count": len(events),
        "feature_summary": {
            "rms_db_median": float(np.nanmedian(rms_db)) if np.any(np.isfinite(rms_db)) else None,
            "rms_db_max": float(np.nanmax(rms_db)) if np.any(np.isfinite(rms_db)) else None,
            "pitch_hz_median": float(np.nanmedian(pitch)) if np.any(np.isfinite(pitch)) else None,
            "pitch_hz_max": float(np.nanmax(pitch)) if np.any(np.isfinite(pitch)) else None,
            "centroid_hz_median": float(np.nanmedian(centroid)) if np.any(np.isfinite(centroid)) else None,
            "centroid_hz_max": float(np.nanmax(centroid)) if np.any(np.isfinite(centroid)) else None,
        },
        "events": [asdict(e) for e in events],
    }

    summary = make_json_serializable(summary)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------
# Графіки
# ---------------------------------------------------------

def build_analysis_figure(
    title: str,
    features: Dict[str, "np.ndarray"],
    events: List[EventRecord],
    event_codes: "np.ndarray",
):
    require_dependencies()

    times = features["times"]
    y = features["signal"]
    sr = int(features["sample_rate"][0])

    waveform_times = np.arange(len(y)) / sr

    rms_db = features["rms_db"]
    pitch = features["pitch_hz"]
    mfcc_1 = features["mfcc_1"]
    centroid = features["spectral_centroid_hz"]
    bandwidth = features["spectral_bandwidth_hz"]

    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=False)
    fig.suptitle(title, fontsize=15, fontweight="bold")

    ax = axes[0]
    ax.plot(waveform_times, y, linewidth=0.7)
    ax.set_title("Хвильова форма сигналу")
    ax.set_ylabel("Амплітуда")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(times, rms_db, label="RMS dB", linewidth=1.0)
    ax.set_title("Енергетичні та тональні характеристики")
    ax.set_ylabel("RMS, dB")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(times, pitch, label="Pitch (Hz)", linewidth=1.0, alpha=0.85)
    ax2.set_ylabel("Pitch, Hz")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax = axes[2]
    ax.plot(times, mfcc_1, label="MFCC[0]", linewidth=1.0)
    ax.set_title("MFCC та спектральні характеристики")
    ax.set_ylabel("MFCC[0]")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(times, centroid, label="Spectral centroid (Hz)", linewidth=1.0, alpha=0.85)
    ax2.plot(times, bandwidth, label="Bandwidth (Hz)", linewidth=1.0, alpha=0.75)
    ax2.set_ylabel("Hz")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax = axes[3]
    ax.plot(times, event_codes, linewidth=1.0, label="Event code")
    ax.set_title("Події")
    ax.set_xlabel("Час, с")
    ax.set_ylabel("Код події")
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["none", "energy", "pitch", "spectral", "combined"])
    ax.grid(True, alpha=0.3)

    color_map = {
        "high_energy": "tab:red",
        "high_pitch": "tab:blue",
        "sharp_spectral": "tab:orange",
        "combined": "tab:purple",
    }

    for event in events:
        color = color_map.get(event.event_type, "gray")
        ax.axvspan(event.start_sec, event.end_sec, alpha=0.20, color=color)

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    return fig

# ---------------------------------------------------------
# Аналіз одного каналу
# ---------------------------------------------------------

def analyze_one_signal(
    y: "np.ndarray",
    sr: int,
    out_dir: Path,
    base_name: str,
    label: str,
    meta: AudioMeta,
    params: dict,
):
    features = extract_features(
        y=y,
        sr=sr,
        n_fft=params["n_fft"],
        hop_length=params["hop_length"],
        frame_length=params["frame_length"],
        fmin=params["fmin"],
        fmax=params["fmax"],
        n_mfcc=params["n_mfcc"],
    )

    events, thresholds, event_codes = detect_events(
        features=features,
        rms_event_threshold_db=params["rms_event_threshold_db"],
        pitch_event_threshold_hz=params["pitch_event_threshold_hz"],
        centroid_event_threshold_hz=params["centroid_event_threshold_hz"],
        min_event_duration_sec=params["min_event_duration_sec"],
        merge_gap_sec=params["merge_gap_sec"],
    )

    fig = build_analysis_figure(
        title=f"Аналіз CVR-файлу: {base_name} [{label}]",
        features=features,
        events=events,
        event_codes=event_codes,
    )

    return {
        "label": label,
        "base_name": base_name,
        "meta": meta,
        "features": features,
        "events": events,
        "thresholds": thresholds,
        "event_codes": event_codes,
        "figure": fig,
    }

# ---------------------------------------------------------
# GUI
# ---------------------------------------------------------
class CVRAnalyzerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CVR Audio Analyzer")
        self.root.geometry("1180x820")
        self.root.minsize(980, 640)

        self.input_file_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value=str(Path.cwd() / "out"))
        self.channel_var = tk.StringVar(value="mix")

        self.target_sr_var = tk.StringVar(value="16000")
        self.normalize_var = tk.BooleanVar(value=False)
        self.trim_var = tk.BooleanVar(value=False)

        self.raw_mode_var = tk.BooleanVar(value=False)
        self.raw_sr_var = tk.StringVar(value="8000")
        self.raw_channels_var = tk.StringVar(value="1")
        self.raw_width_var = tk.StringVar(value="2")
        self.raw_endian_var = tk.StringVar(value="little")
        self.raw_unsigned_var = tk.BooleanVar(value=False)

        self.rms_thr_var = tk.StringVar(value="")
        self.pitch_thr_var = tk.StringVar(value="")
        self.centroid_thr_var = tk.StringVar(value="")
        self.min_event_duration_var = tk.StringVar(value="0.20")
        self.merge_gap_var = tk.StringVar(value="0.12")

        self.n_fft_var = tk.StringVar(value="2048")
        self.hop_length_var = tk.StringVar(value="512")
        self.frame_length_var = tk.StringVar(value="2048")
        self.fmin_var = tk.StringVar(value="50")
        self.fmax_var = tk.StringVar(value="1200")
        self.n_mfcc_var = tk.StringVar(value="13")

        self.is_running = False

        self.analysis_cache = []
        self.current_canvas = None
        self.current_toolbar = None
        self.current_preview_key = tk.StringVar(value="")

        self.save_png_var = tk.BooleanVar(value=True)
        self.save_features_var = tk.BooleanVar(value=True)
        self.save_events_var = tk.BooleanVar(value=True)
        self.save_summary_var = tk.BooleanVar(value=True)

        self._build_ui()
        self._log("Програма готова.")
        self._log(environment_report())

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        file_frame = ttk.LabelFrame(main, text="Файли", padding=10)
        file_frame.pack(fill="x", pady=(0, 8))

        ttk.Label(file_frame, text="Вхідний файл:").grid(row=0, column=0, sticky="w")
        ttk.Entry(file_frame, textvariable=self.input_file_var, width=85).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(file_frame, text="Обрати файл", command=self.choose_input_file).grid(row=0, column=2, padx=4)

        ttk.Label(file_frame, text="Папка результатів:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(file_frame, textvariable=self.output_dir_var, width=85).grid(row=1, column=1, sticky="ew", padx=6, pady=(8, 0))
        ttk.Button(file_frame, text="Обрати папку", command=self.choose_output_dir).grid(row=1, column=2, padx=4, pady=(8, 0))

        file_frame.columnconfigure(1, weight=1)

        params_frame = ttk.LabelFrame(main, text="Параметри аналізу", padding=8)
        params_frame.pack(fill="x", pady=(0, 8))

        params_notebook = ttk.Notebook(params_frame)
        params_notebook.pack(fill="x", expand=False)

        basic_tab = ttk.Frame(params_notebook, padding=10)
        raw_tab = ttk.Frame(params_notebook, padding=10)
        dsp_tab = ttk.Frame(params_notebook, padding=10)
        event_tab = ttk.Frame(params_notebook, padding=10)

        params_notebook.add(basic_tab, text="Основні")
        params_notebook.add(raw_tab, text="RAW")
        params_notebook.add(dsp_tab, text="DSP")
        params_notebook.add(event_tab, text="Події")

        ttk.Label(basic_tab, text="Канал:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            basic_tab,
            textvariable=self.channel_var,
            values=["mix", "all", "1", "2", "3", "4"],
            width=8,
            state="readonly"
        ).grid(row=0, column=1, sticky="w", padx=(6, 18))

        ttk.Label(basic_tab, text="Target SR:").grid(row=0, column=2, sticky="w")
        ttk.Entry(basic_tab, textvariable=self.target_sr_var, width=10).grid(row=0, column=3, sticky="w", padx=(6, 18))

        ttk.Checkbutton(basic_tab, text="Нормалізація", variable=self.normalize_var).grid(row=0, column=4, sticky="w", padx=(0, 18))
        ttk.Checkbutton(basic_tab, text="Підрізати тишу", variable=self.trim_var).grid(row=0, column=5, sticky="w")

        ttk.Checkbutton(raw_tab, text="Увімкнути RAW-режим", variable=self.raw_mode_var).grid(row=0, column=0, sticky="w", columnspan=6, pady=(0, 8))

        ttk.Label(raw_tab, text="Sample rate:").grid(row=1, column=0, sticky="w")
        ttk.Entry(raw_tab, textvariable=self.raw_sr_var, width=10).grid(row=1, column=1, sticky="w", padx=(6, 18))

        ttk.Label(raw_tab, text="Канали:").grid(row=1, column=2, sticky="w")
        ttk.Entry(raw_tab, textvariable=self.raw_channels_var, width=10).grid(row=1, column=3, sticky="w", padx=(6, 18))

        ttk.Label(raw_tab, text="Байт/семпл:").grid(row=1, column=4, sticky="w")
        ttk.Combobox(
            raw_tab,
            textvariable=self.raw_width_var,
            values=["1", "2", "4"],
            width=8,
            state="readonly"
        ).grid(row=1, column=5, sticky="w", padx=(6, 18))

        ttk.Label(raw_tab, text="Endian:").grid(row=1, column=6, sticky="w")
        ttk.Combobox(
            raw_tab,
            textvariable=self.raw_endian_var,
            values=["little", "big"],
            width=8,
            state="readonly"
        ).grid(row=1, column=7, sticky="w", padx=(6, 18))

        ttk.Checkbutton(raw_tab, text="Unsigned", variable=self.raw_unsigned_var).grid(row=1, column=8, sticky="w")

        ttk.Label(dsp_tab, text="n_fft").grid(row=0, column=0, sticky="w")
        ttk.Entry(dsp_tab, textvariable=self.n_fft_var, width=10).grid(row=0, column=1, padx=(6, 18), sticky="w")

        ttk.Label(dsp_tab, text="hop_length").grid(row=0, column=2, sticky="w")
        ttk.Entry(dsp_tab, textvariable=self.hop_length_var, width=10).grid(row=0, column=3, padx=(6, 18), sticky="w")

        ttk.Label(dsp_tab, text="frame_length").grid(row=0, column=4, sticky="w")
        ttk.Entry(dsp_tab, textvariable=self.frame_length_var, width=10).grid(row=0, column=5, padx=(6, 18), sticky="w")

        ttk.Label(dsp_tab, text="fmin").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(dsp_tab, textvariable=self.fmin_var, width=10).grid(row=1, column=1, padx=(6, 18), sticky="w", pady=(8, 0))

        ttk.Label(dsp_tab, text="fmax").grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(dsp_tab, textvariable=self.fmax_var, width=10).grid(row=1, column=3, padx=(6, 18), sticky="w", pady=(8, 0))

        ttk.Label(dsp_tab, text="n_mfcc").grid(row=1, column=4, sticky="w", pady=(8, 0))
        ttk.Entry(dsp_tab, textvariable=self.n_mfcc_var, width=10).grid(row=1, column=5, padx=(6, 18), sticky="w", pady=(8, 0))

        ttk.Label(event_tab, text="RMS threshold dB").grid(row=0, column=0, sticky="w")
        ttk.Entry(event_tab, textvariable=self.rms_thr_var, width=10).grid(row=0, column=1, padx=(6, 18), sticky="w")

        ttk.Label(event_tab, text="Pitch threshold Hz").grid(row=0, column=2, sticky="w")
        ttk.Entry(event_tab, textvariable=self.pitch_thr_var, width=10).grid(row=0, column=3, padx=(6, 18), sticky="w")

        ttk.Label(event_tab, text="Centroid threshold Hz").grid(row=0, column=4, sticky="w")
        ttk.Entry(event_tab, textvariable=self.centroid_thr_var, width=10).grid(row=0, column=5, padx=(6, 18), sticky="w")

        ttk.Label(event_tab, text="Min event sec").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(event_tab, textvariable=self.min_event_duration_var, width=10).grid(row=1, column=1, padx=(6, 18), sticky="w", pady=(8, 0))

        ttk.Label(event_tab, text="Merge gap sec").grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(event_tab, textvariable=self.merge_gap_var, width=10).grid(row=1, column=3, padx=(6, 18), sticky="w", pady=(8, 0))
        control_frame = ttk.Frame(main)
        control_frame.pack(fill="x", pady=(0, 8))

        ttk.Button(control_frame, text="Перевірити / встановити бібліотеки", command=self.install_or_check_packages).pack(side="left")
        ttk.Button(control_frame, text="Аналізувати файл", command=self.start_analysis).pack(side="left", padx=8)
        ttk.Button(control_frame, text="Відкрити папку результатів", command=self.open_output_dir).pack(side="left", padx=8)

        self.progress = ttk.Progressbar(control_frame, mode="indeterminate")
        self.progress.pack(side="right", fill="x", expand=True)

        save_frame = ttk.LabelFrame(main, text="Перегляд і збереження результатів", padding=10)
        save_frame.pack(fill="x", pady=(0, 8))

        ttk.Label(save_frame, text="Показати графік:").grid(row=0, column=0, sticky="w")
        self.preview_combo = ttk.Combobox(
            save_frame,
            textvariable=self.current_preview_key,
            values=[],
            width=18,
            state="readonly"
        )
        self.preview_combo.grid(row=0, column=1, sticky="w", padx=(6, 18))
        self.preview_combo.bind("<<ComboboxSelected>>", self.on_preview_selected)

        ttk.Checkbutton(save_frame, text="PNG графік", variable=self.save_png_var).grid(row=0, column=2, sticky="w", padx=(0, 12))
        ttk.Checkbutton(save_frame, text="CSV ознаки", variable=self.save_features_var).grid(row=0, column=3, sticky="w", padx=(0, 12))
        ttk.Checkbutton(save_frame, text="CSV події", variable=self.save_events_var).grid(row=0, column=4, sticky="w", padx=(0, 12))
        ttk.Checkbutton(save_frame, text="JSON summary", variable=self.save_summary_var).grid(row=0, column=5, sticky="w", padx=(0, 12))

        ttk.Button(save_frame, text="Зберегти вибране", command=self.save_selected_results).grid(row=0, column=6, sticky="e")

        preview_frame = ttk.LabelFrame(main, text="Графік", padding=8)
        preview_frame.pack(fill="both", expand=True, pady=(0, 6))

        self.preview_container = ttk.Frame(preview_frame)
        self.preview_container.pack(fill="both", expand=True)

        log_frame = ttk.LabelFrame(main, text="Журнал", padding=8)
        log_frame.pack(fill="x")

        self.log_text = tk.Text(log_frame, wrap="word", height=5)
        self.log_text.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=scrollbar.set)

    def _log(self, text: str):
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.root.update_idletasks()
        
    def render_figure(self, fig):
        require_dependencies()

        for widget in self.preview_container.winfo_children():
            widget.destroy()

        self.current_canvas = FigureCanvasTkAgg(fig, master=self.preview_container)
        self.current_canvas.draw()

        self.current_toolbar = NavigationToolbar2Tk(self.current_canvas, self.preview_container, pack_toolbar=False)
        self.current_toolbar.update()
        self.current_toolbar.pack(side="top", fill="x")

        self.current_canvas.get_tk_widget().pack(fill="both", expand=True)


    def on_preview_selected(self, event=None):
        key = self.current_preview_key.get().strip()
        if not key:
            return

        for item in self.analysis_cache:
            if item["label"] == key:
                self.render_figure(item["figure"])
                self._log(f"Показано графік: {key}")
                break


    def save_selected_results(self):
        if not self.analysis_cache:
            messagebox.showwarning("Увага", "Немає результатів аналізу для збереження.")
            return

        out_dir = ensure_dir(self.output_dir_var.get().strip())

        saved_any = False
        saved_lines = []

        for item in self.analysis_cache:
            label = item["label"]
            base_name = item["base_name"]
            meta = item["meta"]
            features = item["features"]
            events = item["events"]
            thresholds = item["thresholds"]
            fig = item["figure"]

            if self.save_features_var.get():
                out_features = out_dir / f"{base_name}_{label}_features.csv"
                save_features_csv(out_features, features)
                saved_any = True
                saved_lines.append(f"CSV ознаки: {out_features}")

            if self.save_events_var.get():
                out_events = out_dir / f"{base_name}_{label}_events.csv"
                save_events_csv(out_events, events)
                saved_any = True
                saved_lines.append(f"CSV події: {out_events}")

            if self.save_summary_var.get():
                out_summary = out_dir / f"{base_name}_{label}_summary.json"
                save_summary_json(out_summary, meta, label, thresholds, events, features)
                saved_any = True
                saved_lines.append(f"JSON summary: {out_summary}")

            if self.save_png_var.get():
                out_png = out_dir / f"{base_name}_{label}_analysis.png"
                fig.savefig(out_png, dpi=150)
                saved_any = True
                saved_lines.append(f"PNG графік: {out_png}")

        if not saved_any:
            messagebox.showwarning("Увага", "Не вибрано жодного типу файлів для збереження.")
            return

        self._log("Збережено файли:")
        for line in saved_lines:
            self._log("  " + line)

        messagebox.showinfo("Готово", "Вибрані результати збережено.")    

    def choose_input_file(self):
        path = filedialog.askopenfilename(
            title="Обрати CVR-файл",
            filetypes=[
                ("Audio / CVR files", "*.wav *.mp3 *.m4a *.dat *.bin *.trs"),
                ("WAV", "*.wav"),
                ("MP3", "*.mp3"),
                ("M4A", "*.m4a"),
                ("DAT", "*.dat"),
                ("BIN", "*.bin"),
                ("TRS", "*.trs"),
                ("All files", "*.*"),
            ]
        )
        if path:
            self.input_file_var.set(path)
            self._log(f"Обрано файл: {path}")

    def choose_output_dir(self):
        path = filedialog.askdirectory(title="Обрати папку результатів")
        if path:
            self.output_dir_var.set(path)
            self._log(f"Папка результатів: {path}")

    def open_output_dir(self):
        out_dir = Path(self.output_dir_var.get().strip())
        ensure_dir(out_dir)
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(out_dir))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(out_dir)])
            else:
                subprocess.Popen(["xdg-open", str(out_dir)])
        except Exception as e:
            messagebox.showerror("Помилка", f"Не вдалося відкрити папку:\n{e}")

    def install_or_check_packages(self):
        report = environment_report()
        self._log(report)

        missing = get_missing_packages()
        if not missing:
            messagebox.showinfo("Готово", "Усі потрібні Python-бібліотеки вже встановлені.")
            return

        answer = messagebox.askyesno(
            "Встановити бібліотеки",
            "Відсутні бібліотеки:\n\n"
            + "\n".join(missing)
            + "\n\nСпробувати встановити автоматично?"
        )
        if not answer:
            return

        self._log("Починаю встановлення бібліотек...")
        self.progress.start(10)

        def worker():
            ok, output = pip_install(missing)
            self.root.after(0, lambda: self.progress.stop())
            self.root.after(0, lambda: self._log(output))
            if ok:
                self.root.after(0, lambda: messagebox.showinfo("Готово", "Бібліотеки встановлено успішно."))
                self.root.after(0, lambda: self._log("Встановлення завершено успішно."))
            else:
                self.root.after(0, lambda: messagebox.showerror("Помилка", "Не вдалося встановити бібліотеки. Дивись журнал."))
                self.root.after(0, lambda: self._log("Встановлення завершилось з помилкою."))

        threading.Thread(target=worker, daemon=True).start()

    def _get_float_or_none(self, value: str) -> Optional[float]:
        value = value.strip()
        if value == "":
            return None
        return float(value)

    def _collect_params(self) -> dict:
        return {
            "target_sr": int(self.target_sr_var.get().strip()),
            "normalize": bool(self.normalize_var.get()),
            "trim_silence": bool(self.trim_var.get()),
            "channel": self.channel_var.get().strip(),

            "raw": bool(self.raw_mode_var.get()),
            "raw_sr": int(self.raw_sr_var.get().strip()),
            "raw_channels": int(self.raw_channels_var.get().strip()),
            "raw_width": int(self.raw_width_var.get().strip()),
            "raw_endian": self.raw_endian_var.get().strip(),
            "raw_signed": not bool(self.raw_unsigned_var.get()),

            "n_fft": int(self.n_fft_var.get().strip()),
            "hop_length": int(self.hop_length_var.get().strip()),
            "frame_length": int(self.frame_length_var.get().strip()),
            "fmin": float(self.fmin_var.get().strip()),
            "fmax": float(self.fmax_var.get().strip()),
            "n_mfcc": int(self.n_mfcc_var.get().strip()),

            "rms_event_threshold_db": self._get_float_or_none(self.rms_thr_var.get()),
            "pitch_event_threshold_hz": self._get_float_or_none(self.pitch_thr_var.get()),
            "centroid_event_threshold_hz": self._get_float_or_none(self.centroid_thr_var.get()),
            "min_event_duration_sec": float(self.min_event_duration_var.get().strip()),
            "merge_gap_sec": float(self.merge_gap_var.get().strip()),
        }

    def start_analysis(self):
        if self.is_running:
            return

        input_file = self.input_file_var.get().strip()
        output_dir = self.output_dir_var.get().strip()

        if not input_file:
            messagebox.showwarning("Увага", "Обери вхідний файл.")
            return

        if not Path(input_file).exists():
            messagebox.showerror("Помилка", "Вхідний файл не існує.")
            return

        try:
            params = self._collect_params()
        except Exception as e:
            messagebox.showerror("Помилка параметрів", f"Неправильні параметри:\n{e}")
            return

        self.is_running = True
        self.progress.start(10)
        self._log("=" * 70)
        self._log("Починаю аналіз...")

        def worker():
            try:
                missing = get_missing_packages()
                if missing:
                    raise RuntimeError(
                        "Відсутні бібліотеки: " + ", ".join(missing) +
                        "\nНатисни 'Перевірити / встановити бібліотеки'."
                    )

                out_dir = ensure_dir(output_dir)

                audio_2d, meta = load_audio_any_format(
                    file_path=input_file,
                    target_sr=params["target_sr"],
                    normalize=params["normalize"],
                    trim_silence=params["trim_silence"],
                    raw=params["raw"],
                    raw_sr=params["raw_sr"],
                    raw_channels=params["raw_channels"],
                    raw_width=params["raw_width"],
                    raw_endian=params["raw_endian"],
                    raw_signed=params["raw_signed"],
                )

                selected = select_channel(audio_2d, params["channel"])
                base_name = safe_stem(input_file)

                logs = []
                logs.append("Аудіо успішно завантажено:")
                logs.append(f"  Файл:          {meta.path}")
                logs.append(f"  Decoder:       {meta.used_decoder}")
                logs.append(f"  Sample rate:   {meta.sample_rate}")
                logs.append(f"  Канали:        {meta.channels}")
                logs.append(f"  Семплів/канал: {meta.samples_per_channel}")
                logs.append(f"  Тривалість, с: {meta.duration_sec:.2f}")

                all_results = []
                for label, signal_1d in selected.items():
                    logs.append("")
                    logs.append(f"Аналіз: {label}")
                    result_data = analyze_one_signal(
                        y=signal_1d,
                        sr=meta.sample_rate,
                        out_dir=out_dir,
                        base_name=base_name,
                        label=label,
                        meta=meta,
                        params=params,
                    )
                    all_results.append(result_data)

                def on_success():
                    self.progress.stop()
                    self.is_running = False
                    self.analysis_cache = all_results

                    for line in logs:
                        self._log(line)

                    if self.analysis_cache:
                        labels = [item["label"] for item in self.analysis_cache]
                        self.preview_combo["values"] = labels
                        self.current_preview_key.set(labels[0])
                        self.render_figure(self.analysis_cache[0]["figure"])
                        self._log("")
                        self._log(f"Графік виведено у вікні програми: {labels[0]}")
                        self._log("Щоб зберегти результати, обери потрібні формати і натисни 'Зберегти вибране'.")

                    messagebox.showinfo("Готово", "Аналіз завершено. Графік показано у вікні програми.")

                self.root.after(0, on_success)

            except Exception as e:
                tb = traceback.format_exc()

                def on_error():
                    self.progress.stop()
                    self.is_running = False
                    self._log("ПОМИЛКА:")
                    self._log(str(e))
                    self._log(tb)
                    messagebox.showerror("Помилка аналізу", str(e))

                self.root.after(0, on_error)

        threading.Thread(target=worker, daemon=True).start()


# ---------------------------------------------------------
# CLI режим
# ---------------------------------------------------------
def run_cli_check():
    print(environment_report())


# ---------------------------------------------------------
# Точка входу
# ---------------------------------------------------------
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        run_cli_check()
        return

    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass

    app = CVRAnalyzerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()