# -*- coding: utf-8 -*-
#
# 使用 Silero VAD 模型检测语音段, 输出每个 movie 的 segments.json.
# 输入: scp 音频列表
# 输出: vad_result/<movie>/segments.json
# 依赖: torch, silero-vad 模型
#

import argparse
from pathlib import Path

import json
import torch
import traceback
from datetime import datetime


parser = argparse.ArgumentParser(description='Silero VAD timestamps generator')
parser.add_argument('--scp_path', '-s', type=str, required=True, help='scp file with one audio path per line')
parser.add_argument('--out_dir', '-o', type=str, required=True, help='Output root directory to store per-movie segments.json')
args = parser.parse_args()

# Model and utility handles will be initialized per worker to avoid forking issues
MODEL_REPO = '$MODEL_ROOT/hub/snakers4_silero-vad_master'

# Globals populated in worker_init
get_speech_timestamps = None
read_audio = None


def worker_init(repo_dir: str = MODEL_REPO):
    """Initialize model utilities inside worker process."""
    global get_speech_timestamps, read_audio, model
    import torch
    model, utils = torch.hub.load(
        repo_or_dir=repo_dir,
        model='silero_vad',
        source='local',
    )
    # ensure model is in eval mode and on CPU
    try:
        model.eval()
    except Exception:
        pass
    (get_speech_timestamps, _, read_audio, _, _) = utils


def format_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}:{millis:03d}"

scp_path = Path(args.scp_path)
out_root = Path(args.out_dir)
out_root.mkdir(parents=True, exist_ok=True)

with scp_path.open('r', encoding='utf-8') as f:
  audio_paths = [line.strip() for line in f if line.strip()]

import multiprocessing as mp
from functools import partial
from tqdm import tqdm


def _process_one(audio_path: str, out_root: Path):
    """Worker: read audio, compute speech timestamps, write segments.json."""

    movie_name = Path(audio_path).stem
    movie_dir = out_root / movie_name
    movie_dir.mkdir(parents=True, exist_ok=True)
    json_path = movie_dir / 'segments.json'
    if json_path.exists():
        return (audio_path, 0, f"segments.json already exists at {json_path}")
    
    try:
        if get_speech_timestamps is None or read_audio is None or model is None:
            raise RuntimeError("Worker not initialized: call worker_init in each worker process")
        wav = read_audio(audio_path)
        # call with the loaded model; if it fails due to model state, try to re-init once
        try:
            speech_timestamps = get_speech_timestamps(wav, model, return_seconds=True)
        except Exception as e_inner:
            # attempt one re-initialization and retry once
            try:
                worker_init(MODEL_REPO)
                speech_timestamps = get_speech_timestamps(wav, model, return_seconds=True)
            except Exception as e_retry:
                raise RuntimeError(f"get_speech_timestamps failed: {e_retry}") from e_retry
        segments = []
        for idx, ts in enumerate(speech_timestamps):
            segments.append({
                'segment_id': f"segment_{idx:04d}",
                'start_time': format_time(ts['start']),
                'end_time': format_time(ts['end']),
            })
        
        with json_path.open('w', encoding='utf-8') as jf:
            json.dump(segments, jf, ensure_ascii=False, indent=2)
        return (audio_path, len(segments), None)
    except Exception as e:
        tb = traceback.format_exc()
        return (audio_path, 0, tb)


# Parallel processing
workers = mp.cpu_count()
parser.add_argument('--workers', '-w', type=int, default=workers, help='Number of worker processes')
args = parser.parse_args()

scp_path = Path(args.scp_path)
out_root = Path(args.out_dir)
out_root.mkdir(parents=True, exist_ok=True)

with scp_path.open('r', encoding='utf-8') as f:
    audio_paths = [line.strip() for line in f if line.strip()]

print(f"Starting processing {len(audio_paths)} files with {args.workers} workers")

error_log_path = out_root / "step1_vad_error.log"
with mp.Pool(processes=args.workers, initializer=worker_init, initargs=(MODEL_REPO,)) as pool:
    func = partial(_process_one, out_root=out_root)
    results = []
    for audio_path, count, err in tqdm(pool.imap_unordered(func, audio_paths), total=len(audio_paths)):
        if err:
            print(f"FAIL {audio_path}: {err}")
            try:
                with open(error_log_path, 'a', encoding='utf-8') as ef:
                    ts = datetime.utcnow().isoformat()
                    ef.write(f"[{ts}] {audio_path}\n")
                    ef.write(err)
                    if not err.endswith('\n'):
                        ef.write('\n')
                    ef.write('-' * 80 + '\n')
            except Exception as e_write:
                print(f"Failed to write to error log: {e_write}")
        else:
            print(f"Saved {count} timestamps to {Path(audio_path).stem}/segments.json")
        results.append((audio_path, count, err))

# summary
succ = sum(1 for _, c, e in results if e is None)
fail = len(results) - succ
print(f"Done. success={succ}, failed={fail}")