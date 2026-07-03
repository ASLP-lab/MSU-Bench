#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
#
# 合并 VAD 与 ASR 时间戳, 按 4-6.5 分钟在静音缝隙处切分, 用 FFmpeg 导出.
# 输入: movie_list, vad_base, huoshan_base
# 输出: split_raw_wavs_result/<movie>/ (分段音频 + 对齐 JSON + total_time.json)
#

import argparse
import json
import os
import subprocess
import logging
import math
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

# =========================================================
# 日志配置
# =========================================================
logger = logging.getLogger(__name__)

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "step1_cut_audio_only.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# =========================================================
# 工具函数
# =========================================================
def parse_time_to_seconds(time_str):
    try:
        parts = time_str.split(':')
        if len(parts) == 3:
            return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 1000.0
        elif len(parts) == 4:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]) + int(parts[3]) / 1000.0
    except: 
        logger.info("不标准的时间格式")
        pass
    return 0.0

def format_seconds_to_time(seconds):
    seconds = max(0.0, seconds)
    m = int(seconds // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{m:02d}:{s:02d}:{ms:03d}"

# =========================================================
# FFmpeg 音频切分单元 (极速版)
# =========================================================
def run_audio_ffmpeg(task):
    start, duration, input_path, wav_out = task
    cmd = [
        'ffmpeg', '-y',
        '-ss', f"{start:.4f}",
        '-i', input_path,
        '-t', f"{duration:.4f}",
        '-vn', 
        '-c:a', 'pcm_s16le', 
        '-ar', '16000',
        '-ac', '1',
        wav_out
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.info(f"FFmpeg Error for {wav_out}: {e.stderr.decode()}")
        return f"FFmpeg Error for {wav_out}: {e.stderr.decode()}"
    except Exception as e:
        logger.info(str(e))
        return str(e)

# =========================================================
# 核心处理逻辑
# =========================================================
def merge_timestamps(vad_json_path, huoshan_json_path, output_merged_path):
    all_intervals = []
    if os.path.exists(vad_json_path):
        with open(vad_json_path, 'r', encoding='utf-8') as f:
            for item in json.load(f):
                s, e = parse_time_to_seconds(item.get('start_time')), parse_time_to_seconds(item.get('end_time'))
                if e > s: all_intervals.append((s, e))
    if os.path.exists(huoshan_json_path):
        with open(huoshan_json_path, 'r', encoding='utf-8') as f:
            for item in json.load(f):
                s, e = parse_time_to_seconds(item.get('start_time')), parse_time_to_seconds(item.get('end_time'))
                if e > s: all_intervals.append((s, e))

    if not all_intervals: return False
    all_intervals.sort(key=lambda x: x[0])

    merged = []
    curr_s, curr_e = all_intervals[0]
    for next_s, next_e in all_intervals[1:]:
        if next_s <= curr_e + 0.1: curr_e = max(curr_e, next_e)
        else:
            merged.append((curr_s, curr_e))
            curr_s, curr_e = next_s, next_e
    merged.append((curr_s, curr_e))

    final_json = [{
        "segment_id": f"seg_{idx+1:04d}",
        "start_time": format_seconds_to_time(s),
        "end_time": format_seconds_to_time(e),
        "start_sec": s, "end_sec": e
    } for idx, (s, e) in enumerate(merged)]

    os.makedirs(os.path.dirname(output_merged_path), exist_ok=True)
    with open(output_merged_path, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)
    return True

def process_movie_audio_splitting(merged_json_path, audio_input_path, output_root_dir):
    json_out_dir = os.path.join(output_root_dir, "part_json_vad_huoshan_only")
    wav_out_dir = os.path.join(output_root_dir, "audio_segments_vad_huoshan_only")
    os.makedirs(json_out_dir, exist_ok=True)
    os.makedirs(wav_out_dir, exist_ok=True)

    with open(merged_json_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)

    try:
        probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', audio_input_path]
        total_duration = float(subprocess.check_output(probe_cmd).strip())
    except:
        total_duration = segments[-1]['end_sec'] + 1.0

    # 严格切分参数
    MIN_D = 240.0   # 4分钟
    MAX_D = 390.0   # 6.5分钟

    chunks = []
    chunk_start = 0.0
    current_idx = 0
    total_seg = len(segments)

    while current_idx < total_seg:
        remaining_time = total_duration - chunk_start

        # 如果剩余时间不足以撑起一个高质量的新片段，直接包进最后一段
        if remaining_time < (MAX_D + 60.0):
            chunks.append({
                "start": chunk_start, 
                "end": total_duration, 
                "segments": segments[current_idx:]
            })
            break

        best_cut_idx = -1
        max_gap = -1.0

        # 在区间内寻找“最大静音缝隙”作为切分参考
        for temp_idx in range(current_idx, total_seg):
            seg = segments[temp_idx]
            curr_dur = seg['end_sec'] - chunk_start

            # 绝不让当前 chunk 的最后一个段的结束时间超过 MAX_D
            if curr_dur > MAX_D:
                break

            if curr_dur >= MIN_D:
                if temp_idx + 1 < total_seg:
                    # 计算当前段结束到下一段开始的缝隙
                    gap = segments[temp_idx + 1]['start_sec'] - seg['end_sec']
                    if gap > max_gap:
                        max_gap = gap
                        best_cut_idx = temp_idx
                else:
                    best_cut_idx = temp_idx

        # 确定物理切分点
        if best_cut_idx != -1:
            curr_seg_end = segments[best_cut_idx]['end_sec']
            next_seg_start = segments[best_cut_idx + 1]['start_sec'] if (best_cut_idx + 1 < total_seg) else total_duration

            # 重要：切在两个语音段的中间，确保不伤及语音
            cut_p = (curr_seg_end + next_seg_start) / 2.0

            # 纠偏：如果中值太往后导致超标，强行压缩到限制内（依然在缝隙里）
            if cut_p - chunk_start > MAX_D:
                cut_p = min(next_seg_start, chunk_start + MAX_D)

            chunk_segments = segments[current_idx : best_cut_idx + 1]
            current_idx = best_cut_idx + 1
        else:
            # 极端情况：如果 4-6.5分钟内连一个缝隙都没有（一直在说话）
            # 我们只能找最接近 MAX_D 的那个段的结束点切
            temp_idx = current_idx
            while temp_idx < total_seg and (segments[temp_idx]['end_sec'] - chunk_start) <= MAX_D:
                temp_idx += 1

            # 取上一个符合条件的段
            actual_idx = max(current_idx, temp_idx - 1)
            cut_p = segments[actual_idx]['end_sec']
            chunk_segments = segments[current_idx : actual_idx + 1]
            current_idx = actual_idx + 1

        chunks.append({"start": chunk_start, "end": cut_p, "segments": chunk_segments})
        chunk_start = cut_p

    # FFmpeg 任务准备
    tasks = []
    chunk_info = []
    for idx, chunk in enumerate(chunks):
        scene = f"part{idx+1:03d}"
        s, e = chunk["start"], chunk["end"]
        dur = e - s
        if dur < 0.1: continue

        chunk_info.append({"scene_id": scene, "total_duration": round(dur, 3), "start_time": round(s, 3), "end_time": round(e, 3)})

        aligned = [{"segment_id": f"seg_{i:03d}", "start_time": format_seconds_to_time(max(0, seg["start_sec"]-s)), "end_time": format_seconds_to_time(max(0, seg["end_sec"]-s))} 
                   for i, seg in enumerate(chunk["segments"], 1)]
        with open(os.path.join(json_out_dir, f"{scene}_vad_huoshan.json"), "w", encoding="utf-8") as f:
            json.dump(aligned, f, ensure_ascii=False, indent=2)

        wav_out = os.path.join(wav_out_dir, f"{scene}.wav")
        tasks.append((s, dur, audio_input_path, wav_out))

    with ProcessPoolExecutor(max_workers=min(os.cpu_count(), 32)) as executor:
        list(tqdm(executor.map(run_audio_ffmpeg, tasks), total=len(tasks), desc="  [Cutting Audio]", leave=False))

    with open(os.path.join(json_out_dir, "total_time.json"), "w", encoding="utf-8") as f:
        json.dump(chunk_info, f, ensure_ascii=False, indent=2)

# =========================================================
# SCP 读取工具
# =========================================================
def load_movies_from_scp(scp_file):
    movies = []
    with open(scp_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            audio_path = line
            movie = os.path.splitext(os.path.basename(audio_path))[0]

            if audio_path:
                movies.append((movie, audio_path))
    return movies

# =========================================================
# Main
# =========================================================
def main(args):
    global logger
    logger = setup_logging(args.log_dir)

    movies = load_movies_from_scp(args.movie_list_file)

    # failure log files
    failed_paths_file = os.path.join(args.log_dir, 'failed_paths.txt')
    failed_detailed_file = os.path.join(args.log_dir, 'failed_detailed.txt')

    def _append_failed(path, reason=None):
        ts = datetime.now().isoformat()
        try:
            # write path-only
            with open(failed_paths_file, 'a', encoding='utf-8') as fp:
                fp.write(f"{path}\n")
            # write detailed (path + reason)
            if reason is not None:
                with open(failed_detailed_file, 'a', encoding='utf-8') as fd:
                    fd.write(f"[{ts}] {path} || {reason}\n")
        except Exception as ex:
            logger.error(f"Failed to write failure records: {ex}")

    for movie, full_audio_path in tqdm(movies, desc="Total Movies"):
        try:
            if not os.path.exists(full_audio_path):
                _append_failed(full_audio_path, 'missing_audio')
                logger.warning(f"Missing audio {full_audio_path}; recorded failure")
                continue

            vad_json = os.path.join(args.vad_base, movie, f"{movie}.json")
            huoshan_json = os.path.join(args.huoshan_base, movie, f"{movie}.json")
            merged_json = os.path.join(args.vad_base, movie, f"{movie}_merged_for_audio.json")

            ok = merge_timestamps(vad_json, huoshan_json, merged_json)
            if not ok:
                _append_failed(full_audio_path, 'merge_timestamps_empty')
                logger.warning(f"merge_timestamps returned empty for {movie}; recorded failure")
                continue

            try:
                process_movie_audio_splitting(merged_json, full_audio_path, os.path.join(args.output_base, movie))
            except Exception as e:
                _append_failed(full_audio_path, f"process_exception: {e}")
                logger.error(f"Error processing {movie}: {e}")
        except Exception as e:
            _append_failed(full_audio_path if full_audio_path else movie, f"unexpected_exception: {e}")
            logger.error(f"Error {movie}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split raw wavs based on VAD and Huoshan timestamps")
    parser.add_argument("--movie_list_file", default="$SILERO_VAD_ROOT/movies.scp", help="Path to movies scp file")
    parser.add_argument("--vad_base", default="$DATA_ROOT/test/test_pipeline/vad_result", help="Base directory for VAD results")
    parser.add_argument("--huoshan_base", default="$DATA_ROOT/test/test_pipeline/huoshan_result", help="Base directory for Huoshan results")
    parser.add_argument("--output_base", default="$DATA_ROOT/test/test_pipeline/split_raw_wavs_result", help="Output root directory")
    parser.add_argument("--log_dir", default="$DATA_ROOT/test/test_pipeline/logs", help="Directory to store logs")

    args = parser.parse_args()
    main(args)