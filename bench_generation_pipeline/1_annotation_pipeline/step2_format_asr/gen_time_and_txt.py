#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
#
# 将火山 ASR 的绝对时间戳裁剪到分段内, 转换为相对毫秒.
# 输入: total_time.json, huoshan ASR JSON
# 输出: asr_huoshan_formatted_txt/partXXX.wav.txt
#

import json
import os
import logging
from argparse import ArgumentParser
from tqdm import tqdm

# ================= 日志配置 =================
logger = logging.getLogger(__name__)


def configure_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "step2_huoshan_asr_for_reference.log")

    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, mode='a', encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

# ================= 时间转换工具 =================

def time_to_ms(val):
    """
    将各种格式的时间转为整数绝对毫秒
    1. 数字或数字字符串 (秒) -> ms
    2. 火山格式 "MM:SS:mmm" -> ms
    """
    if isinstance(val, (int, float)):
        return int(round(val * 1000))

    s_val = str(val).strip()
    if ":" in s_val:
        # 处理 MM:SS:mmm 格式
        parts = s_val.split(':')
        if len(parts) == 4:
            h, m, s, ms = map(int, parts)
            return ((h * 60 + m) * 60 + s) * 1000 + ms
        elif len(parts) == 3:
            m, s, ms = map(int, parts)
            return (m * 60 + s) * 1000 + ms
        elif len(parts) == 2:
            s, ms = map(int, parts)
            return s * 1000 + ms
    else:
        # 处理 "311.875" 这种字符串格式
        try:
            return int(round(float(s_val) * 1000))
        except:
            return 0
    return 0

# ================= 核心切分逻辑 =================

def process_movie_asr_segments(total_time_json_path, raw_huoshan_json_path, output_base_dir, movie_name):
    try:
        # 加载切分依据 (单位: 秒)
        with open(total_time_json_path, 'r', encoding='utf-8') as f:
            scenes = json.load(f)

        # 加载火山原始 ASR (单位: MM:SS:mmm)
        with open(raw_huoshan_json_path, 'r', encoding='utf-8') as f:
            asr_segments = json.load(f)
    except Exception as e:
        logger.error(f"[{movie_name}] 读取失败: {e}")
        return

    print(f"[{movie_name}] 处理火山 ASR，片段数: {len(scenes)}, ASR条数: {len(asr_segments)}")
    # 输出目录
    final_txt_dir = os.path.join(output_base_dir, movie_name, "asr_huoshan_formatted_txt")
    os.makedirs(final_txt_dir, exist_ok=True)

    total_extracted = 0

    for scene in tqdm(scenes, desc=f"Processing {movie_name}", leave=False):
        scene_id = scene.get("scene_id")

        # 1. 获取当前片段的绝对起始和结束时间 (ms)
        s_start_abs = time_to_ms(scene.get("start_time"))
        s_end_abs = time_to_ms(scene.get("end_time"))

        scene_lines = []

        for asr in asr_segments:
            # 2. 获取 ASR 的原始绝对时间 (ms)
            a_start_abs = time_to_ms(asr.get("start_time"))
            a_end_abs = time_to_ms(asr.get("end_time"))

            # 3. 判断 ASR 是否在该片段的时间窗内 (有交集)
            if a_start_abs < s_end_abs and a_end_abs > s_start_abs:

                # 4. 绝对时间裁剪：确保不超出当前 scene 边界
                clamped_start_abs = max(a_start_abs, s_start_abs)
                clamped_end_abs = min(a_end_abs, s_end_abs)

                if clamped_start_abs < clamped_end_abs:
                    # 5. 【核心修改】计算相对时间：绝对时间 - 片段起始时间
                    relative_start = clamped_start_abs - s_start_abs
                    relative_end = clamped_end_abs - s_start_abs

                    text = str(asr.get("text", "")).strip()
                    # 格式: start_time:相对毫秒, end_time:相对毫秒, text:文本
                    line = f"start_time:{relative_start}, end_time:{relative_end}, text: {text}"
                    scene_lines.append(line)

        # 写入文件
        if scene_lines:
            output_file = os.path.join(final_txt_dir, f"{scene_id}.wav.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(scene_lines) + "\n")
            total_extracted += len(scene_lines)

    logger.info(f"[{movie_name}] 完成。生成场景: {len(scenes)}, ASR条数: {total_extracted}")

# ================= 参数解析 =================


def parse_args():
    parser = ArgumentParser(description="Format Huoshan ASR into scene-relative timestamps")
    parser.add_argument(
        "--base-outputs-dir",
        default="$DATA_ROOT/test/test_pipeline/split_raw_wavs_result",
        help="基础输出目录"
    )
    parser.add_argument(
        "--asr-results-dir",
        default="$DATA_ROOT/test/test_pipeline/huoshan_result",
        help="火山 ASR 结果目录"
    )
    parser.add_argument(
        "--list-file",
        default="$SILERO_VAD_ROOT/movies.scp",
        help="SCP 文件路径，每行一个待处理音频的绝对路径"
    )
    parser.add_argument(
        "--log-dir",
        default="$DATA_ROOT/test/test_pipeline_step3.5_outputs/logs",
        help="日志输出目录"
    )
    return parser.parse_args()


# ================= 主入口 =================

def main():
    args = parse_args()
    configure_logging(args.log_dir)

    if not os.path.exists(args.list_file):
        logger.error("任务列表不存在")
        return

    movies = []
    with open(args.list_file, 'r', encoding='utf-8') as f:
        for line in f:
            path = line.strip()
            if not path:
                continue
            movie = os.path.splitext(os.path.basename(path))[0]
            movies.append(movie)

    for movie in tqdm(movies, desc="Total Progress"):
        total_time_path = os.path.join(args.base_outputs_dir, movie, "part_json_vad_huoshan_only", "total_time.json")
        raw_asr_path = os.path.join(args.asr_results_dir, movie, f"{movie}.json")

        if os.path.exists(total_time_path) and os.path.exists(raw_asr_path):
            process_movie_asr_segments(total_time_path, raw_asr_path, args.base_outputs_dir, movie)
        else:
            if not os.path.exists(total_time_path):
                logger.warning(f"缺失 total_time.json: {movie}")
            if not os.path.exists(raw_asr_path):
                logger.warning(f"缺失 火山JSON: {movie}")

if __name__ == '__main__':
    main()