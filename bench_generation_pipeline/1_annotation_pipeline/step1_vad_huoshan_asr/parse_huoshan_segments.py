#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 从火山 ASR 原始输出提取分句, 生成 <movie>/<movie>.json.
# 输入: huoshan_result/<movie>.txt
# 输出: huoshan_result/<movie>/<movie>.json
#

import argparse
import json
from pathlib import Path


def ms_to_hhmmssms(ms: int) -> str:
    total_ms = int(ms)
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    seconds = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{millis:03d}"


def load_utterances(result_path: Path):
    """Load huoshan result JSON and return segments list.
    Raises ValueError on invalid JSON or unexpected format."""
    try:
        with result_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as ex:
        raise ValueError(f"Invalid JSON in {result_path}: {ex}")

    utts = data.get('result', {}).get('utterances', [])
    segments = []
    for idx, utt in enumerate(utts):
        segments.append({
            'segment_id': f"segment_{idx:04d}",
            'text': utt.get('text', ''),
            'speaker': utt.get('additions', {}).get('speaker', ''),
            'start_time': ms_to_hhmmssms(utt.get('start_time', 0)),
            'end_time': ms_to_hhmmssms(utt.get('end_time', 0)),
        })
    return segments


def main():
    parser = argparse.ArgumentParser(description='Extract huoshan ASR segments into per-movie JSON.')
    parser.add_argument('--scp_path', '-s', required=True, help='SCP file with audio paths (used to derive movie names)')
    parser.add_argument('--huoshan_dir', '-i', required=True, help='Directory containing huoshan result txt/json files')
    parser.add_argument('--out_dir', '-o', required=True, help='Output root; writes movie_name/movie_name.json')
    args = parser.parse_args()

    scp_path = Path(args.scp_path)
    huoshan_dir = Path(args.huoshan_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # error log file (append mode)
    error_log = out_root / 'error_log.txt'

    def _append_error(msg: str):
        ts = __import__('datetime').datetime.now().isoformat()
        line = f"[{ts}] {msg}\n"
        # print to stdout for visibility
        print(line.strip())
        try:
            with error_log.open('a', encoding='utf-8') as ef:
                ef.write(line)
        except Exception:
            # If logging fails, still allow program to continue
            print(f"Failed to write to error log: {error_log}")

    with scp_path.open('r', encoding='utf-8') as f:
        audio_paths = [line.strip() for line in f if line.strip()]

    for audio_path in audio_paths:
        movie_name = Path(audio_path).name  # keep extension for file match
        huoshan_file = huoshan_dir / f"{movie_name}.txt"
        if not huoshan_file.is_file():
            _append_error(f"Skip {audio_path}: missing {huoshan_file}")
            continue

        try:
            segments = load_utterances(huoshan_file)
        except Exception as ex:
            _append_error(f"Failed to parse {huoshan_file}: {ex}")
            continue

        movie_dir = out_root / Path(audio_path).stem
        movie_dir.mkdir(parents=True, exist_ok=True)
        out_file = movie_dir / f"{Path(audio_path).stem}.json"

        with out_file.open('w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(segments)} segments to {out_file}")


if __name__ == '__main__':
    main()
