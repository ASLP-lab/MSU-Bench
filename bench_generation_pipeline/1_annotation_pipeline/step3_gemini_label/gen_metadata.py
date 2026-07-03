# -*- coding: utf-8 -*-
#
# 从清洗后的 JSONL 生成按电影分组的 metadata JSON.
# 输入: cleaned_out.jsonl
# 输出: data/<movie>/partXXX.json
#

import json
import os

def jsonl_to_per_utt_json(jsonl_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                print(f"[WARN] line {line_idx} is empty, skipped")
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] line {line_idx} json decode failed: {e}")
                continue

            utt = item.get("utt")
            wav_path = item.get("wav_path")
            audio_segments = (
                item.get("audio_segments", [])
            )
            movie_name = wav_path.split('/')[-3]
            os.makedirs(os.path.join(out_dir, movie_name), exist_ok=True)

            if not utt:
                print(f"[WARN] line {line_idx} missing utt, skipped")
                continue

            out_obj = {
                "video_path": "",
                "audio_path": wav_path,
                "audio_segments": audio_segments
            }

            out_path = os.path.join(out_dir, movie_name, f"{utt}.json")
            with open(out_path, "w", encoding="utf-8") as wf:
                json.dump(out_obj, wf, ensure_ascii=False, indent=2)

    print("Done.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert Gemini JSONL to per-utt JSON files")
    parser.add_argument("--input", "-i", required=True, help="Path to input JSONL file")
    parser.add_argument("--out-dir", "-o", required=True, help="Output directory to write per-utt json files")

    args = parser.parse_args()
    jsonl_path = args.input
    out_dir = args.out_dir

    if not os.path.isfile(jsonl_path):
        print(f"Error: input file {jsonl_path} does not exist")
        exit(2)

    jsonl_to_per_utt_json(jsonl_path, out_dir)