import os
import json
from collections import defaultdict

failed_path = "$PIPELINE_ROOT/gemini_revise/tmp/step4_2/all_failed.scp"
out_jsonl = "$PIPELINE_ROOT/gemini_revise/tmp/step4_2/rerun/jsonl/wav_count_to_paths.jsonl"

# wav数量 -> [extraction_path, ...]
wav_count_to_paths = defaultdict(list)

min_wav_num = 0x3f3f3f3f

with open(failed_path, "r") as f:
    lines = f.readlines()

for line in lines:
    base_path = line.strip()
    if not base_path:
        continue
    import pdb
    scp_path = os.path.join(base_path, "wav.scp")

    if not os.path.isfile(scp_path):
        print(f"[WARN] wav.scp file not exist: {scp_path}")
        continue

    with open(scp_path, "r") as f:
        wav_lines = f.readlines()
    
    wav_count = len(wav_lines)

    wav_count_to_paths[wav_count].append(base_path)
    min_wav_num = min(min_wav_num, wav_count)

print(f"Minimum number of wav files: {min_wav_num}")

# 写 jsonl
with open(out_jsonl, "w", encoding="utf-8") as f:
    for wav_count, paths in sorted(wav_count_to_paths.items()):
        record = {
            "wav_count": wav_count,
            "paths": paths
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Saved wav count distribution to {out_jsonl}")