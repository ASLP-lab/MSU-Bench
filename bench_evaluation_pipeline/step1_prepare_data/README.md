# Step 1 — Prepare Data

Downloads the MSU-Bench dataset from HuggingFace and normalises it into a
single, language-tagged QA index that step 2 / step 3 can consume without
knowing anything about the CN / EN dual-split layout.

## Files
- `download_hf.py` — fetch `ASLP-lab/MSU-Benchmark` via `huggingface_hub`
  (audio + QA JSONs). Falls back to a local snapshot if you already have
  it on disk (e.g. `bench_cn/` and `bench_en/`).
- `merge_cn_en.py` — walk the CN and EN splits and emit
  `qa_index.jsonl`, one line per QA JSON, tagged with `language`,
  `tier`, `task`, `capability`, `scenario`, `qa_len` and `source_audio`.

## Usage
```bash
python download_hf.py \
    --repo_id ASLP-lab/MSU-Benchmark \
    --out_dir ../work/data/raw

python merge_cn_en.py \
    --raw_dir ../work/data/raw \
    --out_index ../work/data/qa_index.jsonl \
    --out_qa_root ../work/data/qa_merged
```

## About the CN/EN dual split
The upstream release contains **two linguistic views of the same audio**:

- `bench_cn/QA_cn/{scenario}/QA_short|QA_long/…/level*/*.json` — Chinese QA
- `bench_en/QA_en/{scenario}/QA_short|QA_long/…/level*/*.json` — English QA

`merge_cn_en.py` unifies them into a flat set of QA JSONs at
`work/data/qa_merged/<language>/<scenario>/<qa_len>/<segment_id>/<level>/<task>.json`,
each carrying an absolute `source_audio` path pointing to the shared WAV.
