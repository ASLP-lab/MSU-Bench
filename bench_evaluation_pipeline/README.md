# MSU-Bench Evaluation Pipeline

This directory contains the **end-to-end evaluation pipeline** for
[MSU-Bench](https://huggingface.co/datasets/ASLP-lab/MSU-Benchmark):
from downloading the dataset, calling a Large Audio-Language Model (LALM)
on every QA item, all the way to producing per-task / per-capability /
per-tier accuracy tables and diagnostic error breakdowns.

The pipeline is deliberately kept **light and reproducible**: every step is
a self-contained script that only reads/writes JSON on disk, so you can
plug in a different model backend (Doubao, Gemini, GPT-4o, …) by changing
one file.

The example scoring backend shipped here is **Gemini 2.5 Flash**
(via a Google `generateContent`-compatible endpoint).
An alternative Doubao-Seed backend script is included as a reference for
how to plug in a different provider.

---

## 📁 Directory Layout

```
bench_evaluation_pipeline/
├── README.md                       # this file
├── run_all.sh                      # one-shot: prepare -> score -> summarize
│
├── step1_prepare_data/             # ── Step 1. Download & prepare data
│   ├── download_hf.py              #   pull ASLP-lab/MSU-Benchmark (audio + QA)
│   ├── merge_cn_en.py              #   merge bench_cn / bench_en → single QA index
│   └── README.md
│
├── step2_run_scoring/              # ── Step 2. Call the model on every QA
│   ├── run_gemini.py               #   Gemini 2.5 Flash scoring (default)
│   ├── run_doubao.py               #   Doubao-Seed reference (drop-in alt.)
│   ├── question_builder.py         #   shared prompt / audio-part builder
│   └── README.md
│
├── step3_summarize_results/        # ── Step 3. Aggregate metrics
│   ├── summarize.py                #   per-task / capability / tier accuracy
│   │                               #   + speaker-referencing (index) breakdown
│   │                               #   + diagnostic error-type breakdown
│   ├── task_taxonomy.py            #   task ↔ capability ↔ tier mapping
│   └── README.md
│
└── configs/
    ├── gemini.env.example          #   copy to gemini.env and fill in secrets
    └── doubao.env.example
```

---

## 🗂️ Language-split note

The upstream benchmark is released in a **CN / EN dual split** (`bench_cn`
holds the same conversations transcribed / QA-generated in Chinese,
`bench_en` in English). The two splits share the same underlying audio
and speaker structure — they are **two linguistic views of the same items**.

`step1_prepare_data/merge_cn_en.py` therefore normalises them into a
**single flat evaluation index** with a `language` field
(`zh` or `en`), so that `step2` and `step3` do not need to know anything
about the CN/EN directory duality. The step3 report always breaks down
metrics by `language` in addition to task / capability / tier.

---

## 🚀 Quick Start

```bash
# 0. Install
pip install -r requirements.txt            # huggingface_hub, requests, tqdm
sudo apt-get install -y ffmpeg              # audio slicing for Gemini payloads

# 1. Configure the model endpoint
cp configs/gemini.env.example configs/gemini.env
$EDITOR configs/gemini.env                  # fill in GEMINI_BASE_URL / GEMINI_API_KEY

# 2. Run the full pipeline
bash run_all.sh
```

`run_all.sh` writes everything under `./work/`:

```
work/
├── data/                           # step1 outputs
│   ├── raw/                        #   HF snapshot (audio + QA jsons)
│   └── qa_index.jsonl              #   flattened, language-tagged QA index
├── scoring/                        # step2 outputs
│   └── gemini-2.5-flash/           #   one QA.json per input, augmented with reference_result
└── report/                         # step3 outputs (final, human-readable)
    ├── summary_by_task.csv
    ├── summary_by_capability.csv
    ├── summary_by_tier.csv
    ├── summary_by_index_scheme.csv
    ├── summary_by_error_type.csv
    └── summary_by_language.csv
```

> Because we already publish the leaderboard-level scores in the paper,
> **this pipeline does not persist any intermediate leaderboard artefact
> beyond the CSVs above** — the goal is *reproducibility of the
> evaluation flow*, not re-releasing scores.

---

## 🔌 Plugging in a different model

To evaluate a new model, only [`step2_run_scoring/`](step2_run_scoring/)
needs a new file. Follow the pattern in `run_gemini.py`:

1. Iterate every QA `.json` under `work/scoring/<model>/…`.
2. For each question item, use `question_builder.build_parts_for_question(...)`
   to obtain `(prompt_text, list_of_audio_clips)` — this handles the
   `no_index / time / transcript / speaker / complex` audio-selection
   rules and pair-verification correctly.
3. Call your model, parse a single letter (A/B/C/D) from the reply, and
   write it back to the question dict as `reference_result`
   (and optionally `reference_raw` / `reference_error`).

Once the score files exist, `step3_summarize_results/summarize.py`
consumes them without modification.
