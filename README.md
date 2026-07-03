<div align="center">

# MSU-Bench

### Towards Speaker-Centric Understanding in Conversational Multi-Speaker Scenarios

**Interspeech 2026**

*A diagnostic benchmark for evaluating how well Large Audio-Language Models (LALMs) understand **who says what**, and **what happens between speakers**, in real multi-speaker conversations.*

[![Demo Page](https://img.shields.io/badge/🌐_Demo-Live-00f0ff)](https://aslp-lab.github.io/msu-bench.github.io/)
[![Paper](https://img.shields.io/badge/arXiv-2606.22868-b31b1b)](https://arxiv.org/abs/2606.22868)
[![Dataset](https://img.shields.io/badge/🤗_HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/ASLP-lab/MSU-Benchmark)
[![License](https://img.shields.io/badge/License-see_below-blue)](#license)

</div>

---

## Overview

Speech understanding is moving from single-task systems toward unified audio-to-text generation. Yet most existing evaluations still focus on **single-speaker** audio or isolated sub-tasks (ASR, speaker attribute recognition, emotion recognition). They cannot answer the core question of real multi-party dialogue:

> When several people keep interacting in one audio stream, does the model actually understand **"who is speaking what"** and **"what is happening between the speakers"**?

**MSU-Bench** is a diagnostic benchmark built around **speaker-centric understanding**. It organizes evaluation into a **two-tier** framework, refined into **5 ability dimensions** and **16 sub-tasks**, with **2,300 human-verified QA items**. All questions are multiple-choice (single correct answer) with diagnostically-designed distractors, so results are both easy to score automatically and easy to analyze for *why* a model fails.

- **Demo Page:** https://aslp-lab.github.io/msu-bench.github.io/
- **Paper (arXiv):** https://arxiv.org/abs/2606.22868
- **Dataset (HuggingFace):** https://huggingface.co/datasets/ASLP-lab/MSU-Benchmark

> Authors: Zhaokai Sun, Shuai Wang, Zhennan Lin, Chengyou Wang, Dehui Gao, Yu'ang Cao, Chunjiang He, Pan Zhou, Lei Xie.
> ASLP@NPU (Audio, Speech and Language Processing Group, Northwestern Polytechnical University), in collaboration with Li Auto.

---

## Task Taxonomy

MSU-Bench uses a two-tier design that moves from basic speaker grounding to complex dialogue reasoning.

### Tier 1 — Speaker Grounding & Identification
Can the model bind speech content, speaker identity, and speaker attributes together?

| Ability Dimension | Sub-tasks |
|---|---|
| **Speaker Identification** | Speaker Retrieval · Reverse Retrieval · Speaker Counting · Speaker Verification · Speaker Opinion Summary |
| **Speaker Attributes** | Gender Recognition · Age Recognition · Emotion Recognition · Accent Recognition · Speaker Profile |

### Tier 2 — Multi-Speaker Dialogue Reasoning
Can the model reason about relationships, structure, and context across speakers?

| Ability Dimension | Sub-tasks |
|---|---|
| **Context Reasoning** | Emotion Interaction · Multi-Speaker Opinion Summary |
| **Scene Reasoning** | Dialogue Background Reasoning · Dialogue Role Identification |
| **Structure Analysis** | Dialogue Act Recognition · QA Structure Recognition |

### Speaker-Referencing Schemes
To test robustness to *how* a target speaker is specified, each applicable task is instantiated under multiple referencing schemes:

| Scheme | How the target speaker is specified |
|---|---|
| **No Index** | A raw audio snippet of the target speaker (acoustic anchor) |
| **Time Index** | A time range (e.g. "the person speaking from 10s to 15s") |
| **Transcript Index** | A quoted transcript line |
| **Speaker Index** | Order of appearance (e.g. "the second speaker") |
| **Complex Index** | A combination of time / text / speaker cues |

---

## Data & Quality Control

MSU-Bench draws from diverse **Chinese and English** multi-speaker sources — **telephone conversations, meetings, podcasts, and film/TV clips** — to test generalization across acoustic conditions and dialogue styles.

Construction pipeline (automatic generation + human review):
1. **Dialogue-quality filtering** — select information-rich, context-complete segments.
2. **Multi-dimensional annotation** — speaker diarization, transcription, speaker identity, sound events, and paralinguistic cues.
3. **QA generation** — produce candidate QA from audio + structured annotation + task-specific prompts, across task types and referencing schemes.
4. **Human review** — audio-literate annotators verify, fix, or remove invalid/ambiguous items.

The full, reproducible pipeline is open-sourced in [`bench_generation_pipeline/`](bench_generation_pipeline/).

---

## Key Findings

We evaluate **9 representative LALMs**, including Qwen2.5-Omni, Qwen3-Omni, Audio-Flamingo-3, Kimi-Audio, Step-Audio-2, MiMo-Audio, and the closed-source Gemini family.

- **Closed-source leads, but nobody solves it.** Gemini-3-Flash reaches the highest overall accuracy (**Avg 0.77**; Tier 1 0.73 / Tier 2 0.84). The best open model, MiMo-Audio, reaches **0.56** (Tier 1 0.52 / Tier 2 0.64).
- **Finding 1 — Time Index is the hardest referencing scheme.** Temporal grounding of a target speaker remains difficult under fast turn-taking and overlap; multi-cue *Complex Index* helps but exposes weak time↔identity alignment.
- **Finding 2 — Strong models still "blame the wrong speaker."** As tasks get harder, weaker models retreat to *Unknown*, while stronger models increasingly make *wrong-speaker attribution* errors (e.g. Gemini-3-Flash reaches 0.67 wrong-speaker rate on Tier 2).
- **Finding 3 — General audio ability ≠ stable speaker-centric understanding.** Omni-style models do not automatically dominate; multi-speaker understanding needs dedicated speaker modeling, temporal alignment, and cross-speaker reasoning.

---

## Repository Structure

```
publish-github/
├── README.md                    # this file — project overview
├── docs/                        # 🌐 interactive demo site (GitHub Pages source)
│   ├── index.html               # single-page demo
│   ├── data.js / qa_data.js / bench_examples.js
│   ├── assets/                  # figures & images
│   ├── video-example/           # annotated video clips
│   ├── annotations/             # speaker-segment annotations for the videos
│   ├── demo_qa/ demo_qa_audio/  # per-video QA + audio snippets
│   ├── bench_examples/          # 16-task gallery (CN + EN example per task, with audio)
│   └── .nojekyll
└── bench_generation_pipeline/   # 🔧 open-sourced QA construction pipeline
    ├── 1_annotation_pipeline/   # audio → speaker/paralinguistic annotation
    ├── 2_qa_generation/         # annotation → QA (per-ability prompts)
    └── 3_samples/               # end-to-end CN & EN worked examples
```

---

## Quick Start

### View the demo online
The demo is served via **GitHub Pages** from the `docs/` folder:
👉 **https://aslp-lab.github.io/msu-bench.github.io/**

To enable Pages on your own fork: *Settings → Pages → Build from branch → `main` / `docs`*.

### Run the demo locally
```bash
cd docs
python3 -m http.server 8080
# then open http://localhost:8080
```

The demo includes:
- **Video + annotation explorer** — synchronized speaker segments with metadata (name, gender, age, emotion).
- **Interactive QA** — try questions under different speaker-referencing schemes, reveal answers.
- **16-task gallery** — one Chinese and one English film example per task, each with reference audio and multiple QA variants.

### Reproduce the QA pipeline
See [`bench_generation_pipeline/README.md`](bench_generation_pipeline/README.md) for the full audio → annotation → QA flow, prompt templates, dependencies, and worked samples. **All API keys are anonymized to `YOUR_*` placeholders — fill in your own before running.**

---

## Dataset

The benchmark data (audio + annotations + QA) is available on HuggingFace:
**https://huggingface.co/datasets/ASLP-lab/MSU-Benchmark**

```python
from datasets import load_dataset
ds = load_dataset("ASLP-lab/MSU-Benchmark", split="test")
```

It ships a flat `data/test.jsonl` (one row per question) referencing audio files under `audio/`. See the dataset card for the full schema and audio-loading snippet.

Because the audio is sourced from copyrighted film/TV, telephone, meeting, and podcast material, the dataset is released **for non-commercial research use only**.

---

## Citation

If you find MSU-Bench useful, please cite:

```bibtex
@inproceedings{sun2026msubench,
  title     = {MSU-Bench: Towards Speaker-Centric Understanding in Conversational Multi-Speaker Scenarios},
  author    = {Sun, Zhaokai and Wang, Shuai and Lin, Zhennan and Wang, Chengyou and Gao, Dehui and Cao, Yu'ang and He, Chunjiang and Zhou, Pan and Xie, Lei},
  booktitle = {Proc. Interspeech},
  year      = {2026}
}
```

---

## License

- **Code** (demo site + generation pipeline): released under the MIT License — see [`LICENSE`](LICENSE).
- **Data** (audio, annotations, QA): for **non-commercial academic research only**, as it is derived from third-party copyrighted media. Do not redistribute the raw media for commercial purposes.

> If your intended use differs, please open an issue or contact the authors.

---

<div align="center">
Made by <b>ASLP@NPU</b> · in collaboration with <b>Li Auto</b>
</div>
