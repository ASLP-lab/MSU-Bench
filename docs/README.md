# MSU-Bench — Demo Page (standalone)

A self-contained interactive demo for **MSU-Bench: Towards Speaker-Centric Understanding
in Conversational Multi-Speaker Scenarios** (Interspeech 2026). Everything the page needs
is in this folder — no build step required.

- 📄 Paper: https://arxiv.org/abs/2606.22868
- 💻 Code: https://github.com/ASLP-lab/MSU-Bench
- 🤗 Dataset: https://huggingface.co/datasets/ASLP-lab/MSU-Benchmark

## Run

```bash
# from this directory
python3 -m http.server 8080
# then open http://localhost:8080
```

Any static file server works (`npx serve`, `php -S`, nginx, GitHub Pages, etc.).
Because the page fetches JSON/audio at runtime, open it via `http://`, not `file://`.

## Contents

```
publish-demo-page/
├── index.html            # the single-page demo
├── data.js               # video/task configuration
├── qa_data.js            # bundled per-video QA
├── bench_examples.js     # 16-task gallery (one CN + one EN example per task)
├── assets/images/        # figures
├── video-example/        # annotated video clips
├── annotations/          # speaker-segment annotations for the videos
├── demo_qa/              # per-video QA (fallback fetch)
├── demo_qa_audio/        # QA audio snippets
└── bench_examples/       # gallery audio + per-task JSON
```

## Features

- **Video + annotation explorer** — synchronized speaker segments with metadata (name, gender, age, emotion).
- **Interactive QA** — try questions under different speaker-referencing schemes and reveal answers.
- **16-task gallery** — one Chinese and one English film example per task, each with reference audio and multiple QA variants.
