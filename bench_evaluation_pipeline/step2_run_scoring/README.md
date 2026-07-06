# Step 2 — Run Scoring

Given the language-tagged QA tree from step 1, calls a Large Audio-Language
Model on every question and writes back a `reference_result` (single letter
A/B/C/D) into each QA JSON.

## Files
- `question_builder.py` — audio-selection & prompt-composition helpers
  shared across backends. Encodes the paper's speaker-referencing rules:
  no-index (target snippet only), verification pair (two clips), and
  full-audio-only for time / transcript / speaker / complex indices;
  Tier 2 no-index questions additionally receive the full audio.
- `run_gemini.py` — **default backend**: Gemini 2.5 Flash / Pro / 3-Flash
  via a `generateContent`-compatible endpoint.
- `run_doubao.py` — reference alternative that shows how to plug a
  chat/completions-shaped Doubao endpoint into the same flow. Kept for
  reference only; the shipped `run_all.sh` uses Gemini.

## Usage (Gemini, default)

```bash
source ../configs/gemini.env

python run_gemini.py \
    --input_root ../work/data/qa_merged \
    --output_root ../work/scoring/${GEMINI_MODEL} \
    --num_workers ${NUM_WORKERS}
```

Each output QA JSON keeps the original `qa_result` list, but every question
item now carries:
- `reference_result`: `A|B|C|D` or `""` on failure
- `reference_raw`   : the model's raw reply (for debugging)
- `reference_error` : short error tag when the reply could not be scored

Files that already contain `reference_result` for every question are
skipped unless `--overwrite` is set — the script is fully resumable.

## Usage (Doubao, reference alternative)

Doubao expects a public **URL** for the audio, so you first need to expose
the merged audio via HTTP (e.g. `python -m http.server`) and pass the
prefix mapping:

```bash
source ../configs/doubao.env
python run_doubao.py \
    --input_root ../work/data/qa_merged \
    --output_root ../work/scoring/${DOUBAO_MODEL} \
    --audio_url_map /abs/local/path/to/audio/=http://your-host/msu-audio/
```
