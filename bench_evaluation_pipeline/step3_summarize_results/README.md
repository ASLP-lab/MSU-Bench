# Step 3 — Summarize Results

Consumes the scored QA JSONs written by step 2 and produces six
diagnostic CSVs — one per lens the paper uses to look at model behaviour.

## Files
- `task_taxonomy.py` — the 5-capability × 16-task hierarchy plus the
  CN/EN task-name → short-code mapping.
- `summarize.py` — walks the scored tree, classifies every question and
  writes the CSVs.

## Usage
```bash
python summarize.py \
    --input_root ../work/scoring/${GEMINI_MODEL} \
    --out_dir ../work/report \
    --model_name ${GEMINI_MODEL}
```

## Outputs

| File | Rows | What it answers |
|---|---|---|
| `summary_by_task.csv` | one row per **16 tasks** | *"How does the model do on speaker retrieval / gender / opinion summary / …?"* |
| `summary_by_capability.csv` | one row per **5 capabilities** (SID / SAR / DSR / DSA / DCR) | *"How does the model do on each capability group?"* |
| `summary_by_tier.csv` | one row per tier (1, 2) | *"Grounding vs. reasoning."* |
| `summary_by_language.csv` | (lang, tier) | *"Same accuracy on the CN and EN split of the same audio?"* |
| `summary_by_index_scheme.csv` | (tier, scheme) | Speaker-referencing scheme breakdown: how the model does on **no_index / time_index / transcript_index / speaker_index / complex_index**, plus each scheme's **share within the tier** — this covers "different level speaker-referencing distribution" from the paper's error analysis. |
| `summary_by_error_type.csv` | one row per tier | Wrong-answer composition: **wrong_speaker / hallucination / unknown / instruction_not_follow** counts and shares. Diagnostic — this is what the paper calls the "error-type composition" table. |

All CSVs carry the `model` column so you can concatenate outputs from
multiple backends.

## Question-level classification rules

- Predicted answer letter is extracted from `reference_result` first,
  falling back to a single letter within the raw reply.
- If no valid A/B/C/D letter can be extracted, the item counts as
  **instruction_not_follow**.
- Otherwise, if the letter equals `answer`, the item is **correct**.
- Otherwise, the option's rationale (from `rationale`, aligned with the
  option list or letter-prefixed) decides which of
  **wrong_speaker / hallucination / unknown** the wrong answer belongs to.
  A rationale that resolves to `ground_truth` on a wrong prediction
  (schema inconsistency) is bucketed as `unknown`.
- `reverse_retrieval` / `reverse_count` question types are excluded by
  default (matching the paper); pass `--keep_reverse` to include them.

Note: since the outer leaderboard already publishes the per-model score,
this pipeline **does not** persist any consolidated leaderboard file —
the six CSVs above are diagnostic-only.
