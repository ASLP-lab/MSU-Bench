#!/usr/bin/env bash
# One-shot MSU-Bench evaluation:
#   Step 1  Download & merge (bench_cn + bench_en) → language-tagged QA index
#   Step 2  Score every QA with Gemini 2.5 Flash
#   Step 3  Summarize into diagnostic CSVs (task / capability / tier /
#           language / speaker-referencing scheme / error type)
#
# Configure the model endpoint via ./configs/gemini.env (copy from
# gemini.env.example first). By default everything is written under
# ./work/. Re-running is safe — every step is resumable.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${HERE}"

# ── Config ─────────────────────────────────────────────────────────────
ENV_FILE="${HERE}/configs/gemini.env"
if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
else
    echo "[warn] ${ENV_FILE} not found; expecting GEMINI_BASE_URL / GEMINI_API_KEY / GEMINI_MODEL / NUM_WORKERS in the environment." >&2
fi

: "${GEMINI_BASE_URL:?Set GEMINI_BASE_URL (see configs/gemini.env.example)}"
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY (see configs/gemini.env.example)}"
: "${GEMINI_MODEL:=gemini-2.5-flash}"
: "${NUM_WORKERS:=8}"

WORK_DIR="${HERE}/work"
RAW_DIR="${WORK_DIR}/data/raw"
QA_MERGED_DIR="${WORK_DIR}/data/qa_merged"
QA_INDEX_JSONL="${WORK_DIR}/data/qa_index.jsonl"
SCORE_DIR="${WORK_DIR}/scoring/${GEMINI_MODEL}"
REPORT_DIR="${WORK_DIR}/report"

mkdir -p "${WORK_DIR}"

# ── Step 1 — prepare data ──────────────────────────────────────────────
echo "==[step1]== download + merge (CN/EN) → ${QA_MERGED_DIR}"
if [[ -n "${MSU_LOCAL_SNAPSHOT:-}" ]]; then
    python "${HERE}/step1_prepare_data/download_hf.py" \
        --from_local "${MSU_LOCAL_SNAPSHOT}" \
        --out_dir    "${RAW_DIR}"
else
    python "${HERE}/step1_prepare_data/download_hf.py" \
        --repo_id ASLP-lab/MSU-Benchmark \
        --out_dir "${RAW_DIR}"
fi

python "${HERE}/step1_prepare_data/merge_cn_en.py" \
    --raw_dir     "${RAW_DIR}" \
    --out_qa_root "${QA_MERGED_DIR}" \
    --out_index   "${QA_INDEX_JSONL}"

# ── Step 2 — score with Gemini ─────────────────────────────────────────
echo "==[step2]== scoring with ${GEMINI_MODEL} → ${SCORE_DIR}"
python "${HERE}/step2_run_scoring/run_gemini.py" \
    --input_root  "${QA_MERGED_DIR}" \
    --output_root "${SCORE_DIR}" \
    --base_url    "${GEMINI_BASE_URL}" \
    --api_key     "${GEMINI_API_KEY}" \
    --model       "${GEMINI_MODEL}" \
    --num_workers "${NUM_WORKERS}"

# ── Step 3 — summarize ─────────────────────────────────────────────────
echo "==[step3]== summarizing → ${REPORT_DIR}"
python "${HERE}/step3_summarize_results/summarize.py" \
    --input_root "${SCORE_DIR}" \
    --out_dir    "${REPORT_DIR}" \
    --model_name "${GEMINI_MODEL}"

echo
echo "== Done =="
echo "Report CSVs:"
ls -1 "${REPORT_DIR}"
