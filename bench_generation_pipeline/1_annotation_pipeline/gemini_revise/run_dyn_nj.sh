#!/bin/bash
# -*- coding: utf-8 -*-
#
# 支持动态并行调度的精修版本.
# ⚠️ 同上, 需配置 config.yaml
#

export PYTHONPATH=./:$PYTHONPATH
cd $PIPELINE_ROOT/gemini_revise

export TMPDIR=$PIPELINE_ROOT/gemini_revise/tmp

source $CONDA_PREFIX/bin/activate llm_pipeline

# Automatically detect number of gpus
if command -v nvidia-smi &> /dev/null; then
    num_gpus=$(nvidia-smi -L | wc -l)
    gpu_list=$(seq -s, 0 $((num_gpus-1)))
else
    num_gpus=-1
    gpu_list="-1"
fi
# You can also manually specify CUDA_VISIBLE_DEVICES
# if you don't want to utilize all available GPU resources.
# export CUDA_VISIBLE_DEVICES="${gpu_list}"
export CUDA_VISIBLE_DEVICES="0,1"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"

stage=3 # start from -1
stop_stage=3

config=conf/config.yaml

# Persistent success/failed lists for stage 3 (will be created by stage 3)
ALIGN_SUCCESS_FILE="${TMPDIR%/}/step4_2/rerun/all_align_dirs.success.list"
ALIGN_FAILED_FILE="${TMPDIR%/}/step4_2/rerun/all_align_dirs.failed.list"

# Parallelism for stage 3 (each job runs one dataset dir)
ALIGN_JOBS=${ALIGN_JOBS:-6}

# Batch size for stage 3 (how many dataset dirs per align.sh invocation)
ALIGN_BATCH=${ALIGN_BATCH:-16}


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Run alignment (parallel jobs=${ALIGN_JOBS})"

    # JSONL source with per-line wav_count and paths array
    WAV_COUNT_JSONL=${WAV_COUNT_JSONL:-"$PIPELINE_ROOT/gemini_revise/tmp/step4_2/rerun/jsonl/wav_count_to_paths.jsonl"}
    if [[ ! -s "$WAV_COUNT_JSONL" ]]; then
        echo "Error: WAV count jsonl file $WAV_COUNT_JSONL not found or empty."
        exit 1
    fi

    mkdir -p "$(dirname "$ALIGN_SUCCESS_FILE")" "$(dirname "$ALIGN_FAILED_FILE")"
    # Ensure success/failed files exist (may be used for resume)
    : > "$ALIGN_SUCCESS_FILE"
    : > "$ALIGN_FAILED_FILE"

    # Generate per-batch commands: for each JSONL line, set nj=wav_count and split its paths in batches of ALIGN_BATCH
    tmp_cmds=$(mktemp)
    echo "Generating batch commands from ${WAV_COUNT_JSONL} (batch=${ALIGN_BATCH}) -> ${tmp_cmds}"
    python3 - "$WAV_COUNT_JSONL" "$ALIGN_BATCH" "$ALIGN_SUCCESS_FILE" "$ALIGN_FAILED_FILE" "${ALIGN_RESUME:-}" > "$tmp_cmds" <<'PY'
import sys, json
jsonl = sys.argv[1]
batch = int(sys.argv[2])
success_file = sys.argv[3]
failed_file = sys.argv[4]
resume_flag = sys.argv[5].lower() if len(sys.argv) > 5 else ''
success_set = set()
if resume_flag in ('1', 'true'):
    try:
        with open(success_file) as f:
            for l in f:
                success_set.add(l.strip())
    except FileNotFoundError:
        pass

with open(jsonl) as f:
    for line in f:
        if not line.strip():
            continue
        j = json.loads(line)
        wav_count = int(j.get('wav_count', 1))
        paths = j.get('paths', [])
        # deduplicate while preserving order
        seen = set(); uniq = []
        for p in paths:
            if p not in seen:
                seen.add(p); uniq.append(p)
        # optionally filter already-successful
        if resume_flag in ('1', 'true'):
            uniq = [p for p in uniq if p not in success_set]
        for i in range(0, len(uniq), batch):
            batch_paths = uniq[i:i+batch]
            if not batch_paths:
                continue
            # escape embedded double quotes
            esc = ' '.join('"{}"'.format(p.replace('"','\\"')) for p in batch_paths)
            cmd = 'bash local/shell/align_dyn_nj.sh 0 2 --feats-nj {nj} --train-nj {nj} --success-list "{succ}" --failed-list "{fail}" {paths}'.format(nj=wav_count, succ=success_file.replace('"','\\"'), fail=failed_file.replace('"','\\"'), paths=esc)
            print(cmd)
PY

    cmds_count=$(wc -l < "$tmp_cmds" || echo 0)
    if [[ "$cmds_count" -eq 0 ]]; then
        echo "No commands generated. Nothing to run."
        rm -f "$tmp_cmds"
        exit 0
    fi

    echo "Running ${cmds_count} batch command(s) with ${ALIGN_JOBS} parallel workers"
    # Run generated commands in parallel
    xargs -a "$tmp_cmds" -d '\n' -P "$ALIGN_JOBS" -I {} bash -c 'exec {}'
    rc=$?

    rm -f "$tmp_cmds"

    echo "Alignment success list: ${ALIGN_SUCCESS_FILE}"
    echo "Alignment failed list:  ${ALIGN_FAILED_FILE}"
    exit $rc
fi
