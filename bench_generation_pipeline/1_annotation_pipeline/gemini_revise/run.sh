#!/bin/bash
# -*- coding: utf-8 -*-
#
# 串联 gemini_revise 全流程: 源分离 + VAD + ASR 重识别 + 说话人日志 + 强制对齐.
# 用法: ./run.sh <input_dir>
# ⚠️ 需配置 conf/config.yaml 中的密钥和模型路径
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

stage=4 # start from -1
stop_stage=4

config=conf/config.yaml

# 输入目录：默认使用 data，或通过第一个参数指定
INPUT_DIR=${1:-data}
if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "Error: input directory ${INPUT_DIR} does not exist"
    exit 1
fi

# Persistent paths file (will be created by stage -1)
PATHS_FILE="${TMPDIR%/}/clean_fixed.scp"

# Persistent alignment dir list (will be created by stage 2)
ALIGN_DIRS_FILE="${TMPDIR%/}/step4_2/align_dirs.list"

# Persistent skipped list for stage 2 (will be created by stage 2)
ALIGN_SKIPPED_FILE="${TMPDIR%/}/step4_2/align_dirs.skipped.list"

# Persistent success/failed lists for stage 3 (will be created by stage 3)
ALIGN_SUCCESS_FILE="${TMPDIR%/}/step4_2/align_dirs.success.list"
ALIGN_FAILED_FILE="${TMPDIR%/}/step4_2/align_dirs.failed.list"

# Parallelism for stage 3 (each job runs one dataset dir)
ALIGN_JOBS=${ALIGN_JOBS:-10}

# Batch size for stage 3 (how many dataset dirs per align.sh invocation)
ALIGN_BATCH=${ALIGN_BATCH:-16}


# Stage -1: Generate the persistent paths file
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    # 自动递归收集符合模式的 json 文件（part*.json），并转换为绝对路径，按路径排序
    mkdir -p "$(dirname "${PATHS_FILE}")"
    json_files=()
    while IFS= read -r f; do
        json_files+=("$(readlink -f "$f")")
    done < <(find "${INPUT_DIR}" -type f -name 'part*.json' | sort)

    if [[ ${#json_files[@]} -eq 0 ]]; then
        echo "Error: no part*.json files found in ${INPUT_DIR}"
        exit 1
    fi


    echo "stage -1: Generating persistent paths file at ${PATHS_FILE}"

    : > "${PATHS_FILE}"
    for f in "${json_files[@]}"; do
        printf '%s\n' "$f" >> "${PATHS_FILE}"
    done
    echo "Wrote ${#json_files[@]} paths to ${PATHS_FILE}"
fi

# reference variable for downstream stages
paths_file="$PATHS_FILE"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Audio extraction"
    if [[ ! -s "$paths_file" ]]; then
        echo "Error: paths file $paths_file not found or empty. Run stage -1 to generate it."
        exit 1
    fi
    python local/bin/audioExtraction.py \
        --config "$config" \
        --paths-file "$paths_file"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Source separation"
    if [[ ! -s "$paths_file" ]]; then
        echo "Error: paths file $paths_file not found or empty. Run stage -1 to generate it."
        exit 1
    fi
    python local/bin/sourceSeparation.py \
        --config "$config" \
        --paths-file "$paths_file"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Prepare alignment dir list"

    if [[ ! -s "$paths_file" ]]; then
        echo "Error: paths file $paths_file not found or empty. Run stage -1 to generate it."
        exit 1
    fi

    align_total=0
    align_skipped=0

    mkdir -p "$(dirname "$ALIGN_DIRS_FILE")" "$(dirname "$ALIGN_SKIPPED_FILE")"
    : > "$ALIGN_DIRS_FILE"
    : > "$ALIGN_SKIPPED_FILE"

    while IFS= read -r p; do
        data_dir_path=$(readlink -f "${p%\.json}")
        if [[ -z "$data_dir_path" ]]; then
            align_skipped=$((align_skipped+1))
            printf '%s\t%s\n' "EMPTY_PATH" "$p" >> "$ALIGN_SKIPPED_FILE"
            continue
        fi
        if [[ ! -d "$data_dir_path" ]]; then
            align_skipped=$((align_skipped+1))
            printf '%s\t%s\t%s\n' "NOT_A_DIR" "$p" "$data_dir_path" >> "$ALIGN_SKIPPED_FILE"
            continue
        fi
        printf '%s\n' "$data_dir_path" >> "$ALIGN_DIRS_FILE"
        align_total=$((align_total+1))
    done < "$paths_file"

    echo "Alignment dir list written to ${ALIGN_DIRS_FILE} (${align_total} dirs, skipped ${align_skipped})."
    echo "Skipped dir list written to ${ALIGN_SKIPPED_FILE}."
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Run alignment (parallel jobs=${ALIGN_JOBS})"

    if [[ ! -s "$ALIGN_DIRS_FILE" ]]; then
        echo "Error: alignment dir list $ALIGN_DIRS_FILE not found or empty. Run stage 2 first."
        exit 1
    fi

    # 支持重跑/跳过已成功的目录：设置 ALIGN_RESUME=1 或 ALIGN_RESUME=true
    mkdir -p "$(dirname "$ALIGN_SUCCESS_FILE")" "$(dirname "$ALIGN_FAILED_FILE")"

    if [[ "${ALIGN_RESUME:-}" == "1" || "${ALIGN_RESUME:-}" == "true" ]] && [[ -s "$ALIGN_SUCCESS_FILE" ]]; then
        echo "ALIGN_RESUME enabled: skipping $(wc -l < "$ALIGN_SUCCESS_FILE") already-successful dirs"
        tmp_align_dirs="${ALIGN_DIRS_FILE}.to_run"
        # Exclude exact matching lines present in success file
        grep -F -x -v -f "$ALIGN_SUCCESS_FILE" "$ALIGN_DIRS_FILE" > "$tmp_align_dirs" || true
        if [[ ! -s "$tmp_align_dirs" ]]; then
            echo "No remaining directories to run after excluding successful ones. Exiting stage 3."
            exit 0
        fi
        # Keep previous success file; reset failed file for this run
        : > "$ALIGN_FAILED_FILE"
    else
        # Fresh run: reset success and failed lists
        : > "$ALIGN_SUCCESS_FILE"
        : > "$ALIGN_FAILED_FILE"
        tmp_align_dirs="$ALIGN_DIRS_FILE"
    fi

    # 批量 + 并行：每次给 align.sh 传 ALIGN_BATCH 个目录（用空格分隔，作为多个参数传入）。
    # align.sh 会把成功/失败的目录追加记录到 success/failed list。
    # xargs 会自动分批以避免命令行参数过长。
    xargs -a "$tmp_align_dirs" -d '\n' -n "$ALIGN_BATCH" -P "$ALIGN_JOBS" bash local/shell/align.sh 0 2 \
        --success-list "$ALIGN_SUCCESS_FILE" \
        --failed-list "$ALIGN_FAILED_FILE"

    # 清理临时文件（如果有）
    if [[ -n "${tmp_align_dirs:-}" && "${tmp_align_dirs}" != "$ALIGN_DIRS_FILE" ]]; then
        rm -f "$tmp_align_dirs"
    fi

    echo "Alignment success list: ${ALIGN_SUCCESS_FILE}"
    echo "Alignment failed list:  ${ALIGN_FAILED_FILE}"
fi

ALIGN_SUCCESS_FILE="$PIPELINE_ROOT/gemini_revise/tmp/step4_2/rerun/all_align_dirs.success.list"
# FORCED_PATHS_FILE=""
# 如果用户在第四步之前指定了 FORCED_PATHS_FILE（环境变量或在脚本外设置），则直接使用该文件，不再生成
: "${FORCED_PATHS_FILE:=}"
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Forced alignment (python post-processing)"
    echo "ALIGN_SUCCESS_FILE is ${ALIGN_SUCCESS_FILE}"

    forced_paths_file="$paths_file"

    # If user provided FORCED_PATHS_FILE, prefer it (must exist and be non-empty)
    if [[ -n "${FORCED_PATHS_FILE:-}" ]]; then
        if [[ -s "$FORCED_PATHS_FILE" ]]; then
            echo "Using provided FORCED_PATHS_FILE: $FORCED_PATHS_FILE"
            forced_paths_file="$FORCED_PATHS_FILE"
        else
            echo "Error: Provided FORCED_PATHS_FILE '$FORCED_PATHS_FILE' not found or empty."
            exit 1
        fi
    # Otherwise, try to generate one from ALIGN_SUCCESS_FILE if available
    elif [[ -s "$ALIGN_SUCCESS_FILE" ]]; then
        # default generated path
        FORCED_PATHS_FILE="${ALIGN_SUCCESS_FILE}.path.json"
        mkdir -p "$(dirname "$FORCED_PATHS_FILE")"
        : > "$FORCED_PATHS_FILE"
        # dedupe while preserving order
        while IFS= read -r line; do
            [[ -z "${line// }" ]] && continue
            # extract first field (in case failed list may have tabs)
            dir=$(printf '%s' "$line" | awk '{print $1}')
            jsonp="${dir}.json"
            if [[ -f "$jsonp" ]]; then
                printf '%s\n' "$jsonp" >> "$FORCED_PATHS_FILE"
            else
                echo "[WARN] corresponding json not found for '$dir' -> expected '$jsonp' (skipping)"
            fi
        done < <(awk '!seen[$0]++' "$ALIGN_SUCCESS_FILE")

        if [[ -s "$FORCED_PATHS_FILE" ]]; then
            echo "Using converted success json list: $FORCED_PATHS_FILE"
            forced_paths_file="$FORCED_PATHS_FILE"
        else
            echo "No valid json paths found from $ALIGN_SUCCESS_FILE; falling back to original paths file"
            forced_paths_file="$paths_file"
        fi
    fi

    if [[ ! -s "$forced_paths_file" ]]; then
        echo "Error: paths file $forced_paths_file not found or empty. Run earlier stages to generate it."
        exit 1
    fi

    python local/bin/forcedAlignment.py \
        --config "$config" \
        --paths-file "$forced_paths_file"
fi