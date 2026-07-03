#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# 串联 step3: 收集时间戳 → Gemini 标注 → 清洗 → 生成 metadata.
# 用法: ./run_gemini.sh <output_root>
# ⚠️ 需填入 YOUR_GEMINI_API_KEY (api.py)
#

set -euo pipefail

# Usage: ./step3/run_gemini.sh $SPEAKER_BENCH_ROOT/data_en_podcast/huoshan_output
# Usage: ./step3/run_gemini.sh $SPEAKER_BENCH_ROOT/data_cn_podcast/huoshan_output
# Usage: ./step3/run_gemini.sh $SPEAKER_BENCH_ROOT/data_movie/labeled_cn_data/huoshan_output
# Usage: ./step3/run_gemini.sh $SPEAKER_BENCH_ROOT/data_movie/labeled_en_data/huoshan_output
# - 读取 Step2 生成的分段音频 SCP（仅含音频路径，无 movie name 列）
# - 自动定位对应的时间戳 txt（asr_huoshan_formatted_txt/<basename>.txt），收集到临时 times 目录
# - 调用 Gemini 分析，并清洗结果、切分副语言片段

OUTPUT_ROOT=${1:?-"Provide absolute output root (same as step1/step2)"}
 
# 工作目录（在 output_root 下集中产物）
SCP_PATH="${OUTPUT_ROOT%/}/split_raw_wavs_result/segments_wav_list.scp"
WORK_ROOT="${OUTPUT_ROOT%/}/gemini_step3"
TIME_DIR="${WORK_ROOT}/times"
OUT_PATH="${WORK_ROOT}/outs/all.jsonl"
ERROR_PATH="${WORK_ROOT}/error_fast.log"
CLEAN_PATH="${WORK_ROOT}/cleaned_out.jsonl"
AUDIO_DIR="${WORK_ROOT}/data"
METADATA_PATH="${WORK_ROOT}/data"

mkdir -p "${WORK_ROOT}" "${TIME_DIR}" "${AUDIO_DIR}" "${WORK_ROOT}/outs"

# 激活环境并运行
# 默认阶段范围：0..2 （0: api.py, 1: clean.py, 2: gen_metadata.py）
STAGE=0
STOP_STAGE=2


# 收集时间戳 txt：从分段 wav 路径推断对应 txt 路径
# 约定：wav 在 .../split_raw_wavs_result/<movie>/audio_segments_vad_huoshan_only/*.wav
#       txt 在 .../split_raw_wavs_result/<movie>/asr_huoshan_formatted_txt/<same_basename>.txt
if [ ${STAGE} -le 0 ] && [ ${STOP_STAGE} -ge 0 ]; then
        while IFS= read -r wav_path; do
                [[ -z "${wav_path}" ]] && continue
                [[ "${wav_path}" = \#* ]] && continue

                base=$(basename "${wav_path}")
                audio_dir=$(dirname "${wav_path}")
                movie_root=$(dirname "${audio_dir}")
                movie=$(basename "${movie_root}")
                # audio_segments_vad_huoshan_only -> asr_huoshan_formatted_txt
                txt_src="${movie_root}/asr_huoshan_formatted_txt/${base}.txt"

                if [[ -f "${txt_src}" ]]; then
                        movie_time_dir="${TIME_DIR}/${movie}"
                        mkdir -p "${movie_time_dir}"
                        ln -sf "${txt_src}" "${movie_time_dir}/${base}.txt"
                else
                        echo "Warn: missing timestamp txt for ${wav_path}" >&2
                        echo "missing ${txt_src}" >> "${ERROR_PATH}" >&2
                fi
        done < "${SCP_PATH}"
fi


source $CONDA_PREFIX/bin/activate wekws

# stage 0: 运行 Gemini 分析（api.py）
if [ ${STAGE} -le 0 ] && [ ${STOP_STAGE} -ge 0 ]; then
    echo "Stage 0: Running Gemini analysis (api.py)..."
    python $PIPELINE_ROOT/test/gemini/api.py \
        --scp_path "${SCP_PATH}" \
        --time_dir "${TIME_DIR}" \
        --out_path "${OUT_PATH}" \
        --error_path "${ERROR_PATH}" \
        --success_scp "${WORK_ROOT}/success_files.scp"
fi

# stage 1: 清洗 Gemini 输出（clean.py）
if [ ${STAGE} -le 1 ] && [ ${STOP_STAGE} -ge 1 ]; then
    echo "Stage 1: Cleaning Gemini output (clean.py)..."
    python $PIPELINE_ROOT/test/gemini/clean.py \
        --input_file "${OUT_PATH}" \
        --output_file "${CLEAN_PATH}" \
        --error_file "${WORK_ROOT}/error_clean.log"
fi

# stage 2: 生成 metadata（gen_metadata.py）
if [ ${STAGE} -le 2 ] && [ ${STOP_STAGE} -ge 2 ]; then
    echo "Stage 2: Generating metadata (gen_metadata.py)..."
    python $PIPELINE_ROOT/test/gemini/gen_metadata.py \
        -i "${CLEAN_PATH}" \
        -o "${METADATA_PATH}"
fi

# 可选: stage 3 切分副语言片段（生成片段音频与 metadata）
# if [ ${STAGE} -le 3 ] && [ ${STOP_STAGE} -ge 3 ]; then
#     echo "Stage 3: Splitting sub-language audio segments (split_audio_segments.py)..."
#     python $PIPELINE_ROOT/test/gemini/split_audio_segments.py \
#         -j "${CLEAN_PATH}" \
#         -o "${AUDIO_DIR}"
# fi 