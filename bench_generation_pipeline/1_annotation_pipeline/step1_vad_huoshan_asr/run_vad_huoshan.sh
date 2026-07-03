#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# 串联 step1 全部子步骤: VAD → 火山ASR → 解析 → 切分.
# 用法: ./run_vad_huoshan.sh <scp> <output_root>
# ⚠️ 需先填入火山密钥(ak/sk/appid/token)到 gen_task_id_multi.py
#

set -euo pipefail
source $CONDA_PREFIX/bin/activate silero-vad
cd $PIPELINE_ROOT/step1/

# Usage: ./run_vad_huoshan.sh /path/to/list.scp /abs/output_root [stage] [stop_stage]
# stage/stop_stage 用法示例：
#   ./run_vad_huoshan.sh list.scp /abs/output_root        # 等价于 stage=0 stop_stage=3
#   ./run_vad_huoshan.sh $SPEAKER_BENCH_ROOT/data_en_podcast/en_podcast.scp $SPEAKER_BENCH_ROOT/data_en_podcast/huoshan_output 0 0    # 仅运行第1到2步
#   ./step1/run_vad_huoshan.sh $SPEAKER_BENCH_ROOT/data_cn_podcast/wav.scp $SPEAKER_BENCH_ROOT/data_cn_podcast/huoshan_output 0 0    # 仅运行第1到2步
#   ./step1/run_vad_huoshan.sh $SPEAKER_BENCH_ROOT/data_cn_podcast/wav.scp $SPEAKER_BENCH_ROOT/data_cn_podcast/huoshan_output 0 0    # 仅运行第1到2步
#   ./step1/run_vad_huoshan.sh $SPEAKER_BENCH_ROOT/data_movie/quality_results/samples/chinese_ge95_200.scp $SPEAKER_BENCH_ROOT/data_movie/labeled_cn_data/huoshan_output 0 0    # 仅运行第1到2步
#   ./step1/run_vad_huoshan.sh $SPEAKER_BENCH_ROOT/data_movie/quality_results/samples/english_ge95_161.scp $SPEAKER_BENCH_ROOT/data_movie/labeled_en_data/huoshan_output 0 0    # 仅运行第1到2步
# 步骤映射（从0开始）：
#  0 - gen_timestamps.py
#  1 - gen_task_id_multi.py
#  2 - parse_huoshan_segments.py
#  3 - split_raw_wavs.py

SCP_PATH=${1:?"Provide scp file path"}
OUTPUT_ROOT=${2:?"Provide absolute output root directory"}
STAGE=0
STOP_STAGE=3

# validate numeric
re='^[0-9]+$'
if ! [[ "${STAGE}" =~ ${re} && "${STOP_STAGE}" =~ ${re} ]]; then
  echo "Error: stage and stop_stage must be integers." >&2
  exit 2
fi

if [ "${STAGE}" -lt 0 ] || [ "${STOP_STAGE}" -lt 0 ] || [ "${STAGE}" -gt "${STOP_STAGE}" ]; then
  echo "Error: invalid stage range. Ensure 0 <= stage <= stop_stage." >&2
  exit 2
fi

VAD_OUT="${OUTPUT_ROOT%/}/vad_result"
HUOSHAN_OUT="${OUTPUT_ROOT%/}/huoshan_result"
SPLIT_OUT="${OUTPUT_ROOT%/}/split_raw_wavs_result"
LOG_DIR="${OUTPUT_ROOT%/}/logs"

mkdir -p "${VAD_OUT}" "${HUOSHAN_OUT}"

# Step 0: gen timestamps
if [ ${STAGE} -le 0 ] && [ ${STOP_STAGE} -ge 0 ]; then
  echo "[stage 0] gen_timestamps"
  python gen_timestamps.py -s "${SCP_PATH}" -o "${VAD_OUT}"
else
  echo "[stage 0] skipped"
fi

# Step 1: generate huoshan task ids
if [ ${STAGE} -le 1 ] && [ ${STOP_STAGE} -ge 1 ]; then
  echo "[stage 1] gen_task_id_multi"
  python gen_task_id_multi.py -s "${SCP_PATH}" -t "${HUOSHAN_OUT}"
else
  echo "[stage 1] skipped"
fi

# Step 2: parse huoshan segments
if [ ${STAGE} -le 2 ] && [ ${STOP_STAGE} -ge 2 ]; then
  echo "[stage 2] parse_huoshan_segments"
  python parse_huoshan_segments.py -s "${SCP_PATH}" -i "${HUOSHAN_OUT}" -o "${HUOSHAN_OUT}"
else
  echo "[stage 2] skipped"
fi

# Step 3: split raw wavs
if [ ${STAGE} -le 3 ] && [ ${STOP_STAGE} -ge 3 ]; then
  echo "[stage 3] split_raw_wavs"
  python split_raw_wavs.py \
    --movie_list_file "${SCP_PATH}" \
    --vad_base "${VAD_OUT}" \
    --huoshan_base "${HUOSHAN_OUT}" \
    --output_base "${SPLIT_OUT}" \
    --log_dir "${LOG_DIR}"
else
  echo "[stage 3] skipped"
fi
