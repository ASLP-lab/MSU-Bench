#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# 激活 silero-vad 环境, 调用 gen_time_and_txt.py.
#

set -euo pipefail

# Usage:  ./step2/gen_time_and_txt.sh $SPEAKER_BENCH_ROOT/data_en_podcast/en_podcast.scp $SPEAKER_BENCH_ROOT/data_en_podcast/huoshan_output
# Usage:  ./step2/gen_time_and_txt.sh $SPEAKER_BENCH_ROOT/data_cn_podcast/wav.scp $SPEAKER_BENCH_ROOT/data_cn_podcast/huoshan_output
# Usage:  ./step2/gen_time_and_txt.sh $SPEAKER_BENCH_ROOT/data_movie/quality_results/samples/chinese_ge95_200.scp $SPEAKER_BENCH_ROOT/data_movie/labeled_cn_data/huoshan_output
# Usage:  ./step2/gen_time_and_txt.sh $SPEAKER_BENCH_ROOT/data_movie/quality_results/samples/english_ge95_161.scp $SPEAKER_BENCH_ROOT/data_movie/labeled_en_data/huoshan_output
# - 格式化火山 ASR 为片段内相对时间，生成每个 part 的 .wav.txt
# - 依赖 step1 的输出结构：
#   * ${output_root}/split_raw_wavs_result/<movie>/part_json_vad_huoshan_only/total_time.json
#   * ${output_root}/huoshan_result/<movie>/<movie>.json
 
SCP_PATH=${1:?-"Provide scp file path"}
OUTPUT_ROOT=${2:?-"Provide absolute output root directory"}

BASE_OUTPUTS_DIR="${OUTPUT_ROOT%/}/split_raw_wavs_result"
ASR_RESULTS_DIR="${OUTPUT_ROOT%/}/huoshan_result"
LOG_DIR="${OUTPUT_ROOT%/}/logs"

source $CONDA_PREFIX/bin/activate silero-vad
cd $PIPELINE_ROOT/step2/

python gen_time_and_txt.py \
	--base-outputs-dir "${BASE_OUTPUTS_DIR}" \
	--asr-results-dir "${ASR_RESULTS_DIR}" \
	--list-file "${SCP_PATH}" \
	--log-dir "${LOG_DIR}"

SCP_OUTPATH="${BASE_OUTPUTS_DIR%/}/segments_wav_list.scp"
python get_scp.py \
    "${BASE_OUTPUTS_DIR}" \
    -o "${SCP_OUTPATH}"