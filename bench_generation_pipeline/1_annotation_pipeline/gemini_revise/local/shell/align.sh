#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# 对单个数据集目录执行对齐流程.
#

pushd $THIRD_PARTY_ROOT

. ./cmd.sh
[ -f path.sh ] && . ./path.sh
set -e
set -o pipefail

# Positional parameters
stage=$1
stop_stage=$2
shift 2

# General parameters
feats_nj=24 #提特征线程数，不能多于文件个数stage1
train_nj=24 #对齐线程数，不能多于文件个数stage2

# Optional per-dataset bookkeeping (safe for parallel runs)
# Use: --success-list /path/to/file --failed-list /path/to/file
success_list=
failed_list=

. utils/parse_options.sh || exit 1;

data_dir=data_5w #不要动
exp_dir=exp_5w #不要改

# data_set='Aishell2 codeswitch_3000 zipformer_9000 Libirspeech WenetSpeech TTS_3W'
# 支持两种调用方式：
# 1) bash align.sh 0 2 --success-list s --failed-list f "dir1 dir2 dir3"  （第三个参数是一个带空格的字符串）
# 2) bash align.sh 0 2 --success-list s --failed-list f dir1 dir2 dir3     （第三个及之后每个参数是一个数据集目录）
data_set="$*"
#需要包含wav.scp和text这两个文件（文件名严格相同），wav.scp内容为：1 wav_path\n(空格分隔) text内容为 1 asr文本\n（空格分隔）

append_line() {
    local file="$1"
    local line="$2"
    [[ -z "$file" ]] && return 0
    mkdir -p "$(dirname "$file")"
    if command -v flock >/dev/null 2>&1; then
        (
            flock -x 200
            printf '%s\n' "$line" >> "$file"
        ) 200>"${file}.lock"
    else
        printf '%s\n' "$line" >> "$file"
    fi
}

record_success() {
    local dir="$1"
    [[ -z "$success_list" ]] && return 0
    append_line "$success_list" "$dir"
}

record_failure() {
    local dir="$1"
    local reason="$2"
    [[ -z "$failed_list" ]] && return 0
    append_line "$failed_list" "${dir}\t${reason}"
}

had_failure=0
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Prepare wav.scp & text & spk2utt & utt2spk started @ `date`"
    for x in ${data_set}; do
        if [[ ! -s "$x/wav.scp" ]] || [[ ! -s "$x/text" ]]; then
            echo "[WARN] missing wav.scp/text in $x"
            record_failure "$x" "MISSING_INPUT(stage0)"
            had_failure=1
            continue
        fi

        if ! awk '{print $1, $1}' "$x/wav.scp" > "$x/utt2spk"; then
            echo "[WARN] failed to write utt2spk for $x"
            record_failure "$x" "UTT2SPK_FAILED(stage0)"
            had_failure=1
            continue
        fi
        if ! awk '{print $1, $1}' "$x/wav.scp" > "$x/spk2utt"; then
            echo "[WARN] failed to write spk2utt for $x"
            record_failure "$x" "SPK2UTT_FAILED(stage0)"
            had_failure=1
            continue
        fi
        if ! $CONDA_PREFIX/envs/llm_pipeline/bin/python local.new/jieBaCut_new.py --text "$x/text"; then
            echo "[WARN] jieBaCut failed for $x"
            record_failure "$x" "JIEBA_FAILED(stage0)"
            had_failure=1
            continue
        fi
    done
    echo "stage 0: Done @ `date`"
fi

#新增utils/fix_data_dir.sh
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: MFCC Feature Extration & CMVN for Training set started @ `date`"

    for x in ${data_set}; do
        # utils/validate_data_dir.sh --no-feats $x
        if ! utils/fix_data_dir.sh "$x"; then
            echo "[WARN] fix_data_dir failed for $x"
            record_failure "$x" "FIX_DATA_DIR_FAILED(stage1)"
            had_failure=1
            continue
        fi

        if ! steps/make_mfcc.sh --cmd "$train_cmd" --nj "$feats_nj" "$x"; then
            echo "[WARN] make_mfcc failed for $x"
            record_failure "$x" "MAKE_MFCC_FAILED(stage1)"
            had_failure=1
            continue
        fi

        if ! utils/fix_data_dir.sh "$x"; then
            echo "[WARN] fix_data_dir (post-mfcc) failed for $x"
            record_failure "$x" "FIX_DATA_DIR2_FAILED(stage1)"
            had_failure=1
            continue
        fi

        if ! steps/compute_cmvn_stats.sh "$x"; then
            echo "[WARN] compute_cmvn_stats failed for $x"
            record_failure "$x" "CMVN_FAILED(stage1)"
            had_failure=1
            continue
        fi

        : > "$x/.complete"
    done
    echo "stage 1: Done @ `date`"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Generate Alignment started @ `date`"
    tag=tri4
    for x in ${data_set}; do
        x_name=$(basename $x)

        ali_dir=${x}/${tag}_ali
        if ! steps/align_fmllr_lats_optional.sh --cmd "$train_cmd" --nj "$train_nj" --boost-silence 0.1 \
            "${x}" "${data_dir}/lang" "${exp_dir}/${tag}" "${ali_dir}" 80 100 50 40000; then
            echo "[WARN] align_fmllr_lats_optional failed for $x"
            record_failure "$x" "ALIGN_FAILED(stage2)"
            had_failure=1
            continue
        fi

        if ! gunzip -c "${ali_dir}"/lat.*.gz | \
            lattice-align-words "${data_dir}/lang/phones/word_boundary.int" \
            "${ali_dir}/final.mdl" ark:- ark:- | \
            lattice-to-ctm-conf --decode-mbr=false --acoustic-scale=0.1 --frame-shift=0.0125 --print-silence=true ark:- - | \
            utils/int2sym.pl -f 5 "${data_dir}/lang/words.txt" > "${ali_dir}/word.ctm"; then
            echo "[WARN] ctm generation pipeline failed for $x"
            record_failure "$x" "CTM_FAILED(stage2)"
            had_failure=1
            continue
        fi

        if [[ -s "${ali_dir}/word.ctm" ]]; then
            record_success "$x"
        else
            echo "[WARN] word.ctm is empty for $x"
            record_failure "$x" "CTM_EMPTY(stage2)"
            had_failure=1
            continue
        fi
    done
    echo "stage 2: Done @ `date`"
fi

popd

if [[ "$had_failure" -ne 0 ]]; then
    exit 1
fi
