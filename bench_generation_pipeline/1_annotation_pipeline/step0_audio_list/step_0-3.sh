#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# 根据 del.log 和 revised.log 执行删除/重命名, 并更新 scp.
# 输入: scp.final_1, del.log, revised.log
# 输出: scp.final
#

set -euo pipefail

ORIG_SCP="$PIPELINE_ROOT/step0/scp.final_1"
DEL_LOG="$PIPELINE_ROOT/step0/new_del.log"
REVISED_LOG="$PIPELINE_ROOT/step0/new_revised.log"

STEP1_SCP="$PIPELINE_ROOT/step0/scp.step1"
FINAL_SCP="$PIPELINE_ROOT/step0/scp.final"
DRY_RUN=0

#######################################
# 检查 revised.log 是否是 TAB 分隔
#######################################
if ! grep -q $'\t' "$REVISED_LOG"; then
    echo "❌ revised.log 不是 TAB 分隔，请修正后再跑"
    exit 1
fi

#######################################
# Step 1: 删除 SCP 中重复行
#######################################
grep -F -v -f "$DEL_LOG" "$ORIG_SCP" > "$STEP1_SCP"

#######################################
# Step 2: mv（或 dry-run）
#######################################
while IFS=$'\t' read -r old new; do
    [ "$old" = "$new" ] && continue
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[DRY-RUN] mv \"$old\" \"$new\""
    else
        mkdir -p "$(dirname "$new")"
        mv "$old" "$new"
    fi
done < "$REVISED_LOG"

#######################################
# Step 3: 精确替换 SCP（文件级）
#######################################
awk -F'\t' '
BEGIN {
    while ((getline < "'"$REVISED_LOG"'") > 0) {
        old[$1] = $2
    }
}
{
    if ($0 in old) {
        print old[$0]
    } else {
        print $0
    }
}
' "$STEP1_SCP" > "$FINAL_SCP"

#######################################
# 校验
#######################################
awk -F'\t' '{print $1}' "$REVISED_LOG" | while read -r old; do
    if grep -F -q "$old/" "$FINAL_SCP"; then
        echo "❌ SCP 中仍存在旧路径前缀: $old"
    fi
done

echo "✅ 完成"