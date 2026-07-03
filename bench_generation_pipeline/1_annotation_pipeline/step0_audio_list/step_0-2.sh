#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# 对同名电影的不同版本音频按父目录语义去重, 区分需删除和需重命名的文件.
# 输入: duplicate.log
# 输出: del.log, revised.log
#

input="$PIPELINE_ROOT/step0/new_duplicate.log"
del_log="$PIPELINE_ROOT/step0/new_del.log"
rev_log="$PIPELINE_ROOT/step0/new_revised.log"

> "$del_log"
> "$rev_log"

awk '
function normalize(s) {
    gsub(/（[^）]*）/, "", s)
    gsub(/\([^)]*\)/, "", s)
    gsub(/[0-9]/, "", s)
    gsub(/[[:space:]]+/, "", s)
    return s
}

function process_group(   i, same, base, fname, dir_path, last_slash, newname, newpath) {
    if (n < 2) return

    same = 1
    base = norm_parent[0]

    for (i = 1; i < n; i++) {
        if (norm_parent[i] != base) {
            same = 0
            break
        }
    }

    if (same) {
        # 情况 1：语义一致（父目录名归一化后相同） -> 保留第一个，其余删除
        for (i = 1; i < n; i++) {
            print paths[i] >> "'"$del_log"'"
        }
    } else {
        # 情况 2：语义不一致 -> 全部重命名为 "父目录_文件名"
        for (i = 0; i < n; i++) {
            # 找到路径中最后一个斜杠的位置
            if (match(paths[i], /.*\//)) {
                dir_path = substr(paths[i], RSTART, RLENGTH)
                fname = substr(paths[i], RSTART + RLENGTH)
                
                # 获取父目录名用于重命名
                split(dir_path, segments, "/")
                parent_raw = segments[length(segments)-1]
                
                newname = parent_raw "_" fname
                newpath = dir_path newname
                
                print paths[i] "\t" newpath >> "'"$rev_log"'"
            }
        }
    }
}

/^==== DUPLICATE FILE:/ {
    process_group()
    delete paths
    delete norm_parent
    n = 0
    next
}

/^\// {
    paths[n] = $0
    # 提取父目录
    split($0, b, "/")
    parent = b[length(b)-1]
    norm_parent[n] = normalize(parent)
    n++
}

END {
    process_group()
}
' "$input"