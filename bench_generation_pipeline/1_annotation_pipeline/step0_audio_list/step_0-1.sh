# -*- coding: utf-8 -*-
#
# 比较两个 scp 文件, 找出差异行.
# 输入: 两个 scp 文件
# 输出: diff 结果
#

awk -F/ '
{
    fname = $NF
    files[fname] = files[fname] ? files[fname] "\n" $0 : $0
    count[fname]++
}
END {
    for (f in count) {
        if (count[f] > 1) {
            print "==== DUPLICATE FILE:", f " ===="
            print files[f]
            print ""
        }
    }
}
' $PIPELINE_ROOT/step0/scp.final > $PIPELINE_ROOT/step0/new_duplicate.log