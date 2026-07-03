# -*- coding: utf-8 -*-
#
# 递归扫描电影音频目录, 生成每行一个 wav 路径的 scp 文件.
# 输入: 音频根目录
# 输出: audio.scp
#

import os

root = "$DATA_ROOT"
exts = (".wav", ".mp3", ".flac")

with open("$PIPELINE_ROOT/step0/audio.scp", "w") as f:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(exts) and not name.endswith("_60s" + os.path.splitext(name)[1]):
                f.write(os.path.join(dirpath, name) + "\n")