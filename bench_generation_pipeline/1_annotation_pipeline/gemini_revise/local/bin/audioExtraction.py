# -*- coding: utf-8 -*-
#
# 从视频/音频文件中提取单声道音频.
# 依赖: ffmpeg
#

import argparse
import logging

import yaml

from local.utils.util import audioExtraction, resultWrite


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='config file')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--paths", nargs='+', help='one or more paths')
    group.add_argument("--paths-file", dest="paths_file", help='file containing one path per line (supports spaces)')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if getattr(args, 'paths_file', None):
        with open(args.paths_file, 'r', encoding='utf-8') as pf:
            paths = [line.rstrip('\n') for line in pf if line.strip()]
    else:
        paths = args.paths
    results = audioExtraction(configs, paths)
    resultWrite(results, "stage 0: Audio extraction")


if __name__ == "__main__":
    main()