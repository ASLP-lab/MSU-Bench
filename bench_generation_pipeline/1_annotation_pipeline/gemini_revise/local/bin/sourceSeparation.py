# -*- coding: utf-8 -*-
#
# 使用 MelBandRoformer 模型分离人声与背景.
# ⚠️ 需下载 MelBandRoformer.ckpt 权重
#

import argparse
import logging

import yaml

from local.utils.util import sourceSeparation, resultWrite


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
    results = sourceSeparation(configs, paths)
    resultWrite(results, "stage 1: Source separation")


if __name__ == "__main__":
    main()