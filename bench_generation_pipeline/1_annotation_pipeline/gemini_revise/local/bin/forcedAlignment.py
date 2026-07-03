# -*- coding: utf-8 -*-
#
# 将 ASR 文本与音频做强制对齐, 输出精确时间戳.
# 依赖: 火山 ASR (需配置 recognition 段)
#

import argparse
import logging

import yaml

from local.utils.util import forcedAlignment, resultWrite


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='config file')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--paths", nargs='+', help='one or more paths')
    group.add_argument("--paths-file", dest="paths_file", help='file containing one path per line (supports spaces)')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=500, help='number of paths to process per forcedAlignment call (default: 500)')
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

    total = len(paths)
    batch_size = int(args.batch_size) if args.batch_size and int(args.batch_size) > 0 else 500
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    print(f"Running forced alignment on {total} paths with batch_size={batch_size}")

    if total == 0:
        print("No paths to process.")
        return

    # Process in batches and write results incrementally
    num_batches = (total + batch_size - 1) // batch_size
    for idx in range(num_batches):
        start = idx * batch_size
        end = min(start + batch_size, total)
        batch_paths = paths[start:end]
        print(f"Processing batch {idx+1}/{num_batches}: paths {start+1}-{end} (count={len(batch_paths)})")
        results = forcedAlignment(configs, batch_paths)
        resultWrite(results, f"Forced alignment batch {idx+1}/{num_batches}")

    print("Forced alignment completed.")


if __name__ == "__main__":
    main()