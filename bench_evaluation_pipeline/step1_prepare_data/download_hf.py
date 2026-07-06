#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1a — Download MSU-Bench from HuggingFace.

Pulls the full snapshot (audio + QA JSONs) of `ASLP-lab/MSU-Benchmark` into
a local directory. The snapshot preserves the upstream `bench_cn/` and
`bench_en/` layout, which `merge_cn_en.py` then unifies.

If you already have a local copy (e.g. under `../../bench_cn` and
`../../bench_en`), pass `--from_local` to skip the network fetch.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def _download_from_hub(repo_id: str, out_dir: Path, revision: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[!] huggingface_hub not installed. `pip install huggingface_hub`",
              file=sys.stderr)
        sys.exit(1)

    print(f"[hf] snapshot_download repo_id={repo_id} revision={revision}")
    print(f"[hf] target: {out_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
        revision=revision,
    )
    print("[hf] done.")


def _copy_from_local(local_root: Path, out_dir: Path) -> None:
    """Copy an existing on-disk MSU-Bench snapshot into `out_dir`.

    The expected layout under `local_root` is any of:
        local_root/bench_cn/... and local_root/bench_en/...
    or  local_root/QA_cn/... and local_root/QA_en/...
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in ("bench_cn", "bench_en", "QA_cn", "QA_en", "annotation_json",
                 "source_audio", "prompts"):
        src = local_root / name
        if not src.exists():
            continue
        dst = out_dir / name
        if dst.exists():
            print(f"[local] {dst} already exists — skipping.")
            continue
        print(f"[local] copy {src} -> {dst}")
        shutil.copytree(src, dst, symlinks=True)
        copied += 1
    if copied == 0:
        print(f"[!] Nothing found under {local_root} to copy. "
              f"Expected bench_cn/ or bench_en/ subdirectories.",
              file=sys.stderr)
        sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", default="ASLP-lab/MSU-Benchmark",
                    help="HuggingFace dataset repo id.")
    ap.add_argument("--revision", default="main",
                    help="HF revision / branch / tag.")
    ap.add_argument("--out_dir", required=True,
                    help="Local target directory for the raw snapshot.")
    ap.add_argument("--from_local", default=None,
                    help="If set, copy from this local path instead of "
                         "downloading (must contain bench_cn/ and/or bench_en/).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_local:
        _copy_from_local(Path(args.from_local).expanduser().resolve(), out_dir)
    else:
        _download_from_hub(args.repo_id, out_dir, args.revision)

    # Sanity check
    n_cn = len(list((out_dir / "bench_cn").rglob("*.json"))) \
        if (out_dir / "bench_cn").exists() else 0
    n_en = len(list((out_dir / "bench_en").rglob("*.json"))) \
        if (out_dir / "bench_en").exists() else 0
    print(f"[done] snapshot at {out_dir}")
    print(f"       bench_cn json count: {n_cn}")
    print(f"       bench_en json count: {n_en}")


if __name__ == "__main__":
    main()
