#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1b — Merge bench_cn / bench_en into a language-tagged QA index.

The upstream release ships two linguistic views of the *same* underlying
audio:
    <raw_dir>/bench_cn/QA_cn/{scenario}/QA_short|QA_long/<seg>/level*/*.json
    <raw_dir>/bench_en/QA_en/{scenario}/QA_short|QA_long/<seg>/level*/*.json

For evaluation we only need a single flat pool of QA files, each carrying
its own `language` tag and the absolute path to the shared audio. This
script does exactly that:

* walks both splits (whichever exists);
* copies each QA JSON to
    <out_qa_root>/<lang>/<scenario>/<qa_len>/<seg>/<level>/<task>.json
  and rewrites its `source_audio` field to an absolute path;
* emits `qa_index.jsonl`, one line per QA file, with the fields step 2/3
  need to break down metrics by language / scenario / tier / task.

Only files that (a) contain a valid `qa_result` (or `result`) list *and*
(b) can be linked to an existing WAV are included in the index.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_LEVEL_RE = re.compile(r"level(\d+)", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────────
def _find_qa_root(raw_dir: Path, lang: str) -> Optional[Path]:
    """Return the directory containing per-scenario QA_short / QA_long trees.

    Handles a few plausible layouts:
      raw_dir/bench_cn/QA_cn/...    (upstream)
      raw_dir/QA_cn/...             (unpacked)
    """
    candidates = [
        raw_dir / f"bench_{lang}" / f"QA_{lang}",
        raw_dir / f"QA_{lang}",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def _find_audio_root(raw_dir: Path, lang: str) -> Optional[Path]:
    for c in [
        raw_dir / f"bench_{lang}" / "source_audio",
        raw_dir / "source_audio",
    ]:
        if c.is_dir():
            return c
    return None


def _iter_qa_files(qa_root: Path):
    """Yield (scenario, qa_len, seg, level, task_stem, path) for each QA .json.

    The expected structure is:
        qa_root/<scenario>/QA_short|QA_long/<seg>/level*/<task>.json
    """
    for scen_dir in sorted(p for p in qa_root.iterdir() if p.is_dir()):
        scenario = scen_dir.name
        for qa_len_dir in sorted(p for p in scen_dir.iterdir() if p.is_dir()):
            qa_len = qa_len_dir.name  # QA_short / QA_long
            for seg_dir in sorted(p for p in qa_len_dir.iterdir() if p.is_dir()):
                seg = seg_dir.name
                for level_dir in sorted(p for p in seg_dir.iterdir() if p.is_dir()):
                    m = _LEVEL_RE.search(level_dir.name)
                    if not m:
                        continue
                    level = f"level{m.group(1)}"
                    for jf in sorted(level_dir.glob("*.json")):
                        # skip .bak / hidden
                        if jf.name.endswith(".bak") or jf.name.startswith("."):
                            continue
                        yield scenario, qa_len, seg, level, jf.stem, jf


def _resolve_audio_path(raw_dir: Path, lang: str, scenario: str,
                        seg_id: str, hint: Optional[str]) -> Optional[Path]:
    """Try, in order:
      1) `hint` (the original `source_audio` relative to the CN/EN root)
      2) source_audio/<scenario>/<seg>.wav
    """
    audio_root = _find_audio_root(raw_dir, lang)

    # 1) honour the JSON's own hint if it resolves
    if hint:
        # hint is usually `source_audio/<scen>/<seg>.wav` relative to bench_{lang}
        for base in (raw_dir / f"bench_{lang}", raw_dir):
            cand = (base / hint).expanduser()
            if cand.exists():
                return cand.resolve()

    # 2) canonical fallback — trim seg to the first "__" segment id
    if audio_root is not None:
        # segment folder names look like R8001_M8004__seg0000__6.90-317.24__idx018-033__96.89-160.05
        # while the underlying wav is R8001_M8004__seg0000__6.90-317.24.wav
        seg_key = "__".join(seg_id.split("__")[:3])  # e.g. R..__seg0000__6.90-317.24
        cand = audio_root / scenario / f"{seg_key}.wav"
        if cand.exists():
            return cand.resolve()

    return None


# ─────────────────────────────────────────────────────────────────────────
# Per-file merge
# ─────────────────────────────────────────────────────────────────────────
def _load_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[skip] cannot parse {path}: {e}", file=sys.stderr)
        return None


def _num_questions(obj: dict) -> int:
    """Count real question items in either qa_result / result / questions."""
    for key in ("qa_result", "result", "questions"):
        arr = obj.get(key)
        if isinstance(arr, list):
            return sum(1 for x in arr if isinstance(x, dict) and "question" in x)
    return 0


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True,
                    help="Directory produced by download_hf.py "
                         "(contains bench_cn/ and/or bench_en/).")
    ap.add_argument("--out_qa_root", required=True,
                    help="Where to write the language-tagged, flat QA tree.")
    ap.add_argument("--out_index", required=True,
                    help="Where to write qa_index.jsonl.")
    ap.add_argument("--languages", nargs="+", default=["cn", "en"],
                    help="Which language splits to include. "
                         "Note: the CN split is emitted with lang code 'zh', "
                         "the EN split with lang code 'en'.")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    out_qa_root = Path(args.out_qa_root).expanduser().resolve()
    out_index = Path(args.out_index).expanduser().resolve()
    out_qa_root.mkdir(parents=True, exist_ok=True)
    out_index.parent.mkdir(parents=True, exist_ok=True)

    lang_map = {"cn": "zh", "zh": "zh", "en": "en"}
    per_lang_stats: Dict[str, Dict[str, int]] = {}
    total_kept = 0
    total_skipped = 0

    with out_index.open("w", encoding="utf-8") as fout:
        for lang_arg in args.languages:
            lang_key = lang_arg.lower()
            if lang_key not in ("cn", "zh", "en"):
                print(f"[warn] ignoring unknown language {lang_arg!r}",
                      file=sys.stderr)
                continue
            lang = lang_map[lang_key]

            # find upstream QA root (bench_cn/QA_cn or bench_en/QA_en)
            hf_lang = "cn" if lang == "zh" else "en"
            qa_root = _find_qa_root(raw_dir, hf_lang)
            if qa_root is None:
                print(f"[skip] language={lang}: no QA root found under {raw_dir}",
                      file=sys.stderr)
                continue
            print(f"[scan] {lang}: {qa_root}")

            stats = {"kept": 0, "no_audio": 0, "no_questions": 0, "bad_json": 0}

            for scenario, qa_len, seg, level, task_stem, jf in _iter_qa_files(qa_root):
                obj = _load_json(jf)
                if obj is None:
                    stats["bad_json"] += 1
                    total_skipped += 1
                    continue

                n_q = _num_questions(obj)
                if n_q == 0:
                    stats["no_questions"] += 1
                    total_skipped += 1
                    continue

                # resolve audio
                hint = obj.get("source_audio")
                audio_path = _resolve_audio_path(
                    raw_dir=raw_dir, lang=hf_lang,
                    scenario=scenario, seg_id=seg, hint=hint,
                )
                if audio_path is None:
                    stats["no_audio"] += 1
                    total_skipped += 1
                    continue

                # rewrite source_audio to absolute path so step2 can read it
                # regardless of cwd
                obj["source_audio"] = str(audio_path)

                # attach normalised metadata used by step3
                obj.setdefault("_msu_meta", {})
                obj["_msu_meta"].update({
                    "language": lang,
                    "scenario": scenario,
                    "qa_len": qa_len,
                    "segment": seg,
                    "level": level,
                    "task_stem": task_stem,
                })

                # write to flat, language-tagged tree
                out_path = (out_qa_root / lang / scenario / qa_len / seg
                            / level / f"{task_stem}.json")
                _write_json(obj, out_path)

                fout.write(json.dumps({
                    "language": lang,
                    "scenario": scenario,
                    "qa_len": qa_len,
                    "segment": seg,
                    "level": level,
                    "task_stem": task_stem,
                    "n_questions": n_q,
                    "qa_json": str(out_path),
                    "source_audio": str(audio_path),
                }, ensure_ascii=False) + "\n")

                stats["kept"] += 1
                total_kept += 1

            per_lang_stats[lang] = stats
            print(f"[done] {lang}: kept={stats['kept']}  "
                  f"no_audio={stats['no_audio']}  "
                  f"no_questions={stats['no_questions']}  "
                  f"bad_json={stats['bad_json']}")

    print(f"\n[summary] total kept={total_kept}, total skipped={total_skipped}")
    print(f"[summary] index written to: {out_index}")
    print(f"[summary] merged QA tree:   {out_qa_root}")


if __name__ == "__main__":
    main()
