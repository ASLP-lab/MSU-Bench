#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 (default backend) — score MSU-Bench with Gemini 2.5 Flash.

Reads the language-tagged QA tree written by step 1
(``work/data/qa_merged/<lang>/<scenario>/<qa_len>/<seg>/<level>/<task>.json``)
and, for every question in every file, calls a Gemini
``generateContent``-compatible endpoint. The reply is parsed down to a
single letter (A/B/C/D) and written back to the question dict as
``reference_result`` (with ``reference_raw`` / ``reference_error`` for
debugging). Files are written to
``work/scoring/<model>/<same relative path>`` so that step 3 can walk the
output tree without further plumbing.

The pipeline is embarrassingly parallel per QA file. Progress is printed
to stdout via ``tqdm`` and mirrored into a JSONL log for post-mortems.

Environment:
    Configure your endpoint by ``source configs/gemini.env`` first,
    or pass ``--base_url / --api_key / --model`` explicitly.
"""

from __future__ import annotations

import argparse
import base64
import json
import multiprocessing as mp
import os
import random
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

# Local shared helpers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from question_builder import (  # noqa: E402
    build_parts_for_question,
    extract_single_letter,
    infer_level,
    parse_options_letters,
    split_speaker_meta_and_questions,
)


# ─────────────────────────────────────────────────────────────────────────
# I/O utils
# ─────────────────────────────────────────────────────────────────────────
def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _atomic_write_json(obj: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json",
                               dir=str(out_path.parent))
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, out_path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def _list_all_qa_json(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*.json") if p.is_file())


# ─────────────────────────────────────────────────────────────────────────
# Gemini call
# ─────────────────────────────────────────────────────────────────────────
def _gemini_generate(
    session: requests.Session,
    base_url: str,
    api_key: str,
    model: str,
    parts: List[Dict[str, Any]],
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    timeout_s: int,
    max_retries: int,
) -> Tuple[Optional[str], Optional[str]]:
    url = f"{base_url.rstrip('/')}/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": temperature,
            "topP": top_p,
            "maxOutputTokens": max_output_tokens,
        },
    }
    headers = {"Content-Type": "application/json"}
    retry_status = {429, 500, 502, 503, 504, 524}
    last_err: Optional[str] = None

    for attempt in range(max_retries + 1):
        try:
            resp = session.post(url, headers=headers, json=payload,
                                timeout=timeout_s)
            if resp.status_code == 200:
                data = resp.json()
                try:
                    return (data["candidates"][0]["content"]["parts"][0]
                            ["text"] or "").strip(), None
                except Exception:
                    return json.dumps(data, ensure_ascii=False)[:500], None
            if resp.status_code in retry_status:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if attempt < max_retries:
                    time.sleep(min(2 ** attempt + random.uniform(0, 1), 30.0))
                    continue
                return None, f"retry_exhausted: {last_err}"
            return None, f"fatal_http {resp.status_code}: {resp.text[:200]}"
        except requests.Timeout:
            last_err = "timeout"
            if attempt < max_retries:
                time.sleep(2)
                continue
            return None, "final_timeout"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                time.sleep(2)
                continue
            return None, f"final_error: {last_err}"
    return None, f"exhausted: {last_err}"


# ─────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────
def _score_one_file(
    qa_json: Path,
    input_root: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> Tuple[str, str]:
    """Score every unanswered question in `qa_json` and write the augmented
    file to the mirrored path under `output_root`. Returns (status, err)."""
    rel = qa_json.relative_to(input_root)
    out_path = output_root / rel

    if not args.overwrite and out_path.exists():
        try:
            with out_path.open("r", encoding="utf-8") as f:
                old = json.load(f)
            _, old_qs = split_speaker_meta_and_questions(old)
            if old_qs and all(q.get("reference_result") for q in old_qs):
                return "skipped", ""
        except Exception:
            pass  # re-score if we can't parse the previous output

    try:
        with qa_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return "error", f"json_load_failed: {e}"

    source_audio = data.get("source_audio")
    if not source_audio or not os.path.exists(source_audio):
        return "error", f"source_audio not found: {source_audio}"

    level = infer_level(str(qa_json), data)
    speaker_meta, questions = split_speaker_meta_and_questions(data)
    language = (data.get("_msu_meta", {}) or {}).get("language", "zh")

    session = requests.Session()

    for q in questions:
        if not args.overwrite and q.get("reference_result"):
            continue

        allowed = parse_options_letters(q.get("options") or [])

        try:
            parts = build_parts_for_question(
                qa_json=str(qa_json),
                level=level,
                speaker_meta=speaker_meta,
                source_audio=source_audio,
                question=q,
                language=language,
                audio_fmt=args.audio_format,
                sample_rate=args.sample_rate,
                mono=args.mono,
                mp3_bitrate=args.mp3_bitrate,
            )
        except Exception as e:
            q["reference_result"] = ""
            q["reference_error"] = f"build_parts_failed: {e}"
            continue

        raw, err = _gemini_generate(
            session=session,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            parts=parts,
            temperature=args.temperature,
            top_p=args.top_p,
            max_output_tokens=args.max_output_tokens,
            timeout_s=args.timeout,
            max_retries=args.retries,
        )

        if err:
            q["reference_result"] = ""
            q["reference_error"] = err
            q["reference_raw"] = raw or ""
            continue

        q["reference_raw"] = raw or ""
        letter = extract_single_letter(raw or "", allowed)
        if letter is None:
            q["reference_result"] = ""
            q["reference_error"] = f"unparseable: {(raw or '')[:200]}"
        else:
            q["reference_result"] = letter
            q.pop("reference_error", None)

    data.setdefault("_reference_meta", {})
    data["_reference_meta"].update({
        "backend": "gemini",
        "model": args.model,
        "base_url": args.base_url,
        "time_unix": int(time.time()),
    })
    _atomic_write_json(data, out_path)
    return "ok", ""


def _worker_main(worker_id: int,
                 shard: List[Path],
                 input_root: Path,
                 output_root: Path,
                 args: argparse.Namespace,
                 progress_q: mp.Queue) -> None:
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"gemini_worker{worker_id}.log"

    def _log(msg: str) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{_now_str()}][W{worker_id}] {msg}\n")

    for qa_json in shard:
        try:
            status, err = _score_one_file(qa_json, input_root, output_root, args)
        except Exception as e:  # pragma: no cover
            status = "error"
            err = f"{type(e).__name__}: {e}"
            if args.print_trace:
                _log(traceback.format_exc(limit=8))
        progress_q.put({
            "ts": _now_str(),
            "worker": worker_id,
            "qa_json": str(qa_json),
            "status": status,
            "error": err,
        })


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True,
                    help="Merged QA root produced by step1 "
                         "(work/data/qa_merged).")
    ap.add_argument("--output_root", required=True,
                    help="Where to write per-QA scored JSONs "
                         "(work/scoring/<model>).")

    ap.add_argument("--base_url", default=os.environ.get("GEMINI_BASE_URL", ""),
                    help="Gemini endpoint base URL (no trailing /v1beta).")
    ap.add_argument("--api_key", default=os.environ.get("GEMINI_API_KEY", ""),
                    help="Gemini API key.")
    ap.add_argument("--model", default=os.environ.get("GEMINI_MODEL",
                                                      "gemini-2.5-flash"))
    ap.add_argument("--num_workers", type=int,
                    default=int(os.environ.get("NUM_WORKERS", 8)))

    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_output_tokens", type=int, default=8)

    ap.add_argument("--audio_format", default="mp3", choices=["mp3", "wav"])
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--mono", action="store_true", default=True)
    ap.add_argument("--mp3_bitrate", default="64k")

    ap.add_argument("--overwrite", action="store_true",
                    help="Re-score even if reference_result already present.")
    ap.add_argument("--log_dir", default="./_logs_gemini")
    ap.add_argument("--print_trace", action="store_true")
    ap.add_argument("--start_method", default="spawn",
                    choices=["spawn", "forkserver"])
    args = ap.parse_args()

    if not args.base_url or not args.api_key:
        raise SystemExit("Missing --base_url / --api_key "
                         "(or GEMINI_BASE_URL / GEMINI_API_KEY env).")

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    files = _list_all_qa_json(input_root)
    if not files:
        raise SystemExit(f"No QA .json under {input_root}")

    shards: List[List[Path]] = [[] for _ in range(args.num_workers)]
    for i, p in enumerate(files):
        shards[i % args.num_workers].append(p)

    ctx = mp.get_context(args.start_method)
    progress_q: mp.Queue = ctx.Queue()

    procs: List[mp.Process] = []
    for wid in range(args.num_workers):
        p = ctx.Process(
            target=_worker_main,
            args=(wid, shards[wid], input_root, output_root, args, progress_q),
            daemon=False,
        )
        p.start()
        procs.append(p)

    total = len(files)
    n_ok = n_skip = n_err = done = 0
    pbar = tqdm(total=total, desc=f"Gemini scoring [{args.model}]",
                dynamic_ncols=True)

    while done < total:
        evt = progress_q.get()
        done += 1
        pbar.update(1)
        st = evt.get("status")
        if st == "ok":
            n_ok += 1
        elif st == "skipped":
            n_skip += 1
        else:
            n_err += 1
            tqdm.write(f"[error] {evt.get('qa_json')}: {evt.get('error')}")

    pbar.close()
    for p in procs:
        p.join()

    print(f"[done] ok={n_ok}  skipped={n_skip}  errors={n_err}  total={total}")


if __name__ == "__main__":
    main()
