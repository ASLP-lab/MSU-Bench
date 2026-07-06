#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 (reference backend) — score MSU-Bench with Doubao-Seed.

This is the *reference* backend that mirrors the Gemini pipeline but
targets a Doubao "passthrough" chat/completions endpoint (OpenAI-shaped
body with an ``input_audio.url`` content part). It is provided so
you can see how to plug a non-Gemini provider into the same
step-1 → step-2 → step-3 flow.

Because Doubao expects a **URL** for the audio (not inline bytes), you
either:
  * host your audio on a small HTTP server and pass ``--audio_url_prefix``
    that maps ``/absolute/path`` → ``http://host/…``; or
  * pass ``--audio_url_map key=value`` pairs to rewrite prefixes.

The default backend for MSU-Bench is Gemini (see run_gemini.py); this file
is here purely as a reference template.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from question_builder import (  # noqa: E402
    build_prompt_text,
    extract_single_letter,
    infer_level,
    parse_options_letters,
    split_speaker_meta_and_questions,
)


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


def _apply_url_map(local_path: str, mapping: List[Tuple[str, str]]) -> str:
    for old, new in mapping:
        if local_path.startswith(old):
            return new + local_path[len(old):]
    return local_path


def _build_headers(app_id: str, app_key: str, provider: str,
                   model: str, timeout_s: int) -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": (
            f"Bearer {app_id}:{app_key}"
            f"?provider={provider}&model={model}&timeout={timeout_s}"
        ),
    }


def _build_payload(audio_url: str, prompt: str, model: str,
                   max_tokens: int) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"url": audio_url}},
                {"type": "text", "text": prompt},
            ],
        }],
        "stream": False,
        "max_tokens": max_tokens,
    }


def _call_doubao(session: requests.Session,
                 url: str, headers: Dict[str, str],
                 payload: Dict[str, Any],
                 timeout_s: int, max_retries: int,
                 retry_delay: float) -> Tuple[Optional[str], Optional[str]]:
    last_err: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.post(url, headers=headers, json=payload,
                                timeout=timeout_s)
        except Exception as e:
            last_err = f"http_error: {e}"
            time.sleep(min(retry_delay * (2 ** attempt), 30.0))
            continue
        if resp.status_code == 200:
            try:
                return resp.json()["choices"][0]["message"]["content"], None
            except Exception as e:
                last_err = f"bad_shape: {e}; body={resp.text[:200]}"
                time.sleep(min(retry_delay * (2 ** attempt), 30.0))
                continue
        last_err = f"http_{resp.status_code}: {resp.text[:200]}"
        # 429 → longer backoff
        if resp.status_code == 429:
            time.sleep(min(15.0 + 5.0 * attempt, 60.0))
        else:
            time.sleep(min(retry_delay * (2 ** attempt), 30.0))
    return None, last_err


def _score_one_file(qa_json: Path, input_root: Path, output_root: Path,
                    args: argparse.Namespace,
                    url_map: List[Tuple[str, str]]) -> Tuple[str, str]:
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
            pass

    try:
        with qa_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return "error", f"json_load_failed: {e}"

    src = data.get("source_audio")
    if not src:
        return "error", "missing source_audio"
    audio_url = _apply_url_map(src, url_map)
    if not (audio_url.startswith("http://") or audio_url.startswith("https://")):
        return "error", f"non-URL source_audio (add --audio_url_map): {src}"

    speaker_meta, questions = split_speaker_meta_and_questions(data)
    language = (data.get("_msu_meta", {}) or {}).get("language", "zh")

    session = requests.Session()
    headers = _build_headers(args.app_id, args.app_key, args.provider,
                             args.model, args.timeout)
    doubao_url = args.url

    for q in questions:
        if not args.overwrite and q.get("reference_result"):
            continue
        allowed = parse_options_letters(q.get("options") or [])
        prompt = build_prompt_text(q, language=language)
        payload = _build_payload(audio_url, prompt, args.model,
                                 args.max_tokens)

        raw, err = _call_doubao(session, doubao_url, headers, payload,
                                args.timeout, args.retries, args.retry_delay)
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
        "backend": "doubao",
        "model": args.model,
        "url": args.url,
        "time_unix": int(time.time()),
    })
    _atomic_write_json(data, out_path)
    return "ok", ""


def _worker_main(worker_id: int, shard: List[Path],
                 input_root: Path, output_root: Path,
                 args: argparse.Namespace,
                 url_map: List[Tuple[str, str]],
                 progress_q: mp.Queue) -> None:
    for qa_json in shard:
        try:
            status, err = _score_one_file(qa_json, input_root,
                                          output_root, args, url_map)
        except Exception as e:
            status, err = "error", f"{type(e).__name__}: {e}"
        progress_q.put({
            "ts": _now_str(), "worker": worker_id,
            "qa_json": str(qa_json), "status": status, "error": err,
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True)
    ap.add_argument("--output_root", required=True)

    ap.add_argument("--url", default=os.environ.get("DOUBAO_URL", ""))
    ap.add_argument("--app_id", default=os.environ.get("DOUBAO_APP_ID", ""))
    ap.add_argument("--app_key", default=os.environ.get("DOUBAO_APP_KEY", ""))
    ap.add_argument("--model", default=os.environ.get("DOUBAO_MODEL",
                                                      "doubao-seed-2-0-lite"))
    ap.add_argument("--provider", default=os.environ.get("DOUBAO_PROVIDER",
                                                         "doubao"))

    ap.add_argument("--num_workers", type=int,
                    default=int(os.environ.get("NUM_WORKERS", 8)))
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--retry_delay", type=float, default=2.0)
    ap.add_argument("--max_tokens", type=int, default=64)

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--audio_url_map", nargs="*", default=[],
                    help="local_prefix=http_prefix pairs, applied to "
                         "source_audio before submitting to Doubao.")

    ap.add_argument("--start_method", default="spawn",
                    choices=["spawn", "forkserver"])
    args = ap.parse_args()

    if not (args.url and args.app_id and args.app_key):
        raise SystemExit("Missing --url / --app_id / --app_key "
                         "(or DOUBAO_URL / DOUBAO_APP_ID / DOUBAO_APP_KEY env).")

    url_map: List[Tuple[str, str]] = []
    for kv in args.audio_url_map:
        if "=" in kv:
            a, b = kv.split("=", 1)
            url_map.append((a, b))

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in input_root.rglob("*.json") if p.is_file())
    if not files:
        raise SystemExit(f"No QA .json under {input_root}")

    shards: List[List[Path]] = [[] for _ in range(args.num_workers)]
    for i, p in enumerate(files):
        shards[i % args.num_workers].append(p)

    ctx = mp.get_context(args.start_method)
    progress_q: mp.Queue = ctx.Queue()

    procs: List[mp.Process] = []
    for wid in range(args.num_workers):
        p = ctx.Process(target=_worker_main,
                        args=(wid, shards[wid], input_root, output_root,
                              args, url_map, progress_q),
                        daemon=False)
        p.start()
        procs.append(p)

    total = len(files)
    n_ok = n_skip = n_err = done = 0
    pbar = tqdm(total=total, desc=f"Doubao scoring [{args.model}]",
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
