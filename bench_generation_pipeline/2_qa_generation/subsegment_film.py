# -*- coding: utf-8 -*-
#
# 将 step1 的 4-6.5 分钟分段进一步切分为约 5 分钟的子段(long_seg/short_seg).
#

# python3 $SPEAKER_BENCH_ROOT/subsegment_film.py \
#   --input_dir $SPEAKER_BENCH_ROOT/data_movie/labeled_cn_data/huoshan_output/gemini_step3/data \
#   --prompt_file $SPEAKER_BENCH_ROOT/subsegment_film_prompt.txt \
#   --out_dir $SPEAKER_BENCH_ROOT/data_movie/labeled_cn_data/short_long_out \
#   --model gemini-2.5-pro \
#   --max_concurrent 64 \
#   --short_ratio 0.5 \
#   --short_min_sec 60 \
#   --short_max_sec 120 \
#   --seed 2026

# python3 $SPEAKER_BENCH_ROOT/subsegment_film.py \
#   --input_dir $SPEAKER_BENCH_ROOT/data_movie/labeled_en_data/huoshan_output/gemini_step3/data \
#   --prompt_file $SPEAKER_BENCH_ROOT/subsegment_film_prompt.txt \
#   --out_dir $SPEAKER_BENCH_ROOT/data_movie/labeled_en_data/short_long_out \
#   --model gemini-2.5-pro \
#   --max_concurrent 64 \
#   --short_ratio 0.5 \
#   --short_min_sec 60 \
#   --short_max_sec 120 \
#   --seed 2026

# python3 $SPEAKER_BENCH_ROOT/subsegment_film.py \
#   --input_dir $SPEAKER_BENCH_ROOT/data_en_podcast/quality_results/samples_jsons \
#   --prompt_file $SPEAKER_BENCH_ROOT/subsegment_film_prompt.txt \
#   --out_dir $SPEAKER_BENCH_ROOT/data_en_podcast/short_long_out \
#   --model gemini-2.5-pro \
#   --max_concurrent 64 \
#   --short_ratio 0.5 \
#   --short_min_sec 60 \
#   --short_max_sec 120 \
#   --seed 2026 \
#   --error_log $SPEAKER_BENCH_ROOT/data_en_podcast/short_long_out/subsegment_error.log




#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import json
import base64
import mimetypes
import random
import argparse
import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Deque
from collections import deque

import aiohttp
import aiofiles

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# -----------------------------
# Utils
# -----------------------------
def strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", t)
        t = re.sub(r"\n```$", "", t)
    # try extract JSON object
    if "{" in t and "}" in t:
        i = t.find("{")
        j = t.rfind("}")
        if 0 <= i < j:
            return t[i : j + 1].strip()
    return t


def parse_json_loose(text: str) -> Tuple[Optional[Any], Optional[str]]:
    try:
        cleaned = strip_code_fences(text)
        return json.loads(cleaned), None
    except Exception as e:
        return None, f"JSON parse failed: {e}"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


async def read_text(p: Path) -> str:
    async with aiofiles.open(p, "r", encoding="utf-8") as f:
        return (await f.read()).strip()


async def write_json(p: Path, obj: Any) -> None:
    ensure_dir(p.parent)
    async with aiofiles.open(p, "w", encoding="utf-8") as f:
        await f.write(json.dumps(obj, ensure_ascii=False, indent=2))


async def append_line(p: Path, line: str) -> None:
    ensure_dir(p.parent)
    async with aiofiles.open(p, "a", encoding="utf-8") as f:
        await f.write(line.rstrip("\n") + "\n")


def remove_tmp_keys_in_segment(seg: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in seg.items() if ".tmp" not in k}


# -----------------------------
# Build segment list for prompt
# -----------------------------
def build_segments_block(label_obj: Dict[str, Any], max_text_len: int = 80) -> str:
    """
    把 audio_segments 变成“每行一个 JSON”的简洁列表，便于 Gemini 输出 idx。
    """
    lines = []
    segs = label_obj.get("audio_segments", [])
    for idx, s in enumerate(segs):
        if not isinstance(s, dict):
            continue
        seg_id = str(s.get("segment_id", f"seg_{idx:04d}"))
        spk = str(s.get("speaker_id", ""))
        try:
            st_ms = int(s.get("start_time"))
            ed_ms = int(s.get("end_time"))
        except Exception:
            continue
        st = st_ms / 1000.0
        ed = ed_ms / 1000.0
        txt = str(s.get("text", "")).strip().replace("\n", " ")
        if len(txt) > max_text_len:
            txt = txt[:max_text_len] + "…"

        lines.append(json.dumps({
            "idx": idx,
            "segment_id": seg_id,
            "speaker_id": spk,
            "start_sec": round(st, 3),
            "end_sec": round(ed, 3),
            "text": txt
        }, ensure_ascii=False))

    return "\n".join(lines)


# -----------------------------
# Audio helpers
# -----------------------------
def run_ffmpeg(cmd: List[str]) -> Tuple[bool, str]:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        ok = p.returncode == 0
        msg = (p.stderr.decode("utf-8", errors="ignore") + "\n" + p.stdout.decode("utf-8", errors="ignore")).strip()
        return ok, msg
    except Exception as e:
        return False, str(e)


def cut_audio_ffmpeg(
    ffmpeg_bin: str,
    src_audio: Path,
    dst_audio: Path,
    start_sec: float,
    end_sec: float,
    force_mono_channel0: bool = False,
) -> Tuple[bool, str]:
    ensure_dir(dst_audio.parent)
    base_cmd = [
        ffmpeg_bin, "-y",
        "-ss", f"{start_sec:.3f}",
        "-to", f"{end_sec:.3f}",
        "-i", str(src_audio),
    ]
    if force_mono_channel0:
        cmd1 = base_cmd + ["-map_channel", "0.0.0", "-ac", "1", "-c:a", "pcm_s16le", str(dst_audio)]
        ok, msg = run_ffmpeg(cmd1)
        if ok:
            return True, msg
        cmd2 = base_cmd + ["-filter:a", "pan=mono|c0=c0", "-c:a", "pcm_s16le", str(dst_audio)]
        ok2, msg2 = run_ffmpeg(cmd2)
        return ok2, msg + "\n---fallback---\n" + msg2

    cmd = base_cmd + ["-c:a", "pcm_s16le", str(dst_audio)]
    return run_ffmpeg(cmd)


def read_audio_bytes(ffmpeg_bin: str, audio_path: Path, channel0_only: bool = False) -> Tuple[Optional[bytes], Optional[str], str]:
    mime_type, _ = mimetypes.guess_type(str(audio_path))
    if not mime_type:
        mime_type = "audio/wav" if audio_path.suffix.lower() == ".wav" else "audio/mpeg"

    if not channel0_only:
        try:
            return audio_path.read_bytes(), None, mime_type
        except Exception as e:
            return None, f"read audio failed: {e}", mime_type

    # 用 ffmpeg 输出单通道 wav 到 stdout
    cmd = [
        ffmpeg_bin, "-v", "error",
        "-i", str(audio_path),
        "-map_channel", "0.0.0",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-f", "wav",
        "pipe:1"
    ]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if p.returncode == 0:
            return p.stdout, None, "audio/wav"
        return None, p.stderr.decode("utf-8", errors="ignore"), "audio/wav"
    except Exception as e:
        return None, str(e), "audio/wav"


# -----------------------------
# Gemini call
# -----------------------------
async def gemini_generate(
    session: aiohttp.ClientSession,
    api_base: str,
    api_key: str,
    model: str,
    audio_bytes: bytes,
    mime_type: str,
    prompt_text: str,
    max_retries: int = 4,
    timeout_sec: int = 1000,
) -> Tuple[Optional[Any], Optional[str], Optional[str]]:
    url = f"{api_base.rstrip('/')}/v1beta/models/{model}:generateContent?key={api_key}"
    b64 = base64.b64encode(audio_bytes).decode("utf-8")

    req = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_text},
                    {"inline_data": {"mime_type": mime_type, "data": b64}},
                ]
            }
        ]
    }
    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries + 1):
        try:
            async with session.post(url, headers=headers, json=req, timeout=timeout_sec) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    raw_text = None
                    try:
                        raw_text = data["candidates"][0]["content"]["parts"][0].get("text")
                    except Exception:
                        raw_text = None

                    if not raw_text:
                        return data, None, json.dumps(data, ensure_ascii=False)

                    parsed, jerr = parse_json_loose(raw_text)
                    if jerr:
                        return None, jerr, raw_text
                    return parsed, None, raw_text

                if resp.status in (429, 500, 502, 503, 504, 524):
                    txt = await resp.text()
                    if attempt < max_retries:
                        wait = (2 ** attempt) + random.uniform(0, 1.0)
                        await asyncio.sleep(wait)
                        continue
                    return None, f"HTTP {resp.status} retry exhausted: {txt[:300]}", None

                txt = await resp.text()
                return None, f"HTTP {resp.status}: {txt[:300]}", None

        except asyncio.TimeoutError:
            if attempt < max_retries:
                await asyncio.sleep((2 ** attempt) + 0.5)
                continue
            return None, f"timeout retry exhausted (timeout_sec={timeout_sec})", None
        except Exception as e:
            if attempt < max_retries:
                await asyncio.sleep((2 ** attempt) + 0.5)
                continue
            return None, f"request failed retry exhausted: {e}", None

    return None, "unknown error", None


# -----------------------------
# Segment selection logic
# -----------------------------
def film_and_part_from_path(json_path: Path) -> Tuple[str, str]:
    film_name = json_path.parent.name
    part_name = json_path.stem
    return film_name, part_name


def find_all_part_jsons(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.rglob("part*.json") if p.is_file()])


def get_seg_bounds_ms(segs: List[Dict[str, Any]], start_idx: int, end_idx: int) -> Optional[Tuple[int, int]]:
    if start_idx < 0 or end_idx >= len(segs) or end_idx < start_idx:
        return None
    try:
        s_ms = int(segs[start_idx]["start_time"])
        e_ms = int(segs[end_idx]["end_time"])
        return s_ms, e_ms
    except Exception:
        return None


def duration_sec_from_bounds(s_ms: int, e_ms: int) -> float:
    return max(0.0, (e_ms - s_ms) / 1000.0)


def adjust_idx_range_to_duration(
    segs: List[Dict[str, Any]],
    start_idx: int,
    end_idx: int,
    min_sec: float,
    max_sec: float,
) -> Tuple[int, int]:
    """
    保持“连续 segments 区间”，必要时用相邻 segment 扩展或收缩以满足时长约束。
    只改变边界 idx，不改内部顺序。
    """
    n = len(segs)
    start_idx = max(0, min(start_idx, n - 1))
    end_idx = max(0, min(end_idx, n - 1))
    if end_idx < start_idx:
        start_idx, end_idx = end_idx, start_idx

    # shrink if too long
    while True:
        b = get_seg_bounds_ms(segs, start_idx, end_idx)
        if not b:
            break
        dur = duration_sec_from_bounds(*b)
        if dur <= max_sec or end_idx == start_idx:
            break
        # 优先从更长一侧缩
        end_idx -= 1

    # expand if too short
    while True:
        b = get_seg_bounds_ms(segs, start_idx, end_idx)
        if not b:
            break
        dur = duration_sec_from_bounds(*b)
        if dur >= min_sec:
            break
        if end_idx < n - 1:
            end_idx += 1
        elif start_idx > 0:
            start_idx -= 1
        else:
            break

    return start_idx, end_idx


def speaker_count_in_idx_range(segs: List[Dict[str, Any]], start_idx: int, end_idx: int) -> int:
    spks = set()
    for i in range(start_idx, end_idx + 1):
        sid = segs[i].get("speaker_id")
        if sid:
            spks.add(str(sid))
    return len(spks)


def slice_label_by_idx_range(
    label_obj: Dict[str, Any],
    start_idx: int,
    end_idx: int,
    new_audio_path: str,
) -> Dict[str, Any]:
    """
    仅保留 idx 区间内 segments，并把 start/end 改成相对时间（毫秒字符串），不重新标注。
    """
    segs = label_obj.get("audio_segments", [])
    b = get_seg_bounds_ms(segs, start_idx, end_idx)
    if not b:
        raise ValueError("invalid idx range for bounds")
    start_ms, end_ms = b

    out = dict(label_obj)
    out["audio_path"] = new_audio_path

    new_segs: List[Dict[str, Any]] = []
    for i in range(start_idx, end_idx + 1):
        seg = segs[i]
        if not isinstance(seg, dict):
            continue
        ns = remove_tmp_keys_in_segment(seg)

        s_ms = int(seg["start_time"])
        e_ms = int(seg["end_time"])
        # 相对平移
        ns["start_time"] = str(s_ms - start_ms)
        ns["end_time"] = str(e_ms - start_ms)

        new_segs.append(ns)

    out["audio_segments"] = new_segs
    out["_slice_by_segments"] = {
        "start_index": start_idx,
        "end_index": end_idx,
        "start_segment_id": str(segs[start_idx].get("segment_id", "")),
        "end_segment_id": str(segs[end_idx].get("segment_id", "")),
        "start_ms_abs": start_ms,
        "end_ms_abs": end_ms,
    }
    return out


def resolve_idx_range_from_gemini(
    parsed: Dict[str, Any],
    segs: List[Dict[str, Any]],
) -> Optional[Tuple[int, int]]:
    """
    支持两种输出：
    - selected.start_index / selected.end_index
    - selected.segment_ids: [...]  (取其在 segs 中的最小/最大 idx)
    """
    if not isinstance(parsed, dict):
        return None
    if not parsed.get("keep", False):
        return None

    sel = parsed.get("selected")
    if not isinstance(sel, dict):
        return None

    # 1) idx
    si = sel.get("start_index")
    ei = sel.get("end_index")
    if isinstance(si, (int, float)) and isinstance(ei, (int, float)):
        return int(si), int(ei)

    # 2) segment_ids
    ids = sel.get("segment_ids")
    if isinstance(ids, list) and ids:
        id2idx = {}
        for i, s in enumerate(segs):
            sid = s.get("segment_id")
            if sid:
                id2idx[str(sid)] = i
        idxs = [id2idx.get(str(x)) for x in ids]
        idxs = [i for i in idxs if i is not None]
        if idxs:
            return min(idxs), max(idxs)

    # 3) start/end segment id
    sid = sel.get("start_segment_id")
    eid = sel.get("end_segment_id")
    if isinstance(sid, str) and isinstance(eid, str):
        id2idx = {}
        for i, s in enumerate(segs):
            ss = s.get("segment_id")
            if ss:
                id2idx[str(ss)] = i
        if sid in id2idx and eid in id2idx:
            return id2idx[sid], id2idx[eid]

    return None


# -----------------------------
# Main pipeline: short/long
# -----------------------------
async def process_one_short(
    session: aiohttp.ClientSession,
    api_base: str,
    api_key: str,
    model: str,
    prompt_template: str,
    json_path: Path,
    out_dir: Path,
    ffmpeg_bin: str,
    short_min_sec: float,
    short_max_sec: float,
    max_retries: int,
    timeout_sec: int,
    channel0_for_gemini: bool,
    channel0_for_output: bool,
    error_log: Path,
) -> Tuple[bool, str]:

    try:
        label_obj = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        await append_line(error_log, f"[SHORT][READ_JSON_FAIL] {json_path} | {e}")
        return False, "read_json_fail"

    audio_path = Path(label_obj.get("audio_path", ""))
    if not audio_path.exists():
        await append_line(error_log, f"[SHORT][AUDIO_NOT_FOUND] {json_path} | audio_path={audio_path}")
        return False, "audio_not_found"

    segs = label_obj.get("audio_segments", [])
    if not isinstance(segs, list) or len(segs) == 0:
        await append_line(error_log, f"[SHORT][NO_SEGMENTS] {json_path}")
        return False, "no_segments"

    film_name, part_name = film_and_part_from_path(json_path)

    # 把 segments 列表塞进 prompt
    seg_block = build_segments_block(label_obj)
    prompt_text = (
        prompt_template.strip()
        + "\n\n"
        + "【Segments 列表（每行一个 JSON）】\n"
        + seg_block
        + "\n"
    )

    audio_bytes, aerr, mime_type = read_audio_bytes(ffmpeg_bin, audio_path, channel0_only=channel0_for_gemini)
    if aerr or audio_bytes is None:
        await append_line(error_log, f"[SHORT][READ_AUDIO_FAIL] {json_path} | {aerr}")
        return False, "read_audio_fail"

    parsed, err, raw_text = await gemini_generate(
        session=session,
        api_base=api_base,
        api_key=api_key,
        model=model,
        audio_bytes=audio_bytes,
        mime_type=mime_type,
        prompt_text=prompt_text,
        max_retries=max_retries,
        timeout_sec=timeout_sec,
    )
    if err or parsed is None:
        await append_line(error_log, f"[SHORT][GEMINI_FAIL] {json_path} | {err}")
        return False, "gemini_fail"

    idx_range = resolve_idx_range_from_gemini(parsed, segs)
    if idx_range is None:
        sel_dir = out_dir / "short_seg" / film_name / part_name / "select"
        ensure_dir(sel_dir)
        await write_json(sel_dir / f"{part_name}.select_failed.json", {
            "source_part_json": str(json_path),
            "source_audio": str(audio_path),
            "gemini": parsed,
            "raw": raw_text,
        })
        return False, "no_selected_range"

    start_idx, end_idx = idx_range
    start_idx, end_idx = adjust_idx_range_to_duration(segs, start_idx, end_idx, short_min_sec, short_max_sec)

    # 至少 2 个说话人（硬约束兜底）
    if speaker_count_in_idx_range(segs, start_idx, end_idx) < 2:
        return False, "speaker_cnt<2"

    b = get_seg_bounds_ms(segs, start_idx, end_idx)
    if not b:
        return False, "invalid_bounds"
    start_ms, end_ms = b
    start_sec, end_sec = start_ms / 1000.0, end_ms / 1000.0
    seg_name = f"{part_name}__idx{start_idx:03d}-{end_idx:03d}__{start_sec:.2f}-{end_sec:.2f}"

    base = out_dir / "short_seg" / film_name / part_name
    out_wav = base / "wav" / f"{seg_name}.wav"
    out_json = base / "json" / f"{seg_name}.json"
    out_select = base / "select" / f"{seg_name}.select.json"

    ok, msg = cut_audio_ffmpeg(
        ffmpeg_bin=ffmpeg_bin,
        src_audio=audio_path,
        dst_audio=out_wav,
        start_sec=start_sec,
        end_sec=end_sec,
        force_mono_channel0=channel0_for_output,
    )
    if not ok:
        await append_line(error_log, f"[SHORT][FFMPEG_CUT_FAIL] {json_path} | {seg_name} | {msg[:300]}")
        return False, "ffmpeg_cut_fail"

    sliced = slice_label_by_idx_range(label_obj, start_idx, end_idx, new_audio_path=str(out_wav))
    sliced["_source_part_json"] = str(json_path)
    sliced["_source_audio_path"] = str(audio_path)

    await write_json(out_json, sliced)
    await write_json(out_select, {
        "source_part_json": str(json_path),
        "source_audio": str(audio_path),
        "chosen": {"start_index": start_idx, "end_index": end_idx, "start_sec": start_sec, "end_sec": end_sec},
        "gemini": parsed,
        "raw": raw_text,
    })

    return True, "ok"


async def process_one_long(
    json_path: Path,
    out_dir: Path,
    ffmpeg_bin: str,
    channel0_for_output: bool,
    error_log: Path,
    overwrite: bool,
) -> None:
    try:
        label_obj = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        await append_line(error_log, f"[LONG][READ_JSON_FAIL] {json_path} | {e}")
        return

    audio_path = Path(label_obj.get("audio_path", ""))
    if not audio_path.exists():
        await append_line(error_log, f"[LONG][AUDIO_NOT_FOUND] {json_path} | audio_path={audio_path}")
        return

    film_name, part_name = film_and_part_from_path(json_path)

    base = out_dir / "long_seg" / film_name / part_name
    out_wav = base / "wav" / f"{part_name}{audio_path.suffix.lower()}"
    out_json = base / "json" / f"{part_name}.json"

    if out_json.exists() and (not overwrite):
        return

    ensure_dir(out_wav.parent)
    ensure_dir(out_json.parent)

    if channel0_for_output:
        # 用 ffmpeg 重写成单通道 wav（全长）
        # 这里用“从0到很长”，ffmpeg 会自动截断到实际长度
        ok, msg = cut_audio_ffmpeg(ffmpeg_bin, audio_path, out_wav, 0.0, 36000.0, force_mono_channel0=True)
        if not ok:
            await append_line(error_log, f"[LONG][FFMPEG_COPY_FAIL] {json_path} | {msg[:300]}")
            return
    else:
        try:
            shutil.copy2(audio_path, out_wav)
        except Exception as e:
            await append_line(error_log, f"[LONG][COPY_AUDIO_FAIL] {json_path} | {e}")
            return

    new_obj = dict(label_obj)
    new_obj["audio_path"] = str(out_wav)

    segs = label_obj.get("audio_segments", [])
    new_segs = []
    for seg in segs if isinstance(segs, list) else []:
        if isinstance(seg, dict):
            new_segs.append(remove_tmp_keys_in_segment(seg))
    new_obj["audio_segments"] = new_segs

    new_obj["_source_part_json"] = str(json_path)
    new_obj["_source_audio_path"] = str(audio_path)

    await write_json(out_json, new_obj)


async def main_async(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    prompt_file = Path(args.prompt_file).expanduser().resolve()
    error_log = Path(args.error_log).expanduser().resolve()

    ensure_dir(out_dir)
    ensure_dir(error_log.parent)

    api_key = args.key or os.environ.get(args.key_env, "")
    if not api_key:
        raise SystemExit(f"[FATAL] missing api key. Use --key or env {args.key_env}")

    prompt_template = await read_text(prompt_file)

    all_jsons = find_all_part_jsons(input_dir)
    if not all_jsons:
        raise SystemExit(f"[FATAL] no part*.json found under: {input_dir}")

    rnd = random.Random(args.seed)
    rnd.shuffle(all_jsons)

    target_short = int(len(all_jsons) * args.short_ratio)
    target_short = max(0, min(len(all_jsons), target_short))

    short_pool: Deque[Path] = deque(all_jsons[:target_short])
    long_pool: Deque[Path] = deque(all_jsons[target_short:])
    in_flight=0

    final_long: List[Path] = []
    short_success: List[Path] = []

    connector = aiohttp.TCPConnector(limit=args.max_concurrent, limit_per_host=args.max_concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        pbar = tqdm(total=target_short, desc="Selecting short", dynamic_ncols=True) if tqdm else None
        lock = asyncio.Lock()

        async def worker():
            nonlocal short_pool, long_pool, in_flight

            while True:
                async with lock:
                    # 已经够了就停
                    if len(short_success) >= target_short:
                        return

                    # 只有在“成功数 + 正在处理数 < target_short”时才继续取新任务
                    if (len(short_success) + in_flight) >= target_short:
                        return

                    if short_pool:
                        jp = short_pool.popleft()
                    elif long_pool:
                        # 只在确实还缺 short 时才从 long_pool 补
                        jp = long_pool.popleft()
                    else:
                        return

                    in_flight += 1  # 预占一个 short 名额

                ok, reason = await process_one_short(
                    session=session,
                    api_base=args.api_base,
                    api_key=api_key,
                    model=args.model,
                    prompt_template=prompt_template,
                    json_path=jp,
                    out_dir=out_dir,
                    ffmpeg_bin=args.ffmpeg,
                    short_min_sec=args.short_min_sec,
                    short_max_sec=args.short_max_sec,
                    max_retries=args.max_retries,
                    timeout_sec=args.timeout_sec,
                    channel0_for_gemini=args.channel0_for_gemini,
                    channel0_for_output=args.channel0_for_output,
                    error_log=error_log,
                )

                async with lock:
                    in_flight -= 1
                    if ok:
                        if len(short_success) < target_short:
                            short_success.append(jp)
                            if pbar:
                                pbar.update(1)
                        else:
                            # 极端情况下仍可能超额（并发已起飞），直接回退 long
                            final_long.append(jp)
                    else:
                        final_long.append(jp)
                        # 可选：失败才补一个新 short
                # （上面已经允许从 long_pool 补，但有 in_flight 约束，不会无限补）

        workers = [asyncio.create_task(worker()) for _ in range(max(1, args.max_concurrent))]
        await asyncio.gather(*workers)
        if pbar:
            pbar.close()

    final_long.extend(list(long_pool))

    pbar2 = tqdm(total=len(final_long), desc="Copying long", dynamic_ncols=True) if tqdm else None
    for jp in final_long:
        await process_one_long(
            json_path=jp,
            out_dir=out_dir,
            ffmpeg_bin=args.ffmpeg,
            channel0_for_output=args.channel0_for_output_long,
            error_log=error_log,
            overwrite=args.overwrite,
        )
        if pbar2:
            pbar2.update(1)
    if pbar2:
        pbar2.close()

    summary = {
        "input_dir": str(input_dir),
        "total_part_json": len(all_jsons),
        "target_short": target_short,
        "short_success": len(short_success),
        "long_total": len(final_long),
        "out_dir": str(out_dir),
        "prompt_file": str(prompt_file),
    }
    await write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("Build short/long segments from 5-min film labels with Gemini (segment-aware).")

    ap.add_argument("--input_dir", required=True, type=str)
    ap.add_argument("--prompt_file", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--error_log", default="", type=str)

    ap.add_argument("--api_base", default="https://apim1tocn.cheapapi.ai", type=str)
    ap.add_argument("--model", default="gemini-2.5-pro", type=str)
    ap.add_argument("--key_env", default="GEMINI_API_KEY", type=str)
    ap.add_argument("--key", default="YOUR_GEMINI_API_KEY", type=str)

    ap.add_argument("--ffmpeg", default="ffmpeg", type=str)

    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--short_ratio", default=0.5, type=float)
    ap.add_argument("--short_min_sec", default=60.0, type=float)
    ap.add_argument("--short_max_sec", default=120.0, type=float)

    ap.add_argument("--max_concurrent", default=6, type=int)
    ap.add_argument("--max_retries", default=4, type=int)
    ap.add_argument("--timeout_sec", default=1000, type=int)

    ap.add_argument("--channel0_for_gemini", action="store_true")
    ap.add_argument("--channel0_for_output", action="store_true")
    ap.add_argument("--channel0_for_output_long", action="store_true")

    ap.add_argument("--overwrite", action="store_true")

    return ap


def main():
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not args.error_log:
        args.error_log = str(out_dir / "build_short_long.error.log")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
