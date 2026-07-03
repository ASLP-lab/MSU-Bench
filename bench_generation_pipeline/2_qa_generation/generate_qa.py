# -*- coding: utf-8 -*-
#
# 单任务、单 seg 调用 Gemini-2.5-Pro 生成一道能力的 QA 样例.
# 输入: prompt 模板 + reference QA.json (从中读取标注路径和音频路径)
# 输出: <能力>.json (含 source_movie_json, source_audio, prompt_file, result[])
# 用法:
# python generate_qa.py --prompt_file <prompt.txt> --out_dir <.../levelX/> --model gemini-2.5-pro --small_label_json --overwrite
# ⚠️ 需配置: --api_key (或环境变量 GEMINI_API_KEY), --api_base (Gemini 端点)
#


# python3 $SPEAKER_BENCH_ROOT/reference_gemini_5min_film_shortlong_final_single2.py \
#   --prompt_file $SPEAKER_BENCH_ROOT/prompts_ability_v3_options/level1/说话人属性识别能力-情感识别任务.txt \
#   --out_dir $SPEAKER_BENCH_ROOT/qa_stats_out_tmp/qa_sample_final/telcn/QA_short/A7222_3__seg0004__1234.88-1558.43__idx022-030__146.06-215.14/level1 \
#   --model gemini-2.5-pro \
#   --small_label_json \
#   --overwrite


# $SPEAKER_BENCH_ROOT/prompts_ability_v3_options/level1/说话人属性识别能力-年龄段识别任务.txt
# $SPEAKER_BENCH_ROOT/prompts_ability_v3_options/level1/说话人属性识别能力-性别识别任务.txt
# $SPEAKER_BENCH_ROOT/prompts_ability_v3_options/level1/说话人识别能力-说话人观点总结任务.txt
# $SPEAKER_BENCH_ROOT/prompts_ability_v3_options/level1/说话人识别能力-说话人计数任务.txt
# $SPEAKER_BENCH_ROOT/prompts_ability_v3_options/level2/多说话人上下文推理能力-多说话人情感交互.txt
# $SPEAKER_BENCH_ROOT/prompts_ability_v3_options/level1/说话人属性识别能力-口音识别任务.txt
# $SPEAKER_BENCH_ROOT/prompts_ability_v3_options/level1/说话人属性识别能力-情感识别任务.txt
# $SPEAKER_BENCH_ROOT/prompts_ability_v3_options/level1/说话人识别能力-说话人检索任务.txt
# $SPEAKER_BENCH_ROOT/prompts_ability_v3_options/level1/说话人识别能力-说话人反向检索任务.txt
# $SPEAKER_BENCH_ROOT/prompts_ability_v3_options/level2/多说话人对话场景推理能力-对话背景推理.txt

#!/usr/bin
# -*- coding: utf-8 -*-

import argparse
import base64
import json
import os
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

AUDIO_EXTS = [".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus"]
_PLACEHOLDER_RE = re.compile(r"\{\{[^{}]+\}\}")


from pathlib import Path
import subprocess

def ensure_ch0_audio(in_path: str, out_dir: str) -> str:
    in_path = Path(in_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / (in_path.stem + ".ch0.flac")
    if out_path.exists() and out_path.stat().st_size > 0:
        return str(out_path)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-map_channel", "0.0.0",
        "-ac", "1",
        "-c:a", "flac",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return str(out_path)
# -----------------------------
# utils
# -----------------------------
def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def read_json(p: Path) -> Any:
    return json.loads(read_text(p))

def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def guess_mime_type(audio_path: Path) -> str:
    ext = audio_path.suffix.lower()
    if ext == ".wav":
        return "audio/wav"
    if ext == ".mp3":
        return "audio/mpeg"
    if ext == ".flac":
        return "audio/flac"
    if ext == ".m4a":
        return "audio/mp4"
    if ext == ".ogg":
        return "audio/ogg"
    if ext == ".opus":
        return "audio/opus"
    return "application/octet-stream"

def sanitize_prompt(prompt: str) -> str:
    return _PLACEHOLDER_RE.sub("", prompt).strip()

def parse_json_loose(s: str) -> Tuple[Optional[Any], Optional[str]]:
    s2 = (s or "").strip()
    try:
        return json.loads(s2), None
    except Exception:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", s2, flags=re.DOTALL)
    if not m:
        return None, "no_json_found"
    try:
        return json.loads(m.group(1)), None
    except Exception as e:
        return None, f"json_parse_failed: {e}"

# -----------------------------
# time conversion (robust)
# -----------------------------
def _to_float(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None
    return None

def convert_times_for_prompt(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    把 audio_segments 里的 start/end（含 *.tmp）统一转成“秒”；
    规则：数值 > 1000 视为 ms（/1000），否则视为 s。
    """
    out = deepcopy(obj)
    segs = out.get("audio_segments", [])
    if not isinstance(segs, list):
        return out

    keys = ("start_time", "end_time", "start_time.tmp", "end_time.tmp")
    for seg in segs:
        if not isinstance(seg, dict):
            continue
        for k in keys:
            if k not in seg:
                continue
            val = _to_float(seg.get(k))
            if val is None:
                continue
            sec = val / 1000.0 if val > 1000.0 else val
            seg[k] = round(sec, 3)
    return out

def build_small_label_for_prompt(label_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    从完整标注 JSON 中抽取精简版字段，避免请求体过大。
    仅保留：spk_id, spk_name, seg_id, emotion, text, accent
    """
    out: Dict[str, Any] = {"audio_segments": []}
    segs = label_obj.get("audio_segments") or label_obj.get("segments") or []
    if not isinstance(segs, list):
        return out

    for seg in segs:
        if not isinstance(seg, dict):
            continue
        item = {
            "seg_id": seg.get("segment_id") or seg.get("seg_id") or seg.get("id"),
            "spk_id": seg.get("speaker_id") or seg.get("spk_id") or seg.get("speaker"),
            "spk_name": seg.get("speaker_name") or seg.get("spk_name") or seg.get("name"),
            "emotion": seg.get("emotion"),
            "age": seg.get("age"),
            "gender": seg.get("gender"),
            "text": seg.get("text") or seg.get("transcript") or seg.get("utt") or "",
            "accent": seg.get("accent"),
        }
        # drop empty keys
        item = {k: v for k, v in item.items() if v not in (None, "")}
        out["audio_segments"].append(item)

    return out

def compose_final_prompt(
    prompt_txt: str,
    label_obj_for_prompt: Dict[str, Any],
    label_title: str,
) -> str:
    prompt_txt = sanitize_prompt(prompt_txt)
    label_json_str = json.dumps(label_obj_for_prompt, ensure_ascii=False, indent=2)
    return (
        prompt_txt
        + f"\n\n【{label_title}】\n"
        + label_json_str
        + "\n\n【输出要求】只输出严格 JSON（不要 Markdown，不要多余解释，不要代码块）。"
    )


# -----------------------------
# discover reference inputs from existing QA.json
# -----------------------------
def find_reference_qa_json(out_dir: Path, target_name: str) -> Optional[Path]:
    """
    在 out_dir 下找一个“不是目标输出文件”的 qa.json 作为参考；
    优先选择包含 source_movie_json/source_audio 字段的。
    """
    cands = sorted([p for p in out_dir.glob("*.json") if p.is_file() and p.name != target_name])
    for p in cands:
        try:
            obj = read_json(p)
            if isinstance(obj, dict) and ("source_movie_json" in obj or "_source_json" in obj):
                return p
        except Exception:
            continue
    return cands[0] if cands else None

def extract_paths_from_reference(ref_obj: Dict[str, Any]) -> Tuple[Optional[Path], Optional[Path]]:
    """
    从已有 QA.json wrapper 中拿到：
      - label_json_path: source_movie_json 或 _source_json
      - audio_path: source_audio 或 _source_audio
    """
    label_json = ref_obj.get("source_movie_json") or ref_obj.get("_source_json")
    audio = ref_obj.get("source_audio") or ref_obj.get("_source_audio")
    label_p = Path(label_json) if isinstance(label_json, str) and label_json else None
    audio_p = Path(audio) if isinstance(audio, str) and audio else None
    return label_p, audio_p

def infer_prompt_root(prompt_file: Path) -> Path:
    # 优先：父目录名为 prompts_ability_v3_options
    for parent in prompt_file.parents:
        if parent.name == "prompts_ability_v3_options":
            return parent
    # 其次：.../levelX/task.txt -> 取 levelX 的父目录
    if re.match(r"^level\d+$", prompt_file.parent.name):
        return prompt_file.parent.parent
    return prompt_file.parent.parent

# -----------------------------
# HTTP call (requests if available, else urllib)
# -----------------------------
def http_post_json(url: str, payload: Dict[str, Any], timeout: int) -> Tuple[int, str]:
    try:
        import requests  # type: ignore
        r = requests.post(url, json=payload, timeout=timeout)
        return r.status_code, r.text
    except Exception:
        # fallback to urllib
        import urllib.request
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.getcode(), resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            return 0, str(e)

def gemini_generate(
    api_base: str,
    api_key: str,
    model: str,
    prompt_text: str,
    audio_path: Path,
    timeout_sec: int,
    max_retries: int,
    retry_sleep: float,
) -> Tuple[Optional[str], Optional[str]]:
    url = f"{api_base.rstrip('/')}/v1beta/models/{model}:generateContent?key={api_key}"
    audio_path=Path(ensure_ch0_audio(audio_path, str(audio_path.parent / "_upload_audio")))
    audio_bytes = audio_path.read_bytes()
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt_text},
                    {
                        "inline_data": {
                            "mime_type": guess_mime_type(audio_path),
                            "data": base64.b64encode(audio_bytes).decode("utf-8"),
                        }
                    },
                ],
            }
        ]
    }

    last_err = None
    for attempt in range(max_retries + 1):
        code, text = http_post_json(url, payload, timeout=timeout_sec)
        if code != 200:
            last_err = f"HTTP {code}: {text[:500]}"
            time.sleep(min(retry_sleep * (attempt + 1), 8.0))
            continue
        try:
            obj = json.loads(text)
            cands = obj.get("candidates", [])
            if not cands:
                return None, f"empty_candidates: {text[:500]}"
            content = cands[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                return None, f"empty_parts: {text[:500]}"
            out = "".join([p.get("text", "") for p in parts if isinstance(p, dict)])
            out = out.strip()
            if not out:
                return None, "empty_text"
            return out, None
        except Exception as e:
            last_err = f"bad_response_json: {e} | {text[:200]}"
            time.sleep(min(retry_sleep * (attempt + 1), 8.0))

    return None, last_err or "unknown_error"

# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Generate one task QA.json for one audio slice (no multiprocessing).")
    ap.add_argument("--prompt_file", required=True, type=str, help="单个任务 prompt txt 路径")
    ap.add_argument("--out_dir", required=True, type=str, help="输出目录（通常是 .../<segment>/levelX/）")
    ap.add_argument("--ref_qa_json", default=None, type=str, help="可选：指定用哪个已有QA.json做reference")
    ap.add_argument("--overwrite", action="store_true", help="若目标已存在则覆盖")
    ap.add_argument("--save_raw", action="store_true", help="保存 raw 响应到同目录 .raw.txt")
    ap.add_argument(
        "--small_label_json",
        action="store_true",
        help="将 prompt 中的标注 JSON 精简为 spk_id/spk_name/seg_id/emotion/text/accent，并在 out_dir 下落一个 *.small_label.txt",
    )


    ap.add_argument("--api_base", default="https://apim1tocn.cheapapi.ai", type=str)
    ap.add_argument("--model", default="gemini-2.5-pro", type=str)
    ap.add_argument("--api_key", default="YOUR_GEMINI_API_KEY", type=str, help="直接传 key；不传则从环境变量取")
    ap.add_argument("--key_env", default="GEMINI_API_KEY", type=str, help="环境变量名（默认 GEMINI_API_KEY）")
    ap.add_argument("--timeout_sec", default=180, type=int)
    ap.add_argument("--max_retries", default=5, type=int)
    ap.add_argument("--retry_sleep", default=1.0, type=float)

    args = ap.parse_args()

    prompt_file = Path(args.prompt_file).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not prompt_file.is_file():
        print(f"[FATAL] prompt_file not found: {prompt_file}", file=sys.stderr)
        sys.exit(2)
    if not out_dir.is_dir():
        print(f"[FATAL] out_dir not found: {out_dir}", file=sys.stderr)
        sys.exit(2)

    task_name = prompt_file.stem
    out_path = out_dir / f"{task_name}.json"
    if out_path.exists() and not args.overwrite:
        print(f"[FATAL] output exists (use --overwrite): {out_path}", file=sys.stderr)
        sys.exit(2)

    # 1) 找 reference QA.json
    if args.ref_qa_json:
        ref_path = Path(args.ref_qa_json).expanduser().resolve()
        if not ref_path.is_file():
            print(f"[FATAL] ref_qa_json not found: {ref_path}", file=sys.stderr)
            sys.exit(2)
    else:
        ref_path = find_reference_qa_json(out_dir, target_name=out_path.name)
        if ref_path is None:
            print(f"[FATAL] no existing QA.json found in {out_dir} to infer audio/label paths.", file=sys.stderr)
            sys.exit(2)

    ref_obj = read_json(ref_path)
    if not isinstance(ref_obj, dict):
        print(f"[FATAL] reference QA.json is not a dict: {ref_path}", file=sys.stderr)
        sys.exit(2)

    label_json_path, audio_path = extract_paths_from_reference(ref_obj)
    if label_json_path is None or not label_json_path.exists():
        print(f"[FATAL] cannot locate label json from reference: {ref_path}", file=sys.stderr)
        sys.exit(2)

    label_obj = read_json(label_json_path)
    if not isinstance(label_obj, dict):
        print(f"[FATAL] label json is not a dict: {label_json_path}", file=sys.stderr)
        sys.exit(2)

    # audio 优先 reference 的 source_audio；如果没有就用 label_obj["audio_path"]
    if (audio_path is None) or (not audio_path.exists()):
        apath = label_obj.get("audio_path")
        if isinstance(apath, str) and apath:
            audio_path = Path(apath)
    if audio_path is None or not audio_path.exists():
        # fallback: 同目录同 stem 找音频
        stem = label_json_path.stem
        parent = label_json_path.parent
        for ext in AUDIO_EXTS:
            cand = parent / f"{stem}{ext}"
            if cand.exists():
                audio_path = cand
                break
    if audio_path is None or not audio_path.exists():
        print(f"[FATAL] cannot locate audio path (from reference or label json): {ref_path}", file=sys.stderr)
        sys.exit(2)

    # 2) 组 prompt
    prompt_txt = read_text(prompt_file)

    small_label_txt_path: Optional[Path] = None
    if args.small_label_json:
        # 精简标注 JSON，避免请求体过大
        label_for_prompt = build_small_label_for_prompt(label_obj)
        small_label_txt_path = out_dir / f"{label_json_path.stem}.small_label.txt"
        small_label_txt_path.write_text(
            json.dumps(label_for_prompt, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        label_title = "标注 JSON（字段精简版：仅 spk_id/spk_name/seg_id/emotion/text/accent）"
    else:
        label_for_prompt = convert_times_for_prompt(label_obj)
        label_title = "标注 JSON（已将 start_time/end_time 统一为秒 s）"

    final_prompt = compose_final_prompt(prompt_txt, label_for_prompt, label_title)

    # 3) key
    api_key = args.api_key or os.getenv(args.key_env)
    if not api_key:
        print(f"[FATAL] api_key is empty. Provide --api_key or set env {args.key_env}.", file=sys.stderr)
        sys.exit(2)

    # 4) call gemini
    raw_text, err = gemini_generate(
        api_base=args.api_base,
        api_key=api_key,
        model=args.model,
        prompt_text=final_prompt,
        audio_path=audio_path,
        timeout_sec=args.timeout_sec,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
    )
    if err or raw_text is None:
        print(f"[FATAL] gemini_generate failed: {err}", file=sys.stderr)
        sys.exit(2)

    parsed, perr = parse_json_loose(raw_text)
    if perr or parsed is None:
        print(f"[FATAL] model output is not valid JSON: {perr}", file=sys.stderr)
        if args.save_raw:
            (out_dir / f"{task_name}.raw.txt").write_text(raw_text, encoding="utf-8")
        sys.exit(2)

    # 5) 写输出（与现有 wrapper 对齐）
    prompt_root = infer_prompt_root(prompt_file)
    try:
        prompt_relpath = prompt_file.relative_to(prompt_root).as_posix()
    except Exception:
        prompt_relpath = prompt_file.name

    payload = {
        "source_movie_json": str(label_json_path),
        "source_audio": str(audio_path),
        "prompt_file": str(prompt_file),
        "prompt_relpath": prompt_relpath,
        "model": args.model,
        "result": parsed,
    }

    write_json(out_path, payload)

    if args.save_raw:
        (out_dir / f"{task_name}.raw.txt").write_text(raw_text, encoding="utf-8")

    print(f"[OK] wrote: {out_path}")
    print(f"     ref_qa: {ref_path}")
    print(f"     label : {label_json_path}")
    print(f"     audio : {audio_path}")
    print(f"     prompt: {prompt_relpath}")

if __name__ == "__main__":
    main()