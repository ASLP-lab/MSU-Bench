#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared prompt / audio-part builder used by every scoring backend
(step2_run_scoring/run_gemini.py, run_doubao.py, ...).

Only depends on the standard library and (optionally) `ffmpeg` on PATH for
slicing/re-encoding audio into a compact base64 payload.

The "how do we present the audio to the model?" rules live here so the
per-backend files can stay tiny.
"""

from __future__ import annotations

import base64
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────
# QA structure helpers
# ─────────────────────────────────────────────────────────────────────────
_LEVEL_RE = re.compile(r"level(\d+)")
_OPT_LETTER_RE = re.compile(r"\s*([A-Za-z])\s*[\.\)、]")
_CHOICE_RE = re.compile(r"(?<![A-Z])([ABCD])(?![A-Z])")


def infer_level(qa_json_path: str, data: Dict[str, Any]) -> int:
    p = qa_json_path.replace("\\", "/")
    m = re.search(r"/level(\d+)/", p)
    if m:
        return int(m.group(1))
    m2 = _LEVEL_RE.search(str(data.get("_msu_meta", {}).get("level", "")))
    if m2:
        return int(m2.group(1))
    m3 = _LEVEL_RE.search(str(data.get("prompt_file", "")))
    if m3:
        return int(m3.group(1))
    return 1


def split_speaker_meta_and_questions(
    data: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Return (speaker_meta, question_list) regardless of upstream schema.

    Handles the three variants seen in bench_cn / bench_en / demo QA:
      * data['qa_result'] = [{"speaker_meta": ...}, q1, q2, ...]
      * data['result']    = [{"speaker_meta": ...}, q1, q2, ...]
      * data['result']    = [q1, q2, ...] (no speaker_meta)
    """
    speaker_meta: Dict[str, Any] = data.get("speaker_meta") or {}
    for key in ("qa_result", "result", "questions"):
        arr = data.get(key)
        if isinstance(arr, list) and arr:
            questions: List[Dict[str, Any]] = []
            for item in arr:
                if not isinstance(item, dict):
                    continue
                if "speaker_meta" in item and not questions:
                    speaker_meta = item.get("speaker_meta") or speaker_meta
                    continue
                if "question" in item or "options" in item:
                    questions.append(item)
            if questions:
                return speaker_meta, questions
    return speaker_meta, []


def parse_options_letters(options: List[str]) -> List[str]:
    letters: List[str] = []
    for opt in options or []:
        m = _OPT_LETTER_RE.match(opt or "")
        if m:
            letters.append(m.group(1).upper())
    if not letters and options:
        letters = [chr(ord("A") + i) for i in range(len(options))]
    seen, out = set(), []
    for x in letters:
        x = x.upper()
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def extract_single_letter(resp: str, allowed: List[str]) -> Optional[str]:
    """Best-effort A/B/C/D extraction from a free-form model reply."""
    if not resp:
        return None
    s = resp.strip().upper()
    if len(s) == 1 and "A" <= s <= "Z" and s in allowed:
        return s
    for ch in allowed:
        if re.search(rf"\b{re.escape(ch)}\b", s):
            return ch
    m = _CHOICE_RE.search(s)
    if m and m.group(1) in allowed:
        return m.group(1)
    return None


# ─────────────────────────────────────────────────────────────────────────
# Prompt & speaker-referencing rules
# ─────────────────────────────────────────────────────────────────────────
_INSTR_ZH = (
    "\n\n【输出格式要求】\n"
    "在回复的最后一行按照固定格式输出：最后答案：A/B/C/D。\n"
)
_INSTR_EN = (
    "\n\n[Output format]\n"
    "On the very last line of your reply, output only: Final answer: A/B/C/D.\n"
)


def build_prompt_text(question: Dict[str, Any], language: str = "zh") -> str:
    q = (question.get("question") or "").strip()
    options = question.get("options") or []
    opt_text = "\n".join(str(o) for o in options)
    tail = _INSTR_EN if language == "en" else _INSTR_ZH
    label = "Options:" if language == "en" else "选项："
    return f"{q}\n\n{label}\n{opt_text}{tail}"


def resolve_pair_ref(ref: str, speaker_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve a `spk_a[0]` / `spk_b[1]` style reference used by
    verification pairs."""
    m = re.match(r"^(spk_[A-Za-z0-9]+)\[(\d+)\]$", (ref or "").strip())
    if not m:
        raise ValueError(f"bad pair ref: {ref}")
    key, idx = m.group(1), int(m.group(2))
    arr = speaker_meta.get(key)
    if not isinstance(arr, list) or idx >= len(arr):
        raise KeyError(f"pair ref not found: {ref}")
    return arr[idx]


def _first_registration_seg(speaker_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if isinstance(speaker_meta, dict):
        arr = speaker_meta.get("spk_a")
        if isinstance(arr, list) and arr:
            return arr[0]
        for _, v in speaker_meta.items():
            if isinstance(v, list) and v:
                return v[0]
    return None


# ─────────────────────────────────────────────────────────────────────────
# ffmpeg → bytes
# ─────────────────────────────────────────────────────────────────────────
def audio_to_bytes(
    audio_path: str,
    out_fmt: str = "mp3",
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    sample_rate: int = 16000,
    mono: bool = True,
    mp3_bitrate: str = "64k",
) -> Tuple[str, bytes]:
    """Slice + re-encode audio via ffmpeg. Returns (mime, raw_bytes)."""
    if out_fmt not in ("mp3", "wav"):
        raise ValueError(f"unsupported out_fmt={out_fmt}")

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start_s is not None:
        cmd += ["-ss", str(float(start_s))]
    if end_s is not None:
        cmd += ["-to", str(float(end_s))]
    cmd += ["-i", audio_path]
    if mono:
        cmd += ["-ac", "1"]
    if sample_rate:
        cmd += ["-ar", str(sample_rate)]
    if out_fmt == "wav":
        cmd += ["-f", "wav", "-acodec", "pcm_s16le", "pipe:1"]
        mime = "audio/wav"
    else:
        cmd += ["-f", "mp3", "-codec:a", "libmp3lame", "-b:a", mp3_bitrate, "pipe:1"]
        mime = "audio/mpeg"

    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed: " + p.stderr.decode("utf-8", errors="ignore")[:200]
        )
    return mime, p.stdout


# ─────────────────────────────────────────────────────────────────────────
# Gemini-shape "parts" builder
# ─────────────────────────────────────────────────────────────────────────
def build_parts_for_question(
    qa_json: str,
    level: int,
    speaker_meta: Dict[str, Any],
    source_audio: str,
    question: Dict[str, Any],
    language: str = "zh",
    audio_fmt: str = "mp3",
    sample_rate: int = 16000,
    mono: bool = True,
    mp3_bitrate: str = "64k",
) -> List[Dict[str, Any]]:
    """Produce a Gemini-compatible `parts` list with the audio-selection
    rules from the paper:

      * ``no_index`` + verification pair  → 2 clips (one per pair member)
      * ``no_index`` (non-verification)   → the target speaker's first
                                            registration snippet
      * ``level >= 2`` + ``no_index``     → additionally the full audio
      * every other index scheme          → the full audio only
    """
    parts: List[Dict[str, Any]] = [
        {"text": build_prompt_text(question, language=language)}
    ]
    qtype = (question.get("question_type") or "").strip().lower()

    def _emit_audio(label: str,
                    start_t: Optional[float] = None,
                    end_t: Optional[float] = None) -> None:
        parts.append({"text": label})
        mime, raw = audio_to_bytes(
            audio_path=source_audio, out_fmt=audio_fmt,
            start_s=start_t, end_s=end_t,
            sample_rate=sample_rate, mono=mono,
            mp3_bitrate=mp3_bitrate,
        )
        parts.append({
            "inline_data": {
                "mime_type": mime,
                "data": base64.b64encode(raw).decode("ascii"),
            }
        })

    is_verification_pair = (
        isinstance(question.get("pair"), dict)
        and question["pair"].get("left")
        and question["pair"].get("right")
    )

    if qtype == "no_index" and is_verification_pair:
        left = resolve_pair_ref(question["pair"]["left"], speaker_meta)
        right = resolve_pair_ref(question["pair"]["right"], speaker_meta)
        for meta in (left, right):
            if meta.get("start_time") is None or meta.get("end_time") is None:
                raise ValueError("pair segment has null time")
        _emit_audio("Clip 1:" if language == "en" else "片段一：",
                    float(left["start_time"]), float(left["end_time"]))
        _emit_audio("Clip 2:" if language == "en" else "片段二：",
                    float(right["start_time"]), float(right["end_time"]))
        if level >= 2:
            _emit_audio("Full conversation audio:" if language == "en"
                        else "完整对话音频：")
        return parts

    if qtype == "no_index":
        seg = _first_registration_seg(speaker_meta)
        if seg is None:
            raise ValueError("no_index question but speaker_meta empty")
        if seg.get("start_time") is None or seg.get("end_time") is None:
            raise ValueError("registration segment has null time")
        _emit_audio("Target speaker snippet:" if language == "en"
                    else "注册/目标说话人音频：",
                    float(seg["start_time"]), float(seg["end_time"]))
        if level >= 2:
            _emit_audio("Full conversation audio:" if language == "en"
                        else "完整对话音频：")
        return parts

    # every other index scheme → full audio only
    _emit_audio("Full conversation audio:" if language == "en"
                else "完整对话音频：")
    return parts
