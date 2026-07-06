#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3 — Aggregate scored QA JSONs into human-readable CSVs.

Consumes the per-QA scored files written by step 2
    <input_root>/<lang>/<scenario>/<qa_len>/<seg>/<level>/<task>.json
and produces:

    summary_by_task.csv          accuracy for each of the 16 sub-tasks
    summary_by_capability.csv    accuracy for each of the 5 capabilities
    summary_by_tier.csv          Tier-1 / Tier-2 accuracy
    summary_by_language.csv      zh / en accuracy per tier
    summary_by_index_scheme.csv  no / time / transcript / speaker / complex
                                 accuracy, per tier — this is the
                                 "speaker-referencing scheme" breakdown
    summary_by_error_type.csv    wrong_speaker / hallucination / unknown /
                                 instruction_not_follow error share, per tier

Because the outer leaderboard already reports the overall score, this
script *does not* try to persist any consolidated leaderboard artefact —
each CSV is a standalone diagnostic view.

Predicted answer field:  ``reference_result``  (single letter)
Ground-truth field:      ``answer``            (single letter)
Rationale for wrong-answer classification: ``rationale``  (aligned with
options), following the mapping:
    "wrong speaker"  → wrong_speaker
    "hallucination"  → hallucination
    "unknown"        → unknown
    "ground truth"   → correct answer marker
Any other rationale text falls back to "unknown".
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task_taxonomy import (  # noqa: E402
    CAPABILITY_LABEL,
    ERROR_TYPES,
    INDEX_SCHEMES,
    TASK_INFO,
    classify_task,
)


# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────
_LEVEL_RE = re.compile(r"level(\d+)", re.IGNORECASE)
_OPT_LETTER_RE = re.compile(r"\s*([A-Za-z])\s*[\.\)、]")
_CHOICE_RE = re.compile(r"(?<![A-Z])([ABCD])(?![A-Z])")
_RATIONALE_LEADING_RE = re.compile(r"^\s*([A-Za-z])\s*[\.\)、]\s*(.*)$")

# The paper deliberately excludes reverse-retrieval-style items from the
# main leaderboard (they collapse to speaker-count answers). Kept here as
# a switch — set --keep_reverse to include them.
_REVERSE_QTYPE = {"reverse_count", "reverse_retrival", "reverse_retrieval"}


# ─────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────
def _safe_div(a: float, b: float) -> float:
    return (a / b) if b else 0.0


def _fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def _normalize_letter(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().upper()
    if len(s) == 1 and s in "ABCD":
        return s
    m = _CHOICE_RE.search(s)
    return m.group(1) if m else None


def _parse_option_letters(options: List[str]) -> List[str]:
    letters: List[str] = []
    for opt in options or []:
        m = _OPT_LETTER_RE.match(opt or "")
        if m:
            letters.append(m.group(1).upper())
    if not letters and options:
        letters = [chr(ord("A") + i) for i in range(len(options))]
    seen: set = set()
    out: List[str] = []
    for x in letters:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _normalize_rationale_tag(tag: str) -> str:
    """Map a rationale text (CN or EN) to one of:
       wrong_speaker / hallucination / unknown / ground_truth / other."""
    s = (tag or "").strip().lower()
    # CN
    if "幻觉" in s:
        return "hallucination"
    if any(k in s for k in ("无法判断", "证据不足", "信息不足")):
        return "unknown"
    if "错" in s and "说话人" in s:
        return "wrong_speaker"
    if "正确" in s or "标准答案" in s or "参考答案" in s:
        return "ground_truth"
    # EN
    if any(k in s for k in ("wrong speaker", "wrong_speaker", "wrong-speaker")):
        return "wrong_speaker"
    if "hallucination" in s or "hallucinate" in s:
        return "hallucination"
    if "unknown" in s or "cannot determine" in s or "insufficient" in s:
        return "unknown"
    if "ground truth" in s or "ground_truth" in s or "correct" in s:
        return "ground_truth"
    return "other"


def _rationale_map(question: Dict[str, Any],
                   allowed_letters: List[str]) -> Dict[str, str]:
    """Return {letter → tag} for wrong-answer classification.

    ``rationale`` may be either
      * ["A. wrong speaker", "B. hallucination", ...] (letter-prefixed), or
      * ["wrong speaker", "hallucination", ...]      (aligned with options).
    """
    rat = question.get("rationale")
    if not isinstance(rat, list) or not rat:
        return {}
    letter_map: Dict[str, str] = {}
    saw_letter = False
    for entry in rat:
        m = _RATIONALE_LEADING_RE.match(str(entry or ""))
        if m:
            saw_letter = True
            letter_map[m.group(1).upper()] = _normalize_rationale_tag(m.group(2))
    if saw_letter:
        return letter_map
    for i, entry in enumerate(rat):
        if i >= len(allowed_letters):
            break
        letter_map[allowed_letters[i]] = _normalize_rationale_tag(str(entry or ""))
    return letter_map


# ─────────────────────────────────────────────────────────────────────────
# Aggregation buckets
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class Bucket:
    total: int = 0
    correct: int = 0
    wrong_speaker: int = 0
    hallucination: int = 0
    unknown: int = 0
    other_wrong: int = 0
    instruction_not_follow: int = 0

    def merge(self, other: "Bucket") -> None:
        for f in (
            "total", "correct", "wrong_speaker", "hallucination",
            "unknown", "other_wrong", "instruction_not_follow",
        ):
            setattr(self, f, getattr(self, f) + getattr(other, f))

    @property
    def acc(self) -> float:
        return _safe_div(self.correct, self.total)

    @property
    def total_wrong(self) -> int:
        return (self.wrong_speaker + self.hallucination + self.unknown
                + self.other_wrong + self.instruction_not_follow)

    def error_share(self) -> Dict[str, float]:
        n = self.total_wrong
        return {
            "wrong_speaker": _safe_div(self.wrong_speaker, n),
            "hallucination": _safe_div(self.hallucination, n),
            "unknown": _safe_div(self.unknown + self.other_wrong, n),
            "instruction_not_follow": _safe_div(self.instruction_not_follow, n),
        }


@dataclass
class QuestionRecord:
    """Everything step 3 needs about a single question, extracted once."""
    language: str
    scenario: str
    tier: int
    capability: str
    task_code: str
    task_label: str
    index_scheme: str
    is_correct: bool
    error_class: str  # correct | wrong_speaker | hallucination | unknown |
                     # other_wrong | instruction_not_follow


# ─────────────────────────────────────────────────────────────────────────
# QA JSON walker
# ─────────────────────────────────────────────────────────────────────────
def _language_from_path(qa_path: Path, root: Path) -> str:
    try:
        rel = qa_path.relative_to(root)
        first = rel.parts[0].lower()
        if first in ("zh", "cn"):
            return "zh"
        if first == "en":
            return "en"
    except Exception:
        pass
    return "unknown"


def _tier_from_path(qa_path: Path) -> int:
    m = _LEVEL_RE.search(str(qa_path).replace("\\", "/"))
    return int(m.group(1)) if m else 0


def _iter_questions(obj: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for key in ("qa_result", "result", "questions"):
        arr = obj.get(key)
        if isinstance(arr, list):
            for it in arr:
                if isinstance(it, dict) and ("question" in it or "options" in it):
                    yield it
            return


def _process_one(qa_path: Path, root: Path,
                 keep_reverse: bool) -> Tuple[List[QuestionRecord], Optional[str]]:
    try:
        with qa_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return [], f"load_failed: {e}"

    meta = data.get("_msu_meta", {}) or {}
    language = meta.get("language") or _language_from_path(qa_path, root)
    scenario = meta.get("scenario") or (
        qa_path.relative_to(root).parts[1]
        if len(qa_path.relative_to(root).parts) > 1 else "unknown"
    )
    tier_hint = meta.get("level") or _tier_from_path(qa_path)
    tier = 0
    if isinstance(tier_hint, int):
        tier = tier_hint
    elif isinstance(tier_hint, str):
        m = _LEVEL_RE.search(tier_hint)
        if m:
            tier = int(m.group(1))

    task_stem = (meta.get("task_stem")
                 or qa_path.stem.replace(".bak", ""))
    info = classify_task(task_stem)
    if info is None:
        # Uncategorised file (e.g. a stray non-QA json) — skip.
        return [], "unknown_task"
    _, capability, task_code, task_label = info

    # By taxonomy the tier is fixed by the task, but if the on-disk tier
    # disagrees we prefer the taxonomy's answer.
    tier = TASK_INFO[task_code][0]

    out: List[QuestionRecord] = []
    for q in _iter_questions(data):
        qtype = str(q.get("question_type") or "").strip().lower()
        if not keep_reverse and qtype in _REVERSE_QTYPE:
            continue
        options = q.get("options") or []
        allowed = _parse_option_letters(options)
        gt = _normalize_letter(q.get("answer"))
        pred = _normalize_letter(q.get("reference_result"))

        if pred is None or (allowed and pred not in allowed):
            error_class = "instruction_not_follow"
            is_correct = False
        elif gt is not None and pred == gt:
            error_class = "correct"
            is_correct = True
        else:
            rat = _rationale_map(q, allowed).get(pred, "other")
            if rat == "ground_truth":  # inconsistent labelling — treat as other
                rat = "other"
            error_class = {
                "wrong_speaker": "wrong_speaker",
                "hallucination": "hallucination",
                "unknown": "unknown",
            }.get(rat, "other_wrong")
            is_correct = False

        # Speaker-referencing scheme normalisation
        scheme = qtype if qtype in INDEX_SCHEMES else "unknown_index"

        out.append(QuestionRecord(
            language=language, scenario=scenario,
            tier=tier, capability=capability,
            task_code=task_code, task_label=task_label,
            index_scheme=scheme,
            is_correct=is_correct,
            error_class=error_class,
        ))
    return out, None


def _add_to_bucket(b: Bucket, rec: QuestionRecord) -> None:
    b.total += 1
    if rec.is_correct:
        b.correct += 1
    else:
        # Every non-correct record contributes to exactly one error bucket.
        setattr(b, rec.error_class, getattr(b, rec.error_class) + 1)


# ─────────────────────────────────────────────────────────────────────────
# CSV writers
# ─────────────────────────────────────────────────────────────────────────
def _write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True,
                    help="Scored QA root, e.g. work/scoring/gemini-2.5-flash")
    ap.add_argument("--out_dir", required=True,
                    help="Where to write summary_*.csv")
    ap.add_argument("--model_name", default=None,
                    help="Optional label for the CSV rows "
                         "(defaults to the input dir's basename).")
    ap.add_argument("--keep_reverse", action="store_true",
                    help="Include reverse_retrieval / reverse_count question "
                         "types in the metrics (excluded by default, "
                         "matching the paper).")
    ap.add_argument("--workers", type=int,
                    default=min(32, (os.cpu_count() or 8)))
    args = ap.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_root.is_dir():
        raise SystemExit(f"input_root not found: {input_root}")

    model_name = args.model_name or input_root.name
    files = [p for p in input_root.rglob("*.json") if p.is_file()]
    if not files:
        raise SystemExit(f"No .json under {input_root}")

    # ── Buckets ─────────────────────────────────────────────────────────
    by_task: Dict[str, Bucket] = defaultdict(Bucket)
    by_capability: Dict[str, Bucket] = defaultdict(Bucket)
    by_tier: Dict[int, Bucket] = defaultdict(Bucket)
    by_language_tier: Dict[Tuple[str, int], Bucket] = defaultdict(Bucket)
    by_index_tier: Dict[Tuple[int, str], Bucket] = defaultdict(Bucket)
    by_error_tier: Dict[int, Bucket] = defaultdict(Bucket)  # same as by_tier

    n_files = 0
    n_skipped = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_process_one, p, input_root, args.keep_reverse)
                for p in files]
        for fut in as_completed(futs):
            recs, err = fut.result()
            if err is not None:
                n_skipped += 1
                continue
            n_files += 1
            for rec in recs:
                _add_to_bucket(by_task[rec.task_code], rec)
                _add_to_bucket(by_capability[rec.capability], rec)
                _add_to_bucket(by_tier[rec.tier], rec)
                _add_to_bucket(by_language_tier[(rec.language, rec.tier)], rec)
                _add_to_bucket(by_index_tier[(rec.tier, rec.index_scheme)], rec)
                _add_to_bucket(by_error_tier[rec.tier], rec)

    # ── Console overview ────────────────────────────────────────────────
    total_records = sum(b.total for b in by_tier.values())
    total_correct = sum(b.correct for b in by_tier.values())
    print(f"[summary] model={model_name}  scored_files={n_files}  "
          f"skipped_files={n_skipped}  questions={total_records}  "
          f"acc={_safe_div(total_correct, total_records):.4f}")

    # ── CSV #1: by task ────────────────────────────────────────────────
    task_rows: List[List[Any]] = []
    for code, (tier, cap, _, label) in TASK_INFO.items():
        b = by_task.get(code, Bucket())
        task_rows.append([
            model_name, tier, cap, code, label,
            b.total, b.correct, _fmt(b.acc),
        ])
    _write_csv(
        out_dir / "summary_by_task.csv",
        ["model", "tier", "capability", "task_code", "task_label",
         "total", "correct", "accuracy"],
        task_rows,
    )

    # ── CSV #2: by capability ──────────────────────────────────────────
    cap_rows: List[List[Any]] = []
    for cap in ("SID", "SAR", "DSR", "DSA", "DCR"):
        b = by_capability.get(cap, Bucket())
        tier = TASK_INFO[
            [k for k, v in TASK_INFO.items() if v[1] == cap][0]
        ][0]
        cap_rows.append([
            model_name, tier, cap, CAPABILITY_LABEL[cap],
            b.total, b.correct, _fmt(b.acc),
        ])
    _write_csv(
        out_dir / "summary_by_capability.csv",
        ["model", "tier", "capability_code", "capability_label",
         "total", "correct", "accuracy"],
        cap_rows,
    )

    # ── CSV #3: by tier ────────────────────────────────────────────────
    tier_rows: List[List[Any]] = []
    for tier in sorted(by_tier.keys()):
        b = by_tier[tier]
        tier_rows.append([
            model_name, tier, b.total, b.correct, _fmt(b.acc),
        ])
    _write_csv(
        out_dir / "summary_by_tier.csv",
        ["model", "tier", "total", "correct", "accuracy"],
        tier_rows,
    )

    # ── CSV #4: by language × tier ─────────────────────────────────────
    lang_rows: List[List[Any]] = []
    for (lang, tier), b in sorted(by_language_tier.items()):
        lang_rows.append([
            model_name, lang, tier, b.total, b.correct, _fmt(b.acc),
        ])
    _write_csv(
        out_dir / "summary_by_language.csv",
        ["model", "language", "tier", "total", "correct", "accuracy"],
        lang_rows,
    )

    # ── CSV #5: by index scheme × tier ─────────────────────────────────
    idx_rows: List[List[Any]] = []
    for tier in sorted(by_tier.keys()):
        tier_total = by_tier[tier].total or 1
        for scheme in INDEX_SCHEMES:
            b = by_index_tier.get((tier, scheme), Bucket())
            idx_rows.append([
                model_name, tier, scheme,
                b.total, _fmt(_safe_div(b.total, tier_total)),
                b.correct, _fmt(b.acc),
            ])
    _write_csv(
        out_dir / "summary_by_index_scheme.csv",
        ["model", "tier", "index_scheme",
         "total", "share_within_tier",
         "correct", "accuracy"],
        idx_rows,
    )

    # ── CSV #6: by error type × tier ───────────────────────────────────
    err_rows: List[List[Any]] = []
    for tier in sorted(by_tier.keys()):
        b = by_tier[tier]
        share = b.error_share()
        err_rows.append([
            model_name, tier,
            b.total, b.correct, b.total_wrong,
            b.wrong_speaker, _fmt(share["wrong_speaker"]),
            b.hallucination, _fmt(share["hallucination"]),
            b.unknown + b.other_wrong, _fmt(share["unknown"]),
            b.instruction_not_follow, _fmt(share["instruction_not_follow"]),
        ])
    _write_csv(
        out_dir / "summary_by_error_type.csv",
        ["model", "tier",
         "total", "correct", "total_wrong",
         "wrong_speaker_cnt", "wrong_speaker_share",
         "hallucination_cnt", "hallucination_share",
         "unknown_cnt", "unknown_share",
         "instruction_not_follow_cnt", "instruction_not_follow_share"],
        err_rows,
    )

    print(f"[summary] CSVs written under {out_dir}")


if __name__ == "__main__":
    main()
