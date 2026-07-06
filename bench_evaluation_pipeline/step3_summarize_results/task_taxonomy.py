#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task ↔ Capability ↔ Tier mapping for MSU-Bench.

Both Chinese and English task-file stems from the upstream release are
covered so we can bucket a QA JSON regardless of which language split it
came from (bench_cn uses CN task names, bench_en uses their English
counterparts).

The 5-capability × 16-task taxonomy follows the paper:

  Tier 1 — Speaker Grounding & Identification
    ├─ SID · Speaker Identification
    │    · Reverse Speaker Retrieval  (RSR)
    │    · Speaker Retrieval          (SR)
    │    · Speaker-specific Viewpoint Summarization (SVS)
    │    · Speaker Counting           (SC)
    │    · Speaker Verification       (SV)
    └─ SAR · Speaker Attribute Recognition
         · Accent Identification  (AI)
         · Age Recognition        (AR)
         · Gender Identification  (GI)
         · Emotion Recognition    (ER)
         · Speaker Profiling      (SP)

  Tier 2 — Multi-Speaker Dialogue Reasoning
    ├─ DSR · Dialogue Scene Reasoning
    │    · Background Inference           (BI)
    │    · Role/Identity Identification   (RII)
    ├─ DSA · Dialogue Structure Analysis
    │    · Dialogue Act Recognition       (DAR)
    │    · Q&A Structure Identification   (QASI)
    └─ DCR · Dialogue Contextual Reasoning
         · Emotion Interaction Reasoning  (EIR)
         · Multi-speaker Viewpoint Summ.  (MSVS)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

# Canonical short code → (tier, capability_code, task_code, human_label)
TASK_INFO: Dict[str, Tuple[int, str, str, str]] = {
    # Tier 1 — SID
    "RSR":  (1, "SID", "RSR",  "Reverse Speaker Retrieval"),
    "SR":   (1, "SID", "SR",   "Speaker Retrieval"),
    "SVS":  (1, "SID", "SVS",  "Speaker-specific Viewpoint Summarization"),
    "SC":   (1, "SID", "SC",   "Speaker Counting"),
    "SV":   (1, "SID", "SV",   "Speaker Verification"),
    # Tier 1 — SAR
    "AI":   (1, "SAR", "AI",   "Accent Identification"),
    "AR":   (1, "SAR", "AR",   "Age Recognition"),
    "GI":   (1, "SAR", "GI",   "Gender Identification"),
    "ER":   (1, "SAR", "ER",   "Emotion Recognition"),
    "SP":   (1, "SAR", "SP",   "Speaker Profiling"),
    # Tier 2 — DSR
    "BI":   (2, "DSR", "BI",   "Background Inference"),
    "RII":  (2, "DSR", "RII",  "Role/Identity Identification"),
    # Tier 2 — DSA
    "DAR":  (2, "DSA", "DAR",  "Dialogue Act Recognition"),
    "QASI": (2, "DSA", "QASI", "Q&A Structure Identification"),
    # Tier 2 — DCR
    "EIR":  (2, "DCR", "EIR",  "Emotion Interaction Reasoning"),
    "MSVS": (2, "DCR", "MSVS", "Multi-speaker Viewpoint Summarization"),
}


CAPABILITY_LABEL: Dict[str, str] = {
    "SID": "Speaker Identification",
    "SAR": "Speaker Attribute Recognition",
    "DSR": "Dialogue Scene Reasoning",
    "DSA": "Dialogue Structure Analysis",
    "DCR": "Dialogue Contextual Reasoning",
}


# ─────────────────────────────────────────────────────────────────────────
# task-file-stem → canonical short code
#
# The upstream files are named after their prompt, e.g.
#   说话人识别能力-说话人检索任务.json      (CN)
#   speaker-identification-speaker-retrieval.json   (EN, if present)
# We match on any distinctive substring so that we tolerate small
# formatting variations across languages / releases.
# ─────────────────────────────────────────────────────────────────────────
_STEM_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    # Tier 1 SID
    "RSR":  ("反向检索", "反向说话人检索", "reverse-retrieval",
             "reverse_retrieval", "reverse-speaker-retrieval"),
    "SR":   ("说话人检索任务", "speaker-retrieval", "speaker_retrieval"),
    "SVS":  ("说话人观点总结", "speaker-viewpoint",
             "speaker_viewpoint", "speaker-specific-viewpoint"),
    "SC":   ("说话人计数", "speaker-counting", "speaker_counting"),
    "SV":   ("说话人验证", "speaker-verification", "speaker_verification"),
    # Tier 1 SAR
    "AI":   ("口音识别", "accent-identification", "accent_identification",
             "accent-recognition"),
    "AR":   ("年龄段识别", "年龄识别", "age-recognition",
             "age_recognition"),
    "GI":   ("性别识别", "gender-identification", "gender_identification",
             "gender-recognition"),
    "ER":   ("情感识别", "emotion-recognition", "emotion_recognition"),
    "SP":   ("说话人画像", "speaker-profiling", "speaker_profiling"),
    # Tier 2 DSR
    "BI":   ("对话背景推理", "background-inference", "background_inference"),
    "RII":  ("对话身份识别", "role-identification", "identity-identification",
             "role_identity", "role/identity"),
    # Tier 2 DSA
    "DAR":  ("对话行为识别", "dialogue-act", "dialogue_act",
             "dialog-act"),
    "QASI": ("问答结构识别", "qa-structure", "qa_structure",
             "q&a-structure"),
    # Tier 2 DCR
    "EIR":  ("多说话人情感交互", "emotion-interaction",
             "emotion_interaction"),
    "MSVS": ("多说话人观点总结", "multi-speaker-viewpoint",
             "multi_speaker_viewpoint", "viewpoint-summarization"),
}


def classify_task(task_stem: str) -> Optional[Tuple[int, str, str, str]]:
    """Return (tier, capability, task, human_label) or None if unknown."""
    if not task_stem:
        return None
    stem = task_stem.strip()
    stem_lc = stem.lower()
    # 1) exact short code
    if stem in TASK_INFO:
        return TASK_INFO[stem]
    # 2) keyword match (order-preserving; longer keyword first)
    keyword_pairs = []
    for code, kws in _STEM_KEYWORDS.items():
        for kw in kws:
            keyword_pairs.append((len(kw), kw, code))
    keyword_pairs.sort(reverse=True)  # longer keywords match first
    for _, kw, code in keyword_pairs:
        if kw.lower() in stem_lc:
            return TASK_INFO[code]
    return None


# Speaker-referencing schemes used by the paper
INDEX_SCHEMES = ("no_index", "time_index", "transcript_index",
                 "speaker_index", "complex_index")

# Diagnostic error types used by the paper
ERROR_TYPES = ("wrong_speaker", "hallucination", "unknown",
               "instruction_not_follow")
