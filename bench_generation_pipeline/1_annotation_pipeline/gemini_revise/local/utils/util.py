# -*- coding: utf-8 -*-
#
# gemini_revise 通用工具函数.
#

from collections import defaultdict
import json
import logging
import os
from pathlib import Path
import queue
import re
import subprocess

import concurrent
import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from local.model.melBandRoformer import mel_band_roformer

err_dir = "$DATA_ROOT/step4/logs"
report_pth = "$DATA_ROOT/step4/reports"

def safe_int(x, default=None):
    try:
        if x is None or x == '':
            return default
        return int(x)
    except (ValueError, TypeError):
        return default

def write_err_log(err_dir, msg, level="warning", filename="error.log"):
    if level == "error":
        logging.error(msg)
    else:
        logging.warning(msg)

    try:
        os.makedirs(err_dir, exist_ok=True)
        err_log = os.path.join(err_dir, filename)
        with open(err_log, 'a', encoding='utf-8') as ef:
            ef.write(msg + "\n")
    except Exception:
        logging.exception("Failed to write error log %s", err_log)


def audioExtraction(configs, paths):
    results = {path: jsonLoad(path) for path in paths}

    for key, value in results.items():
        audio_path = value["audio_path"]
        audio_segments = value["audio_segments"]

        audio_pcm, sample_rate = torchaudio.load(audio_path, normalize=False)
        audio_duration = audioDurationGet(audio_path)

        for segment_index, audio_segment in enumerate(audio_segments):

            # ---------- segment_id ----------
            segment_id = audio_segment.get("segment_id")
            if segment_id is None:
                write_err_log(
                    err_dir,
                    f"[ERROR] segment {key} {segment_index} missing segment_id, skipped",
                    level="error",
                )
                continue

            # ---------- text ----------
            text = audio_segment.get("text") or audio_segment.get("text_and_paralanguage")
            if text is None or (isinstance(text, str) and not text.strip()):
                write_err_log(
                    err_dir,
                    f"[WARN] segment {key} {segment_index} missing text, skipped",
                )
                continue

            text = re.sub(r'<[^>]+>', '', text).strip()

            # ---------- start / end ----------
            start_time = safe_int(audio_segment.get("start_time"))
            end_time = safe_int(audio_segment.get("end_time"))

            if start_time is None or end_time is None:
                write_err_log(
                    err_dir,
                    f"[WARN] segment {key} {segment_index} invalid start_time or end_time, skipped",
                )
                continue

            # ---------- neighbor times ----------
            last_end = (
                safe_int(audio_segments[segment_index - 1].get("end_time"), default=None)
                if segment_index > 0
                else 0
            )

            next_start = (
                safe_int(
                    audio_segments[segment_index + 1].get("start_time"),
                    default=None,
                )
                if segment_index < len(audio_segments) - 1
                else audio_duration
            )

            if last_end is None or next_start is None:
                write_err_log(
                    err_dir,
                    f"[WARN] segment {key} {segment_index} invalid neighbor segment time, skipped",
                )
                continue

            # ---------- boundary adjustment ----------
            if last_end > start_time or start_time - last_end > 1000:
                start_time -= 1000
            else:
                start_time = last_end

            if end_time > next_start or next_start - end_time > 1000:
                end_time += 1000
            else:
                end_time = next_start

            # ---------- waveform slicing ----------
            start_idx = int(start_time * sample_rate / 1000)
            end_idx = int(end_time * sample_rate / 1000)

            if start_idx >= end_idx:
                write_err_log(
                    err_dir,
                    f"[WARN] start_time:{start_time} end_time:{end_time},  segment {key} {segment_index} empty slice ({start_idx}, {end_idx}), skipped",
                )
                continue

            segment_pcm = audio_pcm[:, start_idx:end_idx]

            segment_path = os.path.join(os.path.dirname(key), Path(key).stem, "extraction", f"{segment_id}.wav")
            os.makedirs(os.path.dirname(segment_path), exist_ok=True)
            torchaudio.save(segment_path, segment_pcm, sample_rate)

            # ---------- save tmp fields ----------
            audio_segment["start_time.tmp"] = start_time
            audio_segment["end_time.tmp"] = end_time
            audio_segment["text.tmp"] = text
            audio_segment["extraction.tmp"] = segment_path

    return results

def sourceSeparation(configs, paths):
    sample_rate = configs["separation"]["sample_rate"]
    max_step = sample_rate * configs["separation"]["max_step"]
    smooth_window = sample_rate * configs["separation"]["smooth_window"]

    num_gpus = torch.cuda.device_count()
    if configs["device"] == "cpu":
        num_gpus = 1
        devices = ["cpu"]
    else:
        devices = [f"{configs['device']}:{i}" for i in range(num_gpus)]

    predictor_queue = queue.Queue()
    for device in devices:
        pred = mel_band_roformer.Predictor(args=configs["separation"], device=device)
        predictor_queue.put(pred)
    
    results = {path : jsonLoad(path) for path in paths}

    def process(item):
        segment_id = item["segment_id"]
        extracted_path = item["extraction.tmp"]
        separated_path = os.path.join(os.path.dirname(os.path.dirname(extracted_path)), "separation", f"{segment_id}.wav")
        predictor = predictor_queue.get()
        try:
            os.makedirs(os.path.dirname(separated_path), exist_ok=True)
            mix, _ = librosa.load(extracted_path, mono=False, sr=sample_rate)
            if mix.ndim == 2 and mix.shape[0] != 2:
                mix = librosa.to_mono(mix)
            # rescale back to (-1, 1)
            if np.max(np.abs(mix)) > 1:
                mix /= np.max(np.abs(mix))
            
            # import pdb; pdb.set_trace()
            vocals = predictor.predict(mix, rate=sample_rate)
            sf.write(separated_path, vocals[0,:], sample_rate)
            item["separation.tmp"] = separated_path
        except Exception as e:
            msg = f"[ERROR] Failed to separate file {extracted_path}: {e}"
            logging.error(msg)
            write_err_log(err_dir, msg, level="error", filename="step2-error.log")
            item["separation.tmp"] = None
        finally:
            predictor_queue.put(predictor)

    for key, value in results.items():
        audio_path = value["audio_path"]
        audio_segments = value["audio_segments"]

        if not audio_segments:
            msg = f"[WARN] No audio segments found in {key} in processing {audio_path}, skipped source separation."
            logging.warning(msg)
            write_err_log(err_dir, msg, level="warning", filename="step2-error.log")
            continue
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
        # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(process, item) for item in audio_segments]
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Source separation"):
                pass
            
        # Write text, wav.scp and separation report
        success_entries = []
        failure_entries = []
        with open(os.path.join(os.path.dirname(key), Path(key).stem, "text"), "w") as text, open(os.path.join(os.path.dirname(key), Path(key).stem, "wav.scp"), "w") as scp:
            for audio_segment in audio_segments:
                if audio_segment.get("text.tmp"):
                    print(audio_segment["segment_id"], audio_segment["text.tmp"], file=text)
                    sep = audio_segment.get("separation.tmp")
                    if sep:
                        try:
                            abs_sep = os.path.abspath(sep)
                            print(audio_segment["segment_id"], abs_sep, file=scp)
                            success_entries.append({"segment_id": audio_segment["segment_id"], "path": abs_sep})
                        except Exception as e:
                            msg = f"[ERROR] Failed to write separation path for {audio_segment['segment_id']}: {e}"
                            logging.error(msg)
                            write_err_log(err_dir, msg, level="error", filename="step2-error.log")
                            failure_entries.append({"segment_id": audio_segment["segment_id"], "error": str(e)})
                    else:
                        msg = f"[WARN] Missing separation for {audio_segment['segment_id']} in processing {audio_path}"
                        logging.warning(msg)
                        write_err_log(err_dir, msg, level="warning", filename="step2-error.log")
                        failure_entries.append({"segment_id": audio_segment["segment_id"], "error": audio_segment.get("separation.error", "Missing separation")})
        # generate separation report
        report_path = configs["separation"].get("report_path") if configs.get("separation") else None
        if not report_path:
            report_path = os.path.join(report_pth, f"{Path(key).stem}_separation_report.json")
        try:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            report = {
                "audio_path": audio_path,
                "total_segments": len(audio_segments),
                "success_count": len(success_entries),
                "failure_count": len(failure_entries),
                "successes": success_entries,
                "failures": failure_entries,
            }
            with open(report_path, "w", encoding="utf-8") as rf:
                json.dump(report, rf, ensure_ascii=False, indent=4)
        except Exception as e:
            msg = f"[ERROR] Failed to write separation report for {key}: {e}"
            logging.error(msg)
            write_err_log(err_dir, msg, level="error", filename="step2-error.log")
    return results

def forcedAlignment(configs, paths):
    results = {}
    # Load json files with per-file error handling
    for path in paths:
        try:
            results[path] = jsonLoad(path)
        except Exception as e:
            msg = f"[ERROR] Failed to load json '{path}': {e}"
            logging.exception(msg)
            write_err_log(err_dir, msg, level="error", filename="step4-error.log")

    from tqdm import tqdm as _tqdm

    # Outer progress: files
    for key, value in _tqdm(list(results.items()), desc="Forced alignment (files)"):
        try:
            audio_path = value.get("audio_path")
            audio_segments = value.get("audio_segments") or []

            ctm_path = os.path.join(os.path.dirname(key), Path(key).stem, "tri4_ali", "word.ctm")
            try:
                ctms = ctmLoad(ctm_path)
            except Exception as e:
                msg = f"[ERROR] Failed to load CTM '{ctm_path}' for '{key}': {e}"
                logging.exception(msg)
                write_err_log(err_dir, msg, level="error", filename="step4-error.log")
                # mark all segments as trash and skip
                for seg in audio_segments:
                    seg["trash.tmp"] = True
                continue

            try:
                audio_pcm, sample_rate = torchaudio.load(audio_path, normalize=False)
                # audio_duration = audioDurationGet(audio_path)
            except Exception as e:
                msg = f"[ERROR] Failed to load audio '{audio_path}' for '{key}': {e}"
                logging.exception(msg)
                write_err_log(err_dir, msg, level="error", filename="step4-error.log")
                for seg in audio_segments:
                    seg["trash.tmp"] = True
                continue

            # Inner progress: segments for current file
            inner_desc = f"Segments {Path(key).stem}"
            for segment_index, audio_segment in enumerate(_tqdm(audio_segments, desc=inner_desc, leave=False)):
                try:
                    segment_id = audio_segment.get("segment_id")
                    if not segment_id:
                        write_err_log(err_dir, f"[WARN] missing segment_id in {key} idx {segment_index}", level="warning", filename="step4-error.log")
                        audio_segment["trash.tmp"] = True
                        continue

                    start_time = audio_segment.get("start_time.tmp")
                    end_time = audio_segment.get("end_time.tmp")
                    extraction_path = audio_segment.get("extraction.tmp")

                    if segment_id not in ctms:
                        audio_segment["trash.tmp"] = True
                        continue

                    min_start = min([item["start"] for item in ctms[segment_id]])
                    max_end = max([item["end"] for item in ctms[segment_id]])

                    revise_start = min_start + (start_time if start_time is not None else 0)
                    revise_end = max_end + (start_time if start_time is not None else 0)

                    start_time = revise_start
                    end_time = revise_end

                    start_idx = int(start_time * sample_rate / 1000)
                    end_idx = int(end_time * sample_rate / 1000)

                    if start_idx >= end_idx:
                        write_err_log(err_dir, f"[WARN] start_idx >= end_idx for {key} segment {segment_id} ({start_idx},{end_idx}), skipped", level="warning", filename="step4-error.log")
                        audio_segment["trash.tmp"] = True
                        continue

                    segment_pcm = audio_pcm[:, start_idx:end_idx]
                    segment_path = os.path.join(os.path.dirname(key), Path(key).stem, "data_revise", f"{segment_id}.wav")

                    try:
                        if not os.path.exists(segment_path):
                            os.makedirs(os.path.dirname(segment_path), exist_ok=True)
                            torchaudio.save(segment_path, segment_pcm, sample_rate)
                    except Exception as e:
                        msg = f"[ERROR] Failed to save segment wav '{segment_path}' for '{key}' segment '{segment_id}': {e}"
                        logging.exception(msg)
                        write_err_log(err_dir, msg, level="error", filename="step4-error.log")
                        audio_segment["trash.tmp"] = True
                        continue

                    audio_segment["start_time.tmp"] = start_time
                    audio_segment["end_time.tmp"] = end_time
                    audio_segment["path.tmp"] = segment_path
                    audio_segment["trash.tmp"] = False
                except Exception as e:
                    msg = f"[ERROR] Exception processing segment idx {segment_index} for '{key}': {e}"
                    logging.exception(msg)
                    write_err_log(err_dir, msg, level="error", filename="step4-error.log")
                    audio_segment["trash.tmp"] = True
                    continue
        except Exception as e:
            msg = f"[ERROR] Exception processing '{key}': {e}"
            logging.exception(msg)
            write_err_log(err_dir, msg, level="error", filename="step4-error.log")
            continue
    return results

def audioDurationGet(file):
    info = torchaudio.info(file)
    duration = info.num_frames / info.sample_rate
    return duration

def jsonLoad(file):
    with open(file, "r") as fp:
        return json.load(fp)

def jsonDump(obj, file):
    with open(file, "w") as fp:
        json.dump(obj, fp, indent = 4, ensure_ascii=False)

def ctmLoad(file):
    ctm = defaultdict(list)
    try:
        with open(file, "r") as fp:
            lines = fp.readlines()
    except Exception as e:
        msg = f"[ERROR] Failed to open CTM file '{file}': {e}"
        logging.exception(msg)
        write_err_log(err_dir, msg, level="error", filename="step4_4-error.log")
        return ctm

    for line_num, line in enumerate(lines, start=1):
        try:
            items = line.strip().split()
            if len(items) < 5:
                msg = f"[WARN] Malformed CTM line {line_num} in '{file}': '{line.strip()}'"
                logging.warning(msg)
                write_err_log(err_dir, msg, level="warning", filename="step4_4-error.log")
                continue
            # items expected: utt start dur word _ _ (varies) -- we use indices 0,2,3,4 conservatively
            if items[4] != "<eps>":
                start_ms = int(float(items[2]) * 1000)
                end_ms = int((float(items[2]) + float(items[3])) * 1000)
                ctm[items[0]].append({
                    "start": start_ms,
                    "end": end_ms,
                    "text": items[4],
                })
        except Exception as e:
            msg = f"[ERROR] Exception parsing CTM line {line_num} in '{file}': {e}"
            logging.exception(msg)
            write_err_log(err_dir, msg, level="error", filename="step4-error.log")
            continue
    return ctm

def scpLoad(file):
    scp = {}
    with open(file, "r") as fp:
        lines = fp.readlines()
    for line in lines:
        items = line.strip().split(maxsplit=1)
        if len(items) == 2:
            scp[items[0]] = items[1]
    return scp

def resultWrite(results, stage):
    if not results or type(results) != dict:
        logging.error("Invalid json result provided in {}".format(stage))
    for key, value in tqdm(results.items(), desc="{} -- Save Result".format(stage)):
        save_path = key
        try:
            jsonDump(value, save_path)
        except Exception as e:
            logging.error("Failed to extract data {}: {}".format(key, e))
