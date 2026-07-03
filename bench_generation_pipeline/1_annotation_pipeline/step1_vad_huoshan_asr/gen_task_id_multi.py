#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# 批量提交火山 ASR 任务, run_vad_huoshan.sh 默认调用此版本.
# ⚠️ 需配置: config 字典中的 ak, sk, key
# 输出: huoshan_result/<movie>.txt
#

"""
并行化的 huoshan ASR 提交脚本（基于 gen_task_id_new.py）：
- 支持并发上传到 TOS（每个 worker 创建自己的客户端）
- 使用 submit + query 的轮询方式采集结果
- 当 query 返回 55000000 时会重新 submit（最多可配置次数）
- 支持保存响应 JSON 到指定目录，并记录失败到 error.log

用法示例：
  python gen_task_id_multi.py -s files.scp -t /path/to/out -w 8 --max-resubmit 3
"""

import argparse
import json
import os
import time
import uuid
import requests
from typing import List, Tuple
from functools import partial
import multiprocessing as mp
from tqdm import tqdm

# TOS client will be imported inside worker to avoid forking issues
try:
    import tos
    from tos import HttpMethodType
except Exception:
    tos = None


def with_retry(fn, retries=3, base_sleep=1.0, max_sleep=8.0, retry_exceptions=(Exception,)):
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except retry_exceptions as e:
            if attempt == retries:
                raise
            sleep = min(max_sleep, base_sleep * (2 ** (attempt - 1)))
            print(f"Retry {attempt}/{retries} after error: {e}. Sleeping {sleep:.1f}s")
            time.sleep(sleep)


def submit_task_headers(file_url: str, key: str, resource: str) -> Tuple[str, str]:
    """提交任务并返回 (task_id, x_tt_logid) 或抛出异常"""
    submit_url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"
    task_id = str(uuid.uuid4())

    headers = {
        "Content-Type": "application/json",
        "x-api-key": key,
        "X-Api-Resource-Id": resource,
        "X-Api-Request-Id": task_id,
        "X-Api-Sequence": "-1"
    }

    request = {
        "user": {"uid": "YOUR_HUOSHAN_UID"},
        "audio": {"url": file_url, "format": "wav", "codec": "raw", "rate": 16000, "bits": 16, "channel": 1},
        "request": {"model_name": "bigmodel", "enable_itn": False, "enable_punc": True, "show_utterances": True}
    }

    resp = with_retry(lambda: requests.post(submit_url, data=json.dumps(request), headers=headers), retries=3)
    if 'X-Api-Status-Code' in resp.headers and resp.headers['X-Api-Status-Code'] == '20000000':
        return task_id, resp.headers.get('X-Tt-Logid', '')
    else:
        # raise to trigger outer retry / handling
        raise RuntimeError(f"Submit failed: headers={resp.headers} status={resp.status_code} text={resp.text}")


def query_task_headers(task_id: str, x_tt_logid: str, key: str, resource: str):
    query_url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/query"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": key,
        "X-Api-Resource-Id": resource,
        "X-Api-Request-Id": task_id,
        "X-Tt-Logid": x_tt_logid
    }
    return with_retry(lambda: requests.post(query_url, json.dumps({}), headers=headers), retries=3)


def worker_process(file_path: str, txt_dir: str, config: dict):
    """单个文件的处理流程（运行在 worker 中）：
    1. 上传到 TOS 并获取 presigned url
    2. submit_task -> poll query
    3. 处理 55000000 重试逻辑
    4. 保存结果或记录失败
    """
    # Create TOS client inside worker
    ak = config['ak']
    sk = config['sk']
    endpoint = config['endpoint']
    region = config['region']
    bucket_name = config['bucket']
    pre_object_key = config.get('pre_object_key', 'test')
    key = config['key']
    resource = config['resource']
    max_resubmit = config.get('max_resubmit', 3)

    basename = os.path.basename(file_path)
    out_json = os.path.join(txt_dir, f"{basename}.txt")
    if os.path.exists(out_json):
        return (file_path, True, 'already exists')
    os.makedirs(txt_dir, exist_ok=True)

    try:
        if tos is None:
            raise RuntimeError("tos library not available in worker process")
        client = tos.TosClientV2(ak, sk, endpoint, region)

        object_key = os.path.join(pre_object_key, basename)

        def upload_and_sign():
            client.put_object_from_file(bucket_name, object_key, file_path)
            return client.pre_signed_url(HttpMethodType.Http_Method_Get, bucket_name, object_key, expires=3600)

        presigned = with_retry(upload_and_sign, retries=3)
        file_url = presigned.signed_url

        # submit and poll with resubmit-on-55000000
        task_id, x_tt_logid = submit_task_headers(file_url, key, resource)
        resubmit_attempts = 0
        while True:
            resp = query_task_headers(task_id, x_tt_logid, key, resource)
            code = resp.headers.get('X-Api-Status-Code', '')
            if code == '20000000':
                with open(out_json, 'w', encoding='utf-8') as f:
                    json.dump(resp.json(), f, ensure_ascii=False, indent=2)
                return (file_path, True, '')
            elif code == '55000000':
                if resubmit_attempts < max_resubmit:
                    resubmit_attempts += 1
                    print(f"[{basename}] received 55000000, resubmitting ({resubmit_attempts}/{max_resubmit})")
                    task_id, x_tt_logid = submit_task_headers(file_url, key, resource)
                    time.sleep(1)
                    continue
                else:
                    err = f"{basename} exceeded resubmit attempts"
                    open(os.path.join(txt_dir, 'error.log'), 'a', encoding='utf-8').write(f"{file_path} {err}\n")
                    return (file_path, False, err)
            elif code not in ('20000001', '20000002'):
                err = f"{basename} task failed with code {code}"
                open(os.path.join(txt_dir, 'error.log'), 'a', encoding='utf-8').write(f"{file_path} {err}\n")
                return (file_path, False, err)
            time.sleep(1)

    except Exception as e:
        open(os.path.join(txt_dir, 'error.log'), 'a', encoding='utf-8').write(f"{file_path} exception {e}\n")
        return (file_path, False, str(e))


def _worker_star(args_tuple):
    """Top-level picklable helper to unpack arguments for worker_process."""
    return worker_process(*args_tuple)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scp_path', '-s', type=str, required=True)
    parser.add_argument('--txt_dir', '-t', type=str, required=True)
    parser.add_argument('--workers', '-w', type=int, default=16)
    parser.add_argument('--endpoint', type=str, default='tos-cn-beijing.volces.com')
    parser.add_argument('--region', type=str, default='cn-beijing')
    parser.add_argument('--bucket', type=str, default="tos-test-hs-bj-avl-school")
    parser.add_argument('--resource', type=str, default='volc.seedasr.auc')
    parser.add_argument('--pre_object_key', type=str, default='test')
    parser.add_argument('--max_resubmit', type=int, default=3)

    args = parser.parse_args()

    with open(args.scp_path, 'r', encoding='utf-8') as f:
        file_paths = [l.strip() for l in f if l.strip()]

    os.makedirs(args.txt_dir, exist_ok=True)

    config = {
        'ak': "YOUR_TOS_ACCESS_KEY",
        'sk': "YOUR_TOS_SECRET_KEY",
        'endpoint': args.endpoint,
        'region': args.region,
        'bucket': args.bucket,
        'pre_object_key': args.pre_object_key,
        'key': "YOUR_HUOSHAN_KEY",
        'resource': args.resource,
        'max_resubmit': args.max_resubmit
    }

    tasks = [(p, args.txt_dir, config) for p in file_paths]

    print(f"Starting processing {len(tasks)} files with {args.workers} workers")

    with mp.Pool(processes=args.workers) as pool:
        results = []
        for res in tqdm(pool.imap_unordered(_worker_star, tasks), total=len(tasks)):
            results.append(res)
            file_path, ok, msg = res
            if ok:
                print(f"OK: {file_path}")
            else:
                print(f"FAIL: {file_path} -> {msg}")

    # summary
    succ = sum(1 for _, ok, _ in results if ok)
    fail = len(results) - succ
    print(f"Done. success={succ}, failed={fail}")


if __name__ == '__main__':
    main()
