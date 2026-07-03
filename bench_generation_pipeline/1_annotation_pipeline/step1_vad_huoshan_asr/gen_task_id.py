# -*- coding: utf-8 -*-
#
# 上传音频到 TOS 对象存储, 提交火山 bigmodel ASR 任务, 轮询结果.
# ⚠️ 需配置: ak, sk, appid, token, uid
# 输出: huoshan_result/<movie>.txt
#

import json
import os
import tos
import time
import uuid
import requests
from tos import HttpMethodType
from io import BytesIO 
import argparse


def with_retry(fn, retries=3, base_sleep=1.0, max_sleep=8.0, retry_exceptions=(Exception,)):
    """Simple retry helper with exponential backoff."""
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except retry_exceptions as e:
            if attempt == retries:
                raise
            sleep = min(max_sleep, base_sleep * (2 ** (attempt - 1)))
            print(f"Retry {attempt}/{retries} after error: {e}. Sleeping {sleep:.1f}s")
            time.sleep(sleep)


def submit_task():

    submit_url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"

    task_id = str(uuid.uuid4())

    headers = {
        "X-Api-App-Key": appid,
        "X-Api-Access-Key": token,
        "X-Api-Resource-Id": "volc.seedasr.auc",
        "X-Api-Request-Id": task_id,
        "X-Api-Sequence": "-1"
    }

    request = {
        "user": {
            "uid": uid
        },
        "audio": {
            "url": file_url,
            "format": "wav", #mp3
            "codec": "raw",
            "rate": sr,
            "bits": 16,
            "channel": 1
        },
        "request": {
            "model_name": "bigmodel",
            "enable_itn": False,
            "enable_punc": True,
            # "enable_ddc": True,
            "show_utterances": True,
            "show_speech_rate": True,
            "show_volume": True,
            "enable_lid": True,
            "enable_emotion_detection": True,
            "enable_gender_detection": True,
            # "enable_channel_split": True,
            # "vad_segment": True,
            "enable_speaker_info": True,
            "corpus": {
                # "boosting_table_name": "test",
                "correct_table_name": "",
                "context": ""
            }
        }
    }
    print(f'Submit task id: {task_id}')
    response = with_retry(lambda: requests.post(submit_url, data=json.dumps(request), headers=headers), retries=3, base_sleep=1.0)
    if 'X-Api-Status-Code' in response.headers and response.headers["X-Api-Status-Code"] == "20000000":
        print(f'Submit task response header X-Api-Status-Code: {response.headers["X-Api-Status-Code"]}')
        print(f'Submit task response header X-Api-Message: {response.headers["X-Api-Message"]}')
        x_tt_logid = response.headers.get("X-Tt-Logid", "")
        print(f'Submit task response header X-Tt-Logid: {response.headers["X-Tt-Logid"]}\n')
        return task_id, x_tt_logid
    else:
        print(f'Submit task failed and the response headers are: {response.headers}')
        exit(1)
    return task_id

def query_task(task_id, x_tt_logid):
    query_url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/query"

    headers = {
        "X-Api-App-Key": appid,
        "X-Api-Access-Key": token,
        "X-Api-Resource-Id": "volc.seedasr.auc",
        "X-Api-Request-Id": task_id,
        "X-Tt-Logid": x_tt_logid  # 固定传递 x-tt-logid
    }

    response = with_retry(lambda: requests.post(query_url, json.dumps({}), headers=headers), retries=3, base_sleep=1.0)

    if 'X-Api-Status-Code' in response.headers:
        print(f'Query task response header X-Api-Status-Code: {response.headers["X-Api-Status-Code"]}')
        print(f'Query task response header X-Api-Message: {response.headers["X-Api-Message"]}')
        print(f'Query task response header X-Tt-Logid: {response.headers["X-Tt-Logid"]}\n')
    else:
        print(f'Query task failed and the response headers are: {response.headers}')
        exit(1)
    return response

def main(wav_name, txt_dir="$DATA_ROOT/test/huoshan/qvwan_out/txt"):
    task_id, x_tt_logid = submit_task()
    resubmit_attempts = 0
    max_resubmit_attempts = 3
    while True:
        query_response = query_task(task_id, x_tt_logid)
        code = query_response.headers.get('X-Api-Status-Code', "")
        if code == '20000000':  # task finished
            #print(query_response.json())
            os.makedirs(txt_dir, exist_ok=True)
            with open(f"{txt_dir}/{wav_name}.txt", "w", encoding="utf-8") as f:
                json.dump(query_response.json(), f, ensure_ascii=False, indent=2)
            return(query_response.json()["result"]["text"])
            print("SUCCESS!")
            exit(0)
        elif code == '55000000':  # need to re-submit
            if resubmit_attempts < max_resubmit_attempts:
                resubmit_attempts += 1
                print(f"Received 55000000 for {wav_name}, re-submitting task (attempt {resubmit_attempts}/{max_resubmit_attempts})")
                task_id, x_tt_logid = submit_task()
                time.sleep(1)
                continue
            else:
                print(f"{wav_name} exceeded resubmit attempts ({max_resubmit_attempts}). Giving up.")
                return("")
                exit(0)
        elif code != '20000001' and code != '20000002':  # task failed
            print(wav_name)
            print("FAILED!")
            return("")
            exit(0)
        time.sleep(1)

def infer_asr(wav_name, wav_path, txt_dir):
    global file_url 
    file_url = wav_path
    return main(wav_name, txt_dir)


parser = argparse.ArgumentParser()  
parser.add_argument('--scp_path', "-s", type=str, default="$DATA_ROOT/test/qvwan/test.scp", help='path to scp file')
parser.add_argument('--txt_dir', "-t", type=str, default="$DATA_ROOT/test/huoshan/qvwan_out/txt", help='directory to save txt files')

args = parser.parse_args()
scp_path = args.scp_path
txt_dir = args.txt_dir
# for TOS
ak = "YOUR_TOS_ACCESS_KEY"
sk = "YOUR_TOS_SECRET_KEY"

endpoint = "tos-cn-beijing.volces.com"
region = "cn-beijing"
bucket_name = "tos-test-hs-bj-avl-school"

pre_object_key = "test"
# dir_name = "$DATA_ROOT/test/huoshan/short_wavs"
# file_names = os.listdir(dir_name)
# scp_path = "$DATA_ROOT/test/qvwan/test.scp"

with open(scp_path, "r", encoding="utf-8") as f:
    file_paths = [line.strip() for line in f if line.strip()]

# for ASR
uid = "YOUR_HUOSHAN_UID"

file_url = ""
sr = 16000

appid = "YOUR_HUOSHAN_APPID" #需填写
token = "YOUR_HUOSHAN_TOKEN" #需填写

try:
    client = tos.TosClientV2(ak, sk, endpoint, region)

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        object_key = os.path.join(pre_object_key, file_name)

        def upload_and_sign():
            client.put_object_from_file(bucket_name, object_key, file_path)
            return client.pre_signed_url(HttpMethodType.Http_Method_Get, bucket_name, object_key, expires=3600)

        presigned_url = with_retry(upload_and_sign, retries=3, base_sleep=1.0)

        text = infer_asr(file_name, presigned_url.signed_url, txt_dir) 
        print(presigned_url.signed_url)
        print(text)
except tos.exceptions.TosClientError as e:
    # 操作失败，捕获客户端异常，一般情况为非法请求参数或网络异常
    print('fail with client error, message:{}, cause: {}'.format(e.message, e.cause))
except tos.exceptions.TosServerError as e:
    # 操作失败，捕获服务端异常，可从返回信息中获取详细错误信息
    print('fail with server error, code: {}'.format(e.code))
    # request id 可定位具体问题，强烈建议日志中保存
    print('error with request id: {}'.format(e.request_id))
    print('error with message: {}'.format(e.message))
    print('error with http code: {}'.format(e.status_code))
    print('error with ec: {}'.format(e.ec))
    print('error with request url: {}'.format(e.request_url))
except Exception as e:
    print('fail with unknown error: {}'.format(e))