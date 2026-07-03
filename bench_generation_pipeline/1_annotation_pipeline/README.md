# 阶段一：标注流程（Annotation Pipeline）

来源：`llm_pipeline`。把原始电影音频加工为带说话人/副语言标注的结构化 JSON。

## 顺序执行

1. **step0_audio_list** — `gen_scp.py` 递归扫描音频目录生成 `audio.scp`（每行一个 wav 路径）；`step_0-{1,2,3}.sh` 做同名/重复去重。
2. **step1_vad_huoshan_asr** — `run_vad_huoshan.sh <scp> <output_root>` 串联：
   - `gen_timestamps.py`：Silero VAD 生成 `vad_result/<movie>/segments.json`
   - `gen_task_id_multi.py`：上传 TOS + 提交火山 ASR 任务，轮询结果 → `huoshan_result/<movie>.txt`
   - `parse_huoshan_segments.py`：解析火山结果 → `huoshan_result/<movie>/<movie>.json`
   - `split_raw_wavs.py`：合并 VAD+ASR 时间戳，按 4–6.5 分钟在静音缝隙切分 → `split_raw_wavs_result/<movie>/`（`audio_segments_vad_huoshan_only/partXXX.wav`、`part_json_vad_huoshan_only/partXXX_vad_huoshan.json` + `total_time.json`）
   - ⚠️ 需填入火山 `appid/token/ak/sk`（见 `gen_task_id*.py`，已匿名化为 `YOUR_*`）
3. **step2_format_asr** — `gen_time_and_txt.sh <scp> <output_root>` → `asr_huoshan_formatted_txt/partXXX.wav.txt`（把绝对时间戳裁剪到分段内，转相对毫秒）。
4. **step3_gemini_label** — `run_gemini.sh <output_root>` 串联 `api.py`（Gemini-2.5-Pro 听音标注）→ `clean.py`（清洗）→ `gen_metadata.py` → `gemini_step3/data/<movie>/partXXX.json`。
   - ⚠️ 需填入 `YOUR_GEMINI_API_KEY`（`api.py` / 默认经 `https://apim1tocn.cheapapi.ai` 中转，可换官方端点）
5. **gemini_revise** — `run.sh <input_dir>` / `run_revised.sh` / `run_dyn_nj.sh`：源分离 + VAD + 火山 ASR 重识别 + wespeaker 说话人日志 + 强制对齐，精修标注 → `short_long_out/long_seg/<movie>/partXXX/json/partXXX.json`。
   - ⚠️ 需填入 `conf/config.yaml` 中的 `tos.ak/sk`、`recognition.appid/token/uid`、`diarization.use_auth_token`（均已匿名化）
   - ⚠️ 需自行下载模型权重：`MelBandRoformer.ckpt`、wespeaker 模型、dnsmos 模型、silero-vad 模型

## 各 step 产物对应关系（与 3_samples/ 对照）

```
1_huoshan_split/   ← step1 + step2 产物
2_gemini_label/    ← step3 产物
3_revised_label/   ← gemini_revise 产物
```

详见 `step1/README.md`、`step2/README.md`（从原仓库原样复制）。
