# Bench Generation Pipeline — 评测数据构造流程

本目录梳理并合并了「从原始电影音频 → 中间标注结果 → 最终 QA 样例」的完整构造流程，
含必要脚本、prompt 模板，以及中英文各一部电影的一个分段（seg）作为端到端样例。
**所有 API Key / 密钥均已做匿名化处理（替换为 `YOUR_*` 占位符），开源前请自行填入自己的密钥。**

## 目录结构

```
bench_generation_pipeline/
├── README.md                       # 本文件：端到端总览
├── 1_annotation_pipeline/          # 阶段一：电影音频 → 说话人/副语言标注（来源：llm_pipeline）
│   ├── step0_audio_list/           # 生成音频清单 + 去重
│   ├── step1_vad_huoshan_asr/      # Silero VAD + 火山 ASR + 长音频切分
│   ├── step2_format_asr/           # ASR 文本按分段格式化为相对毫秒时间戳
│   ├── step3_gemini_label/         # Gemini-2.5-Pro 副语言/说话人结构化标注
│   └── gemini_revise/              # 标注精修：源分离 + 说话人日志 + 强制对齐
├── 2_qa_generation/                # 阶段二：标注结果 → QA 样例（来源：speaker-bench）
│   ├── generate_qa.py              # QA 生成主脚本（逐任务调用 Gemini）
│   ├── subsegment_film.py          # 电影长分段 → 5 分钟子段
│   └── prompts_ability_v3_options/ # 各能力 prompt 模板（level1 / level2）
└── 3_samples/                      # 阶段三：中英文各一部电影、一个 seg 的端到端样例
    ├── cn_<film>/                  # 中文样例：她和她的她 S01E03 part004 / part001
    └── en_<film>/                  # 英文样例：Green Book part001
```

## 端到端流程

```
原始电影音频 (.wav)
   │
   │ 1_annotation_pipeline/step0  gen_scp.py → audio.scp（音频清单，去重）
   ▼
audio.scp
   │
   │ 1_annotation_pipeline/step1  run_vad_huoshan.sh
   │   ├─ gen_timestamps.py     Silero VAD → segments.json
   │   ├─ gen_task_id_multi.py  上传 TOS + 提交火山 ASR 任务 → <movie>.txt
   │   ├─ parse_huoshan_segments.py → <movie>/<movie>.json（分句）
   │   └─ split_raw_wavs.py     按 4–6.5 分钟切块 → split_raw_wavs_result/<movie>/
   ▼
split_raw_wavs_result/<movie>/   （分段音频 + VAD/ASR 对齐 json + total_time.json）
   │
   │ 1_annotation_pipeline/step2  gen_time_and_txt.py
   ▼
asr_huoshan_formatted_txt/partXXX.wav.txt   （分段内相对毫秒时间戳 + 文本）
   │
   │ 1_annotation_pipeline/step3  run_gemini.sh → api.py(clean.py / gen_metadata.py)
   │   Gemini-2.5-Pro 听音标注：speaker_id / speaker_name / 情感 / 副语言 / 时间戳 …
   ▼
gemini_step3/data/<movie>/partXXX.json   （结构化标注，audio_segments[]）
   │
   │ 1_annotation_pipeline/gemini_revise  run.sh / run_revised.sh
   │   源分离(MelBandRoformer) + VAD + 火山 ASR 重识别 + wespeaker 说话人日志 + 强制对齐
   ▼
short_long_out/long_seg/<movie>/partXXX/json/partXXX.json   （精修后的说话人标注）
short_long_out/long_seg/<movie>/partXXX/wav/partXXX.wav     （对应音频）
   │
   │ 2_qa_generation/generate_qa.py  + prompts_ability_v3_options/level{1,2}/*.txt
   │   逐能力 prompt + 标注 json + 音频 → Gemini → 解析为 QA
   ▼
short_long_out/QA_results/QA_long/<movie>/partXXX/level{1,2}/<能力>.json   （最终 QA 样例）
```

## 样例说明（3_samples/）

每个样例目录包含同一 seg（`part001`）在四个阶段的产物，可直接对照查看数据流：

| 子目录 | 阶段 | 内容 |
|---|---|---|
| `1_huoshan_split/` | step1+2 | `part001.wav.txt`（ASR 时间戳文本）、`part001_vad_huoshan.json`、`total_time.json` |
| `2_gemini_label/` | step3 | `part001.json`（Gemini 副语言/说话人标注） |
| `3_revised_label/` | gemini_revise | `json/part001.json`（精修后说话人标注）+ `wav/part001.wav`（音频） |
| `4_qa_results/` | QA 生成 | `level1/*.json`（10 个能力）+ `level2/*.json`（6 个能力） |

> 注：样例仅纳入 `part001` 一个 seg；`1_huoshan_split` 未重复纳入分段音频（与 `3_revised_label/wav/part001.wav` 同源）。

## 能力分类体系

- **level1（单说话人属性 / 识别）**：说话人验证、说话人检索、说话人反向检索、说话人计数、说话人观点总结、说话人画像、性别识别、年龄段识别、口音识别、情感识别
- **level2（多说话人上下文 / 对话）**：多说话人观点总结、多说话人情感交互、对话背景推理、对话身份识别、问答结构识别、对话行为识别

## QA 样例格式（最终 json 字段）

```
source_movie_json  : 精修标注 json 路径（3_revised_label/json/part001.json）
source_audio       : 对应音频路径（3_revised_label/wav/part001.wav）
prompt_file        : 使用的 prompt 模板路径
prompt_relpath     : prompt 相对路径（level1/xxx.txt）
model              : gemini-2.5-pro
result[]           : speaker_meta + 多道题（question / question_type / pair / options / answer / answer_text / rationale）
```

## API Key 匿名化

以下密钥均已被替换为占位符，**开源前请确认无残留**：

| 占位符 | 原用途 | 出现位置 |
|---|---|---|
| `YOUR_GEMINI_API_KEY` | Gemini API Key（sk-…，经 cheapapi 中转） | step3/api.py、2_qa_generation/generate_qa.py |
| `YOUR_TOS_ACCESS_KEY` | 火山 TOS 对象存储 AK | step1/gen_task_id*.py、gemini_revise/conf/config.yaml |
| `YOUR_TOS_SECRET_KEY` | 火山 TOS 对象存储 SK | 同上 |
| `YOUR_HUOSHAN_TOKEN` | 火山 ASR token | step1/gen_task_id*.py、config.yaml |
| `YOUR_HUOSHAN_APPID` | 火山 ASR appid | step1/gen_task_id*.py、config.yaml |
| `YOUR_HUOSHAN_UID` | 火山 uid | step1/gen_task_id*.py、config.yaml |
| `YOUR_HUOSHAN_KEY` | 火山 bigmodel key | step1/gen_task_id_new.py、gen_task_id_multi.py |
| `YOUR_HF_TOKEN` | HuggingFace token（wespeaker 模型下载） | gemini_revise/conf/config.yaml |

模型中转地址 `https://apim1tocn.cheapapi.ai` 为第三方代理，可替换为官方 Gemini 端点。

## 依赖与环境

- Python 3.10+，关键依赖：`torch`、`silero-vad`、`ffmpeg`、`aiohttp`、`aiofiles`、`tqdm`、`requests`、`tos`（火山 TOS SDK）、`dashscope`（可选）
- gemini_revise 额外依赖：`MelBandRoformer` 权重（`conf/melBandRoformer/MelBandRoformer.ckpt`，体积大未纳入，请从开源仓库获取）、`wespeaker` 模型、`dnsmos` 模型、`silero-vad` 模型
- Conda 环境名：`silero-vad`（step1/2）、`wekws`（step3）、`llm_pipeline`（gemini_revise）

## 注意事项

- 脚本中保留了大量绝对路径（`/home/work_nfs20/zksun/...`、`/home/work_nfs23/...`），为原作者运行环境路径，迁移时需按自身环境调整。
- `run_vad_huoshan.sh` / `run_gemini.sh` / `run.sh` 等会 `source` 指定 conda 环境，请按需修改。
- 本目录只整理「电影 → 标注 → QA」这条线；podcast / meeting / 电话等其它数据源的分支脚本未纳入。
