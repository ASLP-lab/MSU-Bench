# Step1：VAD + 火山 ASR 切分流程

## 作用概览
- 输入：包含原始音频路径的 `scp` 文件。
- 处理：Silero VAD 生成语音段，火山引擎 ASR 拉取转写并解析，合并时间戳后按静音缝隙切分长音频。
- 输出：`vad_result/`、`huoshan_result/`、`split_raw_wavs_result/` 三类产物（时间戳、ASR 文本、分段音频+对齐 JSON）。

## 快速开始
1) 准备 `scp`：每行一个音频。格式支持：
   - 仅音频路径：`/abs/path/movie.wav`（movie 名取文件名去扩展名）。
   - 显式指定 movie：`movie_name /abs/path/movie.wav`。
2) 运行：
   ```bash
   cd $PIPELINE_ROOT/step1
   ./run_vad_huoshan.sh /abs/path/list.scp /abs/path/output_root
   ```
3) 产出目录（均位于 `output_root`）：
   - `vad_result/<movie>/segments.json`
   - `huoshan_result/<movie>.txt` 与 `huoshan_result/<movie>/<movie>.json`
   - `split_raw_wavs_result/<movie>/` 下的分段音频与对齐 JSON（含 `logs/`）。

## 目录与脚本
- [step1/run_vad_huoshan.sh](step1/run_vad_huoshan.sh)：一键串联全流程，需提供 `scp` 与输出根目录。
- [step1/gen_timestamps.py](step1/gen_timestamps.py)：调用本地 Silero VAD，生成 `<movie>/segments.json`（start/end 毫秒格式）。
- [step1/gen_task_id.py](step1/gen_task_id.py)：上传音频到 TOS，提交火山 ASR 任务并轮询结果，保存 `<movie>.txt`。请替换文件内的 `appid/token/ak/sk` 为你自己的密钥。
- [step1/parse_huoshan_segments.py](step1/parse_huoshan_segments.py)：从火山 ASR 结果提取分句，生成 `<movie>/<movie>.json`。
- [step1/split_raw_wavs.py](step1/split_raw_wavs.py)：合并 VAD 与 ASR 时间戳，按照 4–6.5 分钟切块，在静音最大缝隙处切分并用 FFmpeg 导出分段音频与对齐 JSON。

## 依赖与环境
- 默认通过 `run_vad_huoshan.sh` 激活 Conda 环境 `silero-vad`。
- 关键依赖：PyTorch、Silero VAD 模型（本地路径 `$MODEL_ROOT/hub/snakers4_silero-vad_master`）、ffmpeg、requests、tqdm、tos SDK。
- 若在新环境运行，请预先安装 ffmpeg 并确保上述 Python 依赖可用。

## 输出结构示例
```
output_root/
├─ vad_result/<movie>/segments.json
├─ huoshan_result/
│   ├─ <movie>.txt
│   └─ <movie>/<movie>.json
├─ split_raw_wavs_result/<movie>/
│   ├─ audio_segments_vad_huoshan_only/partXXX.wav
│   ├─ part_json_vad_huoshan_only/partXXX_vad_huoshan.json
│   ├─ part_json_vad_huoshan_only/total_time.json
│   └─ logs/step1_cut_audio_only.log
└─ logs/
```

