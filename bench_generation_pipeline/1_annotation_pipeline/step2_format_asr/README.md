# Step2：火山 ASR 片段文本格式化

## 作用概览
- 输入：
  - `split_raw_wavs_result/<movie>/part_json_vad_huoshan_only/total_time.json`（来自 Step1 切分结果）。
  - `huoshan_result/<movie>/<movie>.json`（火山 ASR 原始分句）。
  - `scp` 列表（与 Step1 相同，每行一个音频路径，可选自定义 movie 名）。
- 输出：每个 movie 的 `asr_huoshan_formatted_txt/partXXX.wav.txt`，时间戳为分段内的相对毫秒，便于后续对齐或文本处理。

## 文件说明
- [step2/gen_time_and_txt.py](step2/gen_time_and_txt.py)：
  - 读取 `total_time.json`（场景起止绝对时间，秒）与火山 ASR JSON（绝对时间 MM:SS:mmm）。
  - 将 ASR 句子裁剪到对应片段内，转换为相对毫秒，输出 `start_time:<ms>, end_time:<ms>, text: ...` 行格式。
  - 日志写入 `step2_huoshan_asr_for_reference.log`。
- [step2/gen_time_and_txt.sh](step2/gen_time_and_txt.sh)：
  - 包装执行脚本，激活 `silero-vad` 环境，传递路径参数并调用 `gen_time_and_txt.py`。

## 快速开始
1) 确保已完成 Step1，得到 `split_raw_wavs_result/` 与 `huoshan_result/`。
2) 准备 `scp` 列表（与 Step1 相同）。
3) 运行：
   ```bash
   cd $PIPELINE_ROOT/step2
   ./gen_time_and_txt.sh /abs/path/list.scp /abs/path/output_root
   ```
   - `output_root` 与 Step1 保持一致，内部会解析：
     - `split_raw_wavs_result` 用作切分时间来源
     - `huoshan_result` 用作 ASR 来源
     - `logs` 存放日志

## 输出结构示例
```
output_root/
├─ split_raw_wavs_result/<movie>/
│   ├─ part_json_vad_huoshan_only/total_time.json
│   └─ asr_huoshan_formatted_txt/
│       ├─ part001.wav.txt
│       ├─ part002.wav.txt
│       └─ ...
└─ logs/step2_huoshan_asr_for_reference.log
```

## 参数与默认值（gen_time_and_txt.py）
- `--base-outputs-dir`：默认 `$DATA_ROOT/test/test_pipeline/split_raw_wavs_result`
- `--asr-results-dir`：默认 `$DATA_ROOT/test/test_pipeline/huoshan_result`
- `--list-file`：默认 `$SILERO_VAD_ROOT/movies.scp`
- `--log-dir`：默认 `$DATA_ROOT/test/test_pipeline_step3.5_outputs/logs`
- `gen_time_and_txt.sh` 会根据传入的 `output_root` 自动填好前两个目录和日志目录。

## 注意事项
- 路径需与 Step1 的输出保持一致，否则会提示缺失 `total_time.json` 或火山 JSON。
- 输出时间为“片段内相对毫秒”，用于与切分音频对齐；源 JSON 为绝对时间（分钟:秒:毫秒）。
- 若日志中看到缺失提示，请检查对应 movie 的 Step1 产物是否齐全。
