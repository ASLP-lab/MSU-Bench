# 阶段二：QA 生成（QA Generation）

来源：`speaker-bench`。把阶段一的精修标注 JSON + 音频，借助 Gemini 生成各能力的 QA 样例。

## 脚本

- **generate_qa.py**（原 `reference_gemini_5min_film_shortlong_final_single2.py`）：QA 生成主脚本。
  单任务、单 seg 调用一次 Gemini，输出一个 `<能力>.json`。
  - 输入：一个 prompt 模板（`--prompt_file`）+ 一个已存在的 reference QA.json（从中读取 `source_movie_json` 标注路径与 `source_audio` 音频路径）。
  - 流程：ffmpeg 提取 ch0 音频 → 组装 prompt（prompt 模板 + 精简标注 JSON）→ Gemini-2.5-Pro → 解析 JSON → 写 `<能力>.json`（含 `source_movie_json` / `source_audio` / `prompt_file` / `model` / `result[]`）。
  - ⚠️ 默认 `--api_key` 已匿名化为 `YOUR_GEMINI_API_KEY`，请用 `--api_key` 传入或设置环境变量 `GEMINI_API_KEY`。

  示例：
  ```bash
  python generate_qa.py \
    --prompt_file prompts_ability_v3_options/level1/说话人识别能力-说话人验证任务.txt \
    --out_dir <...>/<segment>/level1 \
    --model gemini-2.5-pro \
    --small_label_json --overwrite
  ```

- **subsegment_film.py**：把 step1 的长分段切分为 ~5 分钟子段（long_seg / short_seg 的来源之一）。

## prompt 模板（prompts_ability_v3_options/）

- `level1/`：10 个单说话人能力（验证、检索、反向检索、计数、观点总结、画像、性别、年龄段、口音、情感）
- `level2/`：6 个多说话人能力（观点总结、情感交互、对话背景推理、对话身份识别、问答结构识别、对话行为识别）

每个 `.txt` 内含该能力的题面模板与输出 JSON 约定；`generate_qa.py` 会把标注 JSON 填入后送 Gemini。

## 产物

`<output_root>/short_long_out/QA_results/QA_long/<movie>/partXXX/level{1,2}/<能力>.json`
（与 `3_samples/*/4_qa_results/` 同构）

## 评测脚本（未纳入本目录，见 speaker-bench 根）

`test_gpt4o.py`、`test_gemini2-5_flash.py`、`test_raidar.py`、`final_qa_level_task_acc.py` 等用于跑模型评测与统计，属评测侧，不在数据构造流程内。
