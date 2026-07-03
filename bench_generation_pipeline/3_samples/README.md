# 阶段三：端到端样例（Samples）

中英文各挑一部电影、一个分段（`part001`），展示从电影音频到最终 QA 的完整数据流。

| 语言 | 电影 | seg |
|---|---|---|
| 中文 (cn) | 她和她的她 S01E03（part004 切分） | part001 |
| 英文 (en) | Green Book 2018（part001 切分） | part001 |

## 每个样例目录的结构

```
<lang>_<film>/
├── 1_huoshan_split/                 # step1 + step2 产物
│   ├── part001.wav.txt              #   ASR 文本（分段内相对毫秒时间戳）
│   ├── part001_vad_huoshan.json     #   VAD + 火山 ASR 对齐
│   └── total_time.json              #   分段起止绝对时间
├── 2_gemini_label/
│   └── part001.json                 # step3 Gemini 副语言/说话人标注（audio_segments[]）
├── 3_revised_label/
│   ├── json/part001.json            # gemini_revise 精修后的说话人标注（QA 的 source_movie_json）
│   └── wav/part001.wav              # 对应音频（QA 的 source_audio）
└── 4_qa_results/
    ├── level1/*.json                # 10 个 level1 能力的最终 QA
    └── level2/*.json                # 6 个 level2 能力的最终 QA
```

## 如何对照查看数据流

1. 听 `3_revised_label/wav/part001.wav`，对照 `3_revised_label/json/part001.json` 中的 `audio_segments[]`（每个 segment 有 `speaker_id` / `speaker_name` / `text` / `start_time` / `end_time` / 情感 / 副语言 …）。
2. 打开 `4_qa_results/level1/说话人识别能力-说话人验证任务.json`：
   - 顶部 `source_movie_json` / `source_audio` 指向上面两个文件；
   - `result[0].speaker_meta` 列出本题涉及的说话人片段；
   - `result[1:]` 为多道选择题（`question` / `options` / `answer` / `answer_text` / `rationale`）。
3. `2_gemini_label/part001.json` 与 `3_revised_label/json/part001.json` 结构相同（均含 `audio_segments[]`），后者多了 `_source_part_json` / `_source_audio_path` 并经过说话人日志/对齐精修。

## 说明

- 样例仅取 `part001` 一个 seg，体积可控（音频约 12MB/个）。
- `1_huoshan_split/` 未重复纳入分段音频，因其与 `3_revised_label/wav/part001.wav` 同源。
- QA 中的 `source_movie_json` / `source_audio` 为原作者环境的绝对路径，迁移后需按实际路径替换。
