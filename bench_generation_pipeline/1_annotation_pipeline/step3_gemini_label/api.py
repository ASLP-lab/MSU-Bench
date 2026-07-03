# -*- coding: utf-8 -*-
#
# 异步批量调用 Gemini 进行音频副语言/说话人结构化标注.
# 输入: scp 音频列表 + 时间戳目录
# 输出: JSONL 标注结果
# ⚠️ 需配置: url (Gemini 端点), key (API Key)
#

import os
import asyncio
import aiohttp
import aiofiles
import base64
import json
import mimetypes
from tqdm.asyncio import tqdm
import random
import argparse


def create_analysis_prompt(segment: str = None) -> str:
    """创建语音分析提示词"""
    if segment is None:
        return create_analysis_prompt_raw()

    return f"""
你是一位经过专门训练的副语言分析专家，擅长从人类声音中提取细腻的声学信号。你的能力包括：对说话者的语气、韵律、情感、音质等声学特征进行细致分析；在严格区分听觉属性的前提下进行融合判断。仅利用听觉线索和提供的句级别时间戳解决各种可能存在的歧义；通过听觉“脑补”构建场景。

**任务目标：**
输入一个影视剧音频和下面的句级时间戳（含该句子对应的文本转录）。输出对音频中所有语音片段进行逐段、结构化标注。每个片段对应单一说话者的一段连续语音。

核心原则：
- 我提供的时间戳和文本仅作为定位参考。在该基础上，你必需要进行 ‘二次校准与补全’ ：
  1. 文本校对：如果发现参考文本与音频实际内容不符，请务必以音频为准，纠正所有错别字和识别错误。
  2. 说话人拆分：若参考的一行文本中包含多个说话人，你必须根据实际听感将其拆分为多行，并重新计算每一句准确的起止时间戳。
  3. 缺失补全：若音频中存在参考文本未记录的对话（如重叠语音、漏掉的短句），你必须根据你的理解将其识别出来并插入到正确的时间位置，给出精确的时间戳。
  4. 参考时间戳： 请严格以我提供的时间戳为准。除需补全文本未记录的语音（如重叠对话、遗漏短句）这一特殊情况外，默认原时间戳是精确的（即使存在错别字或错误名词），请务必保持不变，不要随意修改。
  最终原则：参考文本是草稿，你的输出必须是基于音频还原的‘绝对真相’，确保每一毫秒的时间戳都与该句语音的起止点完美契合。
- 纯听觉分析：所有的标注必须完全基于听觉信号。不得臆造任何无法通过声音感知的视觉动作（如“点头”、“挥手”）。
- 全覆盖标注：你的标注必须覆盖给定的整个音频片段。请从头分析到尾，不得中途跳段或漏段。
- 准确性与可验证性：你标注的所有内容必须准确、具体、可听觉验证。
- 属性定义： 
  1. 对于语音属性 gender（性别）、age（年龄）、pitch（音高）、speed（语速）、emotion（情感）、emotion_level（情感强度）、accent（口音）、tone（语气）、rhythm（节奏）、texture（音质）、pronunciation（发音）、paralinguistic（副语言事件）和 contextual_inference（语境推断），所有的标注内容必须完全基于听觉内容。输出必须仅反映音频的声学和语言学属性。
  2. 对于 `environment`（声学环境），需结合混响、空间感和背景声判断物理空间。
  3. 对于 background_sound（背景音），仅基于非语音的听觉信号描述环境声（音乐、噪声、音效等）。使用精炼短语，如“持续的风噪”、“远处的车辆声”。
  4. 对于 caption（摘要），生成一段自然语言摘要，综合该片段已标注的声学属性（如音高、语速、情感等）。不要引入不存在的新的听觉。确保它源自核心声学属性（性别、年龄、音高、语速、情感、口音、语气、节奏、音质、发音）。只有当描述了显著的副语言事件（例如“紧张的笑声”、“听得见的叹息”）时，才将其包含在摘要中。绝对不要在摘要中包含“没有副语言事件”或“无任何副语言”之类的短语。
- 防幻觉与真实性约束：
  1. 拒绝幻觉：仅描述听到的内容，拒绝过度推断。若不确定，请归类为最广泛的合理类别，严禁捏造不存在的声音、情绪或事件。
  2. 仅基于声学证据：你的所有描述必须有明确的听觉来源。严禁描述无法通过耳朵直接感知的物理动作（如“点头”、“耸肩”、“皱眉”、“看手机”）。
  3. 宁缺毋滥：对于背景音（background_sound）和声学环境（environment）等，如果音频非常干净或无法确定，请填写“无”，绝对不要为了丰富描述而捏造不存在的“远处的警笛声”、“微弱的风声”或“人群嘈杂声”。

---

---
** 时间戳如下：**

{segment}

每一行都是一个时间戳和对应的文本，格式为start_time: xxx, end_time: xxx(单位为毫秒)，text: xxx 表示该片段的开始和结束时间以及该片段的文本内容。例如：
start_time: 12345, end_time: 23456, text: 你好
表示该片段从12秒345毫秒开始，到23秒456毫秒结束，内容为“你好”。

---

**输出结构**
输出结构必须是有效且可解析的 JSON（所有字段和对应的值必须使用双引号；确保没有遗漏"{" 和 "}"；确保所有字段名（如 "segment_id"）与下方指定的完全一致）。返回一个包含 `audio_segments` 列表的 JSON 对象，其中每一项代表单个说话者的一段连续语音片段：

{{
"audio_segments": [
{{
    "segment_id": "当前语音片段的唯一标识符（例如：seg_001、seg_010）。",
    "speaker_id": "说话人的内部唯一ID（例如：spk_01、spk_10）。",
    "speaker_name": "说话人名称。需要识别并标注说话人的具体姓名或身份，以及对应的声学特征（年龄段），格式须严格遵循“姓名_年龄段”（例如：“张翠花_青年"，"三毛_儿童"，"医生_中年"）。推断应基于音频中的称谓指代、音高频率、语调起伏及对话逻辑；若姓名未知，须结合声学属性进行占位描述（如：“神秘男_中年”、“女旁白_青年”），并确保同一说话人在整段音频中的标签唯一且一致。",
    "text": "该片段的逐字转录内容，包含口语化表达。转录语言应严格遵循说话者实际使用的口语语种。",
    "text_and_paralanguage": "基于text内容基础上，插入副语言事件标记的片段转录。若有明显的副语言事件，从["<呼吸>", "<吸鼻>", "<笑声>", "<哭声>", "<咳嗽>", "<清嗓>", "<打喷嚏>", "<叹气>", "<倒吸气>", "<打鼾>", "<哈欠>", "<鼓掌>", "<吹口哨>", "<哼唱>", "<嘘声>","<嘶声>", "<咂嘴>", "<呻吟>", "<迟疑>", "<确认>", "<不满>", "<惊讶>", "<疑问>"]中选择，须在对应位置以方括号标注，比如：太棒了</笑声>！/ 我很伤心</哭声>我错了。若有明显发音的副语言事件须同时把发音转录标记出来，比如：<笑声>哈哈哈</笑声>！/<哭声>呜呜</哭声>我太难受了。",
    "start_time": "片段开始时间，格式为 xxx(例如78452)，xxx 为毫秒",
    "end_time": "片段结束时间，格式为 xxx(例如92346)，xxx 为毫秒",
    "language": "使用 ISO 639-1 语言代码标注语种，从["zh","en","ja"...]中选择",
    "background_sound": "描述片段中可听到的非语音背景声（如“微弱鸟鸣”、“持续的引擎轰鸣”）。若无明显背景声，填写“无”",
    "environment": "基于音频中的混响、回声、空间感和底噪类型等，推断说话者所处的物理环境（如“强混响且空旷，推断为大型空荡室内或地下室”、“声音干涩无回声，推断为录音棚或狭小软包房间”、“伴随风噪和远处的城市喧嚣，推断为室外开放环境”）。如果实在无法进行推断，填写“无”",
    "gender": "基于声音的音色、共鸣特性与音高范围判断声音的性别，从["男声","女声"]中选择",
    "age": "基于声音的成熟度、气息控制与音高稳定性判断声音的年龄段，从["儿童","少年","青年","中年","老年"]中选择（儿童<12岁，少年13-18岁，青年19-35岁，中年36-60岁，老年>60岁）",
    "pitch": "描述语音的整体音高水平及变化特征（如“高而轻快”、“低沉且带下降语调”、“音高波动明显的儿童声”）",
    "speed": "描述语音的语速与流畅度（如“语速较快且音节短促”、“缓慢并伴随较长停顿”）。若片段内存在明显加速或减速，应加以说明。",
    "emotion": "仅基于语音表现（语调、力度、颤抖等）描述说话人呈现的内部情绪状态，从['生气', '厌恶', '恐惧', '快乐', '悲伤', '惊讶', '中性']中选择其一。",
    "emotion_level": "基于语音表现（音量变化、语速、音高波动、气息紧张度等）判断情绪强度，从["极弱", "较弱", "中等", "较强", "极强"]中选择其一。",
    "accent": "请尽可能具体描述说话人的口音（如“通用美式英语”、“标准普通话、”“北京话”）。若口音不明确，可使用更宽泛类别；若为非母语口音，需注明可感知的来源影响（如“带西班牙语口音的非母语英语”等）。",
    "tone": "描述说话人的语调显示的态度与立场（如“带有轻微升调的安慰语气”、“讽刺且平淡”、“急切且短促的语气”）。",
    "rhythm": "描述语音的韵律模式（如“断奏且急促”、“单调且重音均匀”、“犹豫且频繁停顿”）。",
    "texture": "描述语音的音色与质感（如“气声且单薄”、“粗糙并带颗粒感”、“温暖且浑厚”）。不得加入情绪或态度解读。",
    "pronunciation": "描述语音的发音清晰度与咬字特征（如“咬字清晰有力”、“含混且辅音弱化”、“不清晰且发声松散”）。仅关注语音的可懂度。",
    "paralinguistic": "描述非词汇性质的副语言事件（如“说话前的啜泣声”、“句中急促吸气”、“尾音伴随紧张笑声”）。若无明显副语言事件，填写“无”",
    "contextual_inference": "仅基于语音表现和听觉背景，对说话者可能的角色或情境进行推断（如“一名医生试图掩饰紧迫感”、“压低声音耳语，推断为秘密交谈”、“语气正式且平稳，推断为播报或汇报场景”）。不得引用无法从音频中感知的剧情信息",
    "caption": "基于核心声学属性（性别、年龄、音高、语速、情绪、口音、语气、节奏、音质、发音）生成 1–4 句自然语言摘要。仅当 paralinguistic 中存在明显的副语言事件时，才将其纳入摘要；否则摘要不得提及副语言事件缺失。需使用具体感知到的描述词，避免空泛的表述。示例1（无副语言事件）：“一位年轻成年女性用清晰、浑厚的声音说话，操着通用美式英语口音，语速为正常的对话速度，节奏流畅。她的语气带有讽刺和被逗乐的意味，通过中等音高传达，常带有疑问语调的上升，咬字标准清晰。” 示例2（有副语言事件）：“一位中年男性的声音，特点是温暖的音质和缓慢审慎的语速，带有美国南部口音，传达出深深的悲伤。这一点通过颤抖的节奏和句子中间急促的吸气得到强调，突出了他话语的情感分量。”"
}},
...
]
}}
---

**约束与最佳实践**。
- 开始时间和结束时间必须严格遵循 `xxx` 格式，xxx 为毫秒，与 text_and_paralanguage 的内容严格对齐。
- 我提供的时间戳和文本仅作为定位参考。在该基础上，你必需要进行 ‘二次校准与补全’ ：
  1. 文本校对：如果发现参考文本与音频实际内容不符，请务必以音频为准，纠正所有错别字和识别错误。
  2. 说话人拆分：若参考的一行文本中包含多个说话人，你必须根据实际听感将其拆分为多行，并重新计算每一句准确的起止时间戳。
  3. 缺失补全：若音频中存在参考文本未记录的对话（如重叠语音、漏掉的短句），你必须根据你的理解将其识别出来并插入到正确的时间位置，给出精确的时间戳。
  最终原则：参考文本是草稿，你的输出必须是基于音频还原的‘绝对真相’，确保每一毫秒的时间戳都与该句语音的起止点完美契合。
- 所有描述必须是简洁的短语或句子（最好不少于 6 个汉字），而不是单个词语。例外：caption 可以是 1-4 个句子；pitch，speed，emotion，accent，tone等语音属性，如果为了更详细地说明，可以使用稍长的短语。
- 进行语音属性标注时，请避免使用通用的术语，如“中等”、“正常”或“平均”。要使用生动、具体的描述词。
- 除了"text_and_paralanguage"、"start_time"、"end_time"、"language"、"speaker_id" 和 "segment_id"字段对应的值以外，其他所有字段（background_sound，age等）对应的值用**简体中文**输出。但是特别注意，转录内容（text_and_paralanguage）的语言应该遵循说话人的口语语种。
- 不要产生幻觉（Hallucinate）：进行属性标注时，不确定的时候给出最广泛合理的类别（如“较宽泛的美式英语口音”）。不得创造不存在的声音、情绪、人物或事件。
- 识别说话人时还应考虑不同说话人之间的声音特征的差异。
- 语言（language）字段必须反映片段内实际的口语语言（如：zh, en, ja）。不能根据电影主要语言推断。
- 确保一致性：caption（摘要）应直接反映标注的语音属性，不得与上面单独标注的语音属性相互矛盾。
"""

def create_analysis_prompt_raw():
    return """
    你是一位经过专门训练的副语言分析专家，擅长从人类声音中提取细腻的声学信号。你的能力包括：对说话者的语气、韵律、情感、音质等声学特征进行细致分析；在严格区分听觉属性的前提下进行融合判断。

    **任务目标：**
    输入一个段影视剧音频。输出对音频中所有语音片段进行逐段、结构化标注。每个片段对应单一说话者的一段连续语音。

    核心原则：
    - 你的标注必须覆盖给定的整个音频片段。请从头分析到尾，不得中途跳段或漏段。每次说话者转换或停顿超过 1 秒视为新片段。
    - 你标注的所有内容必须准确、具体、可听觉验证。
    - 对于语音属性 gender（性别）、age（年龄）、pitch（音高）、speed（语速）、emotion（情感）、accent（口音）、tone（语气）、rhythm（节奏）、texture（音质）、pronunciation（发音）、paralinguistic（副语言事件）和 contextual（语境推断），所有的标注内容必须完全基于听觉内容。输出必须仅反映音频的声学和语言学属性。
    - 对于 background_sound（背景音），仅基于非语音的听觉信号描述环境声（音乐、噪声、音效等）。使用精炼短语，如“持续的风噪”、“远处的车辆声”。
    - 对于 caption（摘要），生成一段自然语言摘要，综合该片段已标注的声学属性（如音高、语速、情感等）。不要引入不存在的新的听觉。
    - 对于 caption（摘要）属性，确保它源自核心声学属性（性别、年龄、音高、语速、情感、口音、语气、节奏、音质、发音）。只有当描述了显著的副语言事件（例如“紧张的笑声”、“听得见的叹息”）时，才将其包含在摘要中。绝对不要在摘要中包含“没有副语言事件”或“无任何副语言”之类的短语。

    ---

    **输出结构**
    输出结构必须是有效且可解析的 JSON（所有字段和对应的值必须使用双引号；确保没有遗漏左花括号和右花括号；确保所有字段名（如 "segment_id"）与下方指定的完全一致）。返回一个包含 `audio_segments` 列表的 JSON 对象，其中每一项代表单个说话者的一段连续语音片段：

    {{
    "audio_segments": [
    {{
        "segment_id": "当前语音片段的唯一标识符（例如：seg_001、seg_010）。",
        "speaker_id": "说话人的内部唯一ID（例如：spk_01、spk_10）。",
        "speaker_confidence": "说话人识别置信度：high / medium / low。",
        "text_and_paralanguage": "该片段的转录内容，包含口语化表达。若有明显的副语言（如叹息、笑声等），须一并标注，比如：我很开心（大笑）。转录内容的语言应该遵循说话者的口语语种。",
        "start_time": "片段开始时间，格式为 mm:ss:xxx（例如：02:12:455）。mm 为分钟，ss 为秒，xxx 为毫秒，中间用英文冒号分隔。",
        "end_time": "片段结束时间，格式为 mm:ss:xxx（例如：02:18:932）。mm 为分钟，ss 为秒，xxx 为毫秒，中间用英文冒号分隔。",
        "language": "使用 ISO 639-1 语言代码（例如：“zh”、“en”、“ja”）。",
        "background_sound": "描述片段中可听到的非语音背景声（如“微弱鸟鸣”、“持续的引擎轰鸣”）。若无，填写“无明显背景音”。",
        "gender": "说话人的性别。基于声音的音色、共鸣和音高判断：填写“男性”或“女性”。",
        "age": "说话人的年龄段。基于声音成熟度、呼吸控制、音高稳定性等判断：从“儿童”、“青年”、“年轻成人”、“中年”和“老年”中选择其一。",
        "pitch": "描述语音的音高水平及变化特征（如“高而轻快”、“低沉且带下降语调”、“音高波动明显的儿童声”）。",
        "speed": "描述语音的语速与流畅度（如“急促且音节短促”、“缓慢并带长停顿”）。如出现明显加速或减速需注明。",
        "emotion": "仅基于语音表现（语调、力度、颤抖等）描述说话人呈现的内部情绪状态，从['生气', '厌恶', '恐惧', '快乐', '悲伤', '惊讶', '中性']中选择其一。",
        "emotion_level": "基于语音表现（音量变化、语速、音高波动、气息紧张度等）判断情绪强度，从["极弱", "较弱", "中等", "较强", "极强"]中选择其一。",
        "accent": "请尽可能具体描述说话人的口音（如“通用美式英语”、“标准普通话、”“北京话”）。若口音不明确，可使用更宽泛类别；若为非母语口音，需注明可感知的来源影响（如“带西班牙语口音的非母语英语”等）。",
        "tone": "描述说话人的语调显示的态度与立场（如“带有轻微升调的安慰语气”、“讽刺且平淡”、“急切且短促的语气”）。",
        "rhythm": "描述语音的韵律模式（如“断奏且急促”、“单调且重音均匀”、“犹豫且频繁停顿”）。",
        "texture": "描述语音的音色与质感（如“气声且单薄”、“粗糙并带颗粒感”、“温暖且浑厚”）。不得加入情绪或态度解读。",
        "pronunciation": "描述语音的发音清晰度与咬字特征（如“咬字清晰有力”、“含混且辅音弱化”、“不清晰且发声松散”）。仅关注语音的可懂度。",
        "paralinguistic": "描述非词汇性质的副语言事件（如“说话前的啜泣声”、“句中急促吸气”、“尾音伴随紧张笑声”）。若无明显副语言事件，填写“无明显副语言事件”。",
        "contextual_inference": "仅基于语音表现推断说话者可能的角色或情境（如“努力保持冷静的担心父母”、“模仿权威口吻的儿童”、“一名医生试图掩饰紧迫感”）。",
        "caption": "基于核心声学属性（性别、年龄、音高、语速、情绪、口音、语气、节奏、音质、发音）生成 1–3 句自然语言摘要。仅当 paralinguistic 中存在明显的副语言事件时，才将其纳入摘要；否则摘要不得提及副语言事件缺失。需使用具体感知到的描述词，避免空泛的表述。示例1（无副语言事件）：“一位年轻成年女性用清晰、浑厚的声音说话，操着通用美式英语口音，语速为正常的对话速度，节奏流畅。她的语气带有讽刺和被逗乐的意味，通过中等音高传达，常带有疑问语调的上升，咬字标准清晰。” 示例2（有副语言事件）：“一位中年男性的声音，特点是温暖的音质和缓慢审慎的语速，带有美国南部口音，传达出深深的悲伤。这一点通过颤抖的节奏和句子中间急促的吸气得到强调，突出了他话语的情感分量。”"
    }},
    ...
    
    ]
    }}
    ---

    **约束与最佳实践**
    - 一个片段应代表单个说话者的一段连续话语。每当说话者发生变化或出现超过 1 秒的停顿时，必须拆分片段。
    - 开始时间和结束时间必须严格遵循 `mm:ss:xxx` 格式，由英文冒号分隔的三个部分组成，你他妈的必须保证你给的时间戳是精确的，给出每个时间戳时必须检查三遍，与 text_and_paralanguage 的内容严格对齐。此外，请确保时间戳标注时不要超出该电影片段的结束时间。
    - 所有描述必须是简洁的短语或句子（最好不少于 6 个汉字），而不是单个词语。例外：caption 可以是 1-3 个句子；pitch，speed，emotion，accent，tone等语音属性，如果为了更详细地说明，可以使用稍长的短语。
    - 进行语音属性标注时，请避免使用通用的术语，如“中等”、“正常”或“平均”。要使用生动、具体的描述词。
    - 除了"text_and_paralanguage"、"start_time"、"end_time"、"language"、"speaker_id" 和 "segment_id"字段对应的值以外，其他所有字段（background_sound，age等）对应的值用简体中文输出。但是特别注意，转录内容（text_and_paralanguage）的语言应该遵循说话人的口语语种。
    - 不要产生幻觉（Hallucinate）：进行属性标注时，不确定的时候给出最广泛合理的类别（如“较宽泛的美式英语口音”）。不得创造不存在的声音、情绪、人物或事件。
    - 识别说话人时还应考虑不同说话人之间的声音特征的差异。
    - 语言（language）字段必须反映片段内实际的口语语言（如：zh, en, ja）。不能根据电影主要语言推断。
    - 确保一致性：caption（摘要）应直接反映标注的语音属性，不得与上面单独标注的语音属性相互矛盾。
    """

async def analyze_audio_with_text(session, url, key, audio_path, prompt_text, max_retries=3):
    """
    异步读取音频文件并调用 API，包含重试机制
    """
    api_url_base = f"{url}/v1beta/models/gemini-2.5-pro:generateContent"
    full_api_url = f"{api_url_base}?key={key}"

    # 1. 异步读取文件
    try:
        async with aiofiles.open(audio_path, "rb") as f:
            audio_bytes = await f.read()
    except FileNotFoundError:
        return None, f"文件未找到: {audio_path}"
    except Exception as e:
        return None, f"读取文件出错: {e}"

    # 2. 准备请求数据
    base64_encoded_audio = base64.b64encode(audio_bytes).decode('utf-8')
    mime_type, _ = mimetypes.guess_type(audio_path)
    if mime_type is None:
        mime_type = "audio/mpeg"

    request_data = {
        "contents": [{
            "parts": [
                {"text": prompt_text},
                {"inline_data": {"mime_type": mime_type, "data": base64_encoded_audio}}
            ]
        }]
    }
    headers = {"Content-Type": "application/json"}

    # 3. 发送请求（带重试循环）
    for attempt in range(max_retries + 1):
        try:
            # 设置较长的 ClientTimeout，防止客户端先断开
            # total=None 表示不限制整体时间，但建议 connect 设置短一点
            # timeout = aiohttp.ClientTimeout(total=500, connect=10, sock_read=300)
            
            async with session.post(full_api_url, headers=headers, json=request_data, timeout=1000) as resp:
                
                # 成功情况
                if resp.status == 200:
                    data = await resp.json()
                    try:
                        inner_text = data["candidates"][0]["content"]["parts"][0]['text'].strip()
                        if inner_text.startswith("```json"):
                             inner_text = inner_text[7:]
                        if inner_text.startswith("```"):
                             inner_text = inner_text[3:]
                        if inner_text.endswith("```"):
                             inner_text = inner_text[:-3]
                        response_json = json.loads(inner_text)
                    except Exception:
                        response_json = data
                    return response_json, None

                # 需要重试的状态码
                # 429: Too Many Requests (速率限制)
                # 500, 502, 503, 504: 服务器端错误
                # 524: Cloudflare Timeout
                if resp.status in [429, 500, 502, 503, 504, 524]:
                    error_text = await resp.text()
                    if attempt < max_retries:
                        # 指数退避策略：等待 2秒, 4秒, 8秒... 加上随机抖动
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        print(f"⚠️ {os.path.basename(audio_path)} 遇到 {resp.status}，正在进行第 {attempt+1} 次重试，等待 {wait_time:.2f}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        return None, f"HTTP错误(重试失败): {resp.status}, 内容: {error_text[:200]}"
                
                # 其他不需要重试的错误（如 400 参数错误，403 权限错误）
                else:
                    text = await resp.text()
                    return None, f"HTTP致命错误: {resp.status}, 内容: {text[:200]}"

        except asyncio.TimeoutError:
            if attempt < max_retries:
                print(f"⚠️ {os.path.basename(audio_path)} 请求超时，重试中...")
                await asyncio.sleep(2)
                continue
            return None, "请求最终超时"
        except aiohttp.ClientError as e:
            if attempt < max_retries:
                await asyncio.sleep(2)
                continue
            return None, f"网络错误: {e}"
        except Exception as e:
            return None, f"未知错误: {e}"

    return None, "重试次数耗尽"


async def process_all_files(audio_scp_path, out_path, error_log_path, url, key, segment_dir=None, max_concurrent=5, success_scp=None):
    """
    异步批量处理
    """
    # 限制并发量 Semaphore
    sem = asyncio.Semaphore(max_concurrent)

    # 优化 connector 设置
    connector = aiohttp.TCPConnector(limit=max_concurrent, force_close=True)

    async with aiohttp.ClientSession(connector=connector) as session, \
               aiofiles.open(audio_scp_path, 'r') as fin, \
               aiofiles.open(out_path, 'a') as fout, \
               aiofiles.open(error_log_path, 'a') as ferr:

        # 可选：成功记录文件
        fsuccess = None
        if success_scp is not None:
            try:
                fsuccess = await aiofiles.open(success_scp, 'a')
            except Exception as e:
                print(f"⚠️ 无法打开 success_scp {success_scp}: {e}")
                fsuccess = None

        # 预读取所有行，方便使用 tqdm 显示总进度
        lines = await fin.readlines()
        
        # 定义单个任务的逻辑
        async def run_task(line):
            async with sem:  # 核心并发控制
                wav_path = line.strip()
                if not wav_path: return
                basename = os.path.basename(wav_path)
                utt = os.path.basename(wav_path).split('.')[0]
                movie_name = wav_path.split('/')[-3]
                
                segment_data = None
                if segment_dir is not None:
                    segment_file = os.path.join(segment_dir, movie_name, f"{basename}.txt")
                    if os.path.exists(segment_file):
                        try:
                            # 这里的 open 是同步的，读取小文本文件通常没问题，也可以改用 aiofiles
                            with open(segment_file, 'r', encoding='utf-8') as sf:
                                segment_data = sf.read().strip()
                        except Exception:
                            print(f"⚠️ 读取时间戳文件失败: {segment_file}")
                    else:
                        print(f"⚠️ 时间戳文件不存在: {segment_file}")
                        pass

                # 调用处理函数
                result, error = await analyze_audio_with_text(
                    session, url, key, wav_path, create_analysis_prompt(segment_data)
                )

                if result:
                    res = {
                        'utt': utt,
                        'wav_path': wav_path,
                        'gemini_res': result
                    }
                    await fout.write(json.dumps(res, ensure_ascii=False) + '\n')
                    await fout.flush()
                    # 记录成功的 wav 路径到 success_scp（若提供）
                    if fsuccess is not None:
                        try:
                            await fsuccess.write(f"{wav_path}\n")
                            await fsuccess.flush()
                        except Exception as e:
                            print(f"⚠️ 写入 success_scp 失败: {e}")
                else:
                    await ferr.write(f'{utt}|{wav_path}|{error}\n')
                    await ferr.flush()

        # 创建所有任务
        tasks = [run_task(line) for line in lines]
        
        # 使用 tqdm 监控任务完成进度
        # return_exceptions=True 防止单个任务报错中断整个循环
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
            await f

        # 关闭 success 文件（如果打开）
        if fsuccess is not None:
            try:
                await fsuccess.close()
            except Exception:
                pass

if __name__ == "__main__":
    url = "https://apim1tocn.cheapapi.ai"
    key = "YOUR_GEMINI_API_KEY"  # 记得填回你的 Key

    # audio_scp_path = "$DATA_ROOT/test/qvwan/test.scp"
    # out_path = "$DATA_ROOT/test/huoshan/qvwan_out/out.jsonl"
    # error_log_path = "$DATA_ROOT/test/huoshan/qvwan_out/error_fast.log"
    # segment_dir = "$DATA_ROOT/test/huoshan/qvwan_out/times"
    parser = argparse.ArgumentParser(description="批量处理音频文件进行副语言分析")
    parser.add_argument("--scp_path", "-s", type=str, required=True, help="包含音频文件路径的文本文件，每行一个路径")
    parser.add_argument("--out_path", "-o", type=str, required=True, help="输出结果的 JSONL 文件路径")
    parser.add_argument("--error_path", "-e", type=str, required=True, help="错误日志文件路径")
    parser.add_argument("--time_dir", "-t", type=str, default=None, help="必须的时间戳文件目录路径")
    parser.add_argument("--success_scp", type=str, default=None, help="可选：记录处理成功文件路径的 scp 文件路径（每行一个 wav 路径）")

    args = parser.parse_args()
    audio_scp_path = args.scp_path
    out_path = args.out_path
    error_log_path = args.error_path
    segment_dir = args.time_dir
    success_scp = args.success_scp

    # !!! 关键修改：将并发数从 100 降低到 5 或 8 !!!
    # 音频处理很慢，并发太高必然 524
    max_concurrent_nums = 64
    
    print(f"开始处理，并发数限制为: {max_concurrent_nums}")
    asyncio.run(process_all_files(audio_scp_path, out_path, error_log_path, url, key, segment_dir, max_concurrent=max_concurrent_nums, success_scp=success_scp))