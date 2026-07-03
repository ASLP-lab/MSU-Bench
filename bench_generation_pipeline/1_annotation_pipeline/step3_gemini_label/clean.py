# -*- coding: utf-8 -*-
#
# 清洗 Gemini 原始输出, 移除格式错误/不完整条目.
# 输入: all.jsonl
# 输出: cleaned_out.jsonl
#

import json
import os
import re
from tqdm import tqdm
import argparse

# 尝试导入 json_repair，如果没有安装则使用备用方案
try:
    import json_repair
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False
    print("⚠️ 警告: 未检测到 json_repair 库。强烈建议运行 'pip install json_repair' 以获得最佳修复效果。")
    print("正在使用简易后备模式...")

def basic_repair(json_str):
    """
    简易后的备用修复逻辑（仅当没有安装 json_repair 时使用）
    主要尝试闭合未闭合的括号
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # 简单的尝试闭合逻辑
        stack = []
        for char in json_str:
            if char == '{': stack.append('}')
            elif char == '[': stack.append(']')
            elif char == '}' or char == ']':
                if stack: stack.pop()
        
        # 补全剩余的括号
        json_str += "".join(reversed(stack))
        try:
            return json.loads(json_str)
        except:
            return None


def fix_unquoted_keys(text: str) -> str:
    """给未加引号的键名补引号，覆盖常见 Gemini 泄漏键（如 sem）。"""

    def repl(match: re.Match) -> str:
        indent, key, colon = match.groups()
        return f'{indent}"{key}"{colon}'

    # 仅匹配行首的未加引号键名，避免破坏已合法的 JSON。
    return re.sub(r"(?m)^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", repl, text)


def strip_noise_before_brace(text: str) -> str:
    """去掉行首单个字母+花括号的噪声行（如 "e    {"）。"""

    def repl(match: re.Match) -> str:
        indent = match.group(1)
        return f"{indent}{{"

    # 只处理形如 "e    {" 或 "x   {" 的行，保留缩进，防止破坏正常 JSON。
    return re.sub(r"(?m)^(\s*)[A-Za-z]\s*\{", repl, text)

def _normalize_segments(segments):
    """修正常见字段问题，比如 segmentid 拼写、起止时间转字符串。"""
    normalized = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        seg = dict(seg)  # 拷贝，避免原地修改

        # 修复常见拼写：segmentid -> segment_id
        if "segmentid" in seg and "segment_id" not in seg:
            seg["segment_id"] = seg.pop("segmentid")

        # 起止时间统一转字符串，保持下游一致性
        for k in ("start_time", "end_time"):
            if k in seg:
                seg[k] = str(seg[k])

        normalized.append(seg)
    return normalized


def extract_and_repair_segments(data):
    raw_text = None

    # 1. 已经是正确结构
    if "audio_segments" in data and isinstance(data["audio_segments"], list):
        return _normalize_segments(data["audio_segments"])

    # 2. Gemini 原始文本
    candidates = data.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        if parts:
            raw_text = parts[0].get("text", "")

    if not raw_text:
        return None

    # 3. 清 Markdown
    clean_text = raw_text.strip()
    for prefix in ("```json", "```"):
        if clean_text.startswith(prefix):
            clean_text = clean_text[len(prefix):]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    clean_text = clean_text.strip()

    # 4. 预清理：去噪并修补未加引号的键
    clean_text = strip_noise_before_brace(clean_text)
    clean_text = fix_unquoted_keys(clean_text)

    # 5. 修复 JSON
    try:
        if HAS_JSON_REPAIR:
            parsed = json_repair.repair_json(clean_text, return_objects=True)
        else:
            parsed = basic_repair(clean_text)
    except Exception:
        return None

    # 6. ⭐ 关键：强制保证返回 List[Dict]
    if isinstance(parsed, dict):
        segments = parsed.get("audio_segments")

        # Gemini 常见：audio_segments 是字符串
        if isinstance(segments, str):
            try:
                segments = json.loads(segments)
            except Exception:
                return None

        if isinstance(segments, list):
            # 再防御一层
            if all(isinstance(s, dict) for s in segments):
                return _normalize_segments(segments)

    return None

def process_jsonl(input_path, output_path, error_path):
    success_count = 0
    fail_count = 0
    
    print(f"正在处理: {input_path}")
    
    # 使用宽容的解码以避免因文件中存在非法字节而抛出异常
    with open(input_path, 'r', encoding='utf-8', errors='replace') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout, \
         open(error_path, 'w', encoding='utf-8') as ferr:
        
        # 用二进制方式计数行数，避免在计数阶段触发解码错误
        with open(input_path, 'rb') as bf:
            total_lines = sum(1 for _ in bf)
        fin.seek(0)
        
        for line_num, line in tqdm(enumerate(fin, 1), total=total_lines, desc="Cleaning"):
            line = line.strip()
            if not line:
                continue
            
            try:
                original_data = json.loads(line)
                
                base_info = {
                    "utt": original_data.get("utt"),
                    "wav_path": original_data.get("wav_path")
                }
                
                segments = extract_and_repair_segments(original_data.get('gemini_res', {}))  # 关键修改处
                
                if segments is not None:
                    new_entry = {
                        **base_info,
                        "audio_segments": segments
                    }
                    fout.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
                    success_count += 1
                else:
                    # 记录提取失败的行，方便后续检查
                    fail_count += 1
                    ferr.write(f"Line {line_num} | {base_info['utt']} | 解析失败或无数据\n")
                    
            except json.JSONDecodeError:
                fail_count += 1
                ferr.write(f"Line {line_num} | 未知 | 原始行JSON错误\n")
                continue

    print("-" * 30)
    print(f"处理完成！")
    print(f"成功清洗: {success_count} 行")
    print(f"失败/丢弃: {fail_count} 行 (详情见 error_clean.log)")
    print(f"结果已保存至: {output_path}")

if __name__ == "__main__":
    # --- 配置路径 ---
    parser = argparse.ArgumentParser(description="Clean Gemini output JSONL files")
    parser.add_argument('--input_file', "-i", type=str, required=True, help='Path to input JSONL file')
    parser.add_argument('--output_file', "-o", type=str, help='Path to output cleaned JSONL file')
    parser.add_argument('--error_file', "-e", type=str, default=None, help='Path to error log file (optional)')

    args = parser.parse_args()
    input_file = args.input_file

    output_file = args.output_file
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_clean{ext}"
    
    error_file = args.error_file
    if error_file is None:
        base, ext = os.path.splitext(input_file)
        error_file = os.path.join(os.path.dirname(input_file), "error_clean.log")

    
    process_jsonl(input_file, output_file, error_file)