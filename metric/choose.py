#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import json
import re
import argparse
from tqdm import tqdm

# 要筛选的文件前缀关键词
TARGET_KEYWORDS = ["color", "distance", "height", "reverse", "spatial", "count"]


def normalize_answer(ans: str) -> str:
    """
    清洗与标准化模型输出答案文本，支持包含 <think> 思维链的回答。
    """
    if not isinstance(ans, str):
        return ""
    ans = ans.strip()

    # === ① 去除思维链部分 ===
    think_pattern = re.compile(r"</think>\s*(.*)", re.DOTALL)
    m = think_pattern.search(ans)
    if m:
        ans = m.group(1).strip()

    if "<think>" in ans:
        ans = re.sub(r"<think>.*", "", ans, flags=re.DOTALL).strip()

    # === ② 移除 LaTeX 与包裹符 ===
    ans = ans.replace("\\boxed", "")
    ans = ans.replace("<//boxed>", "")
    ans = ans.replace("<answer>", "")
    ans = ans.replace("</answer>", "")
    ans = ans.replace("{", "").replace("}", "")

    # === ③ 去除换行、空格、标点等杂质 ===
    ans = ans.replace("\n", " ").replace("\t", " ").strip()
    ans = re.sub(r"\s+", " ", ans)
    ans = re.sub(r'[^\w\d\s-]', '', ans.lower()).strip()

    return ans


def extract_choice_letter(ans: str) -> str:
    """
    尝试从答案中提取首个选项字母（A/B/C/D/E...）
    例如：
        "a gray" → A
        "A." → A
        "option B" → B
    """
    m = re.match(r"([a-e])(\b|\.|\s|$)", ans.lower())
    if m:
        return m.group(1).upper()
    return ""


def compute_accuracy(file_path: str):
    """
    计算单个 JSON 文件的预测精度
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total, correct = 0, 0

    if isinstance(data, dict):
        if "predicted_answer" in data and "ground_truth" in data:
            qa_list = [data]
        else:
            qa_list = list(data.values())
    elif isinstance(data, list):
        qa_list = data
    else:
        return 0, 0

    for qa in qa_list:
        if not isinstance(qa, dict):
            continue
        if "predicted_answer" not in qa or "ground_truth" not in qa:
            continue

        pred = normalize_answer(qa["predicted_answer"])
        gt = normalize_answer(qa["ground_truth"])

        # --- 新增逻辑：选项字母匹配 ---
        if pred == gt:
            correct += 1
        else:
            pred_choice = extract_choice_letter(pred)
            gt_choice = extract_choice_letter(gt)
            if pred_choice and gt_choice and pred_choice == gt_choice:
                correct += 1

        total += 1

    return correct, total


def main():
    parser = argparse.ArgumentParser(description='计算选择题 QA 的准确率')
    parser.add_argument('--file', type=str, required=True, help='要评估的 JSON 文件路径')
    args = parser.parse_args()

    file_path = args.file
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在: {file_path}")
        return

    fname = os.path.basename(file_path)
    correct, total = compute_accuracy(file_path)
    acc = correct / total if total > 0 else 0.0

    print(f"\n=== 文件精度统计 ===")
    print(f"{fname:<50}  准确率: {acc*100:.2f}%  ({correct}/{total})")

    if total > 0:
        print(f"\n总体精度: {acc*100:.2f}%  ({correct}/{total})")
    else:
        print("\n未找到符合条件的样本。")


if __name__ == "__main__":
    main()
