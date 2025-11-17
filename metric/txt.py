#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import argparse
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 平滑函数，用于避免BLEU为0的情况
smooth = SmoothingFunction().method1


def compute_bleu_scores(reference, prediction):
    """计算 BLEU-1 ~ BLEU-4"""
    ref_tokens = reference.strip().split()
    pred_tokens = prediction.strip().split()

    bleu1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu2 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu3 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
    bleu4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    return bleu1, bleu2, bleu3, bleu4


def process_file(file_path):
    """读取单个文件并计算平均BLEU"""
    total_bleu = [0, 0, 0, 0]
    count = 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 允许文件是列表或字典格式
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.values()
        else:
            return [0, 0, 0, 0]

        for item in items:
            pred = item.get("predicted_answer", "").strip()
            gt = item.get("ground_truth", "").strip()
            if not pred or not gt:
                continue

            bleu_scores = compute_bleu_scores(gt, pred)
            total_bleu = [x + y for x, y in zip(total_bleu, bleu_scores)]
            count += 1
    except Exception as e:
        print(f"[WARN] Failed to process {file_path}: {e}")

    if count == 0:
        return [0, 0, 0, 0]
    return [x / count for x in total_bleu]


def main():
    parser = argparse.ArgumentParser(description='计算文本 QA 的 BLEU 分数')
    parser.add_argument('--file', type=str, required=True, help='要评估的 JSON 文件路径')
    args = parser.parse_args()

    file_path = args.file
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在: {file_path}")
        return

    fname = os.path.basename(file_path)
    bleu_scores = process_file(file_path)

    print("\n========= 评估结果 =========")
    print(f"[{fname}] BLEU-1: {bleu_scores[0]:.4f}, BLEU-2: {bleu_scores[1]:.4f}, BLEU-3: {bleu_scores[2]:.4f}, BLEU-4: {bleu_scores[3]:.4f}")

    avg_bleu = sum(bleu_scores) / 4
    print(f"\n========= 平均 BLEU =========")
    print(f"[{fname}] Avg BLEU: {avg_bleu:.4f}")


if __name__ == "__main__":
    main()
