#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import json
import argparse
import numpy as np
from tqdm import tqdm

# -------------------------------
# IoU计算函数
# -------------------------------
def compute_iou(boxA, boxB):
    """计算两个bbox的IoU"""
    if not (isinstance(boxA, (list, tuple)) and isinstance(boxB, (list, tuple))):
        return 0
    if len(boxA) != 4 or len(boxB) != 4:
        return 0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = areaA + areaB - interArea
    if unionArea == 0:
        return 0
    return interArea / unionArea

# -------------------------------
# 提取bbox函数
# -------------------------------
def extract_bboxes(text):
    """从字符串中提取bbox列表"""
    pattern = r"<bbox>\s*(.*?)\s*</bbox>"
    match = re.search(pattern, text)
    if not match:
        return []
    try:
        bboxes = json.loads(match.group(1))
        return bboxes
    except Exception:
        return []

# -------------------------------
# 精度计算函数
# -------------------------------
def evaluate_bbox_file(file_path, iou_threshold=0.3):
    """计算单个JSON文件的bbox预测精度"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 处理当文件为 dict (id->sample) 情况，转成 list
    if isinstance(data, dict):
        items = list(data.values())
    elif isinstance(data, list):
        items = data
    else:
        return 0.0, 0, 0

    total = 0
    correct = 0

    for item in items:
        pred_bboxes = extract_bboxes(item.get("predicted_answer", ""))
        gt_bboxes = extract_bboxes(item.get("ground_truth", ""))

        if not gt_bboxes or not pred_bboxes:
            continue

        total += len(gt_bboxes)
        for gt in gt_bboxes:
            # 判断是否有匹配的预测框
            matched = any(compute_iou(gt, pred) >= iou_threshold for pred in pred_bboxes)
            if matched:
                correct += 1

    precision = correct / total if total > 0 else 0.0
    return precision, total, correct

# -------------------------------
# 主程序
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='计算 BBox QA 的精度')
    parser.add_argument('--file', type=str, required=True, help='要评估的 JSON 文件路径')
    args = parser.parse_args()

    file_path = args.file
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在: {file_path}")
        exit(1)

    fname = os.path.basename(file_path)
    precision, total, correct = evaluate_bbox_file(file_path)

    print(f"\n文件: {fname}")
    print(f"精度: {precision:.4f} ({correct}/{total})")
