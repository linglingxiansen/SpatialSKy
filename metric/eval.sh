#!/bin/bash

set -e

# 如果提供了命令行参数，使用该参数作为目录，否则使用默认目录
if [ $# -ge 1 ]; then
    BASE_DIR="$1"
else
    BASE_DIR="/e2e-data/evad-tech-vla/zhangyuchen/sky/global_step_100"
fi

# 检查目录是否存在
if [ ! -d "$BASE_DIR" ]; then
    echo "错误: 目录不存在: $BASE_DIR"
    exit 1
fi

CHOICE_KEYWORDS=("color" "distance" "height" "reverse" "spatial" "count" )

echo "Base directory: $BASE_DIR"
echo "Scanning JSON files..."
echo ""

for f in "$BASE_DIR"/*.json; do
    fname=$(basename "$f")

    # ---------- choose QA ----------
    for kw in "${CHOICE_KEYWORDS[@]}"; do
        if [[ "$fname" == ${kw}* ]]; then
            echo "[Choice] $fname"
            python3 choose.py --file "$f"
            continue 2
        fi
    done

    # ---------- text QA ----------
    if [[ "$fname" == single* || "$fname" == multi* ]]; then
        echo "[Text BLEU] $fname"
        python3 txt.py --file "$f"
        continue
    fi

    # ---------- bbox QA ----------
    if [[ "$fname" == bbox* ]]; then
        echo "[BBox] $fname"
        python3 box.py --file "$f"
        continue
    fi

    # ---------- point QA ----------
    if [[ "$fname" == pointing* ]]; then
        echo "[BBox] $fname"
        python3 point.py --file "$f"
        continue
    fi

    # ---------- point QA ----------
    if [[ "$fname" == freespace* ]]; then
        echo "[BBox] $fname"
        python3 freespace.py --file "$f"
        continue
    fi

    # ---------- landing QA ----------
    if [[ "$fname" == landing* ]]; then
        echo "[BBox] $fname"
        python3 landing.py --file "$f"
        continue
    fi

    # ---------- function QA ----------
    if [[ "$fname" == function* ]]; then
        echo "[BBox] $fname"
        python3 function.py --file "$f"
        continue
    fi
done

echo ""
echo "All evaluation completed!"
