from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
import json
import os
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import argparse
from pathlib import Path

# =====================================================
# 基础配置
# =====================================================
benchmark_paths = [
    "benchmark/pointing.json",
    "benchmark/freespace.json",
    "benchmark/color.json",
    "benchmark/counting.json",
    "benchmark/distance.json",
    "benchmark/height.json",
    "benchmark/bbox.json",
    "benchmark/function.json",
    "benchmark/landing.json",
    "benchmark/reverse.json",
    "benchmark/single_image_caption.json",
    "benchmark/spatial.json",
    "benchmark/multi_image_caption.json",
]

max_pixels = 1605632
min_pixels = 256 * 28 * 28

gen_args = {
    "temperature": 0,
    "top_p": 1,
    "max_new_tokens": 1024,
    "repetition_penalty": 1.05,
    "do_sample": False,
}

use_flash_attention = False
save_every = 10

NUM_GPUS = 8  # 总GPU数量


# =====================================================
# 辅助函数：从ckpt路径提取后两个目录作为输出路径
# =====================================================
def get_output_dir_from_ckpt(ckpt_path):
    """
    从checkpoint路径提取后两个目录名
    例如: /path/to/cvpr-rl_cvpr_only_freespace_avg_20251108-1948/global_step_20
    返回: cvpr-rl_cvpr_only_freespace_avg_20251108-1948/global_step_20
    """
    path = Path(ckpt_path)
    # 获取最后两个目录
    if len(path.parts) >= 2:
        output_dir = os.path.join(path.parts[-2], path.parts[-1])
    else:
        # 如果路径不够长，使用basename
        output_dir = path.name
    return output_dir


def get_model_name_from_ckpt(ckpt_path):
    """
    从checkpoint路径提取模型名称
    例如: global_step_20 -> step20
    """
    path = Path(ckpt_path)
    step_name = path.name  # 例如: global_step_20
    if "step" in step_name.lower():
        # 提取数字部分
        import re
        match = re.search(r'step[_-]?(\d+)', step_name, re.IGNORECASE)
        if match:
            return f"step{match.group(1)}"
    return step_name


# =====================================================
# 单个GPU处理函数
# =====================================================
def process_benchmark_on_gpu(gpu_id, benchmark_path, model_path, output_dir, model_name):
    """在指定GPU上处理单个benchmark"""
    
    # 设置当前进程使用的GPU（不要用CUDA_VISIBLE_DEVICES）
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    
    filename = os.path.basename(benchmark_path)
    prefix = filename.split("_")[0]
    output_file = os.path.join(output_dir, f"{prefix}_{model_name}.json")
    
    print(f"\n[GPU {gpu_id}] Processing Task: {prefix}")
    print(f"[GPU {gpu_id}] Benchmark path: {benchmark_path}")
    
    # 加载模型
    print(f"[GPU {gpu_id}] Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="flash_attention_2" if use_flash_attention else None,
    )
    processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels, min_pixels=min_pixels)
    print(f"[GPU {gpu_id}] Model loaded successfully on {device}.")
    
    # 加载 benchmark 数据
    try:
        with open(benchmark_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"[GPU {gpu_id}] Failed to load {benchmark_path}: {e}")
        return
    
    if not isinstance(dataset, list):
        print(f"[GPU {gpu_id}] Invalid data format in {benchmark_path}, expected a list.")
        return
    
    results = []
    
    # 进度条
    progress_bar = tqdm(
        enumerate(dataset),
        total=len(dataset),
        desc=f"[GPU {gpu_id}][{prefix}]",
        position=gpu_id,
        dynamic_ncols=True
    )
    
    # 推理循环
    for idx, sample in progress_bar:
        images_paths = sample.get("images", None)
        if images_paths is None:
            images_paths = [sample.get("image", "")]
        
        question = sample.get("question", "")
        ground_truth = sample.get("ground_truth", "")
        
        loaded_images = []
        skip_sample = False
        for img_path in images_paths:
            if not os.path.exists(img_path):
                progress_bar.write(f"[GPU {gpu_id}][Warning] Image not found: {img_path}")
                skip_sample = True
                break
            try:
                img = Image.open(img_path).convert("RGB")
                loaded_images.append(img)
            except Exception as e:
                progress_bar.write(f"[GPU {gpu_id}][Warning] Failed to open image {img_path}: {e}")
                skip_sample = True
                break
        if skip_sample:
            continue
        
        # 构造输入消息
        image_messages = [{"type": "image", "image": img} for img in loaded_images]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": image_messages + [{"type": "text", "text": question}]},
        ]
        
        # 应用模板
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=loaded_images, return_tensors="pt", padding=True).to(device)
        
        # 模型推理
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **gen_args)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        results.append({
            "image": images_paths,
            "question": question,
            "predicted_answer": output_text,
            "ground_truth": ground_truth,
        })
        
        # 每隔 save_every 保存一次中间结果
        if (idx + 1) % save_every == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
        
        progress_bar.set_postfix({"saved": len(results)})
    
    # 保存最终结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    progress_bar.close()
    print(f"[GPU {gpu_id}] ✅ Task {prefix} complete. Results saved to: {output_file}")
    
    # 清理GPU内存
    del model
    del processor
    torch.cuda.empty_cache()


# =====================================================
# 主程序：分批处理
# =====================================================
if __name__ == "__main__":
    # 设置multiprocessing启动方法为spawn（CUDA要求）
    mp.set_start_method('spawn', force=True)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Multi-GPU Parallel Inference for Vision-Language Model")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the checkpoint directory (e.g., /path/to/cvpr-rl_xxx/global_step_20)"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use (default: 8)"
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default=None,
        help="Base directory for output (default: auto-generated from ckpt_path)"
    )
    
    args = parser.parse_args()
    
    # 设置路径
    model_path = args.ckpt_path
    
    # 自动生成输出目录
    if args.output_base_dir:
        output_dir = args.output_base_dir
    else:
        output_dir = get_output_dir_from_ckpt(model_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 自动生成模型名称
    model_name = get_model_name_from_ckpt(model_path)
    
    # 更新GPU数量
    num_gpus = args.num_gpus
    
    # 打印配置信息
    print("=" * 80)
    print("Configuration:")
    print(f"  Checkpoint Path: {model_path}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Model Name: {model_name}")
    print(f"  Number of GPUs: {num_gpus}")
    print(f"  Total Benchmarks: {len(benchmark_paths)}")
    print("=" * 80)
    
    # 第一批：前num_gpus个benchmark分配到num_gpus个GPU
    batch1 = benchmark_paths[:num_gpus]
    # 第二批：剩余benchmark分配到对应数量的GPU
    batch2 = benchmark_paths[num_gpus:]
    
    print(f"\nBatch 1: {len(batch1)} benchmarks on {num_gpus} GPUs")
    print(f"Batch 2: {len(batch2)} benchmarks on {min(len(batch2), num_gpus)} GPUs")
    print("=" * 80)
    
    # 处理第一批
    print("\n🚀 Starting Batch 1...")
    processes = []
    for gpu_id, benchmark_path in enumerate(batch1):
        p = mp.Process(
            target=process_benchmark_on_gpu,
            args=(gpu_id, benchmark_path, model_path, output_dir, model_name)
        )
        p.start()
        processes.append(p)
    
    # 等待第一批完成
    for p in processes:
        p.join()
    
    print("\n✅ Batch 1 completed!")
    
    # 处理第二批
    if batch2:
        print("\n🚀 Starting Batch 2...")
        processes = []
        for gpu_id, benchmark_path in enumerate(batch2):
            p = mp.Process(
                target=process_benchmark_on_gpu,
                args=(gpu_id, benchmark_path, model_path, output_dir, model_name)
            )
            p.start()
            processes.append(p)
        
        # 等待第二批完成
        for p in processes:
            p.join()
        
        print("\n✅ Batch 2 completed!")
    
    print("\n" + "=" * 80)
    print("🎉 All tasks completed successfully!")
    print(f"📁 Results saved in: {output_dir}")
    print("=" * 80)