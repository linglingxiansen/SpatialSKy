import os
import json
from tqdm import tqdm
from openai import OpenAI

# =========================================================
# 基础配置
# =========================================================
API_KEY = "your api key"
API_BASE_URL = "your api base url"

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
    default_headers={"X-Model-Provider-Id": "azure_openai"}
)

# =========================================================
# 模型打分函数
# =========================================================
def score_prediction(question, predicted_answer, ground_truth):
    """
    使用模型对预测答案进行1-10分的评分。
    """

    prompt = f"""
You are an expert UAV (drone) safety evaluator. 
You are given:
1️⃣ The **Question** (what the model was asked)
2️⃣ The **Predicted Answer** (the model’s output)
3️⃣ The **Ground Truth** (the correct reference answer)

Please carefully compare the predicted answer with the ground truth.

Rate the predicted answer **strictly from 1 to 10** based on:
- Accuracy of safety assessment
- Correctness and completeness of key elements (hazards, landing feasibility, etc.)
- Consistency with ground truth
- Usefulness and factual precision

Output **only JSON**:
{{
  "score": <integer from 1 to 10>,
  "reason": "short explanation (1-2 sentences)"
}}

---
Question:
{question}

Predicted Answer:
{predicted_answer}

Ground Truth:
{json.dumps(ground_truth, indent=2, ensure_ascii=False)}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a strict evaluator of UAV safety reports."},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        return result.get("score", None)
    except Exception as e:
        print(f"[Error Scoring] {e}")
        return None


# =========================================================
# 主程序
# =========================================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='计算 Landing QA 的模型评分')
    parser.add_argument('--file', type=str, required=True, help='要评估的 JSON 文件路径')
    args = parser.parse_args()

    file_path = args.file
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在: {file_path}")
        return

    filename = os.path.basename(file_path)
    print(f"\n📂 处理文件: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[Error Loading JSON] {file_path}: {e}")
        return

    # 处理当文件为 dict (id->sample) 情况，转成 list
    if isinstance(data, dict):
        items = list(data.values())
    elif isinstance(data, list):
        items = data
    else:
        print(f"错误: 不支持的文件格式，必须是 list 或 dict")
        return

    scores = []
    for item in tqdm(items, desc=f"Scoring {filename}", leave=False):
        q = item.get("question", "")
        p = item.get("predicted_answer", "")
        g = item.get("ground_truth", "")

        score = score_prediction(q, p, g)
        if score is not None:
            scores.append(score)

    if scores:
        avg = sum(scores) / len(scores)
        print(f"✅ {filename} 平均得分: {avg:.2f}/10 ({len(scores)}/{len(items)} 个有效评分)")
    else:
        print(f"⚠️ {filename} 无有效评分结果。")


if __name__ == "__main__":
    main()
