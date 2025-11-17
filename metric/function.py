import json
import os
import numpy as np
from PIL import Image
from pathlib import Path
from scipy import ndimage
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# UAVScenes的颜色映射
cmap = {
    0: {'name': 'background', 'RGB': [0, 0, 0]},
    1: {'name': 'roof', 'RGB': [119, 11, 32]},
    2: {'name': 'dirt_motor_road', 'RGB': [180, 165, 180]},
    3: {'name': 'paved_motor_road', 'RGB': [128, 64, 128]},
    4: {'name': 'river', 'RGB': [173, 216, 230]},
    5: {'name': 'pool', 'RGB': [0, 80, 100]},
    6: {'name': 'bridge', 'RGB': [150, 100, 100]},
    7: {'name': 'unknown_7', 'RGB': [150, 120, 90]},
    8: {'name': 'unknown_8', 'RGB': [70, 70, 70]},
    9: {'name': 'container', 'RGB': [250, 170, 30]},
    10: {'name': 'airstrip', 'RGB': [81, 0, 81]},
    11: {'name': 'traffic_barrier', 'RGB': [102, 102, 156]},
    12: {'name': 'unknown_12', 'RGB': [190, 153, 153]},
    13: {'name': 'green_field', 'RGB': [107, 142, 35]},
    14: {'name': 'wild_field', 'RGB': [210, 180, 140]},
    15: {'name': 'solar_board', 'RGB': [220, 220, 0]},
    16: {'name': 'umbrella', 'RGB': [153, 153, 153]},
    17: {'name': 'transparent_roof', 'RGB': [0, 0, 90]},
    18: {'name': 'car_park', 'RGB': [250, 170, 160]},
    19: {'name': 'paved_walk', 'RGB': [244, 35, 232]},
    20: {'name': 'sedan', 'RGB': [0, 0, 142]},
    21: {'name': 'unknown_21', 'RGB': [224, 224, 192]},
    22: {'name': 'unknown_22', 'RGB': [220, 20, 60]},
    23: {'name': 'unknown_23', 'RGB': [192, 64, 128]},
    24: {'name': 'truck', 'RGB': [0, 0, 70]},
    25: {'name': 'unknown_25', 'RGB': [0, 60, 100]},
}


class PointingQAEvaluator:
    def __init__(self, label_root_dir):
        self.label_root_dir = Path(label_root_dir)
        self.rgb2id = {tuple(v['RGB']): k for k, v in cmap.items()}
        self.id2name = {k: v['name'] for k, v in cmap.items()}
        self.encoded_to_id = {
            (r << 16) + (g << 8) + b: cid for (r, g, b), cid in self.rgb2id.items()
        }
        print(f"Label根目录: {self.label_root_dir}")
        print(f"加载了 {len(cmap)} 个类别")

    def _get_label_path_from_image(self, image_path):
        image_path = Path(image_path)
        subdir_name = image_path.parts[-3]
        label_filename = image_path.stem + '.png'
        return self.label_root_dir / subdir_name / "interval5_CAM_label_color" / label_filename

    def _rgb_to_label_id(self, label_rgb):
        h, w = label_rgb.shape[:2]
        rgb_encoded = (label_rgb[:, :, 0].astype(np.int32) << 16) + \
                      (label_rgb[:, :, 1].astype(np.int32) << 8) + \
                      label_rgb[:, :, 2].astype(np.int32)
        label_id = np.zeros((h, w), dtype=np.int32)
        for encoded_color, class_id in self.encoded_to_id.items():
            mask = (rgb_encoded == encoded_color)
            label_id[mask] = class_id
        return label_id

    def _extract_connected_components(self, mask, min_area=100):
        labeled_mask, num_components = ndimage.label(mask)
        components = []
        for i in range(1, num_components + 1):
            comp = (labeled_mask == i)
            if comp.sum() >= min_area:
                components.append(comp)
        return components

    def _find_closest_component(self, components, gt_points):
        if not components:
            return None
        if len(components) == 1:
            return components[0]
        max_overlap, best = -1, components[0]
        for comp in components:
            overlap = sum(
                1 for (x, y) in gt_points if 0 <= y < comp.shape[0] and 0 <= x < comp.shape[1] and comp[y, x]
            )
            if overlap > max_overlap:
                max_overlap, best = overlap, comp
        return best

    def _parse_points(self, s):
        if not s:
            return []
        match = re.search(r'<point>(.*?)</point>', s, re.DOTALL)
        if not match:
            return []
        try:
            return json.loads(match.group(1))
        except Exception:
            return []

    def evaluate_point_sample(self, sample, min_area=100):
        label_path = self._get_label_path_from_image(sample['image'])
        if not label_path.exists():
            return {'success': False, 'error': f'Label file not found: {label_path}'}

        label_rgb = np.array(Image.open(label_path))
        if len(label_rgb.shape) == 2:
            label_rgb = np.stack([label_rgb] * 3, axis=-1)
        elif label_rgb.shape[2] == 4:
            label_rgb = label_rgb[:, :, :3]

        label_id = self._rgb_to_label_id(label_rgb)
        gt_points = self._parse_points(sample['ground_truth'])
        pred_points = self._parse_points(sample['predicted_answer'])

        if not gt_points or not pred_points:
            return {'success': True, 'type': 'point',
                    'num_gt_points': len(gt_points),
                    'num_pred_points': len(pred_points),
                    'num_correct': 0,
                    'accuracy': 0.0,
                    'target_class_id': None,
                    'target_class_name': None}

        first_gt_point = gt_points[0]
        y, x = first_gt_point[1], first_gt_point[0]
        if not (0 <= y < label_id.shape[0] and 0 <= x < label_id.shape[1]):
            return {'success': True, 'type': 'point',
                    'num_gt_points': len(gt_points),
                    'num_pred_points': len(pred_points),
                    'num_correct': 0,
                    'accuracy': 0.0,
                    'target_class_id': None,
                    'target_class_name': None}

        target_class_id = label_id[y, x]
        class_mask = (label_id == target_class_id)
        comps = self._extract_connected_components(class_mask, min_area)
        if not comps:
            return {'success': True, 'type': 'point',
                    'num_gt_points': len(gt_points),
                    'num_pred_points': len(pred_points),
                    'num_correct': 0,
                    'accuracy': 0.0,
                    'target_class_id': int(target_class_id),
                    'target_class_name': self.id2name.get(target_class_id, 'unknown')}

        target_mask = self._find_closest_component(comps, gt_points)
        num_correct = sum(
            1 for (x, y) in pred_points
            if 0 <= y < target_mask.shape[0] and 0 <= x < target_mask.shape[1] and target_mask[y, x]
        )
        acc = num_correct / len(pred_points) if pred_points else 0.0

        return {'success': True, 'type': 'point',
                'num_gt_points': len(gt_points), 'num_pred_points': len(pred_points),
                'num_correct': num_correct, 'accuracy': acc,
                'target_class_id': int(target_class_id),
                'target_class_name': self.id2name.get(target_class_id, 'unknown')}

    def evaluate_choice_sample(self, sample):
        gt = sample['ground_truth'].strip().lower()
        pred = sample['predicted_answer'].strip().lower()
        correct = int(gt == pred)
        return {'success': True, 'type': 'choice', 'correct': correct}

    def evaluate_open_sample(self, sample):
        gt = sample['ground_truth'].strip().lower()
        pred = sample['predicted_answer'].strip().lower()
        if not gt or not pred:
            return {'success': False, 'error': 'Empty answer'}
        gt_tokens = gt.split()
        pred_tokens = pred.split()
        smooth = SmoothingFunction().method1
        bleu_scores = [
            sentence_bleu([gt_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth),
            sentence_bleu([gt_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth),
            sentence_bleu([gt_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth),
            sentence_bleu([gt_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth),
        ]
        return {'success': True, 'type': 'open', 'bleu1': bleu_scores[0],
                'bleu2': bleu_scores[1], 'bleu3': bleu_scores[2],
                'bleu4': bleu_scores[3], 'avg_bleu': np.mean(bleu_scores)}

    def evaluate_file(self, result_file, min_area=100):
        with open(result_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)

        results = {'point': [], 'choice': [], 'open': [], 'failed': []}

        for s in samples:
            try:
                question = s.get("question", "").lower()
                gt = s.get("ground_truth", "")
                pred = s.get("predicted_answer", "")
                if "<point>" in gt or "<point>" in pred:
                    r = self.evaluate_point_sample(s, min_area)
                elif "choose from the follow object categories" in question:
                    r = self.evaluate_choice_sample(s)
                else:
                    r = self.evaluate_open_sample(s)
                if r['success']:
                    results[r['type']].append(r)
                else:
                    results['failed'].append(r)
            except Exception as e:
                results['failed'].append({'success': False, 'error': str(e)})

        # 汇总结果
        if results['point']:
            total_pred_points = sum(max(r['num_pred_points'], 1) for r in results['point'])
            total_correct = sum(r['num_correct'] for r in results['point'])
            point_acc = total_correct / total_pred_points
        else:
            point_acc = 0

        choice_acc = np.mean([r['correct'] for r in results['choice']]) if results['choice'] else 0
        open_bleu = np.mean([r['avg_bleu'] for r in results['open']]) if results['open'] else 0

        types_count = sum([1 for v in [results['point'], results['choice'], results['open']] if len(v) > 0])
        final_score = (point_acc + choice_acc + open_bleu) / types_count if types_count > 0 else 0

        return {
            'file': str(result_file),
            'point_acc': point_acc,
            'choice_acc': choice_acc,
            'open_bleu': open_bleu,
            'final_score': final_score,
            'counts': {k: len(v) for k, v in results.items()}
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='计算 Function QA 的综合评分')
    parser.add_argument('--file', type=str, required=True, help='要评估的 JSON 文件路径')
    parser.add_argument('--label_dir', type=str, default="/e2e-data/evad-tech-vla/lingfeng/UAVScenes/interval5_CAM_label", help='Label 图像根目录')
    parser.add_argument('--min_area', type=int, default=100, help='最小连通组件面积')
    args = parser.parse_args()

    file_path = args.file
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在: {file_path}")
        return

    evaluator = PointingQAEvaluator(args.label_dir)

    print(f"\n==============================")
    print(f"开始评估文件: {file_path}")
    print("==============================")
    try:
        stats = evaluator.evaluate_file(file_path, min_area=args.min_area)
        print(f"✅ 文件评估完成: {os.path.basename(file_path)}")
        print(f"Point任务: {stats['point_acc']*100:.2f}% | 选择题: {stats['choice_acc']*100:.2f}% | 开放问答BLEU: {stats['open_bleu']*100:.2f}% | 平均得分: {stats['final_score']*100:.2f}%")
    except Exception as e:
        print(f"❌ 评估失败 {file_path}: {e}")


if __name__ == "__main__":
    main()
