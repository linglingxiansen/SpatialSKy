import json
import os
import numpy as np
from PIL import Image
from pathlib import Path
from scipy import ndimage
from collections import defaultdict
import re


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
    25: {'name': 'unknown_25', 'RGB': [0, 60, 100]}
}


class PointingQAEvaluator:
    def __init__(self, label_root_dir):
        """
        初始化评估器

        Args:
            label_root_dir: label图像的根目录（字符串或Path）
        """
        self.label_root_dir = Path(label_root_dir)

        # 构建RGB到ID的映射
        self.rgb2id = {tuple(v['RGB']): k for k, v in cmap.items()}
        self.id2name = {k: v['name'] for k, v in cmap.items()}

        # 创建编码映射
        self.encoded_to_id = {}
        for rgb_tuple, class_id in self.rgb2id.items():
            encoded = (rgb_tuple[0] << 16) + (rgb_tuple[1] << 8) + rgb_tuple[2]
            self.encoded_to_id[encoded] = class_id

        print(f"Label根目录: {self.label_root_dir}")
        print(f"加载了 {len(cmap)} 个类别")

    def _normalize_image_field(self, image_field):
        """
        兼容 image 字段为 str 或 list 的情况，返回 string（第一个路径）
        """
        if image_field is None:
            return None
        if isinstance(image_field, list):
            if len(image_field) == 0:
                return None
            first = image_field[0]
            if isinstance(first, (str, Path)):
                return str(first)
            else:
                # 如果 list 里仍然是list/其他类型，尝试转成 str
                return str(first)
        if isinstance(image_field, (str, Path)):
            return str(image_field)
        # 其他类型，直接转字符串返回，但后面会检测文件存在性
        return str(image_field)

    def _get_label_path_from_image(self, image_path):
        """
        根据image路径获取对应的label路径
        Args:
            image_path: 原始图像路径（字符串）
        Returns:
            对应的label路径（Path）
        """
        if not image_path:
            return None

        image_path = Path(image_path)

        # 防护：路径层级不足时直接使用文件名所在目录名作为子目录
        parts = image_path.parts
        if len(parts) >= 3:
            subdir_name = parts[-3]
        elif len(parts) >= 2:
            subdir_name = parts[-2]
        else:
            subdir_name = parts[-1]

        label_filename = image_path.stem + '.png'
        label_path = self.label_root_dir / subdir_name / "interval5_CAM_label_color" / label_filename
        return label_path

    def _rgb_to_label_id(self, label_rgb):
        """将RGB格式的label转换为ID格式"""
        h, w = label_rgb.shape[:2]

        rgb_encoded = (label_rgb[:, :, 0].astype(np.int32) << 16) + \
                      (label_rgb[:, :, 1].astype(np.int32) << 8) + \
                      label_rgb[:, :, 2].astype(np.int32)

        label_id = np.zeros((h, w), dtype=np.int32)

        # 对常见场景优化：先用 np.vectorize/映射
        # 但为了简单清晰，这里保持原循环匹配
        for encoded_color, class_id in self.encoded_to_id.items():
            mask = (rgb_encoded == encoded_color)
            if mask.any():
                label_id[mask] = class_id

        return label_id

    def _extract_connected_components(self, mask, min_area=100):
        """提取连通组件"""
        labeled_mask, num_components = ndimage.label(mask)
        components = []
        for i in range(1, num_components + 1):
            component_mask = (labeled_mask == i)
            area = int(component_mask.sum())
            if area >= min_area:
                components.append(component_mask)
        return components

    def _find_closest_component(self, components, gt_points):
        """
        找到与GT点集最接近的连通组件
        """
        if not components:
            return None
        if len(components) == 1:
            return components[0]

        max_overlap = -1
        best_component = components[0]
        for component in components:
            overlap_count = 0
            for point in gt_points:
                if not (isinstance(point, (list, tuple)) and len(point) >= 2):
                    continue
                x, y = int(point[0]), int(point[1])
                if 0 <= y < component.shape[0] and 0 <= x < component.shape[1]:
                    if component[y, x]:
                        overlap_count += 1
            if overlap_count > max_overlap:
                max_overlap = overlap_count
                best_component = component
        return best_component

    def _parse_points(self, point_string):
        """
        解析点坐标字符串
        输入例子: "<point>[[1640,153],[1678,149]]</point>"
        返回 [[1640,153], [1678,149]]
        """
        if not point_string:
            return []
        # 如果输入本身已经是 list/tuple，直接返回
        if isinstance(point_string, (list, tuple)):
            return point_string

        # 提取<point>标签内的内容（宽松匹配）
        match = re.search(r'<point>(.*?)</point>', str(point_string), re.DOTALL)
        content = None
        if match:
            content = match.group(1)
        else:
            # 尝试直接把整个字符串当作JSON数组解析
            content = str(point_string)

        try:
            points = json.loads(content)
            # 确保元素是 [x,y] 的形式
            norm = []
            for p in points:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    norm.append([int(p[0]), int(p[1])])
            return norm
        except Exception:
            # 退化解析：匹配数字对的正则
            pts = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]', content)
            if not pts:
                return []
            return [[int(x), int(y)] for x, y in pts]

    def evaluate_single_sample(self, sample, min_area=100):
        """
        评估单个样本
        """
        try:
            # 兼容 image 字段为 list 或 str
            raw_image_field = sample.get('image', None)
            image_path_str = self._normalize_image_field(raw_image_field)
            if not image_path_str:
                return {'success': False, 'error': 'No image path provided'}

            label_path = self._get_label_path_from_image(image_path_str)
            if label_path is None or not label_path.exists():
                return {
                    'success': False,
                    'error': f'Label file not found: {label_path}'
                }

            # 加载label（注意通道）
            try:
                pil_img = Image.open(str(label_path)).convert('RGB')
            except Exception as e:
                return {'success': False, 'error': f'Failed to open label image: {e}'}

            label_rgb = np.array(pil_img)

            if label_rgb.ndim == 2:
                label_rgb = np.stack([label_rgb] * 3, axis=-1)
            elif label_rgb.shape[2] == 4:
                label_rgb = label_rgb[:, :, :3]

            label_id = self._rgb_to_label_id(label_rgb)

            # 解析GT和预测点
            gt_points = self._parse_points(sample.get('ground_truth', None))
            pred_points = self._parse_points(sample.get('predicted_answer', None))

            if not gt_points:
                return {'success': False, 'error': 'No GT points found'}

            if not pred_points:
                # 如果没有预测点，返回0准确率但任务成功（可改为失败视需求）
                return {
                    'success': True,
                    'num_gt_points': len(gt_points),
                    'num_pred_points': 0,
                    'num_correct': 0,
                    'accuracy': 0.0
                }

            first_gt_point = gt_points[0]
            if not (isinstance(first_gt_point, (list, tuple)) and len(first_gt_point) >= 2):
                return {'success': False, 'error': f'GT point format invalid: {first_gt_point}'}

            gx, gy = int(first_gt_point[0]), int(first_gt_point[1])
            # 注意索引顺序：label_id[y, x]
            if not (0 <= gy < label_id.shape[0] and 0 <= gx < label_id.shape[1]):
                return {'success': False, 'error': f'GT point out of bounds: {first_gt_point}'}

            target_class_id = int(label_id[gy, gx])
            class_mask = (label_id == target_class_id)

            components = self._extract_connected_components(class_mask, min_area)

            if not components:
                return {'success': False, 'error': f'No components found for class {target_class_id}'}

            target_mask = self._find_closest_component(components, gt_points)
            if target_mask is None:
                return {'success': False, 'error': f'Could not find a target component for class {target_class_id}'}

            num_correct = 0
            for pred_point in pred_points:
                if not (isinstance(pred_point, (list, tuple)) and len(pred_point) >= 2):
                    continue
                px, py = int(pred_point[0]), int(pred_point[1])
                if 0 <= py < target_mask.shape[0] and 0 <= px < target_mask.shape[1]:
                    if target_mask[py, px]:
                        num_correct += 1

            accuracy = num_correct / len(pred_points) if pred_points else 0.0

            return {
                'success': True,
                'num_gt_points': len(gt_points),
                'num_pred_points': len(pred_points),
                'num_correct': num_correct,
                'accuracy': accuracy,
                'target_class_id': int(target_class_id),
                'target_class_name': self.id2name.get(target_class_id, 'unknown')
            }

        except Exception as e:
            return {'success': False, 'error': f'Exception: {e}'}

    def evaluate_file(self, result_file, min_area=100):
        """
        评估整个结果文件
        """
        print(f"\n开始评估: {result_file}")
        print("=" * 70)

        # 读取结果文件
        with open(result_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)

        # 处理当文件为 dict (id->sample) 情况，转成 list
        if isinstance(samples, dict):
            samples_list = list(samples.values())
        elif isinstance(samples, list):
            samples_list = samples
        else:
            raise RuntimeError("Unsupported result file format: must be list or dict")

        print(f"总样本数: {len(samples_list)}")
        print("-" * 70)

        results = []
        class_results = defaultdict(list)

        for idx, sample in enumerate(samples_list, 1):
            result = self.evaluate_single_sample(sample, min_area)
            results.append(result)
            if result.get('success'):
                class_name = result.get('target_class_name', 'unknown')
                class_results[class_name].append(result.get('accuracy', 0.0))

            if idx % 100 == 0 or idx == len(samples_list):
                print(f"  进度: {idx}/{len(samples_list)} ({idx/len(samples_list)*100:.1f}%)")

        successful_results = [r for r in results if r.get('success')]
        failed_results = [r for r in results if not r.get('success')]

        print(f"\n评估完成!")
        print(f"  成功: {len(successful_results)}")
        print(f"  失败: {len(failed_results)}")

        if failed_results:
            print(f"\n失败原因统计:")
            error_counts = defaultdict(int)
            for r in failed_results:
                err = r.get('error', 'Unknown')
                error_counts[err] += 1
            for error, count in sorted(error_counts.items(), key=lambda x: -x[1]):
                print(f"  {error}: {count}")

        if successful_results:
            total_pred_points = sum(r.get('num_pred_points', 0) for r in successful_results)
            total_correct = sum(r.get('num_correct', 0) for r in successful_results)
            overall_accuracy = total_correct / total_pred_points if total_pred_points > 0 else 0.0

            print("\n" + "=" * 70)
            print("总体评估结果")
            print("=" * 70)
            print(f"总预测点数: {total_pred_points}")
            print(f"正确点数: {total_correct}")
            print(f"总体准确率: {overall_accuracy * 100:.2f}%")

            sample_accuracies = [r.get('accuracy', 0.0) for r in successful_results]
            avg_sample_accuracy = float(np.mean(sample_accuracies)) if sample_accuracies else 0.0
            print(f"样本平均准确率: {avg_sample_accuracy * 100:.2f}%")

            if class_results:
                print("\n各类别准确率:")
                print("-" * 70)
                for class_name in sorted(class_results.keys()):
                    accuracies = class_results[class_name]
                    avg_acc = float(np.mean(accuracies)) if accuracies else 0.0
                    print(f"  {class_name:<25}: {avg_acc * 100:6.2f}% (样本数: {len(accuracies)})")

            print("=" * 70)

            return {
                'total_samples': len(samples_list),
                'successful_samples': len(successful_results),
                'failed_samples': len(failed_results),
                'total_pred_points': total_pred_points,
                'total_correct': total_correct,
                'overall_accuracy': overall_accuracy,
                'avg_sample_accuracy': avg_sample_accuracy,
                'class_results': dict(class_results)
            }
        else:
            print("\n❌ 没有成功评估的样本")
            return None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='计算 Pointing QA 的准确率')
    parser.add_argument('--file', type=str, required=True, help='要评估的 JSON 文件路径')
    parser.add_argument('--label_dir', type=str, default="/e2e-data/evad-tech-vla/lingfeng/UAVScenes/interval5_CAM_label", help='Label 图像根目录')
    parser.add_argument('--min_area', type=int, default=100, help='最小连通组件面积')
    args = parser.parse_args()

    file_path = args.file
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在: {file_path}")
        return

    evaluator = PointingQAEvaluator(args.label_dir)
    stats = evaluator.evaluate_file(file_path, min_area=args.min_area)

    if stats:
        print("\n✅ 评估完成")
        print(f"   总体准确率: {stats['overall_accuracy'] * 100:.2f}%")
        print(f"   样本平均准确率: {stats['avg_sample_accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    main()
