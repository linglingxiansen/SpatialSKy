import json
import numpy as np
from PIL import Image
from pathlib import Path
from scipy import ndimage
from scipy.spatial import ConvexHull
from collections import defaultdict
import re


class FreespaceQAEvaluator:
    def __init__(self):
        """
        初始化Freespace评估器
        """
        print(f"初始化Freespace评估器")
    
    def _parse_points(self, point_string):
        """
        解析点坐标字符串
        
        Args:
            point_string: 例如 "<point>[[1640,153],[1678,149],...]</point>"
        
        Returns:
            点列表 [[x1, y1], [x2, y2], ...]
        """
        if not point_string:
            return []
        
        # 提取<point>标签内的内容
        match = re.search(r'<point>(.*?)</point>', point_string, re.DOTALL)
        if not match:
            return []
        
        try:
            points_str = match.group(1)
            points = json.loads(points_str)
            return points
        except Exception as e:
            print(f"  ⚠️ 解析点坐标失败: {e}")
            return []
    
    def _create_region_from_points(self, points, method='convex_hull', expansion=0):
        """
        从点集创建区域
        
        Args:
            points: 点列表 [[x1, y1], [x2, y2], ...]
            method: 'convex_hull' - 凸包, 'bounding_box' - 外接矩形, 'buffer' - 点缓冲区
            expansion: 区域扩展像素数（用于放宽判定）
        
        Returns:
            区域判定函数，输入(x, y)返回是否在区域内
        """
        if not points or len(points) == 0:
            return None
        
        points_array = np.array(points)
        
        if method == 'convex_hull':
            if len(points) < 3:
                # 点数少于3个，使用缓冲区方法
                return self._create_buffer_region(points_array, expansion)
            
            try:
                hull = ConvexHull(points_array)
                hull_points = points_array[hull.vertices]
                
                def point_in_convex_hull(x, y):
                    """判断点是否在凸包内"""
                    point = np.array([x, y])
                    
                    # 使用叉积判断点是否在凸包内
                    for i in range(len(hull_points)):
                        p1 = hull_points[i]
                        p2 = hull_points[(i + 1) % len(hull_points)]
                        
                        # 计算叉积
                        cross = (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])
                        
                        if cross < -expansion:  # 加入expansion作为容差
                            return False
                    
                    return True
                
                return point_in_convex_hull
                
            except Exception as e:
                print(f"  ⚠️ 创建凸包失败，使用缓冲区方法: {e}")
                return self._create_buffer_region(points_array, expansion)
        
        elif method == 'bounding_box':
            min_x = np.min(points_array[:, 0]) - expansion
            max_x = np.max(points_array[:, 0]) + expansion
            min_y = np.min(points_array[:, 1]) - expansion
            max_y = np.max(points_array[:, 1]) + expansion
            
            def point_in_bbox(x, y):
                return min_x <= x <= max_x and min_y <= y <= max_y
            
            return point_in_bbox
        
        elif method == 'buffer':
            return self._create_buffer_region(points_array, expansion)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _create_buffer_region(self, points_array, radius):
        """
        创建点缓冲区（任意点距离给定点集的最小距离小于radius）
        
        Args:
            points_array: numpy数组 shape (n, 2)
            radius: 缓冲半径
        
        Returns:
            区域判定函数
        """
        if radius <= 0:
            radius = 50  # 默认缓冲半径
        
        def point_in_buffer(x, y):
            point = np.array([x, y])
            distances = np.linalg.norm(points_array - point, axis=1)
            return np.min(distances) <= radius
        
        return point_in_buffer
    
    def evaluate_single_sample(self, sample, region_method='convex_hull', expansion=0):
        """
        评估单个样本
        
        Args:
            sample: 包含image, question, predicted_answer, ground_truth的字典
            region_method: 区域创建方法 ('convex_hull', 'bounding_box', 'buffer')
            expansion: 区域扩展像素数
        
        Returns:
            评估结果字典
        """
        try:
            # 解析GT和预测点
            gt_points = self._parse_points(sample['ground_truth'])
            pred_points = self._parse_points(sample['predicted_answer'])
            
            if not gt_points:
                return {
                    'success': False,
                    'error': 'No GT points found'
                }
            
            if not pred_points:
                return {
                    'success': True,
                    'num_gt_points': len(gt_points),
                    'num_pred_points': 0,
                    'num_correct': 0,
                    'accuracy': 0.0
                }
            
            # 从GT点创建区域
            region_checker = self._create_region_from_points(
                gt_points, 
                method=region_method, 
                expansion=expansion
            )
            
            if region_checker is None:
                return {
                    'success': False,
                    'error': 'Failed to create region from GT points'
                }
            
            # 检查预测点是否在区域内
            num_correct = 0
            for pred_point in pred_points:
                x, y = pred_point
                if region_checker(x, y):
                    num_correct += 1
            
            accuracy = num_correct / len(pred_points) if pred_points else 0.0
            
            return {
                'success': True,
                'num_gt_points': len(gt_points),
                'num_pred_points': len(pred_points),
                'num_correct': num_correct,
                'accuracy': accuracy,
                'region_method': region_method
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def evaluate_file(self, result_file, region_method='convex_hull', expansion=0):
        """
        评估整个结果文件
        
        Args:
            result_file: JSON格式的结果文件路径
            region_method: 区域创建方法 ('convex_hull', 'bounding_box', 'buffer')
            expansion: 区域扩展像素数
        
        Returns:
            评估结果统计
        """
        print(f"\n开始评估Freespace任务: {result_file}")
        print(f"区域方法: {region_method}, 扩展: {expansion}像素")
        print("=" * 70)
        
        # 读取结果文件
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 处理当文件为 dict (id->sample) 情况，转成 list
        if isinstance(data, dict):
            samples = list(data.values())
        elif isinstance(data, list):
            samples = data
        else:
            raise RuntimeError("不支持的文件格式: 必须是 list 或 dict")
        
        print(f"总样本数: {len(samples)}")
        print("-" * 70)
        
        # 评估每个样本
        results = []
        
        for idx, sample in enumerate(samples, 1):
            result = self.evaluate_single_sample(sample, region_method, expansion)
            results.append(result)
            
            if idx % 100 == 0:
                print(f"  进度: {idx}/{len(samples)} ({idx/len(samples)*100:.1f}%)")
        
        # 统计结果
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        print(f"\n评估完成!")
        print(f"  成功: {len(successful_results)}")
        print(f"  失败: {len(failed_results)}")
        
        if failed_results:
            print(f"\n失败原因统计:")
            error_counts = defaultdict(int)
            for r in failed_results:
                error_counts[r.get('error', 'Unknown')] += 1
            for error, count in sorted(error_counts.items(), key=lambda x: -x[1]):
                print(f"  {error}: {count}")
        
        # 计算总体指标
        if successful_results:
            total_pred_points = sum(r['num_pred_points'] for r in successful_results)
            total_correct = sum(r['num_correct'] for r in successful_results)
            overall_accuracy = total_correct / total_pred_points if total_pred_points > 0 else 0.0
            
            print("\n" + "=" * 70)
            print("总体评估结果")
            print("=" * 70)
            print(f"总预测点数: {total_pred_points}")
            print(f"正确点数: {total_correct}")
            print(f"总体准确率: {overall_accuracy * 100:.2f}%")
            
            # 按样本计算平均准确率
            sample_accuracies = [r['accuracy'] for r in successful_results]
            avg_sample_accuracy = np.mean(sample_accuracies)
            median_sample_accuracy = np.median(sample_accuracies)
            print(f"样本平均准确率: {avg_sample_accuracy * 100:.2f}%")
            print(f"样本中位准确率: {median_sample_accuracy * 100:.2f}%")
            
            # 准确率分布
            acc_100 = sum(1 for acc in sample_accuracies if acc == 1.0)
            acc_80_100 = sum(1 for acc in sample_accuracies if 0.8 <= acc < 1.0)
            acc_50_80 = sum(1 for acc in sample_accuracies if 0.5 <= acc < 0.8)
            acc_0_50 = sum(1 for acc in sample_accuracies if 0 < acc < 0.5)
            acc_0 = sum(1 for acc in sample_accuracies if acc == 0.0)
            
            print(f"\n准确率分布:")
            print(f"  100%: {acc_100} ({acc_100/len(sample_accuracies)*100:.1f}%)")
            print(f"  80-100%: {acc_80_100} ({acc_80_100/len(sample_accuracies)*100:.1f}%)")
            print(f"  50-80%: {acc_50_80} ({acc_50_80/len(sample_accuracies)*100:.1f}%)")
            print(f"  0-50%: {acc_0_50} ({acc_0_50/len(sample_accuracies)*100:.1f}%)")
            print(f"  0%: {acc_0} ({acc_0/len(sample_accuracies)*100:.1f}%)")
            
            print("=" * 70)
            
            return {
                'total_samples': len(samples),
                'successful_samples': len(successful_results),
                'failed_samples': len(failed_results),
                'total_pred_points': total_pred_points,
                'total_correct': total_correct,
                'overall_accuracy': overall_accuracy,
                'avg_sample_accuracy': avg_sample_accuracy,
                'median_sample_accuracy': median_sample_accuracy,
                'accuracy_distribution': {
                    '100': acc_100,
                    '80-100': acc_80_100,
                    '50-80': acc_50_80,
                    '0-50': acc_0_50,
                    '0': acc_0
                }
            }
        else:
            print("\n 没有成功评估的样本")
            return None


def main():
    """主函数"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='计算 Freespace QA 的准确率')
    parser.add_argument('--file', type=str, required=True, help='要评估的 JSON 文件路径')
    parser.add_argument('--region_method', type=str, default='convex_hull', 
                        choices=['convex_hull', 'bounding_box', 'buffer'],
                        help='区域创建方法: convex_hull (推荐), bounding_box, buffer')
    parser.add_argument('--expansion', type=int, default=10, help='区域扩展像素数（用于放宽判定）')
    args = parser.parse_args()

    file_path = args.file
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在: {file_path}")
        return
    
    # 创建评估器
    evaluator = FreespaceQAEvaluator()
    
    # 评估
    stats = evaluator.evaluate_file(
        file_path, 
        region_method=args.region_method,
        expansion=args.expansion
    )
    
    if stats:
        print("\n✅ 评估完成")
        print(f"   总体准确率: {stats['overall_accuracy'] * 100:.2f}%")
        print(f"   样本平均准确率: {stats['avg_sample_accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    main()