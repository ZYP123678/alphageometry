# 实现对正确路径和错误路径的收集
import os
import json
import time
from typing import List, Dict, Any, Tuple

class RLDataCollector:
    """收集AlphaGeometry的搜索路径，用于强化学习训练"""
    
    def __init__(self, output_dir: str = None):
        """初始化数据收集器
        
        Args:
            output_dir: 数据保存目录，如果为None则使用当前时间创建目录
        """
        if output_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = f"rl_data_{timestamp}"
            
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.examples = []
        self.problem_counters = {}
        
    def add_search_result(self, 
                     problem_id: str,
                     problem_text: str, 
                     successful_paths: List[Tuple[str, float]], 
                     failed_paths: List[Tuple[str, float]],
                     limit_per_problem: int = 200):
        """添加一次搜索的结果作为训练样本"""
        # 初始化或更新问题计数器
        if problem_id not in self.problem_counters:
            self.problem_counters[problem_id] = 0
            
        # 如果已经达到此问题的样本上限，则跳过
        if self.problem_counters[problem_id] >= limit_per_problem:
            return
            
        print(f"添加搜索结果: {len(successful_paths)}个成功路径, {len(failed_paths)}个失败路径")
        print(successful_paths)
        # 处理特殊情况：仅有成功路径或仅有失败路径
        if successful_paths and not failed_paths:
            # 只有成功路径情况 - 创建"自我对比"样本
            success_limit = min(5, len(successful_paths))
            paths = sorted(successful_paths, key=lambda x: x[1], reverse=True)[:success_limit]
            
            for i, (path1, score1) in enumerate(paths):
                for j, (path2, score2) in enumerate(paths):
                    if i != j and score1 > score2:  # 确保比较不同路径，且更高分的为chosen
                        if self.problem_counters[problem_id] >= limit_per_problem:
                            break
                        
                        example = {
                            "problem_id": problem_id,
                            "prompt": problem_text,
                            "chosen": path1,
                            "chosen_score": float(score1),
                            "rejected": path2,
                            "rejected_score": float(score2)
                        }
                        self.examples.append(example)
                        self.problem_counters[problem_id] += 1
            
            print(f"仅使用成功路径生成了{self.problem_counters[problem_id]}个样本")
            return
            
        if failed_paths and not successful_paths:
            # 只有失败路径情况 - 记录最高得分的失败路径
            fail_limit = min(10, len(failed_paths))
            best_failures = sorted(failed_paths, key=lambda x: x[1], reverse=True)[:fail_limit]
            
            for i, (path1, score1) in enumerate(best_failures):
                
                
                for j, (path2, score2) in enumerate(best_failures):
                    if i != j and score1 > score2:  # 分数高的可能更接近正确解
                        if self.problem_counters[problem_id] >= limit_per_problem:
                            break
                        
                        example = {
                            "problem_id": problem_id,
                            "prompt": problem_text,
                            "chosen": path1,
                            "chosen_score": float(score1),
                            "rejected": path2,
                            "rejected_score": float(score2)
                        }
                        self.examples.append(example)
                        self.problem_counters[problem_id] += 1
            
            print(f"仅使用失败路径生成了{self.problem_counters[problem_id]}个样本")
            return
        
        # 原有逻辑：成功和失败路径都存在的情况
        success_limit = min(5, len(successful_paths))
        fail_limit = min(150, len(failed_paths))
        
        # 按分数降序排序
        successful_paths = sorted(successful_paths, key=lambda x: x[1], reverse=True)[:success_limit]
        failed_paths = sorted(failed_paths, key=lambda x: x[1], reverse=True)[:fail_limit]
        
        # 生成训练样本对
        for success_idx, (success_path, success_score) in enumerate(successful_paths):
            for fail_idx, (fail_path, fail_score) in enumerate(failed_paths):
                # 确保样本计数不超过限制
                if self.problem_counters[problem_id] >= limit_per_problem:
                    break
                    
                example = {
                    "problem_id": problem_id,
                    "prompt": problem_text,
                    "chosen": success_path,
                    "chosen_score": float(success_score),
                    "rejected": fail_path,
                    "rejected_score": float(fail_score)
                }
                self.examples.append(example)
                self.problem_counters[problem_id] += 1
        
        print(f"成功生成{self.problem_counters[problem_id]}个样本")
    
    def save(self, filename: str = None):
        """将收集的数据保存到文件"""
        if not self.examples:
            print("警告：没有收集到任何样本，跳过保存")
            return
            
        if filename is None:
            filename = f"rl_data_{len(self.examples)}.jsonl"
            
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            for example in self.examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        
        print(f"保存了 {len(self.examples)} 条强化学习训练样本到 {filepath}")