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
        """添加一次搜索的结果作为训练样本
        
        Args:
            problem_id: 问题标识符
            problem_text: 原始问题文本
            successful_paths: 成功证明的路径及其分数 [(path, score),...]
            failed_paths: 未能证明的路径及其分数 [(path, score),...]
            limit_per_problem: 每个问题最多生成的样本数
        """
        # 初始化或更新问题计数器
        if problem_id not in self.problem_counters:
            self.problem_counters[problem_id] = 0
            
        # 如果已经达到此问题的样本上限，则跳过
        if self.problem_counters[problem_id] >= limit_per_problem:
            return
            
        # 防止样本过多，限制成功和失败路径数量
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
    
    def save(self, filename: str = None):
        """将收集的数据保存到文件
        
        Args:
            filename: 输出文件名，默认为rl_data_{样本数}.jsonl
        """
        if filename is None:
            filename = f"rl_data_{len(self.examples)}.jsonl"
            
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            for example in self.examples:
                f.write(json.dumps(example) + '\n')
        
        print(f"保存了 {len(self.examples)} 条强化学习训练样本到 {filepath}")
        return filepath