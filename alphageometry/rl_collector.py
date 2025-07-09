import os
import json
import time
from collections import defaultdict
from typing import List, Dict, Any, Tuple

class StepwiseRLDataCollector:
    """
    分步收集AlphaGeometry搜索路径，用于DPO训练
    """
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = f"rl_data_{timestamp}"
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.path_steps = []  # 每条路径的分步记录: (problem_id, is_success, [(prompt, output)])
        self.examples = []

    def add_path(self, problem_id: str, is_success: bool, step_records: List[Tuple[str, str]]):
        """
        添加一条完整路径的分步记录
        :param problem_id: 题目ID
        :param is_success: 是否为成功路径
        :param step_records: [(prompt, output)]，每一步的prompt和辅助点构造
        """
        self.path_steps.append((problem_id, is_success, step_records))

    def build_dpo_examples(self):
        """
        根据所有路径的分步记录，生成分步DPO训练样本
        """
        # 按题目分组
        problem2steps = defaultdict(list)
        for problem_id, is_success, steps in self.path_steps:
            problem2steps[problem_id].append((is_success, steps))

        for problem_id, paths in problem2steps.items():
            # 统计每个prompt下的正负样本
            prompt2outs = defaultdict(lambda: {"chosen": set(), "rejected": set()})
            for is_success, steps in paths:
                for i, (prompt, output) in enumerate(steps):
                    if is_success:
                        prompt2outs[prompt]["chosen"].add(output)
                    else:
                        prompt2outs[prompt]["rejected"].add(output)
            # 生成DPO样本
            for prompt, outs in prompt2outs.items():
                for chosen in outs["chosen"]:
                    for rejected in outs["rejected"]:
                        if chosen != rejected:
                            self.examples.append({
                                "problem_id": problem_id,
                                "prompt": prompt,
                                "chosen": chosen,
                                "rejected": rejected
                            })

    def save(self, filename: str = None):
        """
        保存DPO训练样本到文件
        """
        if not self.examples:
            print("没有可保存的DPO样本，请先调用build_dpo_examples()")
            return
        if filename is None:
            filename = f"stepwise_dpo_{len(self.examples)}.jsonl"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            for ex in self.examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"保存了 {len(self.examples)} 条分步DPO训练样本到 {filepath}")