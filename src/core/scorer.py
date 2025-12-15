import numpy as np
import re
import string
import logging
import pandas as pd
# [修正] Import 路徑
from src.model.base_client import BaseModelClient

logger = logging.getLogger("OPRO")

class Scorer:
    def __init__(self, model_client: BaseModelClient, config=None):
        self.client = model_client
        self.instruction_pos = getattr(config, 'instruction_pos', 'A_begin')
        self.is_instruction_tuned = getattr(config, 'is_instruction_tuned', False)
        
        task_name = getattr(config, 'task_name', '').lower()
        self.treat_as_bool = any(k in task_name for k in ['boolean', 'causal', 'web_of_lies'])
        self.treat_as_number = any(k in task_name for k in ['gsm8k', 'arithmetic', 'counting'])

    def _format_prompt(self, instruction: str, question: str) -> str:
        pos = self.instruction_pos
        # Instruction-tuned 模式 (無 QA 標籤)
        if self.is_instruction_tuned:
            if pos == 'Q_begin': return f"{instruction}\n{question}"
            elif pos == 'Q_end': return f"{question}\n{instruction}"
        
        # 一般 QA 模式
        if pos == 'before_Q': return f"{instruction}\n\nQ: {question}\nA:"
        elif pos == 'Q_begin': return f"Q: {instruction}\n{question}\nA:"
        elif pos == 'Q_end': return f"Q: {question}\n{instruction}\nA:"
        return f"Q: {question}\nA: {instruction}"

    def _check_answer(self, prediction: str, target: str) -> float:
        # 這裡保留簡化版的檢查邏輯，可視需要替換為更強的正規化版本
        pred_clean = prediction.strip().lower()
        target_clean = str(target).strip().lower()
        
        # 1. 包含匹配
        if target_clean in pred_clean: return 1.0
        
        # 2. 布林值簡單匹配
        if self.treat_as_bool:
            bool_map = {'yes': 'true', 'no': 'false', 'valid': 'true', 'invalid': 'false'}
            pred_bool = bool_map.get(pred_clean, pred_clean)
            target_bool = bool_map.get(target_clean, target_clean)
            if pred_bool == target_bool: return 1.0

        return 0.0

    def score_instruction(self, instruction: str, dataset: list, num_samples: int = 50) -> dict:
        """評估單一指令"""
        import random
        # [新增] 固定 Seed，確保每次評估抽到的題目一致
        # 如果不固定，Step 1 抽到簡單題，Step 2 抽到難題，會導致優化方向錯誤
        random.seed(0) 
        
        # 防呆：如果資料少於採樣數，全取
        actual_samples = min(len(dataset), num_samples)
        eval_data = random.sample(dataset, actual_samples)
        scores = []
        results_list = []
        
        for example in eval_data:
            prompt = self._format_prompt(instruction, example['input'])
            prediction = ""
            acc = 0.0
            try:
                prediction = self.client.generate_text(prompt)
                acc = self._check_answer(prediction, example['target'])
            except Exception as e:
                logger.error(f"Scoring error: {e}")
                
            scores.append(acc)
            results_list.append({
                'input': example['input'],
                'target': example['target'],
                'prediction': prediction,
                'accuracy': acc
            })
        
        avg_score = np.mean(scores) if scores else 0.0
        return {
            'score': float(avg_score), 
            'num_evals': len(scores),
            'detailed_dataframe': pd.DataFrame(results_list)
        }