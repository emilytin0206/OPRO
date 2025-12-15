import numpy as np
import re
from src.models.base_client import BaseModelClient

class Scorer:
    def __init__(self, model_client: BaseModelClient, metric_type: str = 'accuracy'):
        self.client = model_client
        self.metric_type = metric_type

    def _format_prompt(self, instruction: str, question: str, instruction_pos: str = 'A_begin') -> str:
        """
        根據 instruction_pos 組合 Prompt。
        這裡簡化了原本複雜的邏輯，保留最常用的格式。
        """
        if instruction_pos == 'before_Q':
            return f"{instruction}\n\nQ: {question}\nA:"
        elif instruction_pos == 'Q_begin':
            return f"Q: {instruction}\n{question}\nA:"
        else: # Default 'A_begin' or others
            return f"Q: {question}\nA: {instruction}"

    def _check_answer(self, prediction: str, target: str) -> float:
        """
        簡易的答案檢查邏輯。
        針對 GSM8K (數值) 或 BBH (選擇題/字串) 進行比對。
        """
        # 移除預測中的多餘空白與標點
        pred_clean = prediction.strip().lower()
        target_clean = str(target).strip().lower()

        # 1. 包含匹配 (Ex: "The answer is 42" vs "42")
        if target_clean in pred_clean:
            return 1.0
        
        # 2. 數值匹配 (嘗試提取數字)
        try:
            pred_nums = re.findall(r"[-+]?\d*\.\d+|\d+", pred_clean)
            target_nums = re.findall(r"[-+]?\d*\.\d+|\d+", target_clean)
            if pred_nums and target_nums and float(pred_nums[-1]) == float(target_nums[-1]):
                return 1.0
        except:
            pass
            
        return 0.0

    def score_instruction(self, instruction: str, dataset: list, num_samples: int = 50) -> dict:
        """
        評估單一指令在資料集上的分數。
        
        Args:
            instruction: 要評估的指令文本
            dataset: 資料集 [{'input':..., 'target':...}]
            num_samples: 為了速度，可以只評估部分驗證集
        
        Returns:
            {'score': float, 'details': list}
        """
        # 隨機採樣
        import random
        eval_data = random.sample(dataset, min(len(dataset), num_samples))
        
        scores = []
        # 可擴充為平行處理 (ThreadPoolExecutor)
        for example in eval_data:
            prompt = self._format_prompt(instruction, example['input'])
            
            # 呼叫 LLM
            prediction = self.client.generate_text(prompt)
            
            # 計算分數
            acc = self._check_answer(prediction, example['target'])
            scores.append(acc)
        
        avg_score = np.mean(scores)
        return {'score': float(avg_score), 'num_evals': len(scores)}