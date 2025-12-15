import numpy as np
import re
import string
import logging
import pandas as pd
#
from src.model.base_client import BaseModelClient

logger = logging.getLogger("OPRO")

class Scorer:
    
    def __init__(self, model_client: BaseModelClient, config=None):
        self.client = model_client
        self.config = config
        self.instruction_pos = getattr(config, 'instruction_pos', 'Q_begin') # 預設 Q_begin
        # ... (其他初始化維持不變) ...
        task_name = getattr(config, 'task_name', '').lower()
        dataset_name = getattr(config, 'dataset_name', '').lower()
        self.is_gsm8k = 'gsm8k' in dataset_name or 'gsm8k' in task_name
        self.treat_as_bool = any(k in task_name for k in ['boolean', 'causal', 'web_of_lies'])

    def _format_prompt(self, instruction: str, question: str) -> str:
        # 這裡僅負責組合 User Message 的內容
        pos = self.instruction_pos
        # Q_begin: Instruction 在問題前面
        if pos == 'Q_begin': return f"{instruction}\n\nQ: {question}\nA:"
        elif pos == 'Q_end': return f"Q: {question}\n\n{instruction}\nA:"
        # Fallback
        return f"{instruction}\n{question}"

    def _normalize_answer(self, s):
        """參考官方 metrics.py 的標準化邏輯"""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

    def _extract_gsm8k_answer(self, prediction: str) -> str:
        """針對 GSM8K 提取數值"""
        # 1. 嘗試尋找官方格式 "#### 答案"
        if "####" in prediction:
            return prediction.split("####")[-1].strip()
        
        # 2. 如果沒有，嘗試提取最後一個數字
        # 移除逗號 (1,000 -> 1000)
        clean_pred = prediction.replace(',', '')
        # 尋找所有數字 (包含浮點數與負號)
        numbers = re.findall(r'-?\d+\.?\d*', clean_pred)
        if numbers:
            return numbers[-1]
        return ""

    def _check_answer(self, prediction: str, target: str) -> float:
        """強化的評分邏輯"""
        prediction = str(prediction)
        target = str(target)

        # --- GSM8K 專用邏輯 ---
        if self.is_gsm8k:
            pred_val = self._extract_gsm8k_answer(prediction)
            target_val = self._extract_gsm8k_answer(target) # Target 通常已經是純數字，但也防呆一下
            
            # 數值比對 (允許小數點誤差)
            try:
                if abs(float(pred_val) - float(target_val)) < 1e-6:
                    return 1.0
            except:
                pass # 轉換失敗或無法提取
            
            # 若數值提取失敗，退回嚴格字串比對
            return 1.0 if target_val == pred_val and target_val != "" else 0.0

        # --- 通用邏輯 (Normalization) ---
        pred_norm = self._normalize_answer(prediction)
        target_norm = self._normalize_answer(target)

        # 1. 精確匹配 (Normalized)
        if pred_norm == target_norm:
            return 1.0
            
        # 2. 包含匹配 (加上邊界檢查，防止 "1" match "100")
        # 檢查 target 是否作為一個獨立的詞存在於 prediction 中
        pattern = r'\b{}\b'.format(re.escape(target_norm))
        if re.search(pattern, pred_norm):
            return 1.0

        # 3. 布林值特殊處理
        if self.treat_as_bool:
            bool_map = {'yes': 'true', 'no': 'false', 'valid': 'true', 'invalid': 'false'}
            p_bool = bool_map.get(pred_norm, pred_norm)
            t_bool = bool_map.get(target_norm, target_norm)
            if p_bool == t_bool: return 1.0

        return 0.0

    def score_instruction(self, instruction: str, dataset: list, num_samples: int = None) -> dict:
        """評估單一指令"""
        import random
        # 使用傳入的 dataset，不再次 sample (因為外部已經控制了 train/eval split)
        # 如果 num_samples 設了，則進行抽樣 (用於大資料集快速預覽)
        
        eval_data = dataset
        if num_samples and num_samples < len(dataset):
            random.seed(0) # 確保同一指令在同一批數據上評分
            eval_data = random.sample(dataset, num_samples)

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