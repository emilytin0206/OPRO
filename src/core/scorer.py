import logging
import pandas as pd
import re
from src.model.base_client import BaseModelClient

logger = logging.getLogger("OPRO")

class Scorer:
    def __init__(self, model_client, config):
        self.client = model_client
        self.config = config

    def score_instruction(self, instruction: str, dataset: list) -> dict:
        """
        Score a single instruction (Answer Starting Sentence) on the dataset.
        Returns a dict with 'score' and 'detailed_dataframe'.
        """
        correct_count = 0
        total = len(dataset)
        results = []  # 收集每題結果
        
        for item in dataset:
            input_text = item['input']
            target_text = item['target']
            
            # --- PAPER SPECIFIC LOGIC (Text 4) ---
            # 指令放在 "A:" 之後作為 Answer Starting Sentence
            full_prompt = f"Q: {input_text}\nA: {instruction} "
            
            # Generate prediction
            prediction = self.client.generate_text(full_prompt).strip()
            
            # --- Evaluation ---
            is_correct = self._evaluate_prediction(prediction, target_text)
            
            if is_correct:
                correct_count += 1
            
            # 將每題結果存入列表
            results.append({
                'input': input_text,
                'target': target_text,
                'prediction': prediction,
                'accuracy': 1.0 if is_correct else 0.0
            })
                
        score = correct_count / total if total > 0 else 0.0
        
        # 回傳字典，包含分數與詳細結果 DataFrame
        return {
            'score': score,
            'detailed_dataframe': pd.DataFrame(results)
        }

    def _evaluate_prediction(self, prediction: str, target: str) -> bool:
        """
        參考官方 OPRO (metrics.py) 的答案提取邏輯
        """
        # 基礎清理
        pred_clean = prediction.strip()
        targ_clean = target.strip()
        
        # ==========================================
        # 處理單選題 (MMLU / BBH 等，答案為 A, B, C, D)
        # ==========================================
        if len(targ_clean) == 1 and targ_clean.upper() in ['A', 'B', 'C', 'D', 'E']:
            # 尋找獨立的 A, B, C, D 字母。
            # 涵蓋 "A", "(A)", "Answer: A", "A." 等常見情況
            matches = re.findall(r'\b([A-E])\b', pred_clean.upper())
            if matches:
                # 取最後一個出現的字母，因為模型往往在最後給出結論
                extracted_ans = matches[-1] 
                return extracted_ans == targ_clean.upper()
            return False

        # ==========================================
        # 處理數學題 (GSM8K，答案為數值)
        # ==========================================
        try:
            # 清理千分位逗號
            target_num_str = targ_clean.replace(',', '')
            target_num = float(target_num_str) if '.' in target_num_str else int(target_num_str)
            
            pred_clean_no_comma = pred_clean.replace(',', '')
            # 提取模型輸出中的所有數字 (包含正負號與小數點)
            pred_nums_str = re.findall(r'-?\d*\.?\d+', pred_clean_no_comma)
            
            if pred_nums_str:
                # 取最後一個數字作為最終預測結果
                pred_last_num = float(pred_nums_str[-1]) if '.' in pred_nums_str[-1] else int(pred_nums_str[-1])
                if pred_last_num == target_num:
                    return True
                
                # 容錯處理：如果最後一個不是，但倒數第二個等其他地方有精準命中也算對
                pred_nums = [float(n) if '.' in n else int(n) for n in pred_nums_str]
                if target_num in pred_nums:
                     return True
        except Exception as e:
            pass
            
        # ==========================================
        # 退回嚴格的全字串比對 (兜底邏輯)
        # ==========================================
        return targ_clean.lower() == pred_clean.lower()