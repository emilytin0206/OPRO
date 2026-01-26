import logging
import pandas as pd  # 記得要 import pandas
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
            
            # [修正點] 將每題結果存入列表
            results.append({
                'input': input_text,
                'target': target_text,
                'prediction': prediction,
                'accuracy': 1.0 if is_correct else 0.0
            })
                
        score = correct_count / total if total > 0 else 0.0
        
        # [修正點] 回傳字典，包含分數與詳細結果 DataFrame
        # 這樣 optimization.py 呼叫 res['detailed_dataframe'] 時才不會報錯
        return {
            'score': score,
            'detailed_dataframe': pd.DataFrame(results)
        }

    def _evaluate_prediction(self, prediction: str, target: str) -> bool:
        """
        GSM8K 評分邏輯: 檢查是否包含正確答案
        """
        pred_clean = prediction.lower().strip()
        targ_clean = target.lower().strip()
        
        # 1. 直接字串比對
        if targ_clean in pred_clean:
            return True
            
        # 2. 數字提取比對 (針對 GSM8K)
        try:
            target_num = self._extract_number(targ_clean)
            pred_nums = self._extract_all_numbers(pred_clean)
            if target_num is not None and target_num in pred_nums:
                return True
        except:
            pass
            
        return False

    def _extract_number(self, text):
        import re
        matches = re.findall(r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?', text)
        if matches:
            return float(matches[-1].replace(',', ''))
        return None

    def _extract_all_numbers(self, text):
        import re
        matches = re.findall(r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?', text)
        return [float(m.replace(',', '')) for m in matches]