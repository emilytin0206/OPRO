import numpy as np
import re
import string
import logging
from src.models.base_client import BaseModelClient

logger = logging.getLogger("OPRO")

# --- 移植自 DeepMind metrics.py 的常數與映射表 ---
_WORD_TO_NUM = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
    'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
    'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80,
    'ninety': 90, 'hundred': 100, 'thousand': 1000, 'million': 1000000,
    'billion': 1000000000, 'trillion': 1000000000000,
}

class Scorer:
    def __init__(self, model_client: BaseModelClient, config=None):
        self.client = model_client
        self.instruction_pos = getattr(config, 'instruction_pos', 'A_begin')
        self.is_instruction_tuned = getattr(config, 'is_instruction_tuned', False)
        
        # 根據任務名稱自動判斷處理模式 (可從 config 擴充)
        task_name = getattr(config, 'task_name', '').lower()
        self.treat_as_bool = any(k in task_name for k in ['boolean', 'causal', 'fallacies', 'navigate', 'web_of_lies'])
        self.treat_as_number = any(k in task_name for k in ['gsm8k', 'arithmetic', 'counting', 'math'])

    def _format_prompt(self, instruction: str, question: str) -> str:
        """根據 instruction_pos 與 model 類型組合 Prompt (同前次優化)"""
        pos = self.instruction_pos
        if self.is_instruction_tuned:
            if pos == 'Q_begin': return f"{instruction}\n{question}"
            elif pos == 'Q_end': return f"{question}\n{instruction}"
        
        if pos == 'before_Q': return f"{instruction}\n\nQ: {question}\nA:"
        elif pos == 'Q_begin': return f"Q: {instruction}\n{question}\nA:"
        elif pos == 'Q_end': return f"Q: {question}\n{instruction}\nA:"
        return f"Q: {question}\nA: {instruction}"

    # --- 核心正規化邏輯 (移植自 metrics.py) ---

    def _normalize_text_to_number(self, text: str) -> str:
        """將英文數字單字轉為阿拉伯數字字串 (e.g. 'twenty-five' -> '25')"""
        text = text.lower().replace('-', ' ').replace(',', '')
        words = text.split()
        current_val = 0
        final_val = 0
        has_num = False
        
        for word in words:
            if word in _WORD_TO_NUM:
                has_num = True
                val = _WORD_TO_NUM[word]
                if val >= 100:
                    if current_val == 0: current_val = 1
                    current_val *= val
                    if val >= 1000:
                        final_val += current_val
                        current_val = 0
                else:
                    current_val += val
        
        final_val += current_val
        return str(final_val) if has_num else text

    def _clean_number_string(self, text: str) -> str:
        """移除貨幣符號、逗號等非數字字元"""
        # 保留數字、小數點、負號，移除 $, %, , 等
        if not text: return ""
        # 移除常見貨幣與符號
        text = re.sub(r'[$,€£%]', '', text)
        return text.replace(',', '').strip()

    def _extract_answer_content(self, prediction: str) -> str:
        """從模型輸出中提取 'The answer is ...' 之後的內容"""
        prediction = prediction.strip()
        # 常見的答案引導詞
        patterns = [
            r"the answer is (.*)",
            r"answer:(.*)",
            r"final answer:(.*)",
            r"#### (.*)" # GSM8K 風格
        ]
        for p in patterns:
            match = re.search(p, prediction, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return prediction

    def _extract_choice(self, text: str) -> str:
        """提取括號選項 (e.g., '(A)')"""
        match = re.search(r'\(([a-zA-Z])\)', text)
        if match:
            return match.group(1).lower()
        return text

    # --- 評分邏輯 ---

    def _check_answer(self, prediction: str, target: str) -> float:
        """
        高強度的答案檢查邏輯，包含正規化與提取。
        """
        # 1. 初步清理與提取
        pred_content = self._extract_answer_content(prediction)
        pred_clean = pred_content.strip().lower()
        # 移除末尾句號 (如果最後是數字或單字)
        if pred_clean.endswith('.'): pred_clean = pred_clean[:-1]
        
        target_clean = str(target).strip().lower()

        # 2. 布林值正規化 (Boolean Normalization)
        if self.treat_as_bool:
            bool_map = {
                'yes': 'true', 'no': 'false', 
                'valid': 'true', 'invalid': 'false',
                'true': 'true', 'false': 'false'
            }
            # 嘗試映射
            pred_bool = bool_map.get(pred_clean, pred_clean)
            target_bool = bool_map.get(target_clean, target_clean)
            if pred_bool == target_bool:
                return 1.0

        # 3. 數字正規化 (Numerical Normalization)
        if self.treat_as_number:
            # 嘗試文字轉數字
            p_num_str = self._normalize_text_to_number(pred_clean)
            p_clean = self._clean_number_string(p_num_str)
            t_clean = self._clean_number_string(target_clean)
            
            # 嘗試提取最後一個數字 (針對 'The answer is 5 apples' -> 5)
            p_nums = re.findall(r"[-+]?\d*\.\d+|\d+", p_clean)
            t_nums = re.findall(r"[-+]?\d*\.\d+|\d+", t_clean)
            
            if p_nums and t_nums:
                try:
                    # 比較數值 (允許微小誤差處理浮點數)
                    if abs(float(p_nums[-1]) - float(t_nums[-1])) < 1e-6:
                        return 1.0
                except:
                    pass

        # 4. 選擇題正規化 (Choice Extraction)
        # 如果答案是單一字母 (A, B, C...)
        if len(target_clean) == 1 and target_clean in string.ascii_lowercase:
            pred_choice = self._extract_choice(pred_clean)
            if target_clean in pred_choice: # 寬鬆匹配
                return 1.0

        # 5. 最後防線：包含匹配 (String Inclusion)
        # 官方 metrics.py 也有類似邏輯，只要 target 出現在提取後的 answer 中就算對
        if target_clean in pred_clean:
            return 1.0
            
        return 0.0

    def score_instruction(self, instruction: str, dataset: list, num_samples: int = 50) -> dict:
        """
        評估單一指令，並回傳詳細數據以便存檔。
        """
        import random
        # 如果 dataset 很大，這步隨機採樣對於 "一致性" 會有影響。
        # 在做快取時，理想上應該要評估相同的驗證集，但 OPRO 原作確實是隨機採樣。
        # 為了快取有效性，我們假設分數代表了該指令的 "真實實力"。
        eval_data = random.sample(dataset, min(len(dataset), num_samples))
        
        results_list = []
        scores = []
        
        for example in eval_data:
            prompt = self._format_prompt(instruction, example['input'])
            prediction = ""
            acc = 0.0
            
            try:
                prediction = self.client.generate_text(prompt)
                acc = self._check_answer(prediction, example['target'])
            except Exception as e:
                logger.error(f"Scoring failed: {e}")
                
            scores.append(acc)
            # 記錄詳細資訊
            results_list.append({
                'input': example['input'],
                'target': example['target'],
                'prediction': prediction,
                'accuracy': acc
            })
        
        avg_score = np.mean(scores) if scores else 0.0
        
        # 建立詳細 DataFrame
        detailed_df = pd.DataFrame(results_list)
        
        return {
            'score': float(avg_score),
            'num_evals': len(scores),
            'detailed_dataframe': detailed_df # 新增這個回傳值
        }