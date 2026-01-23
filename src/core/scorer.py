import numpy as np
import re
import string
import logging
import pandas as pd
from tqdm import tqdm
from src.model.base_client import BaseModelClient
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("OPRO")

# --- 移植自 Google DeepMind metrics.py 的常數與輔助字典 ---
_WORD_TO_NUM = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
    'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
    'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
    'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
    'eighty': 80, 'ninety': 90,
}

# 用於捕捉答案的前綴模式
FINAL_ANSWER_BEHIND_PATTERNS_PRIMARY = ['answer is ', 'answer: ', 'answer is: ']
FINAL_ANSWER_BEHIND_PATTERNS_SECONDARY = ['is: ', 'are: ']
FINAL_ANSWER_AHEAD_PATTERNS = [
    ' is the correct answer', ' is the right answer',
    ' is the final answer', ' is the answer',
]
GSM8K_ANSWER_DELIMITER = '#### '

class Scorer:
    def __init__(self, model_client: BaseModelClient, config=None):
        self.client = model_client
        self.config = config
        self.instruction_pos = getattr(config, 'instruction_pos', 'Q_begin')
        
        task_name = getattr(config, 'task_name', '').lower()
        dataset_name = getattr(config, 'dataset_name', '').lower()
        
        # 判定任務類型
        self.is_gsm8k = 'gsm8k' in dataset_name or 'gsm8k' in task_name
        self.treat_as_bool = any(k in task_name for k in ['boolean', 'causal', 'web_of_lies'])
        # MMLU 視為多選題 (非數值)
        self.treat_as_number = self.is_gsm8k 

    def _format_prompt(self, instruction: str, question: str) -> str:
        pos = self.instruction_pos
        if pos == 'Q_begin': return f"{instruction}\n\nQ: {question}\nA:"
        elif pos == 'Q_end': return f"Q: {question}\n\n{instruction}\nA:"
        return f"{instruction}\n{question}"

    # --- 以下為移植自 Google metrics.py 的核心邏輯 (含修改 B: 加強 Choice 提取) ---

    def _is_float(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _extract_choice_letter(self, text: str) -> str | None:
        """
        修法 B (加強版): Robust 選項提取
        優先順序:
        1. \boxed{x} (Latex 格式，最強訊號)
        2. (x)       (括號格式)
        3. x         (獨立字母，例如 'Answer: C' 切割後的 C)
        """
        t = text.lower()

        # 1. \boxed{c}
        # 允許 \boxed{ c } 裡面有空白
        m = re.search(r'\\boxed\{\s*([a-e])\s*\}', t)
        if m: return m.group(1)

        # 2. (c)
        m = re.search(r'\(([a-e])\)', t)
        if m: return m.group(1)

        # 3. "answer is c" / "final answer: c" -> 假設傳進來的 text 已經經過初步清理
        # 使用 \b 確保是完整單字 (避免抓到 apple 的 a)
        m = re.search(r'\b([a-e])\b', t)
        if m: return m.group(1)

        return None

    def _parse_with_treating_as_number(self, prediction_parsed):
        """強化的數值解析邏輯 (GSM8K)"""
        # 移除等號後的內容作為答案候選
        prediction_parsed = prediction_parsed.split('=')[-1]
        
        # 移除貨幣符號與單位
        for c in ['$', ',', '%', '€', '£']:
            prediction_parsed = prediction_parsed.replace(c, '')
        prediction_parsed = prediction_parsed.split(':')[0]
        prediction_parsed = prediction_parsed.strip()

        # 文字轉數字 (twenty -> 20)
        for word, num in _WORD_TO_NUM.items():
            if word in prediction_parsed:
                prediction_parsed = prediction_parsed.replace(word, str(num))

        # 簡單的提取邏輯 (嘗試取最後一個數字或單詞)
        parts = list(reversed(prediction_parsed.split(' ')))
        prediction_parsed = parts[0] # 預設取最後一個
        for part in parts:
            if not part.isalpha(): # 找到第一個非純字母的 token
                prediction_parsed = part
                break
        
        # 移除結尾單位 (如 156kgs -> 156)
        while prediction_parsed and prediction_parsed[-1].isalpha():
            prediction_parsed = prediction_parsed[:-1]
        if prediction_parsed and prediction_parsed[-1] == '-':
            prediction_parsed = prediction_parsed[:-1]

        # 嘗試標準化為浮點數格式
        if self._is_float(prediction_parsed):
            pass 
        else:
            # Regex 提取
            matches = re.search(r'(\d+)(?!.*\d)', prediction_parsed)
            if matches:
                prediction_parsed = matches.group(0)
        
        return prediction_parsed

    def _get_normalized_prediction(self, prediction: str, treat_as_number: bool, treat_as_bool: bool) -> str:
        """核心標準化函式"""
        prediction_parsed = prediction.lower().strip()

        # 1. 移除 'Answer is...' 等前綴
        patterns = FINAL_ANSWER_BEHIND_PATTERNS_PRIMARY if any(p in prediction for p in FINAL_ANSWER_BEHIND_PATTERNS_PRIMARY) else FINAL_ANSWER_BEHIND_PATTERNS_SECONDARY
        delimiters = patterns + [GSM8K_ANSWER_DELIMITER, 'answer:', 'result:'] # 包含 GSM8K 的 ####
        
        answer_indicated = False
        for d in delimiters:
            if d.lower() in prediction_parsed:
                prediction_parsed = prediction_parsed.split(d.lower())[-1]
                answer_indicated = True
        
        # 移除後綴 (is the correct answer)
        for d in FINAL_ANSWER_AHEAD_PATTERNS:
            if d.lower() in prediction_parsed:
                prediction_parsed = prediction_parsed.split(d.lower())[0]
                answer_indicated = True
        
        prediction_parsed = prediction_parsed.strip()
        
        # 移除句號
        while prediction_parsed and prediction_parsed.endswith('.'):
            prediction_parsed = prediction_parsed[:-1]

        # 2. [修改點] 針對選擇題 (MMLU) 使用新的 robust 提取邏輯
        # 如果不是數值題也不是布林題，就假設是選擇題
        if not treat_as_number and not treat_as_bool:
            choice = self._extract_choice_letter(prediction_parsed)
            if choice:
                prediction_parsed = choice
            # 如果回傳 None，則保留原字串 (可能是模型輸出了非預期格式，保留讓後續邏輯或空值處理)

        # 3. 根據類型解析
        if treat_as_bool:
             # 布林邏輯簡化
            bool_map = {'yes': 'true', 'no': 'false', 'valid': 'true', 'invalid': 'false'}
            # 移除標點
            prediction_parsed = prediction_parsed.translate(str.maketrans('', '', string.punctuation)).strip()
            return bool_map.get(prediction_parsed, prediction_parsed)

        if treat_as_number:
            return self._parse_with_treating_as_number(prediction_parsed)
        
        # 一般文字 (MMLU 已經在上方處理過 choice，這裡做最後確保)
        prediction_parsed = prediction_parsed.split('.')[0] # 取第一句或句號前
        return prediction_parsed

    def _normalize_target(self, target: str) -> str:
        """標準化正確答案 (Target)"""
        target = target.lower().strip()
        
        if GSM8K_ANSWER_DELIMITER in target:
            target = target.split(GSM8K_ANSWER_DELIMITER)[-1]
        
        if target.endswith('.'): target = target[:-1]
        
        if self.treat_as_number:
             target = target.replace(',', '')
             
        return target

    def _check_answer(self, prediction: str, target: str) -> float:
        """
        修正後的評分邏輯：嚴格比對 (Exact Match after Normalization)
        """
        # 1. 標準化 Prediction
        pred_norm = self._get_normalized_prediction(
            str(prediction), 
            treat_as_number=self.treat_as_number, 
            treat_as_bool=self.treat_as_bool
        )
        
        # 2. 標準化 Target
        target_norm = self._normalize_target(str(target))

        # 3. 比對邏輯
        if self.treat_as_number:
            try:
                if abs(float(pred_norm) - float(target_norm)) < 1e-6:
                    return 1.0
            except:
                pass 
        
        if pred_norm == target_norm:
            return 1.0

        # 處理括號殘留 (MMLU): 如果 pred 還是 '(a)' 而 target 是 'a'
        if pred_norm.replace('(', '').replace(')', '') == target_norm.replace('(', '').replace(')', ''):
            return 1.0
            
        return 0.0

    def score_instruction(self, instruction: str, dataset: list, num_samples: int = None) -> dict:
        import random
        
        eval_data = dataset
        if num_samples and num_samples < len(dataset):
            random.seed(0)
            eval_data = random.sample(dataset, num_samples)

        scores = []
        results_list = []
        
        def process_sample(example):
            prompt = self._format_prompt(instruction, example['input'])
            try:
                prediction = self.client.generate_text(prompt)
                acc = self._check_answer(prediction, example['target'])
                return {
                    'input': example['input'],
                    'target': example['target'],
                    'prediction': prediction,
                    'accuracy': acc
                }
            except Exception as e:
                logger.error(f"Scoring error: {e}")
                return None

        # 使用 ThreadPoolExecutor 開啟並發
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_sample = {executor.submit(process_sample, ex): ex for ex in eval_data}
            
            for future in tqdm(as_completed(future_to_sample), total=len(eval_data), desc="    Scoring (Parallel)", unit="sample", leave=False):
                res = future.result()
                if res:
                    scores.append(res['accuracy'])
                    results_list.append(res)
        
        avg_score = np.mean(scores) if scores else 0.0
        return {
            'score': float(avg_score), 
            'num_evals': len(scores),
            'detailed_dataframe': pd.DataFrame(results_list)
        }