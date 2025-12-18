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

    # --- 以下為移植自 Google metrics.py 的核心邏輯 ---

    def _is_float(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _extract_bracketed_choice_from_string(self, prediction):
        """處理 (A) -> A 的轉換，這對 MMLU 至關重要"""
        prediction = prediction.lower()
        all_letters = string.ascii_lowercase
        bracketed_letters_list = set([f'({l})' for l in all_letters])
        
        choice_in_pred_all = [item in prediction for item in bracketed_letters_list]
        if sum(choice_in_pred_all) == 1:
            # 如果只出現一個括號選項，提取它
            prediction = re.findall(r'\(.*?\)', prediction)[0]
        return prediction

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
            # GSM8K 預設整數比對，這裡可保留小數位以防萬一
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

        # 2. 處理 MMLU 的括號選項 (A) -> (a)
        prediction_parsed = self._extract_bracketed_choice_from_string(prediction_parsed)

        # 3. 根據類型解析
        if treat_as_bool:
             # 布林邏輯簡化
            bool_map = {'yes': 'true', 'no': 'false', 'valid': 'true', 'invalid': 'false'}
            # 移除標點
            prediction_parsed = prediction_parsed.translate(str.maketrans('', '', string.punctuation)).strip()
            return bool_map.get(prediction_parsed, prediction_parsed)

        if treat_as_number:
            return self._parse_with_treating_as_number(prediction_parsed)
        
        # 一般文字 (MMLU)
        prediction_parsed = prediction_parsed.split('.')[0] # 取第一句或句號前
        return prediction_parsed

    def _normalize_target(self, target: str) -> str:
        """標準化正確答案 (Target)"""
        target = target.lower().strip()
        # 簡單清理，通常 Dataset 裡的 target 已經很乾淨 (如 'A', '20')
        # 但如果是 GSM8K，target 可能包含 ####
        if GSM8K_ANSWER_DELIMITER in target:
            target = target.split(GSM8K_ANSWER_DELIMITER)[-1]
        
        # 移除句號
        if target.endswith('.'): target = target[:-1]
        
        if self.treat_as_number:
            # 處理 target 中的逗號 (1,000 -> 1000)
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
            # 數值比對 (允許誤差)
            try:
                if abs(float(pred_norm) - float(target_norm)) < 1e-6:
                    return 1.0
            except:
                pass # 轉換失敗，退回字串比對
        
        # 字串精確比對 (MMLU / Boolean / 數值Fallback)
        # 注意：這裡不再使用 Regex Search，而是相等比對，避免 False Positive
        if pred_norm == target_norm:
            return 1.0

        # Google 原版針對 MMLU 還有一個括號處理: '(a)' vs 'a'
        # 如果 target 是 'a' 但 pred 是 '(a)' (或反之)，也算對
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
        
        # 定義單個樣本的處理函式
        def process_sample(example):
            prompt = self._format_prompt(instruction, example['input'])
            try:
                # 這裡會並發呼叫，讓 Ollama 同時處理多個
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

        # [關鍵修改] 使用 ThreadPoolExecutor 開啟並發
        # max_workers 建議設為 4 ~ 16，取決於您的顯存大小 (7B 模型通常可以開 8 或 16)
        with ThreadPoolExecutor(max_workers=4) as executor:
            # 提交所有任務
            future_to_sample = {executor.submit(process_sample, ex): ex for ex in eval_data}
            
            # 使用 tqdm 顯示並發進度
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