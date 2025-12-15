import re
import os
import random
import logging
# [修正] Import 路徑
from src.model.base_client import BaseModelClient

logger = logging.getLogger("OPRO")

class Optimizer:
    def __init__(self, model_client: BaseModelClient, config):
        self.client = model_client
        self.config = config
        self.template_content = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        # 讀取 config 設定的路徑
        path = getattr(self.config, 'meta_prompt_path', 'prompt/meta_prompt.txt')
        
        if not os.path.exists(path):
            logger.warning(f"找不到 Prompt 模板檔案: {path}，使用預設模板。")
            return "Your task is to generate the instruction <INS>.\n{few_shot_examples}\n{history}\nNew Instruction:"
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"讀取模板檔案失敗: {e}")
            raise e

    def _bucketize_score(self, score: float, num_buckets: int = 100) -> int:
        return round(score * num_buckets)

    def _format_history_string(self, history: list) -> str:
        """將歷史記錄格式化"""
        # 按分數升序排列 (低 -> 高)
        sorted_history = sorted(history, key=lambda x: x['score'])
        max_num = getattr(self.config, 'max_num_instructions_in_prompt', 20)
        selected_history = sorted_history[-max_num:]
        
        history_str = ""
        for item in selected_history:
            score_val = self._bucketize_score(item['score'])
            inst_text = item['instruction']
            history_str += f"text:\n{inst_text}\nscore:\n{score_val}\n\n"
        return history_str.strip()

    # [新增] 錯誤驅動選題邏輯
    def _format_few_shot_examples(self, dataset: list, wrong_questions_counter: dict = None) -> str:
        num_shots = getattr(self.config, 'num_few_shot_questions', 3)
        criteria = getattr(self.config, 'few_shot_selection_criteria', 'random')
        
        selected_data = []
        
        if criteria == 'accumulative_most_frequent' and wrong_questions_counter:
            # 優先挑錯題
            most_common = wrong_questions_counter.most_common()
            input_to_data = {d['input']: d for d in dataset}
            
            for q_input, _ in most_common:
                if len(selected_data) >= num_shots: break
                if q_input in input_to_data:
                    selected_data.append(input_to_data[q_input])
        
        # 補滿不足的題數 (隨機)
        if len(selected_data) < num_shots:
            remaining = [d for d in dataset if d not in selected_data]
            if remaining:
                selected_data.extend(random.sample(remaining, min(len(remaining), num_shots - len(selected_data))))
        
        # 格式化
        ex_str = ""
        for i, d in enumerate(selected_data):
            ex_str += f"Problem {i+1}:\nQ: {d['input']}\nA: {d['target']}\n\n"
        return ex_str

    def _build_meta_prompt(self, history: list, dataset: list, wrong_questions_counter: dict = None) -> str:
        history_str = self._format_history_string(history)
        # [新增] 只有當 dataset 存在時才生成範例
        examples_str = self._format_few_shot_examples(dataset, wrong_questions_counter) if dataset else ""
        
        prompt = self.template_content
        # 替換佔位符
        if "{few_shot_examples}" in prompt:
            prompt = prompt.replace("{few_shot_examples}", examples_str)
        if "{history}" in prompt:
            prompt = prompt.replace("{history}", history_str)
            
        return prompt

    def generate_new_instructions(self, history: list, dataset: list = None, wrong_questions_counter: dict = None) -> list:
        # [修正] 接收 dataset 和 counter
        meta_prompt = self._build_meta_prompt(history, dataset, wrong_questions_counter)
        
        num_prompts = getattr(self.config, 'num_prompts_to_generate', 4)
        new_instructions = []
        
        for _ in range(num_prompts):
            raw_output = self.client.generate_text(meta_prompt)
            parsed = self._extract_instruction(raw_output)
            if parsed:
                new_instructions.append(parsed)
                
        return new_instructions

    def _extract_instruction(self, text: str) -> str:
        # 簡單的提取邏輯
        match = re.search(r"<INS>(.*?)</INS>", text, re.DOTALL)
        if match: return match.group(1).strip()
        if text.startswith('"') and text.endswith('"'): return text.strip('"')
        if len(text) < 300 and "text:" not in text: return text.strip()
        return ""