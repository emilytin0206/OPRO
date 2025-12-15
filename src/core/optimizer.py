import re
import os
import random
import logging
# [修正] Import 路徑
from src.model.base_client import BaseModelClient

logger = logging.getLogger("OPRO")

class Optimizer:
    def __init__(self, model_client, config):
        self.client = model_client
        self.config = config
        self.instructions_before_exemplars = getattr(config, 'meta_prompt_instructions_before_exemplars', True)

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
        """將歷史記錄格式化為字串"""
        # [新增] 讀取分數門檻，預設 0.1 (可於 config 設定 old_instruction_score_threshold)
        score_threshold = getattr(self.config, 'old_instruction_score_threshold', 0.1)
        
        # 1. 過濾低分指令
        valid_history = [h for h in history if h['score'] >= score_threshold]
        # 2. 排序：按分數由低到高
        sorted_history = sorted(valid_history, key=lambda x: x['score'])
        # 3. 取最近 N 筆
        max_num = getattr(self.config, 'max_num_instructions_in_prompt', 20)
        selected_history = sorted_history[-max_num:]
        # 4. 組合字串
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
        # 1. 準備內容元件
        # Instructions History
        history_str = self._format_history_string(history)
        
        # Exemplars (Few-shot questions)
        examples_str = self._format_few_shot_examples(dataset, wrong_questions_counter)
        
        # 2. 載入並填充模板
        # [修正] 改為使用 _load_prompt_template() 讀取檔案，而非硬編碼
        try:
            template = self._load_prompt_template()
            
            # 檢查模板中是否包含必要的佔位符
            if "{history}" not in template or "{few_shot_examples}" not in template:
                logger.warning("Meta-prompt 模板缺少 {history} 或 {few_shot_examples} 佔位符，將回退至預設邏輯。")
                raise ValueError("Invalid Template")

            # 填充內容
            meta_prompt = template.format(
                history=history_str,
                few_shot_examples=examples_str
            )
            return meta_prompt

        except Exception as e:
            # [Fallback] 如果讀取失敗或模板格式錯誤，使用原本的硬編碼邏輯 (但建議修正以符合論文)
            logger.warning(f"使用模板生成 Prompt 失敗 ({e})，使用預設硬編碼格式。")
            
            intro = "Your task is to generate the instruction <INS> for solving the following type of problems."
            range_desc = "The score ranges from 0 to 100."
            footer = (
                "Generate an instruction that is different from all the instructions <INS> above, "
                "and has a higher score than all the instructions <INS> above.\n"
                "The instruction should begin with <INS> and end with </INS>.\n"
                "The instruction should be concise, effective, and generally applicable to all problems above.\n"
                "New Instruction:"
            )
            
            block_history = f"Below are some previous instructions with their scores.\n{range_desc}\n{history_str}"
            block_examples = f"Here are some examples of the problems:\n{examples_str}"
            
            # 根據 config 決定順序 (若使用模板，順序由模板內的 {history} 位置決定)
            if self.instructions_before_exemplars:
                return f"{intro}\n\n{block_history}\n\n{block_examples}\n\n{footer}"
            else:
                return f"{intro}\n\n{block_examples}\n\n{block_history}\n\n{footer}"

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