import re
import os
import logging
from src.models.base_client import BaseModelClient

logger = logging.getLogger("OPRO")

class Optimizer:
    def __init__(self, model_client: BaseModelClient, config):
        self.client = model_client
        self.config = config
        self.template_content = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """從檔案讀取 Meta-Prompt 模板"""
        # 從 config 取得路徑，若無則使用預設值
        path = getattr(self.config, 'meta_prompt_path', 'prompts/meta_prompt.txt')
        
        if not os.path.exists(path):
            logger.warning(f"找不到 Prompt 模板檔案: {path}，將使用預設內建模板。")
            # Fallback: 內建的預設模板
            return (
                "Your task is to generate the instruction <INS>.\n"
                "Below are some previous instructions with their scores (0-100).\n\n"
                "{history}\n\n"
                "Generate a new instruction <INS> that is different and has a higher score.\n"
                "New Instruction:"
            )
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"讀取模板檔案失敗: {e}")
            raise e

    def _bucketize_score(self, score: float, num_buckets: int = 100) -> int:
        return round(score * num_buckets)

    def _format_few_shot_examples(self, dataset: list, wrong_questions_counter: dict = None) -> str:
        """
        根據策略 (Random vs Error-Driven) 選擇 Few-Shot 範例
        """
        num_shots = getattr(self.config, 'num_few_shot_questions', 3)
        selection_criteria = getattr(self.config, 'few_shot_selection_criteria', 'random')
        
        selected_data = []
        
        # 策略：累積最常錯 (Accumulative Most Frequent)
        if selection_criteria == 'accumulative_most_frequent' and wrong_questions_counter:
            # 1. 取出錯誤次數最多的題目 (由高到低排序)
            most_common_errors = wrong_questions_counter.most_common()
            
            # 2. 建立 lookup table (Input -> Example Dict) 加速查找
            # (若 dataset 很大，建議在 init 時就建好這張表)
            input_to_example = {item['input']: item for item in dataset}
            
            # 3. 依序挑選
            for input_text, count in most_common_errors:
                if len(selected_data) >= num_shots:
                    break
                
                if input_text in input_to_example:
                    selected_data.append(input_to_example[input_text])
            
            # 4. 如果「錯題」不夠湊滿 num_shots (例如剛開始跑，大家都還沒錯)，用隨機補滿
            if len(selected_data) < num_shots:
                remaining_count = num_shots - len(selected_data)
                candidates = [d for d in dataset if d not in selected_data]
                if candidates:
                    padding = random.sample(candidates, min(len(candidates), remaining_count))
                    selected_data.extend(padding)
                    
        else:
            # 策略：完全隨機 (Random)
            selected_data = random.sample(dataset, min(len(dataset), num_shots))
            
        # 格式化輸出字串
        examples_str = ""
        for i, item in enumerate(selected_data):
            examples_str += f"Problem {i+1}:\nQ: {item['input']}\nA: {item['target']}\n\n"
            
        return examples_str.strip()

    def _format_history_string(self, history: list) -> str:
        """將歷史記錄格式化為字串，準備填入模板"""
        # 1. 排序：按分數由低到高
        sorted_history = sorted(history, key=lambda x: x['score'])
        
        # 2. 取最近 N 筆
        max_num = getattr(self.config, 'max_num_instructions_in_prompt', 20)
        selected_history = sorted_history[-max_num:]
        
        # 3. 組合字串
        history_str = ""
        for item in selected_history:
            score_val = self._bucketize_score(item['score'])
            inst_text = item['instruction']
            history_str += f"text:\n{inst_text}\nscore:\n{score_val}\n\n"
            
        return history_str.strip()

    def _build_meta_prompt(self, history: list, dataset: list, wrong_questions_counter: dict = None) -> str:
        # 傳入 counter
        history_string = self._format_history_string(history)
        examples_string = self._format_few_shot_examples(dataset, wrong_questions_counter)
        
        meta_prompt = self.template_content
        if "{few_shot_examples}" in meta_prompt:
            meta_prompt = meta_prompt.replace("{few_shot_examples}", examples_string)
        meta_prompt = meta_prompt.replace("{history}", history_string)
        
        return meta_prompt

    # 修改函式簽名，加入 dataset 參數
    def generate_new_instructions(self, history: list, dataset: list, wrong_questions_counter: dict = None) -> list:
        """
        生成新指令
        Args:
            history: 歷史指令列表
            dataset: 原始資料集 (用於動態採樣)
        """
        # 傳入 dataset 進行構建
        meta_prompt = self._build_meta_prompt(history, dataset, wrong_questions_counter)
        # logger.debug(f"Meta-Prompt with Examples:\n{meta_prompt[:500]}...") 
        
        num_prompts = getattr(self.config, 'num_prompts_to_generate', 4)
        new_instructions = []
        
        for _ in range(num_prompts):
            # 注意：如果希望每次生成的範例都不同，應該把 _build_meta_prompt 移到迴圈內
            # 但通常同一批次使用相同範例即可，這裡維持在迴圈外
            raw_output = self.client.generate_text(meta_prompt)
            parsed_inst = self._extract_instruction(raw_output)
            if parsed_inst:
                new_instructions.append(parsed_inst)
        
        return new_instructions

    def _extract_instruction(self, text: str) -> str:
        """從 LLM 輸出中提取指令"""
        # 嘗試匹配 <INS>content</INS>
        match = re.search(r"<INS>(.*?)</INS>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # 嘗試匹配 <Start>content</Start> (有些模型會混淆)
        match = re.search(r"<Start>(.*?)</Start>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
            
        # 如果模型只輸出了指令本身 (常見於小參數模型)，但包含引號
        if text.startswith('"') and text.endswith('"'):
            return text.strip('"')

        # 如果找不到標籤，檢查是否包含 "text:" 或 "score:" 這種幻覺，將其過濾
        if "text:" in text or "score:" in text:
            return "" # 放棄這個無效生成

        # Fallback: 如果輸出很短，可能整句就是指令
        if len(text) < 300: 
            return text.strip()
            
        return ""