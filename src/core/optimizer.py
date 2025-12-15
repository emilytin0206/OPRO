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

    def _build_meta_prompt(self, history: list) -> str:
        """使用模板構建最終 Prompt"""
        # 取得格式化後的歷史記錄字串
        history_string = self._format_history_string(history)
        
        # 將歷史記錄填入模板中的 {history} 佔位符
        # 使用 replace 而非 format，避免 Prompt 本身包含 {} 導致錯誤
        meta_prompt = self.template_content.replace("{history}", history_string)
        
        return meta_prompt

    def generate_new_instructions(self, history: list) -> list:
        """
        生成新指令的主函式。
        """
        meta_prompt = self._build_meta_prompt(history)
        
        # logger.debug(f"Current Meta-Prompt:\n{meta_prompt}") # 除錯用
        
        num_prompts = getattr(self.config, 'num_prompts_to_generate', 4)
        new_instructions = []
        
        for _ in range(num_prompts):
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