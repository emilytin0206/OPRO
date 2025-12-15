from src.models.base_client import BaseModelClient

class Optimizer:
    def __init__(self, model_client: BaseModelClient, config):
        self.client = model_client
        self.config = config

    def _build_meta_prompt(self, history: list) -> str:
        """
        構建 Meta-Prompt。
        history: list of dict {'instruction': str, 'score': float}
        """
        # 根據分數排序，取最好的幾個作為範例
        sorted_history = sorted(history, key=lambda x: x['score'], reverse=True)
        # 限制顯示的舊指令數量
        top_k_history = sorted_history[:20] 
        # 按照分數由低到高排列顯示 (這是 OPRO 論文中的建議，讓模型看到進步軌跡)
        top_k_history.reverse()

        meta_prompt = (
            "Your task is to generate a new instruction <INS> that helps a language model solve tasks more accurately.\n"
            "Below are some previous instructions with their corresponding accuracy scores (0 to 1).\n\n"
        )

        for item in top_k_history:
            meta_prompt += f"Instruction: <INS>{item['instruction']}</INS>\n"
            meta_prompt += f"Score: {item['score']:.4f}\n\n"

        meta_prompt += (
            "Generate a new instruction that is different from the old ones and likely to achieve a higher score.\n"
            "Wrap your new instruction in <INS> and </INS> tags.\n"
            "New Instruction:"
        )
        return meta_prompt

    def generate_new_instructions(self, history: list) -> list:
        """
        生成一批新指令。
        """
        meta_prompt = self._build_meta_prompt(history)
        
        # 呼叫 LLM 生成
        # 如果需要生成多個，可以呼叫多次 generate_text 或依賴 client 的 batch 能力
        raw_output = self.client.generate_text(meta_prompt)
        
        # 解析輸出 (提取 <INS>...</INS>)
        import re
        new_instructions = re.findall(r"<INS>(.*?)</INS>", raw_output, re.DOTALL)
        
        # 如果模型沒有遵循格式，嘗試直接取用整段文本或做後處理
        if not new_instructions:
            # Fallback: 假設模型直接輸出了指令
            new_instructions = [raw_output.strip()]
            
        return [inst.strip() for inst in new_instructions if inst.strip()]