from typing import Protocol

class BaseModelClient(Protocol):
    """LLM 客戶端的抽象介面"""
    
    def generate_text(self, prompt: str) -> str:
        """根據提示詞生成文本"""
        ...

    def generate_multiple_texts(self, prompt: str, num_generations: int) -> list[str]:
        """生成多個不同的輸出 (用於評估多樣性或計算平均分數)"""
        ...