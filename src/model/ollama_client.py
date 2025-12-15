import requests
import json
from src.models.base_client import BaseModelClient

class OllamaModelClient(BaseModelClient):
    """使用 Ollama API 進行互動的客戶端"""

    def __init__(self, model_name: str, api_url: str, temperature: float, max_output_tokens: int):
        self.model_name = model_name
        self.api_url = api_url
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def _call_ollama_api(self, prompt: str, num_generations: int = 1) -> list[str]:
        """實際呼叫 Ollama API 的函式"""
        # 由於 Ollama API 預設是 streaming, 且沒有內建的 num_generations,
        # 我們需要手動進行多次呼叫或調整 payload。
        
        # 這裡簡化為單次呼叫, 如果需要多次生成, 則在外部循環呼叫 generate_text
        
        # 完整的 Ollama API payload
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                # Ollama 使用 num_predict 作為 max_output_tokens
                "num_predict": self.max_output_tokens
            }
        }
        
        try:
            # 假設 API URL 是 /api/generate
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status() # 檢查 HTTP 錯誤
            
            data = response.json()
            # Ollama 的回應結構中, 'response' 包含生成的文本
            generated_text = data.get('response', '').strip()
            
            if num_generations > 1:
                # 簡單實現：對於多個生成請求, 呼叫多次單次生成
                results = [generated_text] + [
                    self.generate_text(prompt) for _ in range(num_generations - 1)
                ]
                return results
            
            return [generated_text]

        except requests.exceptions.RequestException as e:
            print(f"Ollama API 呼叫失敗: {e}")
            return [""] * num_generations

    def generate_text(self, prompt: str) -> str:
        """實現 BaseModelClient 的單次生成"""
        results = self._call_ollama_api(prompt, num_generations=1)
        return results[0] if results else ""

    def generate_multiple_texts(self, prompt: str, num_generations: int) -> list[str]:
        """實現 BaseModelClient 的多次生成"""
        # 注意: 這裡將依賴 _call_ollama_api 內部實現多次呼叫 (或您決定在外部呼叫)
        return self._call_ollama_api(prompt, num_generations=num_generations)