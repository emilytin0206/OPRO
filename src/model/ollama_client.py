import requests
import json
# [修正] Import 路徑
from src.model.base_client import BaseModelClient

class OllamaModelClient(BaseModelClient):
    """使用 Ollama API 進行互動的客戶端"""

    def __init__(self, model_name: str, api_url: str, temperature: float, max_output_tokens: int, **kwargs):
        self.model_name = model_name
        self.api_url = api_url
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def _call_ollama_api(self, prompt: str, num_generations: int = 1) -> list[str]:
        # 這裡的 payload 結構視您的 Ollama 版本而定
        # 如果是較新版 Ollama，建議使用 /api/generate
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_output_tokens
            }
        }
        
        try:
            # 發送請求
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            generated_text = data.get('response', '').strip()
            
            # 因為 Ollama API 一次只回傳一個，若需要多個生成，需在外部迴圈呼叫
            # 這裡簡單處理：若 num_generations > 1，遞迴呼叫 (效率較低但邏輯正確)
            if num_generations > 1:
                return [generated_text] + self._call_ollama_api(prompt, num_generations - 1)
            
            return [generated_text]

        except requests.exceptions.RequestException as e:
            print(f"Ollama API 呼叫失敗: {e}")
            return [""] * num_generations

    def generate_text(self, prompt: str) -> str:
        results = self._call_ollama_api(prompt, num_generations=1)
        return results[0] if results else ""

    def generate_multiple_texts(self, prompt: str, num_generations: int) -> list[str]:
        return self._call_ollama_api(prompt, num_generations=num_generations)