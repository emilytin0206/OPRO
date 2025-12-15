import requests
import json
import time
import logging
from src.model.base_client import BaseModelClient

logger = logging.getLogger("OPRO")

class OllamaModelClient(BaseModelClient):
    def __init__(self, model_name: str, api_url: str, temperature: float, max_output_tokens: int, **kwargs):
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        
        # [核心邏輯] 解析 Base URL，以便動態組裝 /api/chat 或 /api/generate
        # 假設傳入 http://140.113.86.14:11434/api/chat 或 .../generate，我們先取出 http://...:11434
        if "/api/" in api_url:
            self.base_url = api_url.split("/api/")[0]
        else:
            self.base_url = api_url.rstrip("/")

    def _get_endpoint(self, endpoint_type: str) -> str:
        """根據需求動態取得 URL"""
        return f"{self.base_url}/api/{endpoint_type}"

    def generate_text(self, prompt: str) -> str:
        """Optimizer 使用: 呼叫 /api/generate"""
        url = self._get_endpoint("generate")
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_output_tokens
            }
        }
        return self._post_request(url, payload, response_key='response')

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Scorer 使用: 呼叫 /api/chat"""
        url = self._get_endpoint("chat")
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_output_tokens
            }
        }
        return self._post_request(url, payload, response_key='message')

    def _post_request(self, url, payload, response_key) -> str:
        max_retries = 3
        base_delay = 1
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                
                if response_key == 'message':
                    return data.get('message', {}).get('content', '').strip()
                else:
                    return data.get('response', '').strip()
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Ollama API 失敗 ({url}) - 嘗試 {attempt+1}: {e}")
                time.sleep(base_delay * (2 ** attempt))
        return ""

    def generate_multiple_texts(self, prompt: str, num_generations: int) -> list[str]:
        return [self.generate_text(prompt) for _ in range(num_generations)]