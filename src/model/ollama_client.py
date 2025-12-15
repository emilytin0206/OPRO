import requests
import json
import time # [新增]
import logging
from src.model.base_client import BaseModelClient

logger = logging.getLogger("OPRO")

class OllamaModelClient(BaseModelClient):
    """使用 Ollama API 進行互動的客戶端"""

    def __init__(self, model_name: str, api_url: str, temperature: float, max_output_tokens: int, **kwargs):
        self.model_name = model_name
        self.api_url = api_url
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def _call_ollama_api(self, prompt: str, num_generations: int = 1) -> list[str]:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_output_tokens
            }
        }
        
        # [新增] 重試機制設定
        max_retries = 3
        base_delay = 1
        
        results = []
        for _ in range(num_generations):
            result = ""
            for attempt in range(max_retries):
                try:
                    response = requests.post(self.api_url, json=payload, timeout=60) # 加入 timeout
                    response.raise_for_status()
                    
                    data = response.json()
                    result = data.get('response', '').strip()
                    break # 成功則跳出重試迴圈
                    
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Ollama API 呼叫失敗 (嘗試 {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        sleep_time = base_delay * (2 ** attempt) # 指數退避
                        time.sleep(sleep_time)
                    else:
                        logger.error("API 重試次數已達上限，回傳空字串。")
            
            results.append(result)
            
        return results

    def generate_text(self, prompt: str) -> str:
        results = self._call_ollama_api(prompt, num_generations=1)
        return results[0] if results else ""

    def generate_multiple_texts(self, prompt: str, num_generations: int) -> list[str]:
        return self._call_ollama_api(prompt, num_generations=num_generations)