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
        
        # 用於統計 Token
        self.usage_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "call_count": 0
        }
        
        # 解析 Base URL
        if "/api/" in api_url:
            self.base_url = api_url.split("/api/")[0]
        else:
            self.base_url = api_url.rstrip("/")

    def _get_endpoint(self, endpoint_type: str) -> str:
        return f"{self.base_url}/api/{endpoint_type}"

    def generate_text(self, prompt: str) -> str:
        """
        [修正] 統一改用 Chat 介面來處理，並增加 timeout。
        Scorer 和 Optimizer 雖然呼叫此函式，但我們可以內部轉成 Chat 格式。
        """
        # 1. 改用 chat endpoint
        url = self._get_endpoint("chat") 
        
        # 2. 修正 Payload：將 prompt 包裝成 user message
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_output_tokens
            }
        }
        # 3. 呼叫 chat 專用的解析邏輯
        return self._post_request(url, payload, response_key='message')

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Scorer 若有使用此函式，也需確保 timeout 足夠"""
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
        
        # [關鍵修正] 將 Timeout 從 60 增加到 300 秒 (5分鐘)
        # 對於 MMLU 長思考或較慢的 GPU，60秒通常不夠。
        timeout_seconds = 300 
        
        for attempt in range(max_retries):
            try:
                # 這裡加入 timeout 參數
                response = requests.post(url, json=payload, timeout=timeout_seconds)
                response.raise_for_status()
                data = response.json()
                
                # Token Cost 統計
                input_tokens = data.get('prompt_eval_count', 0)
                output_tokens = data.get('eval_count', 0)
                
                self.usage_stats["prompt_tokens"] += input_tokens
                self.usage_stats["completion_tokens"] += output_tokens
                self.usage_stats["total_tokens"] += (input_tokens + output_tokens)
                self.usage_stats["call_count"] += 1

                # 解析回傳值
                if response_key == 'message':
                    # Chat 介面回傳結構: data['message']['content']
                    return data.get('message', {}).get('content', '').strip()
                else:
                    # Generate 介面回傳結構: data['response']
                    return data.get('response', '').strip()
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Ollama API 失敗 ({url}) - 嘗試 {attempt+1}: {e}")
                time.sleep(base_delay * (2 ** attempt))
        return ""

    def generate_multiple_texts(self, prompt: str, num_generations: int) -> list[str]:
        return [self.generate_text(prompt) for _ in range(num_generations)]