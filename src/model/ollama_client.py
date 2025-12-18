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
        timeout_seconds = 400 
        
        for attempt in range(max_retries):
            try:
                # --- [DEBUG 開始] ---
                import time
                start_ts = time.time()
                # 印出請求當下的時間戳，觀察是否有「並發」 (若時間戳幾乎相同就是並發)
                print(f"[DEBUG] >>> 發送請求 ({start_ts:.2f})", flush=True) 
                
                response = requests.post(url, json=payload, timeout=timeout_seconds)
                
                end_ts = time.time()
                duration = end_ts - start_ts
                # -------------------

                response.raise_for_status()
                data = response.json()
                
                # 統計 Token
                input_tokens = data.get('prompt_eval_count', 0)
                output_tokens = data.get('eval_count', 0)
                self.usage_stats["prompt_tokens"] += input_tokens
                self.usage_stats["completion_tokens"] += output_tokens
                self.usage_stats["total_tokens"] += (input_tokens + output_tokens)
                self.usage_stats["call_count"] += 1

                # 解析內容
                content = ""
                if response_key == 'message':
                    content = data.get('message', {}).get('content', '').strip()
                else:
                    content = data.get('response', '').strip()

                # --- [DEBUG 結束與診斷] ---
                # 1. 檢查耗時：如果這裡顯示 0.1秒，那瓶頸在 Python；如果顯示 10秒，那瓶頸在 GPU
                # 2. 檢查內容：如果回覆了幾千字廢話，那瓶頸在 Max Tokens 設定
                print(f"[DEBUG] <<< 收到回覆! 耗時: {duration:.4f}秒 | 產出Token: {output_tokens} | 內容片段: {content[:30].replace(chr(10), ' ')}...", flush=True)
                # ------------------------

                return content
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Ollama API 失敗 ({url}) - 嘗試 {attempt+1}: {e}")
                time.sleep(base_delay * (2 ** attempt))
        return ""

    def generate_multiple_texts(self, prompt: str, num_generations: int) -> list[str]:
        return [self.generate_text(prompt) for _ in range(num_generations)]