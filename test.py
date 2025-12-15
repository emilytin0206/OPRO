import requests
import json
import time

# 設定目標 URL
url = "http://140.113.86.14:11434/api/chat"

# 請求內容
payload = {
    "model": "qwen2.5:14b",
    "messages": [{"role": "user", "content": "Hello, simply reply 'OK'."}],
    "stream": False,
    "options": {
        "num_predict": 10 # 限制輸出長度，加快測試回應
    }
}

print(f"Connecting to {url}...")
print("請耐心等待，如果模型正在載入，可能需要 30-60 秒...")

start_time = time.time()

try:
    # 修改：將 timeout 設定為 120 秒
    response = requests.post(url, json=payload, timeout=120)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        content = data.get('message', {}).get('content', '') or data.get('response', '')
        print(f"Response ({duration:.2f}s): {content}")
        print("測試成功！連線與模型皆正常。")
    else:
        print("Error response:", response.text)

except requests.exceptions.ReadTimeout:
    print(f"失敗：伺服器超過 120 秒仍未回應。可能模型過大或伺服器卡住。")
except Exception as e:
    print(f"Connection failed: {e}")