import os
import pandas as pd
import json
import re
import string

def load_dataset(dataset_name: str, task_name: str, data_root: str):
    """
    載入指定資料集 (MMLU, BBH, GSM8K)。
    回傳格式統一為 list of dict: [{'input': '...', 'target': '...'}]
    """
    dataset_name = dataset_name.lower()
    raw_data = []

    if dataset_name == "gsm8k":
        # GSM8K 處理邏輯
        file_path = os.path.join(data_root, "gsm_data", f"gsm_{task_name}.tsv")
        df = pd.read_csv(file_path, sep="\t", header=None)
        # 轉換為標準格式
        for _, row in df.iterrows():
            raw_data.append({'input': row[0], 'target': row[1]})

    elif dataset_name == "bbh":
        # BBH 處理邏輯
        file_path = os.path.join(data_root, "BIG-Bench-Hard-data", f"{task_name}.json")
        with open(file_path, "r") as f:
            data = json.load(f)["examples"]
            for d in data:
                 raw_data.append({'input': d['input'], 'target': d['target']})

    # ... 可擴充 MMLU 等其他資料集邏輯 ...
    
    return raw_data

def parse_tag_content(text, prefix="<INS>", suffix="</INS>"):
    """從文本中提取標籤內容，用於提取優化後的指令"""
    pattern = f"{prefix}(.*?){suffix}"
    results = re.findall(pattern, text, re.DOTALL)
    return [r.strip() for r in results]

def instruction_to_filename(instruction):
    """將指令轉為安全檔名 (用於快取)"""
    # 簡單的雜湊或去除特殊字元
    return re.sub(r'[^a-zA-Z0-9]', '_', instruction[:30])