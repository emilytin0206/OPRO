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

import logging
import os
import sys


def setup_logger(log_dir: str, task_name: str):
    """
    設定 Logger，同時輸出到檔案與螢幕。
    檔名範例: ./logs/BBH-boolean_expressions_20231027-103000.log
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 產生帶時間戳的 Log 檔名
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filename = f"{task_name}_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    # 設定 Logging 格式
    logger = logging.getLogger("OPRO")
    logger.setLevel(logging.INFO)
    
    # 防止重複添加 Handler (在 Notebook 或多次呼叫時)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 1. File Handler (寫入檔案)
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. Stream Handler (輸出到終端機)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info(f"Logger initialized. Log file: {log_filepath}")
    
    return logger, log_filepath


def parse_tag_content(text, prefix="<INS>", suffix="</INS>"):
    """從文本中提取標籤內容，用於提取優化後的指令"""
    pattern = f"{prefix}(.*?){suffix}"
    results = re.findall(pattern, text, re.DOTALL)
    return [r.strip() for r in results]

def instruction_to_filename(instruction: str) -> str:
    """
    將指令轉換為唯一的 MD5 雜湊檔名。
    這是實現快取的關鍵，因為指令本身可能包含無法作為檔名的特殊字元。
    """
    # 統一編碼並雜湊
    m = hashlib.md5()
    m.update(instruction.encode('utf-8'))
    return m.hexdigest()

def polish_instruction(instruction: str) -> str:
    """
    簡單的後處理：移除前後空白、多餘的換行。
    (對應 DeepMind 的 polish_sentence)
    """
    return instruction.strip()