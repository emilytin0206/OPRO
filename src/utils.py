import os
import pandas as pd
import json
import re
import string
import hashlib # [新增] 用於雜湊檔名

def load_dataset(dataset_name: str, task_name: str, data_root: str):
    """載入指定資料集 (MMLU, BBH, GSM8K)"""
    dataset_name = dataset_name.lower()
    raw_data = []

    if dataset_name == "gsm8k":
        file_path = os.path.join(data_root, "gsm_data", f"gsm_{task_name}.tsv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, sep="\t", header=None)
            for _, row in df.iterrows():
                raw_data.append({'input': row[0], 'target': row[1]})
        else:
            raise FileNotFoundError(f"找不到 GSM8K 數據: {file_path}")

    elif dataset_name == "bbh":
        file_path = os.path.join(data_root, "BIG-Bench-Hard-data", f"{task_name}.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding='utf-8') as f:
                data = json.load(f)["examples"]
                for d in data:
                     raw_data.append({'input': d['input'], 'target': d['target']})
        else:
            raise FileNotFoundError(f"找不到 BBH 數據: {file_path}")

    # (可擴充 MMLU 等)
    
    return raw_data

def parse_tag_content(text, prefix="<INS>", suffix="</INS>"):
    pattern = f"{prefix}(.*?){suffix}"
    results = re.findall(pattern, text, re.DOTALL)
    return [r.strip() for r in results]

def instruction_to_filename(instruction):
    """將指令轉為安全檔名 (MD5)"""
    m = hashlib.md5()
    m.update(instruction.encode('utf-8'))
    return m.hexdigest()

# [新增] 對應 Google DeepMind 的 polish_sentence
def polish_instruction(instruction: str) -> str:
    """標準化指令格式"""
    instruction = instruction.strip()
    if not instruction:
        return ""
    # 移除 Markdown 強調符號
    instruction = instruction.replace("**", "")
    # 首字大寫
    if len(instruction) > 1:
        instruction = instruction[0].upper() + instruction[1:]
    # 確保句尾有標點 (DeepMind 邏輯)
    if instruction and instruction[-1] not in ".?!":
        instruction += "."
    return instruction

def setup_logger(log_dir: str, task_name: str):
    """設定 Logger"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logger = logging.getLogger("OPRO")
    logger.setLevel(logging.INFO)
    logger.handlers = [] # 清除舊 handler 避免重複打印
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File Handler
    log_file = os.path.join(log_dir, f"{task_name}.log")
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger, log_file