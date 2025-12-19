# src/utils.py
import os
import pandas as pd
import json
import re
import hashlib
import glob

def load_dataset(dataset_cfg):
    """
    根據 config.dataset 的設定載入資料。
    """
    name = dataset_cfg.name.lower()
    split = dataset_cfg.split.lower() # 'train' or 'test'
    data_root = dataset_cfg.data_root
    raw_data = []

    print(f"正在載入資料集: {name} (Split: {split})")

    if name == "mmlu":
        # MMLU 處理邏輯
        mmlu_root = os.path.join(data_root, "mmlu")
        
        # 決定要載入哪些檔案
        target_subsets = dataset_cfg.subsets
        
        # 處理 'all' 的情況：自動搜尋目錄下所有該 split 的 csv
        if not target_subsets or (isinstance(target_subsets, str) and target_subsets.lower() == 'all'):
            print("  偵測到 subsets='all'，正在搜尋所有 CSV...")
            # 搜尋模式: data/mmlu/*_test.csv 或 data/mmlu/test/*_test.csv
            search_patterns = [
                os.path.join(mmlu_root, f"*_{split}.csv"),
                os.path.join(mmlu_root, split, f"*_{split}.csv")
            ]
            found_files = []
            for p in search_patterns:
                found_files.extend(glob.glob(p))
            
            # 從檔名解析出 subset 名稱 (用於顯示)
            files_to_load = found_files
            print(f"  共找到 {len(files_to_load)} 個 MMLU 子集檔案。")
        else:
            # 使用者指定列表
            if isinstance(target_subsets, str): target_subsets = [target_subsets]
            files_to_load = []
            for sub in target_subsets:
                # 嘗試多種路徑組合
                paths = [
                    os.path.join(mmlu_root, f"{sub}_{split}.csv"),
                    os.path.join(mmlu_root, split, f"{sub}_{split}.csv")
                ]
                found = False
                for p in paths:
                    if os.path.exists(p):
                        files_to_load.append(p)
                        found = True
                        break
                if not found:
                    print(f"  [警告] 找不到子集檔案: {sub} ({split})")

        # 開始讀取
        for file_path in files_to_load:
            try:
                # MMLU CSV 無 header: [Question, A, B, C, D, Answer]
                df = pd.read_csv(file_path, header=None)
                subset_name = os.path.basename(file_path).replace(f"_{split}.csv", "")
                
                for _, row in df.iterrows():
                    question = str(row[0])
                    options = f"(A) {str(row[1])}\n(B) {str(row[2])}\n(C) {str(row[3])}\n(D) {str(row[4])}"
                    full_input = f"{question}\n{options}"
                    target = str(row[5])
                    
                    raw_data.append({
                        'input': full_input, 
                        'target': target,
                        'subset': subset_name,
                        'source': 'mmlu'
                    })
            except Exception as e:
                print(f"  [錯誤] 讀取檔案 {file_path} 失敗: {e}")

    elif name == "gsm8k":
        # GSM8K 處理邏輯
        # 假設檔名為 gsm_train.tsv 或 gsm_test.tsv
        file_name = f"gsm_{split}.tsv"
        file_path = os.path.join(data_root, "gsm8k", file_name)
        
        if os.path.exists(file_path):
            try:
                # 假設 TSV 格式: [Question, Answer]
                df = pd.read_csv(file_path, sep="\t", header=None)
                print(f"  讀取 GSM8K 檔案: {file_path}")
                for _, row in df.iterrows():
                    raw_data.append({
                        'input': str(row[0]),
                        'target': str(row[1]),
                        'source': 'gsm8k'
                    })
            except Exception as e:
                print(f"  [錯誤] 讀取 GSM8K 失敗: {e}")
        else:
            print(f"  [錯誤] 找不到 GSM8K 檔案: {file_path}")

    elif name == "bbh":
        # BBH 保留原本邏輯 (略作調整以適應 new config 結構)
        # 假設 subsets 列表的第一個當作 task name
        task_name = dataset_cfg.subsets[0] if dataset_cfg.subsets else "unknown"
        file_path = os.path.join(data_root, "BIG-Bench-Hard-data", f"{task_name}.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding='utf-8') as f:
                data = json.load(f)["examples"]
                for d in data:
                     raw_data.append({'input': d['input'], 'target': d['target'], 'source': 'bbh'})

    print(f"資料載入完成，共 {len(raw_data)} 筆。")
    return raw_data

def parse_tag_content(text, prefix="<INS>", suffix="</INS>"):
    pattern = f"{prefix}(.*?){suffix}"
    results = re.findall(pattern, text, re.DOTALL)
    return [r.strip() for r in results]

def instruction_to_filename(instruction):
    m = hashlib.md5()
    m.update(instruction.encode('utf-8'))
    return m.hexdigest()

def polish_instruction(instruction: str) -> str:
    instruction = instruction.strip()
    if not instruction: return ""
    instruction = instruction.replace("**", "")
    if len(instruction) > 1: instruction = instruction[0].upper() + instruction[1:]
    if instruction and instruction[-1] not in ".?!": instruction += "."
    return instruction

def setup_logger(log_dir: str, task_name: str):
    import logging
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    logger = logging.getLogger("OPRO")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = os.path.join(log_dir, "run.log") # 固定叫 run.log
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger, log_file