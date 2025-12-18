import os
import pandas as pd
import json
import re
import string
import hashlib # [新增] 用於雜湊檔名


def load_dataset(dataset_name: str, task_name: str, data_root: str):
    """
    載入指定資料集。
    針對 MMLU 支援多子集載入，並處理資料量不足 300 筆的情況。
    """
    dataset_name = dataset_name.lower()
    raw_data = []

    if dataset_name == "mmlu":
        # 定義您需要的 5 個子集
        target_subsets = [
            "high_school_mathematics",
            "high_school_world_history",
            "high_school_physics",
            "professional_law",
            "business_ethics"
        ]
        
        # 決定要載入哪些子集 (如果是 'all_specified' 或不在列表內，就載入全部指定的)
        subsets_to_load = [task_name] if task_name in target_subsets else target_subsets
        
        print(f"準備載入 MMLU 子集: {subsets_to_load}")

        for subset in subsets_to_load:
            # 嘗試路徑：先找 test 資料夾，再找根目錄
            # MMLU 通常結構: data/mmlu/test/high_school_mathematics_test.csv
            file_path = os.path.join(data_root, "mmlu", "test", f"{subset}_test.csv")
            if not os.path.exists(file_path):
                 file_path = os.path.join(data_root, "mmlu", f"{subset}_test.csv")

            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, header=None)
                    
                    # --- [修正核心] 安全選取邏輯 ---
                    total_rows = len(df)
                    if total_rows < 300:
                        print(f"  [警告] 子集 '{subset}' 只有 {total_rows} 筆資料 (不足 300)，將全數載入。")
                        df_subset = df # 全部都要
                    else:
                        print(f"  [正常] 子集 '{subset}' 資料充足 ({total_rows} 筆)，取前 300 筆。")
                        df_subset = df.head(300)
             # ----------------------------

                    for _, row in df_subset.iterrows():
                        # MMLU CSV 格式通常為: Question, A, B, C, D, Answer
                        question = str(row[0])
                        options = f"(A) {str(row[1])}\n(B) {str(row[2])}\n(C) {str(row[3])}\n(D) {str(row[4])}"
                        full_input = f"{question}\n{options}"
                        target = str(row[5]) # 答案 (A, B, C, D)
                        
                        raw_data.append({
                            'input': full_input, 
                            'target': target,
                            'subset': subset 
                        })
                except Exception as e:
                    print(f"  [錯誤] 讀取 {subset} 失敗: {e}")
            else:
                print(f"  [缺失] 找不到檔案: {file_path}")

    elif dataset_name == "bbh":
        file_path = os.path.join(data_root, "BIG-Bench-Hard-data", f"{task_name}.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding='utf-8') as f:
                data = json.load(f)["examples"]
                for d in data:
                     raw_data.append({'input': d['input'], 'target': d['target']})
        else:
            raise FileNotFoundError(f"找不到 BBH 數據: {file_path}")

    elif dataset_name == "gsm8k":
        # 簡單示範保留
        file_path = os.path.join(data_root, "gsm_data", f"gsm_{task_name}.tsv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, sep="\t", header=None)
            for _, row in df.iterrows():
                raw_data.append({'input': row[0], 'target': row[1]})


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
    log_file = os.path.join(log_dir, f"{task_name}.log")
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger, log_file