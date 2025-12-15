# 檔案位置: experiments/data/mmlu/load_data.py
import pandas as pd
import os

# 修改這裡指向您存放 MMLU csv 的路徑
MMLU_DATA_PATH = 'path/to/your/mmlu/data/' 

def load_data(task_name, split='test'):
    """
    讀取指定科目的 MMLU 資料
    task_name: 例如 'abstract_algebra'
    split: 'dev', 'test', 或 'val'
    """
    csv_path = os.path.join(MMLU_DATA_PATH, f'{task_name}_{split}.csv')
    
    # MMLU 通常沒有標頭，我們手動指定
    df = pd.read_csv(csv_path, header=None, names=['question', 'A', 'B', 'C', 'D', 'answer'])
    
    inputs = []
    outputs = []
    
    for _, row in df.iterrows():
        # 組合題目與選項格式
        input_text = (f"{row['question']}\n"
                      f"(A) {row['A']}\n"
                      f"(B) {row['B']}\n"
                      f"(C) {row['C']}\n"
                      f"(D) {row['D']}")
        
        inputs.append(input_text)
        outputs.append(row['answer']) # 預期是 'A', 'B', 'C', or 'D'
        
    return inputs, outputs