import os
import pandas as pd
import logging
import random # [新增]
from collections import Counter
from src.core.scorer import Scorer
from src.core.optimizer import Optimizer
from src.utils import load_dataset, instruction_to_filename, polish_instruction
from src.model.base_client import BaseModelClient

logger = logging.getLogger("OPRO")

def run_opro_optimization(
    scorer_client: BaseModelClient,
    optimizer_client: BaseModelClient,
    config
):
    logger.info(f"開始 OPRO 優化任務: {config.task_name} on {config.dataset_name}")

    cache_dir = os.path.join(config.log_dir, "result_by_instruction")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    is_gsm8k = 'gsm8k' in str(config.dataset_name).lower()

    # 1. 載入資料並切分 (Train/Validation Split)
    data_root = './data'
    try:
        full_dataset = load_dataset(config.dataset_name, config.task_name, data_root)
        
        # [修正] 隨機打亂並切分，防止 Overfitting
        random.seed(42) # 固定種子以重現實驗
        random.shuffle(full_dataset)
        
        # 假設 80% 訓練 (用於優化)，20% 測試 (用於最終驗證)
        split_idx = int(len(full_dataset) * 0.8)
        train_dataset = full_dataset[:split_idx]
        test_dataset = full_dataset[split_idx:]
        
        logger.info(f"資料集載入完成: 總數 {len(full_dataset)}, 訓練集 {len(train_dataset)}, 測試集 {len(test_dataset)}")
        
        if len(train_dataset) == 0:
            raise ValueError("訓練集為空，請檢查資料載入邏輯。")

    except Exception as e:
        logger.error(f"資料載入失敗: {e}")
        raise e

    # 2. 初始化
    scorer = Scorer(scorer_client, config)
    optimizer = Optimizer(optimizer_client, config)
    wrong_questions_counter = Counter()
    
    # [修正] 從 config 讀取多樣化的初始指令
    initial_texts = getattr(config, 'initial_instructions', ["Let's think step by step."])
    instruction_pool = [{'instruction': txt, 'score': 0.0, 'step': 0} for txt in initial_texts]

    # --- 評分輔助函式 ---
    def get_score_and_details(inst_text, step_num, dataset_split):
        filename = instruction_to_filename(inst_text)
        # 注意：快取檔名不區分 dataset，實際應用中最好加上 dataset 標籤以免混用
        # 這裡為了簡單先維持原樣，但要小心
        filepath = os.path.join(cache_dir, f"{filename}.csv")
        
        # 評估 (使用傳入的 dataset_split)
        # [注意] 這裡不再依賴快取來直接回傳分數，因為我們可能換了 dataset (train vs test)
        # 如果要用快取，需要更複雜的 key。這裡建議優化時一律重跑 (或只對相同 dataset 的結果做快取)
        
        res = scorer.score_instruction(inst_text, dataset_split, config.num_evals_per_prompt)
        df = res['detailed_dataframe']
        df['instruction_content'] = inst_text 
        df['step_generated'] = step_num
        
        return res['score'], df

    # 4. 評估初始指令 (使用訓練集)
    logger.info("評估初始指令...")
    for item in instruction_pool:
        s, _ = get_score_and_details(item['instruction'], 0, train_dataset)
        item['score'] = s
        logger.info(f"Initial: '{item['instruction'][:30]}...' -> {s:.4f}")

    # 5. 優化迴圈
    for step in range(config.num_iterations):
        logger.info(f"=== Step {step+1} ===")
        
        # 生成 (傳入訓練集與錯誤計數)
        # 這裡的 dataset 參數是用來選 few-shot examples 的
        raw_insts = optimizer.generate_new_instructions(instruction_pool, train_dataset, wrong_questions_counter)
        
        valid_instructions = []
        for raw in raw_insts:
            polished = polish_instruction(raw)
            
            if len(polished) > 500 or len(polished) == 0: continue
            if "<INS>" in polished or "text:" in polished: continue
            if is_gsm8k and any(char.isdigit() for char in polished): continue
            if any(i['instruction'] == polished for i in instruction_pool): continue
                
            valid_instructions.append(polished)
            
        unique_insts = list(set(valid_instructions))
        logger.info(f"生成 {len(raw_insts)} -> 有效 {len(unique_insts)}")

        # 評估 (使用訓練集)
        step_results = []
        for inst in unique_insts:
            s, df = get_score_and_details(inst, step+1, train_dataset)
            
            # 更新錯誤計數 (只用訓練集的錯誤來驅動)
            if df is not None and 'accuracy' in df.columns:
                for _, row in df[df['accuracy'] == 0.0].iterrows():
                    wrong_questions_counter[row['input']] += 1
            
            logger.info(f"評估: '{inst[:30]}...' -> {s:.4f}")
            step_results.append({'instruction': inst, 'score': s, 'step': step+1})
        
        instruction_pool.extend(step_results)
        
        _save_checkpoint(instruction_pool, config)
        best = max(instruction_pool, key=lambda x: x['score'])
        logger.info(f"目前訓練集最佳: {best['score']:.4f}")

    # 6. 最終測試 (使用測試集)
    logger.info("=== 優化結束，執行最終測試 ===")
    sorted_pool = sorted(instruction_pool, key=lambda x: x['score'], reverse=True)
    best_train_inst = sorted_pool[0]
    
    # 在測試集上驗證最佳指令
    test_score, _ = get_score_and_details(best_train_inst['instruction'], -1, test_dataset)
    
    # 回傳結果結構更新
    return {
        'instruction': best_train_inst['instruction'],
        'score': test_score, # 回傳測試集分數
        'train_score': best_train_inst['score'],
        'step': best_train_inst['step']
    }

def _save_checkpoint(pool, config):
    try:
        path = os.path.join(config.log_dir, f"{config.task_name}_checkpoint.csv")
        pd.DataFrame(pool).sort_values('score', ascending=False).to_csv(path, index=False)
    except: pass