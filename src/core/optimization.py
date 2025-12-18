import os
import pandas as pd
import logging
import random
from collections import Counter
from src.core.scorer import Scorer
from src.core.optimizer import Optimizer
from src.utils import load_dataset, instruction_to_filename, polish_instruction
from src.model.base_client import BaseModelClient

logger = logging.getLogger("OPRO")

def run_opro_optimization(scorer_client: BaseModelClient, optimizer_client: BaseModelClient, config):
    logger.info(f"開始 OPRO 優化任務: {config.task_name} on {config.dataset_name}")
    
    # 讀取設定
    is_gsm8k = 'gsm8k' in str(config.dataset_name).lower()
    train_ratio = getattr(config, 'train_ratio', 0.8) # 預設 80%
    eval_interval = getattr(config, 'eval_interval', 3)

    # 1. 載入並分割資料
    data_root = './data'
    # load_dataset 已經處理了 "不足 300 筆就全拿" 的邏輯
    full_dataset = load_dataset(config.dataset_name, config.task_name, data_root)
    
    if not full_dataset:
        raise ValueError("沒有載入任何資料，請檢查路徑或檔案名稱。")

    # 隨機打亂：混合所有子集的題目
    random.seed(42)
    random.shuffle(full_dataset)
    
    # 動態計算分割點 (基於實際載入的總筆數)
    n_total = len(full_dataset)
    n_train = int(n_total * train_ratio)
    
    # 限制訓練集上限 (避免因為全部載入導致太慢，如果有需要可解開註解)
    # n_train = min(n_train, 2)
    
    train_dataset = full_dataset[:n_train]
    # eval_dataset = full_dataset[n_train:n_train + 1]
    eval_dataset = full_dataset[n_train:] # 剩下的做驗證
    
    # 在本框架中，我們暫時用 Eval Set 當作每輪檢查進步的基準，也可以當作最終 Test Set
    test_dataset = eval_dataset 

    logger.info(f"資料統計: 總筆數={n_total}")
    logger.info(f"分割結果: Train={len(train_dataset)}, Eval/Test={len(eval_dataset)}")
    # 2. 初始化模組
    scorer = Scorer(scorer_client, config)
    optimizer = Optimizer(optimizer_client, config)
    wrong_questions_counter = Counter()
    
    # 初始指令
    initial_texts = getattr(config, 'initial_instructions', ["Let's think step by step."])
    instruction_pool = [{'instruction': txt, 'score': 0.0, 'step': 0} for txt in initial_texts]
    cache_dir = os.path.join(config.log_dir, "result_by_instruction")
    os.makedirs(cache_dir, exist_ok=True)

    def evaluate_instruction(inst_text, dataset, step_num, file_suffix="train"):
        """通用評分函式"""
        filename = instruction_to_filename(inst_text)
        # 這裡檔案名稱加上 suffix 以區分 train/eval 結果
        filepath = os.path.join(cache_dir, f"{filename}_{file_suffix}.csv")
        
        # 這裡簡化邏輯：如果是 train，總是重算 (因為有隨機性或我們希望 error-driven 更新)
        # 如果是 eval/test，可考慮 cache
        
        res = scorer.score_instruction(inst_text, dataset) # 不再傳 num_samples，直接用全量 dataset
        
        # 儲存詳細結果
        df = res['detailed_dataframe']
        df['instruction'] = inst_text
        df['step'] = step_num
        df.to_csv(filepath, index=False)
        
        return res['score'], df

    # 3. 評估初始指令 (Train Set)
    logger.info("評估初始指令...")
    total_initial = len(instruction_pool)
    for item in instruction_pool:
        s, df = evaluate_instruction(item['instruction'], train_dataset, 0, "train")
        item['score'] = s
        # 更新錯誤計數
        if df is not None:
            for _, row in df[df['accuracy'] == 0.0].iterrows():
                wrong_questions_counter[row['input']] += 1

    # 4. 優化迴圈
    for step in range(config.num_iterations):
        current_step = step + 1
        logger.info(f"=== Step {current_step} ===")
        
        # 生成新指令
        # 這裡傳入 train_dataset 供 few-shot 選擇 (Error-driven)
        raw_insts = optimizer.generate_new_instructions(instruction_pool, train_dataset, wrong_questions_counter)
        
        valid_insts = []
        for raw in raw_insts:
            polished = polish_instruction(raw)
            # ... (過濾邏輯同前: 長度、標籤、GSM8K 數字檢查) ...
            if "<INS>" in polished or not polished: continue
            if is_gsm8k and any(c.isdigit() for c in polished): continue
            if any(i['instruction'] == polished for i in instruction_pool): continue
            
            # [新增] Few-shot Pre-filtering (預過濾)
            # 在跑整個訓練集前，先用 5 題測一下，如果全錯就丟掉
            pre_screen_score, _ = evaluate_instruction(polished, train_dataset[:5], current_step, "pre_screen")
            if pre_screen_score == 0.0:
                logger.info(f"預過濾淘汰: {polished[:30]}... (Score: 0.0)")
                continue

            valid_insts.append(polished)
        
        unique_insts = list(set(valid_insts))
        total_insts = len(unique_insts)
        logger.info(f"本輪共有 {total_insts} 個新指令需要評估。")

        # 評估新指令 (Train Set)
        step_results = []
        for i, inst in enumerate(unique_insts,1):
            logger.info(f"Evaluating ({i}/{total_insts}): {inst[:50]}...")
            s, df = evaluate_instruction(inst, train_dataset, current_step, "train")
            step_results.append({'instruction': inst, 'score': s, 'step': current_step})
            
            # 更新錯誤計數
            if df is not None:
                for _, row in df[df['accuracy'] == 0.0].iterrows():
                    wrong_questions_counter[row['input']] += 1
            logger.info(f"Train Score: {s:.4f} | {inst[:40]}...")

        instruction_pool.extend(step_results)
        instruction_pool.sort(key=lambda x: x['score'], reverse=True)
        # 截斷 Pool 大小，保留最好的前 N 個 (參考論文)
        instruction_pool = instruction_pool[:20] 

        # [新增] Eval Interval Check (定期驗證)
        if current_step % eval_interval == 0:
            best_inst = instruction_pool[0]['instruction']
            eval_score, _ = evaluate_instruction(best_inst, eval_dataset, current_step, "eval")
            logger.info(f"★ Eval Check (Step {current_step}): {eval_score:.4f} | Best: {best_inst[:30]}...")

    # 5. 最終測試 (Test Set)
    best_instruction = instruction_pool[0]
    logger.info(f"優化結束。最佳指令 (Train: {best_instruction['score']:.4f}): {best_instruction['instruction']}")
    
    test_score, _ = evaluate_instruction(best_instruction['instruction'], test_dataset, -1, "test")
    logger.info(f"最終測試分數 (Test Score): {test_score:.4f}")
    
    top_n_path = os.path.join(config.log_dir, "top_prompts.csv")
    pd.DataFrame(instruction_pool).to_csv(top_n_path, index=False)
    logger.info(f"已將前 {len(instruction_pool)} 名指令存檔至: {top_n_path}")
    
    return best_instruction