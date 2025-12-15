# src/core/optimization.py
import pandas as pd
import os
import logging
from src.core.scorer import Scorer
from src.core.optimizer import Optimizer
from src.utils import load_dataset
from src.models.base_client import BaseModelClient

logger = logging.getLogger("OPRO") # 取得剛剛設定好的 logger

def run_opro_optimization(
    scorer_client: BaseModelClient,
    optimizer_client: BaseModelClient,
    config
):
    logger.info(f"開始 OPRO 優化任務: {config.task_name} on {config.dataset_name}")
    
    # 1. 載入資料
    data_root = './data'  # 或從 config 讀取
    try:
        dataset = load_dataset(config.dataset_name, config.task_name, data_root)
        logger.info(f"資料集載入完成，共 {len(dataset)} 筆數據。")
    except Exception as e:
        logger.error(f"資料載入失敗: {e}")
        raise e

    # 2. 初始化模組
    scorer = Scorer(scorer_client)
    optimizer = Optimizer(optimizer_client, config)

    # 3. 初始化指令池
    instruction_pool = [{'instruction': "Let's think step by step.", 'score': 0.0, 'step': 0}]
    
    logger.info("正在評估初始指令...")
    for item in instruction_pool:
        if item['score'] == 0.0:
            result = scorer.score_instruction(
                item['instruction'], 
                dataset, 
                num_samples=config.num_evals_per_prompt
            )
            item['score'] = result['score']
            logger.info(f"Initial: {item['instruction']} -> Score: {item['score']:.4f}")

    # 4. 優化迴圈
    for step in range(config.num_iterations):
        logger.info(f"=== Step {step + 1}/{config.num_iterations} ===")
        
        # --- Generate (Optimizer Step) ---
        new_instructions_text = optimizer.generate_new_instructions(instruction_pool)
        
        # 去重與過濾
        unique_new_instructions = []
        existing_texts = set(i['instruction'] for i in instruction_pool)
        for inst in new_instructions_text:
            if inst not in existing_texts:
                unique_new_instructions.append(inst)
                existing_texts.add(inst)
        
        logger.info(f"生成了 {len(unique_new_instructions)} 個新指令")

        # --- Evaluate (Scorer Step) ---
        step_results = []
        for inst_text in unique_new_instructions:
            eval_result = scorer.score_instruction(
                inst_text, 
                dataset, 
                num_samples=config.num_evals_per_prompt
            )
            score = eval_result['score']
            logger.info(f"評估: '{inst_text[:50]}...' -> Score: {score:.4f}")
            
            step_results.append({
                'instruction': inst_text, 
                'score': score,
                'step': step + 1
            })

        # --- Update History ---
        instruction_pool.extend(step_results)
        
        # 顯示目前最佳
        best_instruction = max(instruction_pool, key=lambda x: x['score'])
        logger.info(f"目前最佳指令: '{best_instruction['instruction'][:50]}...' (Score: {best_instruction['score']:.4f})")

    # 5. 最終處理：排序並輸出 Top 10 CSV
    logger.info("優化結束，正在整理結果...")
    
    # 根據分數降冪排序 (高分在前)
    sorted_pool = sorted(instruction_pool, key=lambda x: x['score'], reverse=True)
    
    # 取前 10 名
    top_10 = sorted_pool[:10]
    
    # 儲存 CSV
    output_dir = config.log_dir if hasattr(config, 'log_dir') else './logs'
    csv_filename = f"{config.task_name}_top10_prompts.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    df = pd.DataFrame(top_10)
    # 調整欄位順序美觀一點
    cols = ['score', 'step', 'instruction']
    # 確保欄位存在 (防呆)
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    df.to_csv(csv_path, index=False, encoding='utf-8-sig') # utf-8-sig 防止 Excel 開啟亂碼
    
    logger.info(f"Top 10 Prompts 已儲存至: {csv_path}")
    
    # 回傳最佳結果
    return sorted_pool[0]