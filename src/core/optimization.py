import os
import pandas as pd
import logging
from collections import Counter
from src.core.scorer import Scorer
from src.core.optimizer import Optimizer
from src.utils import load_dataset, instruction_to_filename, polish_instruction
from src.models.base_client import BaseModelClient

logger = logging.getLogger("OPRO")

def run_opro_optimization(
    scorer_client: BaseModelClient,
    optimizer_client: BaseModelClient,
    config
):
    logger.info(f"開始 OPRO 優化任務: {config.task_name} on {config.dataset_name}")

    # 0. 準備快取與輸出目錄
    cache_dir = os.path.join(config.log_dir, "result_by_instruction")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    logger.info(f"快取目錄已建立: {cache_dir}")

    is_gsm8k = 'gsm8k' in config.dataset_name.lower()

    # 1. 載入資料
    data_root = './data'
    try:
        dataset = load_dataset(config.dataset_name, config.task_name, data_root)
        logger.info(f"資料集載入完成，共 {len(dataset)} 筆數據。")
    except Exception as e:
        logger.error(f"資料載入失敗: {e}")
        raise e

    # 2. 初始化模組
    scorer = Scorer(scorer_client, config)
    optimizer = Optimizer(optimizer_client, config)

    # 3. 初始化錯誤計數器與指令池
    wrong_questions_counter = Counter()
    instruction_pool = [{'instruction': "Let's think step by step.", 'score': 0.0, 'step': 0}]

    # --- [修改] 內部輔助函式：現在同時回傳分數與詳細資料 ---
    def get_score_and_details(inst_text, step_num):
        filename = instruction_to_filename(inst_text)
        filepath = os.path.join(cache_dir, f"{filename}.csv")
        
        # A. 快取命中
        if os.path.exists(filepath):
            logger.info(f"快取命中: {filename}")
            try:
                cached_df = pd.read_csv(filepath)
                if 'accuracy' in cached_df.columns:
                    avg_acc = float(cached_df['accuracy'].mean())
                    return avg_acc, cached_df  # 回傳分數與 DataFrame
            except Exception as e:
                logger.warning(f"讀取快取失敗 ({e})，將重新評估。")

        # B. 執行評估
        eval_result = scorer.score_instruction(
            inst_text, 
            dataset, 
            num_samples=config.num_evals_per_prompt
        )
        score = eval_result['score']
        df = eval_result['detailed_dataframe']
        
        # C. 寫入快取
        df['instruction_content'] = inst_text 
        df['step_generated'] = step_num
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        return score, df
    # -------------------------------------------

    # 4. 評估初始指令
    logger.info("正在評估初始指令...")
    for item in instruction_pool:
        if item['score'] == 0.0:
            # 這裡我們只需要分數，不需要 dataframe
            score, _ = get_score_and_details(item['instruction'], 0)
            item['score'] = score
            logger.info(f"Initial: {item['instruction']} -> Score: {item['score']:.4f}")

    # 5. 優化迴圈
    for step in range(config.num_iterations):
        logger.info(f"=== Step {step + 1}/{config.num_iterations} ===")
        
        # --- Generate (Optimizer Step) ---
        # 傳入 wrong_questions_counter 讓 Optimizer 挑錯題
        raw_new_instructions = optimizer.generate_new_instructions(
            instruction_pool, 
            dataset, 
            wrong_questions_counter
        )
        
        # --- Filter & Dedup ---
        valid_instructions = []
        for raw_inst in raw_new_instructions:
            polished_inst = polish_instruction(raw_inst)
            
            # 過濾規則
            if len(polished_inst) > 500: continue
            if "<INS>" in polished_inst or "</INS>" in polished_inst: continue
            if is_gsm8k and any(char.isdigit() for char in polished_inst): continue
            if any(item['instruction'] == polished_inst for item in instruction_pool): continue

            valid_instructions.append(polished_inst)

        unique_instructions = list(set(valid_instructions))
        logger.info(f"生成 {len(raw_new_instructions)} -> 有效 {len(unique_instructions)}")

        # --- Evaluate (Scorer Step) ---
        step_results = []
        for inst_text in unique_instructions:
            # [關鍵] 呼叫修改後的函式，取得分數與詳細資料
            score, detailed_df = get_score_and_details(inst_text, step + 1)
            
            # [關鍵] 更新錯誤計數器 (Error-Driven 核心)
            if detailed_df is not None and 'accuracy' in detailed_df.columns:
                # 找出答錯 (accuracy == 0) 的題目
                wrong_df = detailed_df[detailed_df['accuracy'] == 0.0]
                for _, row in wrong_df.iterrows():
                    wrong_questions_counter[row['input']] += 1
            
            logger.info(f"評估: '{inst_text[:50]}...' -> Score: {score:.4f}")
            step_results.append({'instruction': inst_text, 'score': score, 'step': step + 1})

        # --- Update History & Checkpoint ---
        instruction_pool.extend(step_results)
        _save_checkpoint(instruction_pool, config)

        best_instruction = max(instruction_pool, key=lambda x: x['score'])
        logger.info(f"目前最佳: {best_instruction['score']:.4f}")

    # 6. 最終輸出
    logger.info("整理最終結果...")
    sorted_pool = sorted(instruction_pool, key=lambda x: x['score'], reverse=True)
    
    csv_path = os.path.join(config.log_dir, f"{config.task_name}_top10_prompts.csv")
    pd.DataFrame(sorted_pool[:10]).to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"Top 10 已存至: {csv_path}")
    
    return sorted_pool[0]

def _save_checkpoint(pool, config):
    try:
        csv_path = os.path.join(config.log_dir, f"{config.task_name}_checkpoint.csv")
        pd.DataFrame(pool).sort_values('score', ascending=False).to_csv(csv_path, index=False, encoding='utf-8-sig')
    except Exception as e:
        logger.warning(f"Checkpoint 儲存失敗: {e}")