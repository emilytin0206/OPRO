import os
import pandas as pd
import logging
from collections import Counter
# [修正] Import 路徑
from src.core.scorer import Scorer
from src.core.optimizer import Optimizer
from src.utils import load_dataset, instruction_to_filename, polish_instruction
from src.model.base_client import BaseModelClient

logger = logging.getLogger("OPRO")

def run_opro_optimization(scorer_client: BaseModelClient, optimizer_client: BaseModelClient, config):
    logger.info(f"開始 OPRO 優化任務: {config.task_name} on {config.dataset_name}")

    # 0. 準備目錄
    cache_dir = os.path.join(config.log_dir, "result_by_instruction")
    if not os.path.exists(cache_dir): os.makedirs(cache_dir)

    is_gsm8k = 'gsm8k' in str(config.dataset_name).lower()

    # 1. 載入資料
    dataset = load_dataset(config.dataset_name, config.task_name, './data')
    logger.info(f"資料集載入完成，共 {len(dataset)} 筆。")

    # 2. 初始化
    scorer = Scorer(scorer_client, config)
    optimizer = Optimizer(optimizer_client, config)
    wrong_questions_counter = Counter()
    
    instruction_pool = [{'instruction': "Let's think step by step.", 'score': 0.0, 'step': 0}]

    # 輔助：評分與快取
    def get_score_and_update(inst_text, step_num):
        filename = instruction_to_filename(inst_text)
        filepath = os.path.join(cache_dir, f"{filename}.csv")
        
        score = 0.0
        df = None

        if os.path.exists(filepath):
            logger.info(f"快取命中: {filename}")
            try:
                df = pd.read_csv(filepath)
                score = float(df['accuracy'].mean())
            except: pass
        
        if df is None: # 快取無效或不存在
            res = scorer.score_instruction(inst_text, dataset, config.num_evals_per_prompt)
            score = res['score']
            df = res['detailed_dataframe']
            df['instruction_content'] = inst_text
            df['step_generated'] = step_num
            df.to_csv(filepath, index=False, encoding='utf-8-sig')

        # 更新錯誤計數
        if df is not None and 'accuracy' in df.columns:
            wrong_df = df[df['accuracy'] == 0.0]
            for _, row in wrong_df.iterrows():
                wrong_questions_counter[row['input']] += 1
                
        return score

    # 4. 評估初始
    logger.info("評估初始指令...")
    for item in instruction_pool:
        if item['score'] == 0.0:
            item['score'] = get_score_and_update(item['instruction'], 0)
            logger.info(f"Initial: {item['score']:.4f}")

    # 5. 迴圈
    for step in range(config.num_iterations):
        logger.info(f"=== Step {step+1} ===")
        # 生成 (傳入 dataset 與 counter)
        raw_insts = optimizer.generate_new_instructions(instruction_pool, dataset, wrong_questions_counter)
        
        # 過濾
        valid_insts = []
        for r in raw_insts:
            p = polish_instruction(r)
            if len(p) > 500 or "<INS>" in p or (is_gsm8k and any(c.isdigit() for c in p)): continue
            if any(i['instruction'] == p for i in instruction_pool): continue
            valid_insts.append(p)
        
        unique_insts = list(set(valid_insts))
        logger.info(f"生成 {len(raw_insts)} -> 有效 {len(unique_insts)}")

        # 評估
        step_results = []
        for inst in unique_insts:
            s = get_score_and_update(inst, step+1)
            logger.info(f"評估: '{inst[:30]}...' -> {s:.4f}")
            step_results.append({'instruction': inst, 'score': s, 'step': step+1})
        
        instruction_pool.extend(step_results)
        
        best = max(instruction_pool, key=lambda x: x['score'])
        logger.info(f"目前最佳: {best['score']:.4f}")

    # 6. 結束
    sorted_pool = sorted(instruction_pool, key=lambda x: x['score'], reverse=True)
    return sorted_pool[0]