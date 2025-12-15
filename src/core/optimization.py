import os
import pandas as pd
import logging
from collections import Counter
from src.core.scorer import Scorer
from src.core.optimizer import Optimizer
# [修正] 引入 polish_instruction
from src.utils import load_dataset, instruction_to_filename, polish_instruction
from src.models.base_client import BaseModelClient

logger = logging.getLogger("OPRO")

def run_opro_optimization(
    scorer_client: BaseModelClient,
    optimizer_client: BaseModelClient,
    config
):
    logger.info(f"開始 OPRO 優化任務: {config.task_name} on {config.dataset_name}")

    # 0. 準備目錄
    cache_dir = os.path.join(config.log_dir, "result_by_instruction")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # [新增] 判斷是否為 GSM8K (用於防作弊)
    is_gsm8k = 'gsm8k' in str(config.dataset_name).lower()

    # 1. 載入資料
    data_root = './data'
    try:
        dataset = load_dataset(config.dataset_name, config.task_name, data_root)
        logger.info(f"資料集載入完成，共 {len(dataset)} 筆。")
    except Exception as e:
        logger.error(f"資料載入失敗: {e}")
        raise e

    # 2. 初始化
    scorer = Scorer(scorer_client, config)
    optimizer = Optimizer(optimizer_client, config)
    wrong_questions_counter = Counter()
    
    instruction_pool = [{'instruction': "Let's think step by step.", 'score': 0.0, 'step': 0}]

    # --- 評分輔助函式 ---
    def get_score_and_details(inst_text, step_num):
        filename = instruction_to_filename(inst_text)
        filepath = os.path.join(cache_dir, f"{filename}.csv")
        
        # 快取命中
        if os.path.exists(filepath):
            logger.info(f"快取命中: {filename}")
            try:
                df = pd.read_csv(filepath)
                return float(df['accuracy'].mean()), df
            except: pass
        
        # 評估
        res = scorer.score_instruction(inst_text, dataset, config.num_evals_per_prompt)
        df = res['detailed_dataframe']
        df['instruction_content'] = inst_text 
        df['step_generated'] = step_num
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        return res['score'], df

    # 4. 評估初始指令
    for item in instruction_pool:
        if item['score'] == 0.0:
            s, _ = get_score_and_details(item['instruction'], 0)
            item['score'] = s
            logger.info(f"Initial: {s:.4f}")

    # 5. 優化迴圈
    for step in range(config.num_iterations):
        logger.info(f"=== Step {step+1} ===")
        
        # 生成
        raw_insts = optimizer.generate_new_instructions(instruction_pool, dataset, wrong_questions_counter)
        
        # [關鍵修正] 過濾與後處理
        valid_instructions = []
        for raw in raw_insts:
            # 1. 打磨指令 (Polish)
            polished = polish_instruction(raw)
            
            # 2. 長度檢查 (Length Control)
            if len(polished) > 500 or len(polished) == 0:
                logger.debug(f"過濾: 長度異常 ({len(polished)})")
                continue
                
            # 3. 標籤檢查
            if "<INS>" in polished or "text:" in polished:
                continue
                
            # 4. GSM8K 防作弊 (Anti-Cheating)
            # 如果指令包含數字，在數學題中可能是直接洩漏答案，必須丟棄
            if is_gsm8k and any(char.isdigit() for char in polished):
                logger.info(f"過濾 (Anti-Cheating): 指令包含數字 '{polished}'")
                continue
            
            # 5. 重複檢查
            if any(i['instruction'] == polished for i in instruction_pool):
                continue
                
            valid_instructions.append(polished)
            
        unique_insts = list(set(valid_instructions))
        logger.info(f"生成 {len(raw_insts)} -> 有效 {len(unique_insts)}")

        # 評估
        step_results = []
        for inst in unique_insts:
            s, df = get_score_and_details(inst, step+1)
            
            # 更新錯誤計數
            if df is not None and 'accuracy' in df.columns:
                for _, row in df[df['accuracy'] == 0.0].iterrows():
                    wrong_questions_counter[row['input']] += 1
            
            logger.info(f"評估: '{inst[:30]}...' -> {s:.4f}")
            step_results.append({'instruction': inst, 'score': s, 'step': step+1})
        
        instruction_pool.extend(step_results)
        
        # 存檔與顯示最佳
        _save_checkpoint(instruction_pool, config)
        best = max(instruction_pool, key=lambda x: x['score'])
        logger.info(f"目前最佳: {best['score']:.4f}")

    # 6. 結束
    sorted_pool = sorted(instruction_pool, key=lambda x: x['score'], reverse=True)
    return sorted_pool[0]

def _save_checkpoint(pool, config):
    try:
        path = os.path.join(config.log_dir, f"{config.task_name}_checkpoint.csv")
        pd.DataFrame(pool).sort_values('score', ascending=False).to_csv(path, index=False)
    except: pass