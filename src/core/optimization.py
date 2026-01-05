import os
import random
import logging
from collections import Counter
import pandas as pd
from src.core.scorer import Scorer
from src.core.optimizer import Optimizer
from src.utils import load_dataset, instruction_to_filename, polish_instruction
from src.model.base_client import BaseModelClient

logger = logging.getLogger("OPRO")

def run_opro_optimization(scorer_client: BaseModelClient, optimizer_client: BaseModelClient, config):
    # 取得參數
    task_name = getattr(config.optimization, 'task_name', '')
    dataset_name = config.dataset.name
    
    logger.info(f"開始 OPRO 優化任務: {task_name} on {dataset_name}")
    
    is_gsm8k = 'gsm8k' in str(dataset_name).lower()
    
    # 優化參數
    train_ratio = getattr(config.optimization, 'train_ratio', 0.8)
    eval_interval = getattr(config.optimization, 'eval_interval', 3)
    num_iterations = config.optimization.num_iterations

    # 1. 載入資料
    raw_dataset = load_dataset(config.dataset)
    
    if not raw_dataset:
        raise ValueError("沒有載入任何資料，請檢查路徑或檔案名稱。")

    # --- 實作：各子集限量取用 -> 混合 -> 80/20 切分 ---
    
    # 分組
    grouped_data = {}
    for item in raw_dataset:
        sub = item.get('subset', 'default')
        if sub not in grouped_data:
            grouped_data[sub] = []
        grouped_data[sub].append(item)
    
    # 讀取每子集限制 (例如: 50)
    limit = getattr(config.dataset, 'train_limit', None)
    limit_int = None
    if limit and str(limit).lower() != 'all':
        try:
            limit_int = int(limit)
            logger.info(f"設定每子集取樣上限: {limit_int} 筆")
        except ValueError:
            logger.warning(f"無法解析 train_limit: {limit}，將使用全部資料")

    random.seed(42)
    
    combined_pool = []
    
    # Step 1: 遍歷每個子集，取出指定數量 (例如 50) 加入大池子
    for sub, items in grouped_data.items():
        random.shuffle(items) # 先打亂該子集
        
        if limit_int is not None:
            # 只取前 limit_int 筆 (例如 50)
            selected_items = items[:limit_int]
        else:
            selected_items = items
            
        combined_pool.extend(selected_items)

    logger.info(f"各子集取樣完成，混合池總筆數: {len(combined_pool)} (來自 {len(grouped_data)} 個子集)")

    # Step 2: 混合後進行 80/20 切分
    random.shuffle(combined_pool) # 全局打亂
    
    n_total = len(combined_pool)
    n_train = int(n_total * train_ratio)
    
    # 確保至少有 1 筆訓練資料
    if n_train == 0 and n_total > 0:
        n_train = 1

    train_dataset = combined_pool[:n_train]
    eval_dataset = combined_pool[n_train:]
    
    test_dataset = eval_dataset 

    logger.info(f"最終分割結果: Train={len(train_dataset)}, Eval/Test={len(eval_dataset)}")

    if len(train_dataset) == 0:
        raise ValueError("訓練集為空，請檢查資料載入或 train_limit 設定")

    # 2. 初始化模組
    # 傳入完整 config 較為安全
    scorer = Scorer(scorer_client, config)     
    optimizer = Optimizer(optimizer_client, config)
    
    wrong_questions_counter = Counter()
    
    initial_texts = getattr(config.optimization, 'initial_instructions', ["Let's think step by step."])
    instruction_pool = [{'instruction': txt, 'score': 0.0, 'step': 0} for txt in initial_texts]
    
    cache_dir = os.path.join(config.project.log_dir, "result_by_instruction")
    os.makedirs(cache_dir, exist_ok=True)

    def evaluate_instruction(inst_text, dataset, step_num, file_suffix="train"):
        """通用評分函式"""
        if not dataset:
            return 0.0, None

        filename = instruction_to_filename(inst_text)
        filepath = os.path.join(cache_dir, f"{filename}_{file_suffix}.csv")
        
        res = scorer.score_instruction(inst_text, dataset)
        
        df = res['detailed_dataframe']
        if df is not None and not df.empty:
            df['instruction'] = inst_text
            df['step'] = step_num
            df.to_csv(filepath, index=False)
        
        return res['score'], df

    # 3. 評估初始指令 (Train Set)
    logger.info("評估初始指令...")
    for item in instruction_pool:
        s, df = evaluate_instruction(item['instruction'], train_dataset, 0, "train")
        item['score'] = s
        if df is not None:
            for _, row in df[df['accuracy'] == 0.0].iterrows():
                wrong_questions_counter[row['input']] += 1

    # 4. 優化迴圈
    for step in range(num_iterations):
        current_step = step + 1
        logger.info(f"=== Step {current_step} ===")
        
        # 生成新指令
        raw_insts = optimizer.generate_new_instructions(instruction_pool, train_dataset, wrong_questions_counter)
        
        valid_insts = []
        for raw in raw_insts:
            polished = polish_instruction(raw)
            if not polished: continue
            if "<INS>" in polished: continue
            if is_gsm8k and any(c.isdigit() for c in polished): continue
            if any(i['instruction'] == polished for i in instruction_pool): continue
            
            # Few-shot Pre-filtering
            # 確保有足夠資料才做 slice
            screen_data = train_dataset[:5] if len(train_dataset) > 5 else train_dataset
            if screen_data:
                pre_screen_score, _ = evaluate_instruction(polished, screen_data, current_step, "pre_screen")
                if pre_screen_score == 0.0:
                    logger.info(f"預過濾淘汰: {polished[:30]}... (Score: 0.0)")
                    continue

            valid_insts.append(polished)
        
        unique_insts = list(set(valid_insts))
        total_insts = len(unique_insts)
        logger.info(f"本輪共有 {total_insts} 個新指令需要評估。")

        # 評估新指令
        step_results = []
        for i, inst in enumerate(unique_insts, 1):
            logger.info(f"Evaluating ({i}/{total_insts}): {inst[:50]}...")
            s, df = evaluate_instruction(inst, train_dataset, current_step, "train")
            step_results.append({'instruction': inst, 'score': s, 'step': current_step})
            
            if df is not None:
                for _, row in df[df['accuracy'] == 0.0].iterrows():
                    wrong_questions_counter[row['input']] += 1
            logger.info(f"Train Score: {s:.4f} | {inst[:40]}...")

        instruction_pool.extend(step_results)
        instruction_pool.sort(key=lambda x: x['score'], reverse=True)
        instruction_pool = instruction_pool[:20] 

        # Eval Check
        if current_step % eval_interval == 0:
            if instruction_pool: 
                best_inst = instruction_pool[0]['instruction']
                eval_score, _ = evaluate_instruction(best_inst, eval_dataset, current_step, "eval")
                logger.info(f"★ Eval Check (Step {current_step}): {eval_score:.4f} | Best: {best_inst[:30]}...")

    # 5. 最終測試
    if not instruction_pool:
        logger.error("指令池為空，優化失敗。")
        return None

    best_instruction = instruction_pool[0]
    logger.info(f"優化結束。最佳指令 (Train: {best_instruction['score']:.4f}): {best_instruction['instruction']}")
    
    test_score, _ = evaluate_instruction(best_instruction['instruction'], test_dataset, -1, "test")
    logger.info(f"最終測試分數 (Test Score): {test_score:.4f}")
    
    top_n_path = os.path.join(config.project.log_dir, "top_prompts.csv")
    pd.DataFrame(instruction_pool).to_csv(top_n_path, index=False)
    logger.info(f"已將前 {len(instruction_pool)} 名指令存檔至: {top_n_path}")
    
    return best_instruction