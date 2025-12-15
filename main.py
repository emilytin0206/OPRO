import yaml
import os
import argparse
import datetime
import json
import shutil # [新增] 用於複製設定檔
from dataclasses import dataclass, field 
from typing import Type, List            
import logging

from src.utils import setup_logger
from src.model.base_client import BaseModelClient
from src.model.ollama_client import OllamaModelClient
from src.core.optimization import run_opro_optimization

# ... (ModelConfig 和 OptimizationConfig 定義保持不變) ...
@dataclass
class ModelConfig:
    client_type: str
    model_name: str
    api_url: str
    temperature: float
    max_output_tokens: int

@dataclass
class OptimizationConfig:
    num_iterations: int
    num_prompts_to_generate: int
    max_num_instructions_in_prompt: int 
    meta_prompt_path: str
    task_name: str = ""
    dataset_name: str = ""
    log_dir: str = ""
    instruction_pos: str = "A_begin"
    is_instruction_tuned: bool = False
    num_few_shot_questions: int = 3
    few_shot_selection_criteria: str = "random"
    initial_instructions: List[str] = field(default_factory=lambda: ["Let's think step by step."])
    old_instruction_score_threshold: float = 0.1

def load_config(config_path: str):
    # ... (此函式內容保持不變) ...
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到設定檔: {config_path}")
    print(f"正在載入設定檔: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    scorer_cfg = ModelConfig(**config['scorer_model'])
    optimizer_cfg = ModelConfig(**config['optimizer_model'])
    opt_dict = config['optimization']
    known_keys = OptimizationConfig.__annotations__.keys()
    filtered_opt_dict = {k: v for k, v in opt_dict.items() if k in known_keys}
    if 'initial_instructions' in filtered_opt_dict and not isinstance(filtered_opt_dict['initial_instructions'], list):
         filtered_opt_dict['initial_instructions'] = [str(filtered_opt_dict['initial_instructions'])]
    opt_cfg = OptimizationConfig(**filtered_opt_dict)
    return scorer_cfg, optimizer_cfg, opt_cfg, config

def main():
    parser = argparse.ArgumentParser(description="OPRO Optimization Runner")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the configuration YAML file.')
    args = parser.parse_args()

    # 1. 加載配置
    scorer_cfg, optimizer_cfg, opt_cfg, raw_config = load_config(args.config)
    
    project_config = raw_config.get('project', {})
    task_name = project_config.get('task_name', 'default_task')
    base_log_dir = project_config.get('log_dir', './logs')

    # --- [修改] 建立實驗專屬 Folder ---
    def clean_name(name):
        return name.replace(':', '-').replace('/', '_').replace(' ', '_')

    target_model_name = clean_name(scorer_cfg.model_name)
    optimizer_model_name = clean_name(optimizer_cfg.model_name)
    dataset_name = clean_name(opt_cfg.dataset_name)
    
    # 取得當前時間戳記
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 資料夾結構: logs/<target_model>/<timestamp>_<optimizer>_<dataset>/
    experiment_folder_name = f"{timestamp}_{optimizer_model_name}_{dataset_name}"
    experiment_dir = os.path.join(base_log_dir, target_model_name, experiment_folder_name)
    
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    print(f"建立實驗資料夾: {experiment_dir}")

    # 2. 設定 Logger (日誌檔名固定為 run.log，放在實驗資料夾內)
    logger, log_file_path = setup_logger(experiment_dir, "run")
    
    # 3. [新增] 保存參數資訊 (Config Snapshot)
    config_backup_path = os.path.join(experiment_dir, "config_snapshot.yaml")
    with open(config_backup_path, 'w', encoding='utf-8') as f:
        yaml.dump(raw_config, f, allow_unicode=True, default_flow_style=False)
    logger.info(f"參數設定已備份至: {config_backup_path}")

    # 4. 實例化模型
    scorer_client = OllamaModelClient(**scorer_cfg.__dict__)
    optimizer_client = OllamaModelClient(**optimizer_cfg.__dict__)
    
    # 5. 更新 opt_cfg (將 log_dir 指向新建立的實驗資料夾)
    opt_cfg.dataset_name = raw_config['optimization'].get('dataset_name', 'mmlu')
    opt_cfg.task_name = task_name
    opt_cfg.log_dir = experiment_dir # 重要：這樣 top_prompts.csv 就會存在這裡
    opt_cfg.instruction_pos = raw_config['optimization'].get('instruction_pos', 'A_begin')
    opt_cfg.is_instruction_tuned = raw_config['optimization'].get('is_instruction_tuned', False)

    logger.info("="*50)
    logger.info(f"Experiment Start: {experiment_folder_name}")
    logger.info(f"Target Model: {scorer_cfg.model_name}")
    logger.info("="*50)

    # 6. 執行優化
    try:
        best_result = run_opro_optimization(
            scorer_client=scorer_client,
            optimizer_client=optimizer_client,
            config=opt_cfg
        )
        
        # 7. [新增] 統計並儲存 Token Cost 資訊
        token_cost_data = {
            "scorer_model": {
                "name": scorer_cfg.model_name,
                "usage": scorer_client.usage_stats
            },
            "optimizer_model": {
                "name": optimizer_cfg.model_name,
                "usage": optimizer_client.usage_stats
            },
            "total_calls": scorer_client.usage_stats['call_count'] + optimizer_client.usage_stats['call_count']
        }
        
        token_file_path = os.path.join(experiment_dir, "token_cost.json")
        with open(token_file_path, 'w', encoding='utf-8') as f:
            json.dump(token_cost_data, f, indent=4)
            
        logger.info(f"Token 統計資訊已儲存至: {token_file_path}")
        logger.info(f"Scorer Usage: {scorer_client.usage_stats}")
        logger.info(f"Optimizer Usage: {optimizer_client.usage_stats}")

        print(f"\n==========================================")
        print(f"優化完成! 結果已儲存於: {experiment_dir}")
        print(f"==========================================")
        
    except Exception as e:
        logger.exception("程式執行期間發生未預期的錯誤:")
        raise e

if __name__ == '__main__':
    main()