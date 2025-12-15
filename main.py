import yaml
import os
import argparse
import re  # [新增] 用於處理檔名
from dataclasses import dataclass, field 
from typing import Type, List            
import logging

from src.utils import setup_logger
from src.model.base_client import BaseModelClient
from src.model.ollama_client import OllamaModelClient
from src.core.optimization import run_opro_optimization

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
    log_dir = project_config.get('log_dir', './logs')

    # --- [修改] 自定義 Log 檔名 ---
    # 格式: <target_model>_<task_model>_<dataset>
    # 例如: qwen2.5-7b_qwen2.5-32b_mmlu.log
    
    def clean_name(name):
        # 將冒號、斜線等不適合檔名的字元替換掉
        return name.replace(':', '-').replace('/', '_').replace(' ', '_')

    target_model_name = clean_name(scorer_cfg.model_name)   # Scorer (Target)
    optimizer_model_name = clean_name(optimizer_cfg.model_name) # Optimizer (Task)
    dataset_name = clean_name(opt_cfg.dataset_name)

    # 組合新檔名
    log_filename_base = f"{target_model_name}_{optimizer_model_name}_{dataset_name}"
    
    # 2. 設定 Logger (傳入自定義的檔名 base)
    logger, log_file_path = setup_logger(log_dir, log_filename_base)
    
    # --- [新增] 在 Log 開頭記錄所有參數 ---
    logger.info("="*50)
    logger.info(f"Experiment Start: {log_filename_base}")
    logger.info(f"Config File: {args.config}")
    logger.info("-" * 20)
    logger.info(f"[Scorer Model] (Target): {scorer_cfg}")
    logger.info(f"[Optimizer Model]: {optimizer_cfg}")
    logger.info(f"[Optimization Params]: {opt_cfg}")
    logger.info("="*50)
    # --------------------------------------

    # 3. 實例化模型
    scorer_client = OllamaModelClient(**scorer_cfg.__dict__)
    optimizer_client = OllamaModelClient(**optimizer_cfg.__dict__)
    
    # 4. 補充 opt_cfg
    opt_cfg.dataset_name = raw_config['optimization'].get('dataset_name', 'mmlu')
    opt_cfg.task_name = task_name
    opt_cfg.log_dir = log_dir
    opt_cfg.instruction_pos = raw_config['optimization'].get('instruction_pos', 'A_begin')
    opt_cfg.is_instruction_tuned = raw_config['optimization'].get('is_instruction_tuned', False)

    # 5. 執行優化
    try:
        best_result = run_opro_optimization(
            scorer_client=scorer_client,
            optimizer_client=optimizer_client,
            config=opt_cfg
        )
        logger.info("主程式執行完畢。")
        
        print(f"\n==========================================")
        print(f"優化完成! 最佳指令如下:")
        print(f"Instruction: {best_result['instruction']}")
        print(f"Final Score: {best_result['score']}")
        print(f"Step Generated: {best_result.get('step', 'N/A')}")
        print(f"Log saved to: {log_file_path}")
        print(f"==========================================")
        
    except Exception as e:
        logger.exception("程式執行期間發生未預期的錯誤:")
        raise e

if __name__ == '__main__':
    main()