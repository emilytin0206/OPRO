import yaml
from dataclasses import dataclass
from typing import Type
import logging # 加入 logging

# [修正 1] Import 路徑改為單數 'model'
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
    num_evals_per_prompt: int
    num_prompts_to_generate: int
    # [修正] 移除 config 中不存在的 top_k_prompts，改用 max_num_instructions_in_prompt
    max_num_instructions_in_prompt: int 
    meta_prompt_path: str
    # [新增] 支援更多參數的動態屬性
    task_name: str = ""
    dataset_name: str = ""
    log_dir: str = ""
    instruction_pos: str = "A_begin"
    is_instruction_tuned: bool = False
    num_few_shot_questions: int = 3
    few_shot_selection_criteria: str = "random"

def load_config(config_path: str = 'config/config.yaml'):
    """讀取 YAML 配置檔"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    scorer_cfg = ModelConfig(**config['scorer_model'])
    optimizer_cfg = ModelConfig(**config['optimizer_model'])
    
    # 使用解包並過濾掉 dataclass 不支援的 key (如果 yaml 有多寫的話)
    opt_dict = config['optimization']
    # 這裡簡單處理：只取 dataclass 定義的欄位，或者直接傳遞字典
    # 為了嚴謹，我們先用字典初始化部分，後續動態賦值
    known_keys = OptimizationConfig.__annotations__.keys()
    filtered_opt_dict = {k: v for k, v in opt_dict.items() if k in known_keys}
    opt_cfg = OptimizationConfig(**filtered_opt_dict)
    
    return scorer_cfg, optimizer_cfg, opt_cfg, config # 回傳原始 config 以便讀取 project 設定

def main():
    # 1. 加載配置
    scorer_cfg, optimizer_cfg, opt_cfg, raw_config = load_config()
    
    project_config = raw_config.get('project', {})
    task_name = project_config.get('task_name', 'default_task')
    log_dir = project_config.get('log_dir', './logs')

    # 2. 設定 Logger
    logger, _ = setup_logger(log_dir, task_name)
    logger.info("程式啟動，配置載入完成。")

    # 3. 實例化模型
    # 這裡假設兩個都用 Ollama，如果 scorer 用別的需加判斷
    scorer_client = OllamaModelClient(**scorer_cfg.__dict__)
    optimizer_client = OllamaModelClient(**optimizer_cfg.__dict__)
    
    # 4. 補充 opt_cfg 的額外資訊
    opt_cfg.dataset_name = 'bbh' # 暫定，建議從 config 讀取 dataset_name
    opt_cfg.task_name = task_name
    opt_cfg.log_dir = log_dir
    
    # 讀取額外參數 (如果 YAML 有寫)
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
        
        # [修正 2 & 3] 將結果列印移入 try 區塊，並使用正確的變數 best_result
        print(f"\n==========================================")
        print(f"優化完成! 最佳指令如下:")
        print(f"Instruction: {best_result['instruction']}")
        print(f"Final Score: {best_result['score']}")
        print(f"Step Generated: {best_result.get('step', 'N/A')}")
        print(f"==========================================")
        
    except Exception as e:
        logger.exception("程式執行期間發生未預期的錯誤:")
        raise e

if __name__ == '__main__':
    main()