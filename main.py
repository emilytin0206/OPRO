# main.py
import yaml
import os
import argparse
import datetime
import json
from dataclasses import dataclass
from typing import List

from src.utils import setup_logger
from src.model.ollama_client import OllamaModelClient
from src.core.optimization import run_opro_optimization

# 簡單的 Config 類別定義
@dataclass
class ModelConfig:
    client_type: str
    model_name: str
    api_url: str
    temperature: float
    max_output_tokens: int

@dataclass
class DatasetConfig:
    name: str
    split: str
    subsets: List[str] | str
    train_limit: int | str
    data_root: str

@dataclass
class OptimizationConfig:
    num_iterations: int
    num_prompts_to_generate: int
    max_num_instructions_in_prompt: int 
    meta_prompt_path: str
    eval_interval: int
    task_name: str = ""
    dataset_name: str = ""
    instruction_pos: str = "A_begin"
    is_instruction_tuned: bool = False
    num_few_shot_questions: int = 3
    few_shot_selection_criteria: str = "random"
    initial_instructions: List[str] = None
    old_instruction_score_threshold: float = 0.1

@dataclass
class ProjectConfig:
    log_dir: str

@dataclass
class GlobalConfig:
    project: ProjectConfig
    dataset: DatasetConfig
    scorer_model: ModelConfig
    optimizer_model: ModelConfig
    optimization: OptimizationConfig

def clean_name(name):
    """清理名稱中的特殊字元，用於路徑"""
    return name.replace(':', '-').replace('/', '_').replace(' ', '_')

def load_config(config_path: str) -> GlobalConfig:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到設定檔: {config_path}")
    print(f"正在載入設定檔: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    
    # 建立 Config 物件
    proj_cfg = ProjectConfig(**raw['project'])
    
    # Dataset 處理
    ds_raw = raw['dataset']
    # 確保 subsets 是 list 或 str
    if 'subsets' in ds_raw and isinstance(ds_raw['subsets'], list) == False and ds_raw['subsets'] != 'all':
         if ds_raw['subsets'] is None: ds_raw['subsets'] = []
         else: ds_raw['subsets'] = [str(ds_raw['subsets'])]
    ds_cfg = DatasetConfig(**ds_raw)

    scorer_cfg = ModelConfig(**raw['scorer_model'])
    optimizer_cfg = ModelConfig(**raw['optimizer_model'])
    
    # Optimization 處理
    opt_dict = raw['optimization']
    # 過濾多餘的 key
    known_keys = OptimizationConfig.__annotations__.keys()
    filtered_opt = {k: v for k, v in opt_dict.items() if k in known_keys}
    opt_cfg = OptimizationConfig(**filtered_opt)

    return GlobalConfig(proj_cfg, ds_cfg, scorer_cfg, optimizer_cfg, opt_cfg)

def main():
    parser = argparse.ArgumentParser(description="OPRO Optimization Runner")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    args = parser.parse_args()

    # 1. 載入配置
    cfg = load_config(args.config)
    
    # 2. 自動生成實驗資料夾名稱
    # 格式: OPRO_<target>_<opt>_<dataset>_<subset>_<limit>_<date>
    target_name = clean_name(cfg.scorer_model.model_name)
    opt_name = clean_name(cfg.optimizer_model.model_name)
    ds_name = cfg.dataset.name
    
    # 處理 subset 命名
    if isinstance(cfg.dataset.subsets, str) and cfg.dataset.subsets.lower() == 'all':
        sub_name = "all"
    elif isinstance(cfg.dataset.subsets, list) and len(cfg.dataset.subsets) > 0:
        if len(cfg.dataset.subsets) == 1:
            sub_name = cfg.dataset.subsets[0]
        else:
            # 如果有多個，取第一個加後綴，避免檔名過長
            sub_name = f"{cfg.dataset.subsets[0]}_plus_{len(cfg.dataset.subsets)-1}"
    else:
        sub_name = "default"
        
    limit_name = str(cfg.dataset.train_limit)
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    experiment_folder_name = f"OPRO_{target_name}_{opt_name}_{ds_name}_{sub_name}_{limit_name}_{date_str}"
    
    # 完整路徑
    experiment_dir = os.path.join(cfg.project.log_dir, experiment_folder_name)
    
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    
    print(f"建立實驗資料夾: {experiment_dir}")

    # 3. 更新 Config 中的 Log Dir (讓其他模組知道寫入哪裡)
    cfg.project.log_dir = experiment_dir
    
    # 4. 備份 Config
    config_backup_path = os.path.join(experiment_dir, "config_snapshot.yaml")
    with open(config_backup_path, 'w', encoding='utf-8') as f:
        # 將 dataclass 轉回 dict 比較麻煩，這裡直接讀原始檔案再寫入一次最保險，或者用 yaml dump raw
        # 為了簡單我們手動 dump 必要的資訊
        pass # 此處省略，建議在 load_config 時一併回傳 raw dict 以便備份

    # 5. 設定 Logger
    logger, _ = setup_logger(experiment_dir, "run")
    logger.info(f"Experiment: {experiment_folder_name}")

    # 6. 實例化 Client
    scorer_client = OllamaModelClient(**cfg.scorer_model.__dict__)
    optimizer_client = OllamaModelClient(**cfg.optimizer_model.__dict__)

    # 7. 執行優化
    try:
        run_opro_optimization(
            scorer_client=scorer_client,
            optimizer_client=optimizer_client,
            config=cfg # 傳入整個 GlobalConfig
        )
        
        # 8. 統計 Token Cost (簡單版)
        token_cost_data = {
            "scorer_usage": scorer_client.usage_stats,
            "optimizer_usage": optimizer_client.usage_stats
        }
        with open(os.path.join(experiment_dir, "token_cost.json"), 'w') as f:
            json.dump(token_cost_data, f, indent=4)
            
    except Exception as e:
        logger.exception("執行失敗:")
        raise e

if __name__ == '__main__':
    main()