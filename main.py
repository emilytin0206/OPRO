# main.py
import yaml
import os
import argparse
import datetime
import json
from dataclasses import dataclass, asdict
from typing import List, Union

from src.utils import setup_logger
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
class DatasetConfig:
    name: str
    split: str
    subsets: Union[List[str], str]
    train_limit: Union[int, str]
    data_root: str
    shuffle: bool = True  # 預設為 True

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
    
    # [修正重點] 增強 Subsets 解析邏輯，支援逗號分隔字串
    if 'subsets' in ds_raw:
        val = ds_raw['subsets']
        if val is None:
            ds_raw['subsets'] = []
        elif isinstance(val, str):
            if val.lower() == 'all':
                pass  # 保持 'all' 字串
            elif ',' in val:
                # 如果是 "math,physics"，自動切分為 ['math', 'physics']
                ds_raw['subsets'] = [x.strip() for x in val.split(',') if x.strip()]
            else:
                # 單一子集字串轉為列表
                ds_raw['subsets'] = [val]
    else:
        ds_raw['subsets'] = []

    ds_cfg = DatasetConfig(**ds_raw)

    scorer_cfg = ModelConfig(**raw['scorer_model'])
    optimizer_cfg = ModelConfig(**raw['optimizer_model'])
    
    # Optimization 處理
    opt_dict = raw['optimization']
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
    # 格式: OPRO_<target>_<opt>_<dataset>_<Num>Sub_Lim<Limit>_<Shuffle/Seq>_<Date>
    target_name = clean_name(cfg.scorer_model.model_name)
    opt_name = clean_name(cfg.optimizer_model.model_name)
    ds_name = cfg.dataset.name
    
    # 計算子集數量
    if isinstance(cfg.dataset.subsets, list):
        num_sub = len(cfg.dataset.subsets)
        sub_info = f"{num_sub}Sub"
    elif str(cfg.dataset.subsets).lower() == 'all':
        sub_info = "AllSub"
    else:
        sub_info = "1Sub"
        
    limit_val = str(cfg.dataset.train_limit)
    limit_info = f"Lim{limit_val}"
    
    # Shuffle 資訊
    is_shuffle = getattr(cfg.dataset, 'shuffle', True)
    shuffle_info = "Shuffle" if is_shuffle else "Seq"

    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    experiment_folder_name = f"OPRO_{target_name}_{opt_name}_{ds_name}_{sub_info}_{limit_info}_{shuffle_info}_{date_str}"
    
    # 完整路徑
    experiment_dir = os.path.join(cfg.project.log_dir, experiment_folder_name)
    
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    
    print(f"建立實驗資料夾: {experiment_dir}")

    # 3. 更新 Config 中的 Log Dir
    cfg.project.log_dir = experiment_dir
    
    # 4. 備份 Config
    config_backup_path = os.path.join(experiment_dir, "config_snapshot.yaml")
    with open(config_backup_path, 'w', encoding='utf-8') as f:
        print(f"正在備份設定檔至: {config_backup_path}")
        yaml.dump(asdict(cfg), f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    # 5. 設定 Logger
    logger, _ = setup_logger(experiment_dir, "run")
    logger.info(f"Experiment Folder: {experiment_folder_name}")

    # 6. 實例化 Client
    scorer_client = OllamaModelClient(**cfg.scorer_model.__dict__)
    optimizer_client = OllamaModelClient(**cfg.optimizer_model.__dict__)

    # 7. 執行優化
    try:
        run_opro_optimization(
            scorer_client=scorer_client,
            optimizer_client=optimizer_client,
            config=cfg
        )
        
        # 8. 統計 Token Cost
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