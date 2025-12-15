import yaml
from dataclasses import dataclass
from typing import Type
from src.utils import setup_logger
from src.models.base_client import BaseModelClient
from src.models.ollama_client import OllamaModelClient
# 假設您有一個 PaLM 客戶端:
# from src.models.palm_client import PaLMModelClient 
from src.core.optimization import optimize_instructions # 重構後的主邏輯

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
    top_k_prompts: int

def load_config(config_path: str = 'config/config.yaml'):
    """讀取 YAML 配置檔"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 解析模型配置
    scorer_cfg = ModelConfig(**config['scorer_model'])
    optimizer_cfg = ModelConfig(**config['optimizer_model'])
    
    # 解析優化參數
    opt_cfg = OptimizationConfig(**config['optimization'])
    
    return scorer_cfg, optimizer_cfg, opt_cfg

def get_model_client(cfg: ModelConfig) -> BaseModelClient:
    """根據配置檔動態實例化模型客戶端"""
    if cfg.client_type == 'Ollama':
        return OllamaModelClient(
            model_name=cfg.model_name,
            api_url=cfg.api_url,
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens
        )
    # elif cfg.client_type == 'PaLM':
    #     return PaLMModelClient(...) 
    else:
        raise ValueError(f"不支援的 Model Client 類型: {cfg.client_type}")

def main():
    # 1. 加載配置
    scorer_cfg, optimizer_cfg, opt_cfg = load_config()
    
    # 提取專案設定
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        raw_config = yaml.safe_load(f)
        project_config = raw_config.get('project', {})
    
    task_name = project_config.get('task_name', 'default_task')
    log_dir = project_config.get('log_dir', './logs')

    # 2. 設定 Logger (這是關鍵新增)
    # logger 會自動被 src 下的模組使用 (因為我們用了 logging.getLogger("OPRO"))
    logger, _ = setup_logger(log_dir, task_name)
    
    logger.info("程式啟動，配置載入完成。")

    # 3. 實例化模型
    # (這裡假設我們用 Ollama)
    scorer_client = OllamaModelClient(**scorer_cfg.__dict__)
    optimizer_client = OllamaModelClient(**optimizer_cfg.__dict__) # 需注意這裡參數傳遞方式需與 class init 匹配
    
    # 4. 補充 opt_cfg 的額外資訊 (傳給 optimization.py 使用)
    opt_cfg.dataset_name = 'bbh' # 範例，應從 config 讀取
    opt_cfg.task_name = task_name
    opt_cfg.log_dir = log_dir    # 讓 optimization 知道 CSV 存哪

    # 5. 執行優化
    try:
        best_result = run_opro_optimization(
            scorer_client=scorer_client,
            optimizer_client=optimizer_client,
            config=opt_cfg
        )
        logger.info("主程式執行完畢。")
        
    except Exception as e:
        logger.exception("程式執行期間發生未預期的錯誤:")
        raise e

if __name__ == '__main__':
    main()

print(f"\n==========================================")
    print(f"優化完成! 最佳指令如下:")
    print(f"Instruction: {best_prompt['instruction']}")
    print(f"Final Score: {best_prompt['score']}")
    print(f"==========================================")


if __name__ == '__main__':

    main()