#!/bin/bash

# ========================================================
# OPRO 批量實驗自動化腳本
# ========================================================

# 設定暫存 Config 路徑 (腳本會自動產生這個檔案)
TEMP_CONFIG="config/config_auto_run.yaml"

# --------------------------------------------------------
# 1. 定義要測試的 Scorer 模型列表
# --------------------------------------------------------
# 您可以在這裡填入多個 Ollama 模型名稱
SCORERS=("qwen2.5:7b") 
# SCORERS=("qwen2.5:7b" "llama3.1:8b" "gemma2:9b")

# --------------------------------------------------------
# 2. 定義要測試的資料集與子集
# 格式: "Dataset_Name Subset_Name"
# 注意: 如果是 GSM8K，Subset 填 train 即可 (代表使用訓練集進行優化)
# --------------------------------------------------------
TASKS=(
    "mmlu high_school_mathematics"
    "mmlu professional_law"
    "gsm8k train" 
)

# --------------------------------------------------------
# 3. 定義訓練資料量限制 (Train Limit)
# --------------------------------------------------------
LIMITS=(20 50)

# --------------------------------------------------------
# 4. 固定參數 (優化器模型與其他設定)
# --------------------------------------------------------
OPTIMIZER="qwen2.5:32b"
ITERATIONS=10
PROMPTS_PER_STEP=4

# ========================================================
# 主執行迴圈
# ========================================================

mkdir -p logs

for SCORER in "${SCORERS[@]}"; do
    for TASK_INFO in "${TASKS[@]}"; do
        
        # 解析 Dataset 和 Subset
        read -r DATASET SUBSET <<< "$TASK_INFO"
        
        # 判斷 Split (GSM8K 通常跑 train, MMLU 跑 test)
        SPLIT="test"
        if [ "$DATASET" == "gsm8k" ]; then
            SPLIT="train"
        fi

        for LIMIT in "${LIMITS[@]}"; do
            echo "================================================================"
            echo "正在執行實驗 (Running Experiment)"
            echo "----------------------------------------------------------------"
            echo "  Scorer Model : $SCORER"
            echo "  Optimizer    : $OPTIMIZER"
            echo "  Dataset      : $DATASET ($SUBSET)"
            echo "  Split        : $SPLIT"
            echo "  Train Limit  : $LIMIT"
            echo "================================================================"

            # 產生暫時的 Config YAML
            # 使用 cat <<EOF 動態寫入設定檔，這裡對應新版的 Config 結構
            cat > "$TEMP_CONFIG" <<EOF
project:
  log_dir: './logs'

dataset:
  name: '$DATASET'
  split: '$SPLIT'
  subsets: ['$SUBSET'] 
  train_limit: $LIMIT
  data_root: './data'

scorer_model:
  client_type: 'Ollama'
  model_name: '$SCORER'
  api_url: 'http://localhost:11434/api/chat'
  temperature: 0.0
  max_output_tokens: 1024

optimizer_model:
  client_type: 'Ollama'
  model_name: '$OPTIMIZER'
  api_url: 'http://localhost:11434/api/chat'
  temperature: 0.7
  max_output_tokens: 2048

optimization:
  num_iterations: $ITERATIONS
  num_prompts_to_generate: $PROMPTS_PER_STEP
  max_num_instructions_in_prompt: 10
  meta_prompt_path: 'prompt/meta_prompt.txt'
  eval_interval: 3
  instruction_pos: 'Q_begin'
  is_instruction_tuned: true
  num_few_shot_questions: 3
  few_shot_selection_criteria: 'random'
  initial_instructions:
    - "Let's think step by step."
    - "Answer the question directly."
EOF

            # 執行 Python 主程式
            python main.py --config "$TEMP_CONFIG"

            # 檢查執行結果
            if [ $? -eq 0 ]; then
                echo "✅ 實驗完成"
            else
                echo "❌ 實驗發生錯誤"
            fi
            
            echo ""
            # 休息 3 秒讓 GPU 降溫或釋放資源
            sleep 3
        done
    done
done

# 清理暫存檔
rm "$TEMP_CONFIG"
echo "所有實驗已結束。"