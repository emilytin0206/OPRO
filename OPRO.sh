#!/bin/bash

# ========================================================
# OPRO 批量實驗自動化腳本
# ========================================================

# 設定暫存 Config 路徑 (腳本會自動產生這個檔案)
TEMP_CONFIG="config/config_auto_run.yaml"

# --------------------------------------------------------
# 1. 定義要測試的 Scorer 模型列表
# --------------------------------------------------------
SCORERS=("qwen2.5:7b") 

# --------------------------------------------------------
# 2. 定義要測試的資料集與子集
# 格式: "Dataset_Name Subset_List" (子集間用逗號分隔，無空格)
# 修正重點: 必須在開頭指定 mmlu，程式才能正確解析
# --------------------------------------------------------
TASKS=(
    "mmlu high_school_mathematics,high_school_chemistry,high_school_physics,high_school_world_history,business_ethics"
)

# --------------------------------------------------------
# 3. 定義訓練資料量限制 (Train Limit)
# --------------------------------------------------------
LIMITS=(100)

# --------------------------------------------------------
# 4. 固定參數 (已更新為論文最佳設定)
# --------------------------------------------------------
OPTIMIZER="qwen2.5:32b"
ITERATIONS=50             # 論文建議至少跑 50-100 步
PROMPTS_PER_STEP=8        # 論文設定每步生成 8 個指令
OPTIMIZER_TEMP=1.0        # 優化器溫度設為 1.0 以增加創意

# ========================================================
# 主執行迴圈
# ========================================================

mkdir -p logs

for SCORER in "${SCORERS[@]}"; do
    for TASK_INFO in "${TASKS[@]}"; do
        
        # 1. 解析 Dataset 和 Raw Subsets
        # 輸入範例: "mmlu math,physics,history"
        read -r DATASET SUBSET_RAW <<< "$TASK_INFO"
        
        # 2. [關鍵修正] 將逗號分隔的字串轉換為 YAML 列表格式
        # 轉換前: math,physics
        # 轉換後: math','physics (準備放入 YAML 的 ['...'] 中)
        SUBSET_YAML_LIST=$(echo "$SUBSET_RAW" | sed "s/,/','/g")

        # 3. 判斷 Split
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
            echo "  Dataset      : $DATASET"
            echo "  Subsets      : $SUBSET_RAW"
            echo "  Train Limit  : $LIMIT (per subset, shuffled)"
            echo "================================================================"

            # 產生暫時的 Config YAML
            cat > "$TEMP_CONFIG" <<EOF
project:
  log_dir: './logs'

dataset:
  name: '$DATASET'
  split: '$SPLIT'
  # 這裡會被展開為 subsets: ['math','physics','...']
  subsets: ['$SUBSET_YAML_LIST'] 
  train_limit: $LIMIT
  data_root: './data'
  shuffle: true

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
  temperature: $OPTIMIZER_TEMP
  max_output_tokens: 2048

optimization:
  num_iterations: $ITERATIONS
  num_prompts_to_generate: $PROMPTS_PER_STEP
  max_num_instructions_in_prompt: 20
  meta_prompt_path: 'prompt/meta_prompt.txt'
  eval_interval: 5
  instruction_pos: 'Q_begin'
  is_instruction_tuned: true
  num_few_shot_questions: 3
  few_shot_selection_criteria: 'random'
  initial_instructions:
    - "Let's think step by step."
    - "Answer the question directly."
    - "Solve this problem carefully."
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