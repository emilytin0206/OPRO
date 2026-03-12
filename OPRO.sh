#!/bin/bash

# ========================================================
# OPRO 批量實驗自動化腳本 (前 N 筆截斷版)
# ========================================================

TEMP_CONFIG="config/config_auto_run.yaml"

# 1. 基礎設定
SCORERS=("qwen2.5:7b") 
OPTIMIZER="qwen2.5:32b"

# 2. 定義你的子集 (用逗號分隔，中間不要有空白)
DATASET="mmlu"
SUBSETS="high_school_mathematics,high_school_world_history,high_school_physics,professional_law,business_ethics"  # 5 科

# 3. 題數與資料控制設定
PER_SUBSET_LIMIT=100  # 每個子集要取的最前面幾筆

# 自動計算總數量 (計算逗號數量 + 1 即為子集數量)
NUM_SUBSETS=$(echo "$SUBSETS" | tr -cd ',' | wc -c)
NUM_SUBSETS=$((NUM_SUBSETS + 1))
TOTAL_LIMIT=$((PER_SUBSET_LIMIT * NUM_SUBSETS)) # 例如 100 * 5 = 500

SHUFFLE_DATA="true"   # true: 將這 500 筆徹底打散再訓練 / false: 依科目順序訓練

# 4. 固定參數
ITERATIONS=50            
PROMPTS_PER_STEP=8       
OPTIMIZER_TEMP=1.0       

# ========================================================
# 主執行迴圈
for SCORER in "${SCORERS[@]}"; do
    
    SUBSET_YAML_LIST=$(echo "$SUBSETS" | sed "s/,/','/g")
    SPLIT="test"

    echo "================================================================"
    echo " Scorer Model : $SCORER"
    echo " Optimizer    : $OPTIMIZER"
    echo " Dataset      : $DATASET"
    echo " Subsets      : $SUBSETS ($NUM_SUBSETS 科)"
    echo " Per Subset   : $PER_SUBSET_LIMIT 筆 (取最前面)"
    echo " Total Limit  : $TOTAL_LIMIT 筆"
    echo " Shuffle Data : $SHUFFLE_DATA"
    echo "================================================================"

    mkdir -p config

    cat > "$TEMP_CONFIG" <<EOF
project:
  log_dir: './logs'

dataset:
  name: '$DATASET'
  split: '$SPLIT'
  subsets: ['$SUBSET_YAML_LIST']
  train_limit: $TOTAL_LIMIT
  data_root: './data'
  shuffle: $SHUFFLE_DATA

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
    - Solve this problem carefully.
    - Provide a detailed answer.
    - Answer the question directly.
EOF

    python main.py --config "$TEMP_CONFIG"

    if [ $? -eq 0 ]; then
        echo "✅ 實驗完成"
    else
        echo "❌ 實驗發生錯誤"
    fi

    echo "等待 3 秒釋放資源..."
    sleep 3
done

rm -f "$TEMP_CONFIG"