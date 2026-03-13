#!/bin/bash
# ============================================================
#  run_train_single_gpu.sh — Single GPU (ไม่ใช้ DeepSpeed)
#  สำหรับ GPU VRAM ≥ 80GB (A100/H100/H200)
# ============================================================

MODEL="Qwen/Qwen3.5-9B-Base"
DATA_PATH="../data/cleaned/thai_legal_pretrain.jsonl"
OUTPUT_DIR="./output/qwen3.5-9b-thai-law-cpt"

# Hugging Face Hub (ใส่ Token ใน Environment Variable: export HF_TOKEN="your_token")
HF_TOKEN=${HF_TOKEN:-""}
HUB_MODEL_ID="Phonsiri/Qwen3.5-9B-Thai-Law-Base"
DATASET_REPO="Phonsiri/Somdataset"

# Weights & Biases (ใส่ Token ที่นี่ถ้าต้องการ Track Log)
export WANDB_API_KEY=""
export WANDB_PROJECT="Qwen3.5-9B-Thai-Law-CPT"
export WANDB_NAME="run-1-cpt-H100"

mkdir -p $OUTPUT_DIR
mkdir -p "../data/cleaned"

# --- Cloud Data Pull ---
echo "📥 Downloading dataset from Hugging Face Hub..."
export HF_TOKEN=$HF_TOKEN
huggingface-cli download $DATASET_REPO thai_legal_pretrain.jsonl --local-dir "../data/cleaned" --local-dir-use-symlinks False
# ----------------------

# Hyperparameters
MAX_SEQ_LENGTH=4096
NUM_EPOCHS=1
BATCH_SIZE=2                         # per-device batch size
GRAD_ACCUM=16                        # effective batch = 32
LR=2e-5                              # CPT LR

CUDA_VISIBLE_DEVICES=0 python train.py \
    --model "$MODEL" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --warmup_ratio 0.05 \
    --save_steps 500 \
    --logging_steps 10 \
    --seed 42 \
    --bf16 \
    --gradient_checkpointing \
    --attn_implementation sdpa \
    --report_to all \
    ${HF_TOKEN:+--push_to_hub} \
    ${HF_TOKEN:+--hub_token "$HF_TOKEN"} \
    ${HUB_MODEL_ID:+--hub_model_id "$HUB_MODEL_ID"}

echo "✅ Training complete!"
