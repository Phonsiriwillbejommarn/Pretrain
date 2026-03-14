#!/bin/bash
# ============================================================
#  run_train_single_gpu.sh — Single GPU (ไม่ใช้ DeepSpeed)
#  สำหรับ GPU VRAM ≥ 80GB (A100/H100/H200)
# ============================================================

# หยุดทำงานทันทีถ้ามีคำสั่งใด Error
set -e

# หาตำแหน่งของสคริปต์นี้
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

MODEL="Qwen/Qwen3.5-9B-Base"
# ใช้ Absolute Path เสมอเพื่อให้ Python ชัวร์
DATA_PATH="$SCRIPT_DIR/../data/cleaned/thai_legal_pretrain.jsonl"
OUTPUT_DIR="$SCRIPT_DIR/output/qwen3.5-9b-thai-law-cpt"
CHECKPOINT_DIR="$OUTPUT_DIR/last-checkpoint"

# Hugging Face Hub (ใส่ Token ใน Environment Variable: export HF_TOKEN="your_token")
HF_TOKEN=${HF_TOKEN:-""}
HUB_MODEL_ID="Phonsiri/Qwen3.5-9B-Thai-Law-Base"
DATASET_REPO="Phonsiri/Somdataset"
CHECKPOINT_REPO="Phonsiri/Qwen3.5-9B-Thai-Law-Base"

# Weights & Biases (ใส่ Token ที่นี่ถ้าต้องการ Track Log)
export WANDB_API_KEY=""
export WANDB_PROJECT="Qwen3.5-9B-Thai-Law-CPT"
export WANDB_NAME="run-1-cpt-H100"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$SCRIPT_DIR/../data/cleaned"

# --- Cloud Data Pull (Dataset) ---
# บังคับให้โหลดใหม่ถ้าไม่มีไฟล์ เพื่อความปลอดภัย
if [ ! -f "$DATA_PATH" ]; then
    echo "📥 Downloading dataset from Hugging Face Hub (via Python)..."
    export HF_TOKEN=$HF_TOKEN
    # สคริปต์แก้ไขใหม่ จะพยายามเดาชื่อไฟล์ถ้าไม่ตรง
    python "$SCRIPT_DIR/download_cpt_data.py" --repo_id "$DATASET_REPO" --filename "thai_legal_pretrain.jsonl" --local-dir "$SCRIPT_DIR/../data/cleaned"
    
    # ตรวจสอบอีกครั้งว่าโหลดมาแล้วชื่อไฟล์ตรงไหม (ถ้า download_cpt_data.py มีการเดาชื่อไฟล์)
    # เราจะหาไฟล์ .jsonl ในโฟลเดอร์ออกมาเป็น DATA_PATH จริงๆ
    DATA_PATH=$(find "$SCRIPT_DIR/../data/cleaned" -name "*.jsonl" | head -n 1)
    echo "📍 Final DATA_PATH set to: $DATA_PATH"
else
    echo "✅ Dataset already exists locally at $DATA_PATH"
fi

# --- Checkpoint Pull (Resume Logic) ---
if [ ! -f "$CHECKPOINT_DIR/config.json" ]; then
    echo "🔍 Checkpoint not found locally. Attempting to pull from $CHECKPOINT_REPO..."
    mkdir -p "$CHECKPOINT_DIR"
    python "$SCRIPT_DIR/download_checkpoint.py" --repo_id "$CHECKPOINT_REPO" --local-dir "$CHECKPOINT_DIR" || echo "⚠️ Skip checkpoint download (might be first run)"
else
    echo "✅ Checkpoint exists. Preparing to resume..."
fi
# ----------------------

# Hyperparameters
MAX_SEQ_LENGTH=2048
NUM_EPOCHS=1
BATCH_SIZE=2                         # per-device batch size
GRAD_ACCUM=32                        # effective batch = 64
LR=2e-5                              # CPT LR

# ตรวจสอบว่ามี checkpoint หรือไม่ เพื่อส่งเข้า train.py
RESUME_FLAG=""
if [ -f "$CHECKPOINT_DIR/config.json" ]; then
    RESUME_FLAG="--resume_from_checkpoint $CHECKPOINT_DIR"
    echo "🔄 Found valid checkpoint, enabling Resume mode."
fi

echo "🚀 Launching Training..."
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
    $RESUME_FLAG \
    ${HF_TOKEN:+--push_to_hub} \
    ${HF_TOKEN:+--hub_token "$HF_TOKEN"} \
    ${HUB_MODEL_ID:+--hub_model_id "$HUB_MODEL_ID"}

echo "✅ Training complete!"
