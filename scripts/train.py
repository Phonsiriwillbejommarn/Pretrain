"""
train.py — Continued Pretraining (CPT) สำหรับ Qwen3.5-9B-Base
Thai Law Domain Adaptation

Usage:
  # Single GPU
  python train.py --config configs/single_gpu.yaml

  # Multi GPU (DeepSpeed ZeRO-2)
  deepspeed --num_gpus=4 train.py --deepspeed ds_config_zero2.json

  # Multi GPU (DeepSpeed ZeRO-3)
  deepspeed --num_gpus=8 train.py --deepspeed ds_config_zero3.json
"""

import os
import math
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint

from dataset import PackedCPTDataset

logger = logging.getLogger(__name__)

# ─── Arguments ────────────────────────────────────────────────────────────────

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen3.5-9B-Base",
        metadata={"help": "Path to pretrained model or HF Hub name"},
    )
    trust_remote_code: bool = field(default=True)
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "bfloat16 | float16 | float32"},
    )
    attn_implementation: str = field(
        default="sdpa",
        metadata={"help": "sdpa | flash_attention_2 | eager"},
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="../data/cleaned/thai_legal_pretrain.jsonl",
        metadata={"help": "Path to pretrain_data_clean.jsonl"},
    )
    max_seq_length: int = field(
        default=4096,
        metadata={"help": "Max sequence length for packing"},
    )
    shuffle_buffer_size: int = field(
        default=10_000,
        metadata={"help": "Shuffle buffer size for streaming dataset"},
    )
    seed: int = field(default=42)


@dataclass
class CPTTrainingArguments(TrainingArguments):
    # Override defaults สำหรับ CPT
    output_dir: str = field(default="./output")
    num_train_epochs: float = field(default=1.0)
    per_device_train_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=8)
    learning_rate: float = field(default=1e-5)
    lr_scheduler_type: str = field(default="cosine")
    warmup_ratio: float = field(default=0.05)
    weight_decay: float = field(default=0.01)
    max_grad_norm: float = field(default=1.0)
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=500)
    save_total_limit: int = field(default=3)
    dataloader_num_workers: int = field(default=4)
    report_to: str = field(default="tensorboard")
    remove_unused_columns: bool = field(default=False)
    # Gradient checkpointing ช่วยประหยัด VRAM
    gradient_checkpointing: bool = field(default=True)
    gradient_checkpointing_kwargs: dict = field(
        default_factory=lambda: {"use_reentrant": False}
    )


# ─── Estimate total steps ──────────────────────────────────────────────────────

def estimate_steps(data_path: str, max_seq_length: int, batch_size: int,
                   grad_accum: int, num_epochs: float) -> int:
    """ประมาณจำนวน training steps"""
    total_chars = 0
    import json
    import os
    paths = [p.strip() for p in data_path.split(",")]
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    total_chars += len(obj.get("text", ""))
                except:
                    pass
    # ประมาณ token: 1 token ≈ 2 chars สำหรับภาษาไทย
    total_tokens = total_chars // 2
    tokens_per_step = max_seq_length * batch_size * grad_accum
    steps_per_epoch = total_tokens // tokens_per_step
    total_steps = int(steps_per_epoch * num_epochs)
    logger.info(f"Estimated total steps: {total_steps:,} "
                f"({total_tokens:,} tokens, {steps_per_epoch:,} steps/epoch)")
    return total_steps


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CPT for Qwen3.5-9B-Base (Thai Law)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--data_path", type=str, default="../data/cleaned/thai_legal_pretrain.jsonl")
    parser.add_argument("--output_dir", type=str, default="./output/qwen3.5-9b-thai-law")
    parser.add_argument("--push_to_hub", action="store_true", default=False)
    parser.add_argument("--hub_model_id", type=str, default="Phonsiri/Qwen3.5-9B-Thai-Law-Base")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--attn_implementation", type=str, default="sdpa",
                        choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--report_to", type=str, default="all")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    # DeepSpeed (ถ้าใช้)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    # Logging setup
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    set_seed(args.seed)

    # ── Load tokenizer ─────────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer from: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        padding_side="right",
    )
    # Qwen3.5 มี eos_token แต่ถ้าไม่มี pad_token ให้ใช้ eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Load model ─────────────────────────────────────────────────────────────
    logger.info(f"Loading model from: {args.model}")
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
        device_map=None,  # Trainer จัดการ device เอง
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

    # แสดง model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {args.model}")
    logger.info(f"Total parameters    : {total_params / 1e9:.2f}B")
    logger.info(f"Trainable parameters: {trainable_params / 1e9:.2f}B")

    # ── Build Dataset ──────────────────────────────────────────────────────────
    logger.info(f"Building dataset from: {args.data_path}")
    train_dataset = PackedCPTDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        shuffle=True,
        seed=args.seed,
    )

    # ── Training Arguments ─────────────────────────────────────────────────────
    if args.max_steps > 0:
        max_steps = args.max_steps
    else:
        num_gpus = int(os.environ.get("WORLD_SIZE", "1"))
        max_steps = estimate_steps(
            args.data_path,
            args.max_seq_length,
            args.per_device_train_batch_size * num_gpus,
            args.gradient_accumulation_steps,
            args.num_train_epochs
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=200, # ตั้งค่าคงที่ไปเลย (ประมาณ 5% ของ 3000 steps)
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=args.bf16,
        fp16=False,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to=args.report_to,
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=args.deepspeed,
        local_rank=args.local_rank,
        # Checkpoint config (สำคัญสำหรับ resume จากพังบนคลาวด์)
        save_only_model=False, # เซฟ optimizer ด้วย จะได้กลับมาเทรนต่อได้
        # HF Hub Config
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_strategy="checkpoint", # push ทั้ง model และ checkpoint
        hub_token=args.hub_token,
        # Logging
        run_name="qwen3.5-9b-thai-law-cpt",
    )

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # ── Resume checkpoint ──────────────────────────────────────────────────────
    checkpoint = args.resume_from_checkpoint
    if checkpoint is None:
        last_ckpt = get_last_checkpoint(args.output_dir)
        if last_ckpt is not None:
            checkpoint = last_ckpt
            logger.info(f"Resuming from checkpoint: {checkpoint}")

    # ── Train ──────────────────────────────────────────────────────────────────
    logger.info("🚀 Starting CPT training...")
    trainer.train(resume_from_checkpoint=checkpoint)

    # ── Save final model ───────────────────────────────────────────────────────
    logger.info("💾 Saving final model and tokenizer...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    if args.push_to_hub:
        logger.info(f"☁️ Pushing final model to Hub: {args.hub_model_id}...")
        trainer.push_to_hub(commit_message="Training complete!")
        
    logger.info(f"✅ Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
