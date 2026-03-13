import json
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from itertools import chain

class PackedCPTDataset(IterableDataset):
    """
    Dataset สำหรับ Continued Pre-training (CPT) 
    ที่ทำการ pack tokens ให้เต็ม max_length เพื่อประสิทธิภาพสูงสุด
    """
    def __init__(self, data_path, tokenizer, max_length=4096, shuffle=True, seed=42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_path = data_path
        
        # รองรับหลายไฟล์ (คั่นด้วย comma)
        paths = [p.strip() for p in data_path.split(",")]
        
        # โหลด dataset แบบ streaming
        self.dataset = load_dataset("json", data_files=paths, split="train", streaming=True)
        
        if shuffle:
            self.dataset = self.dataset.shuffle(seed=seed, buffer_size=10000)

    def __iter__(self):
        token_buffer = []
        
        for example in self.dataset:
            text = example.get("text", "")
            if not text:
                continue
            
            # Tokenize และเพิ่ม EOS
            tokens = self.tokenizer.encode(text) + [self.tokenizer.eos_token_id]
            token_buffer.extend(tokens)
            
            # เมื่อ buffer เต็มพอที่จะสร้าง chunks
            while len(token_buffer) >= self.max_length:
                chunk = token_buffer[:self.max_length]
                token_buffer = token_buffer[self.max_length:]
                
                # Causal LM: labels == input_ids
                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "labels": torch.tensor(chunk, dtype=torch.long)
                }
