import json
import torch
import random
import os
from torch.utils.data import IterableDataset

class PackedCPTDataset(IterableDataset):
    """
    Dataset สำหรับ Continued Pre-training (CPT) 
    ที่ทำการ pack tokens ให้เต็ม max_length เพื่อประสิทธิภาพสูงสุด
    ขยับมาใช้การอ่าน JSONL แบบ Manual แทน load_dataset เพื่อข้ามขั้นตอนการแปลงไฟล์ของ HF
    """
    def __init__(self, data_path, tokenizer, max_length=4096, shuffle=True, seed=42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_path = data_path
        self.shuffle = shuffle
        self.seed = seed
        
        # รองรับหลายไฟล์ (คั่นด้วย comma)
        self.paths = [p.strip() for p in data_path.split(",")]
        # ทำให้เป็น absolute path เพื่อความชัวร์
        self.paths = [os.path.abspath(p) if not os.path.isabs(p) else p for p in self.paths]
        
        # ตรวจสอบว่าไฟล์มีจริง
        for p in self.paths:
            if not os.path.exists(p):
                print(f"⚠️ Warning: Dataset file not found: {p}")

    def __iter__(self):
        token_buffer = []
        
        # ผสมลำดับไฟล์ถ้ามีการ shuffle
        file_paths = self.paths.copy()
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(file_paths)
            
        for fpath in file_paths:
            if not os.path.exists(fpath):
                continue
                
            with open(fpath, "r", encoding="utf-8") as f:
                # ถ้า Shuffle เราจะอ่านเข้ามาก่อนบางส่วน (buffer) หรืออ่านทีละบรรทัด
                # สำหรับไฟล์ใหญ่มากๆ แนะนำให้อ่านทีละบรรทัด
                for line in f:
                    try:
                        example = json.loads(line)
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
                            
                            yield {
                                "input_ids": torch.tensor(chunk, dtype=torch.long),
                                "labels": torch.tensor(chunk, dtype=torch.long)
                            }
                    except Exception as e:
                        # ข้ามบรรทัดที่พัง
                        continue
