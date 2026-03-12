"""
Download Cleaned Thai Legal Data from HuggingFace
=================================================
ใช้สำหรับดึงไฟล์ jsonl ที่ทำความสะอาดแล้วลง server โดยตรง
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download

# ตั้งค่า path
BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_DIR = BASE_DIR / "data/cleaned"
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

# กำหนด repo และไฟล์
REPO_ID = "Phonsiriwillbejommarn/thai-legal-pretrain"  # เปลี่ยนเป็นชื่อ repo จริงของคุณ
FILENAME = "thai_legal_pretrain.jsonl"
LOCAL_PATH = CLEANED_DIR / FILENAME

def download_data():
    print("=" * 60)
    print(f"📥 Downloading {FILENAME} from HuggingFace...")
    print(f"   Repo: {REPO_ID}")
    print("=" * 60)
    
    try:
        # ดาวน์โหลดไฟล์ (ระบุ repo_type="dataset")
        # ถ้า dataset เป็น private ต้องรัน `huggingface-cli login` ก่อน
        file_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            repo_type="dataset",
            local_dir=CLEANED_DIR,
            local_dir_use_symlinks=False
        )
        
        size_gb = os.path.getsize(file_path) / (1024**3)
        print("\n✅ Download complete!")
        print(f"   Saved to: {file_path}")
        print(f"   Size    : {size_gb:.2f} GB")
        
    except Exception as e:
        print(f"\n❌ Error downloading data: {e}")
        print("\n⚠️ ถ้า dataset เป็น private:")
        print("   1. สมัคร/Login HuggingFace")
        print("   2. รันคำสั่ง: huggingface-cli login")
        print("   3. รัน script นี้ใหม่")

if __name__ == "__main__":
    download_data()
