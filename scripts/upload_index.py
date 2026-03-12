"""
Upload RAG Index to HuggingFace
===============================
อัพโหลดไฟล์ FAISS index และ metadata กลับไปที่ HuggingFace
"""

import sys
from pathlib import Path
from huggingface_hub import HfApi, login

# ตั้งค่า path
BASE_DIR = Path(__file__).resolve().parent.parent
RAG_DIR = BASE_DIR / "data/rag"
INDEX_FILE = RAG_DIR / "faiss_index/legal.index"
METADATA_FILE = RAG_DIR / "chunks_metadata.jsonl"

REPO_ID = "Phonsiriwillbejommarn/thai-legal-pretrain"

def upload_rag_index(token):
    print("=" * 60)
    print("🚀 Uploading RAG Index to HuggingFace...")
    print("=" * 60)
    
    if not INDEX_FILE.exists() or not METADATA_FILE.exists():
        print("❌ ไม่พบไฟล์ index หรือ metadata กรุณารัน build_rag.py ให้เสร็จก่อน")
        print(f"   หาไฟล์: {INDEX_FILE}")
        print(f"   หาไฟล์: {METADATA_FILE}")
        return
        
    try:
        # Login ด้วย token
        print("🔑 Logging in to HuggingFace...")
        login(token=token)
        
        api = HfApi()
        
        # Upload metadata
        print(f"📤 Uploading {METADATA_FILE.name}...")
        api.upload_file(
            path_or_fileobj=str(METADATA_FILE),
            path_in_repo=f"rag/{METADATA_FILE.name}",
            repo_id=REPO_ID,
            repo_type="dataset"
        )
        
        # Upload index
        print(f"📤 Uploading {INDEX_FILE.name}...")
        api.upload_file(
            path_or_fileobj=str(INDEX_FILE),
            path_in_repo=f"rag/faiss_index/{INDEX_FILE.name}",
            repo_id=REPO_ID,
            repo_type="dataset"
        )
        
        print("\n✅ Upload complete!")
        print(f"👉 เช็คไฟล์ได้ที่: https://huggingface.co/datasets/{REPO_ID}/tree/main/rag")
        
    except Exception as e:
        print(f"\n❌ Error uploading: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        token = sys.argv[1]
    else:
        token = input("🔑 ใส่ HuggingFace Token: ").strip()
        
    if not token:
        print("❌ ต้องใส่ Token ถึงจะอัพโหลดได้ครับ")
        sys.exit(1)
        
    upload_rag_index(token)
