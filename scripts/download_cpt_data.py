import os
import argparse
from huggingface_hub import hf_hub_download, list_repo_files

def download_data(repo_id, filename, local_dir, token=None):
    print(f"📥 กำลังตรวจสอบไฟล์ใน Repository: {repo_id}...")
    try:
        # ตรวจสอบไฟล์ที่มีใน repo ก่อน
        files = list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
        print(f"📄 พบไฟล์ใน repository: {files}")
        
        if filename not in files:
            # กรณีหาไม่เจอตรงๆ ลองหาไฟล์ที่ลงท้ายด้วย .jsonl
            jsonl_files = [f for f in files if f.endswith(".jsonl")]
            if jsonl_files:
                print(f"⚠️ ไม่พบไฟล์ '{filename}' แต่พบไฟล์ JSONL อื่นๆ: {jsonl_files}")
                filename = jsonl_files[0]
                print(f"🔄 จะลองดาวน์โหลดไฟล์: {filename} แทน")
            else:
                print(f"❌ ไม่พบไฟล์ที่ลงท้ายด้วย .jsonl ใน repository")
                # แสดง Error เพื่อให้หยุด
                raise FileNotFoundError(f"File '{filename}' not found in repo '{repo_id}'")

        print(f"📥 เริ่มดาวน์โหลด: {filename}...")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            repo_type="dataset",
            token=token,
            local_dir_use_symlinks=False
        )
        print(f"✅ ดาวน์โหลดสำเร็จ: {path}")
        
        # คืนชื่อไฟล์ที่โหลดได้ (เผื่อมีการเปลี่ยนชื่อ)
        return filename
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการดาวน์โหลด: {e}")
        # ถ้าติด Permission/Not Found ให้แสดงรายการไฟล์ให้ผู้ใช้ดู
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset from HF Hub using Python")
    parser.add_argument("--repo_id", type=str, required=True, help="HF Dataset repository ID")
    parser.add_argument("--filename", type=str, default="thai_legal_pretrain.jsonl", help="Filename to download")
    parser.add_argument("--local-dir", dest="local_dir", type=str, default="../data/cleaned", help="Local directory to save")
    
    args = parser.parse_args()
    
    # Get token from environment if available
    token = os.environ.get("HF_TOKEN")
    
    download_data(args.repo_id, args.filename, args.local_dir, token)
