import os
import argparse
from huggingface_hub import snapshot_download

def download_checkpoint(repo_id, local_dir, token=None):
    print(f"📥 Starting checkpoint download from {repo_id} to {local_dir}...")
    try:
        # ใช้ snapshot_download เพื่อโหลดทั้งโฟลเดอร์ (รวม optimizer states)
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            repo_type="model",
            token=token,
            local_dir_use_symlinks=False,
            # โหลดเฉพาะไฟล์ที่จำเป็นสำหรับ checkpoint
            allow_patterns=["*.json", "*.bin", "*.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json"]
        )
        print(f"✅ Checkpoint download complete: {path}")
    except Exception as e:
        print(f"❌ Error downloading checkpoint: {e}")
        # ถ้าเป็นแค่ warning ให้ไปต่อได้ แต่ถ้า Error สำคัญต้องเลิก
        if "404 Client Error" in str(e):
            print("⚠️ Note: Repository might be empty or not found. Starting from base model instead.")
        else:
            exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model checkpoint from HF Hub using Python")
    parser.add_argument("--repo_id", type=str, required=True, help="HF Model repository ID")
    parser.add_argument("--local_dir", type=str, required=True, help="Local directory to save (e.g., ./output/last-checkpoint)")
    
    args = parser.parse_args()
    
    # Get token from environment if available
    token = os.environ.get("HF_TOKEN")
    
    download_checkpoint(args.repo_id, args.local_dir, token)
