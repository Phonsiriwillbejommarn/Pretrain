import os
import argparse
from huggingface_hub import hf_hub_download

def download_data(repo_id, filename, local_dir, token=None):
    print(f"📥 Starting download: {filename} from {repo_id}...")
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            repo_type="dataset",
            token=token,
            local_dir_use_symlinks=False
        )
        print(f"✅ Download complete: {path}")
    except Exception as e:
        print(f"❌ Error downloading file: {e}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset from HF Hub using Python")
    parser.add_argument("--repo_id", type=str, required=True, help="HF Dataset repository ID")
    parser.add_argument("--filename", type=str, default="thai_legal_pretrain.jsonl", help="Filename to download")
    parser.add_argument("--local_dir", type=str, default="../data/cleaned", help="Local directory to save")
    
    args = parser.parse_args()
    
    # Get token from environment if available
    token = os.environ.get("HF_TOKEN")
    
    download_data(args.repo_id, args.filename, args.local_dir, token)
