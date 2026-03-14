import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

def main():
    parser = argparse.ArgumentParser(description="Test Thai Legal LLM with Streaming")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model or checkpoint folder")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"🔄 Checking path: {args.model_path}...")
    
    # แปลงเป็น Absolute Path เพื่อให้ transformers มั่นใจว่าเป็นเครื่องเรา ไม่ใช่ Repo ID
    model_path = os.path.abspath(args.model_path)
    
    # ระบบค้นหาอัตโนมัติ: ถ้าในโฟลเดอร์ที่ส่งมาไม่มี config.json ให้ลองหาในทายาท (เผื่อซ้อนโฟลเดอร์)
    if os.path.isdir(model_path):
        if not os.path.exists(os.path.join(model_path, "config.json")):
            print("🔍 config.json not found in root. Searching in subdirectories...")
            found = False
            for root, dirs, files in os.walk(model_path):
                # ตรวจสอบไฟล์สำคัญของโมเดล
                if "config.json" in files or "model.safetensors" in files or "pytorch_model.bin" in files:
                    model_path = root
                    print(f"📍 Found valid model files at: {model_path}")
                    found = True
                    break
            if not found:
                print("❌ Warning: No valid model files (config.json) found in subdirectories.")
    else:
        print(f"⚠️ Path {model_path} is not a directory. Proceeding as Repo ID...")

    print(f"🔄 Loading model and tokenizer from: {model_path}...")

    try:
        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Load Model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map=args.device if torch.cuda.is_available() else "auto"
        )
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        # ข้อมูลบางครั้ง Checkpoint อาจจะไม่มีครบ (เช่น ขาด tokenizer) 
        # ลอง Fallback ไปที่ Base model สำหรับ Tokenizer ถ้าพัง
        if "tokenizer" in str(e).lower():
            print("🔄 Attempting to load tokenizer from base model (Qwen/Qwen3.5-9B-Base)...")
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B-Base", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                device_map=args.device
            )
        else:
            raise e
    
    # Setup Streamer
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print("\n✅ Model Ready! (Type 'exit' to quit)")
    print("-" * 50)

    while True:
        prompt = input("\n👤 User: ")
        if prompt.lower() in ["exit", "quit", "ออก"]:
            break
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        print("🤖 Assistant: ", end="", flush=True)
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        print("\n" + "-" * 50)

if __name__ == "__main__":
    main()
