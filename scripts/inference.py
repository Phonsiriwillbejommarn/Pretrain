import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

def main():
    parser = argparse.ArgumentParser(description="Test Thai Legal LLM with Streaming")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model or checkpoint folder")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"🔄 Loading model from: {args.model_path}...")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map=args.device
    )
    
    # Setup Streamer
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print("\n✅ Model Ready! (Type 'exit' to quit)")
    print("-" * 50)

    while True:
        prompt = input("\n👤 User: ")
        if prompt.lower() in ["exit", "quit", "ออก"]:
            break
        
        # สำหรับ Base Model (CPT) เรามักจะใช้ข้อความดิบๆ หรือ Prompt เริ่มต้น
        # เนื่องจากยังไม่ได้ทำ SFT (Instruction Tuning) จึงเน้นต่อประโยค
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        
        print("🤖 Assistant: ", end="", flush=True)
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=512,
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
