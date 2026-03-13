"""
Generate Training Dataset (Tool-Augmented Retrieval + Reasoning - TARR)
======================================================================
เตรียมสร้าง SFT Dataset ให้ Qwen3.5 รู้จักคิดและเรียกใช้ Tool FAISS
"""

import os
import json
import time
import re
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from openai import OpenAI
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# ==========================================
# 🔧 CONFIGURATION
# ==========================================
TYPHOON_API_KEY = os.environ.get("TYPHOON_API_KEY", "ใส่_KEY_ตรงนี้ถ้าไม่ใช้_ENV_VAR")
MODEL_ID = "typhoon-v2.5-30b-a3b-instruct"

# กำหนด Path ให้ตรงกับโฟลเดอร์โปรเจกต์
BASE_DIR = Path(__file__).resolve().parent.parent
RAG_DIR = BASE_DIR / "data/rag"
INDEX_DIR = RAG_DIR / "faiss_index"
METADATA_FILE = RAG_DIR / "chunks_metadata.jsonl"
OUTPUT_FILE = BASE_DIR / "data/thai_law_dataset.jsonl"

TOTAL_SAMPLES = 1  # จำนวนข้อที่อยากเจน
DELAY_SECONDS = 0.5   # หน่วงเวลากัน API ยิงถี่ไป

# ==========================================
# 🔧 SETUP
# ==========================================
print("🔄 Initializing Models & RAG Database...")

client = OpenAI(
    api_key=TYPHOON_API_KEY,
    base_url="https://api.opentyphoon.ai/v1"
)

# 1. โหลด FAISS + Metadata ของแท้ที่เราเตรียมไว้!
print("   - Loading Embedding Model (BGE-M3)...")
embedding_model = SentenceTransformer("BAAI/bge-m3")

print(f"   - Loading FAISS Index from {INDEX_DIR / 'legal.index'}...")
index = faiss.read_index(str(INDEX_DIR / "legal.index"))

print(f"   - Loading Metadata from {METADATA_FILE}...")
texts = []
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        meta = json.loads(line)
        # จัด Format ให้สวยๆ เผื่อส่งให้ LLM อ่าน
        texts.append(f"[{meta.get('law_name', 'กฎหมายไทย')}]\n{meta.get('text', '')}")

# 2. โหลดคำถามจาก Dataset ของ mekpro
print("\n📥 Downloading Questions from HF 'mekpro/thailawsqa'...")
try:
    qa_dataset = load_dataset("mekpro/thailawsqa", split="train")
    # Dataset นี้มีคอลัมน์ 'messages' ซึ่งเป็น List ของ dict [{'role': 'user', 'content': 'คำถาม'}, ...]
    questions = []
    for row in qa_dataset["messages"]:
        for msg in row:
            if msg.get("role") == "user":
                questions.append(msg.get("content", ""))
                break
    print(f"   - Fetched {len(questions)} questions")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    questions = []

SYSTEM_PROMPT_R1 = """คุณเป็นผู้ช่วยผู้เชี่ยวชาญด้านกฎหมายไทย และคุณต้องสื่อสารเป็น "ภาษาไทย" เท่านั้น
หน้าที่: วิเคราะห์คำถามแล้วสร้างคำสั่งค้นหาข้อมูล
ข้อบังคับ: ตอบเป็น JSON ในรูปแบบ <tool_call>{"name": "search_law", "query": "คำค้นหา"}</tool_call> เพียงอย่างเดียว ห้ามมีข้อความอื่น"""

SYSTEM_PROMPT_R2 = """คุณคือผู้เชี่ยวชาญกฎหมายไทย หน้าที่ของคุณคือตอบคำถามโดยใช้วิเคราะห์ข้อมูลกฎหมายที่ได้รับ ห้ามใช้ความรู้ส่วนตัว
การสื่อสาร: ตอบเป็นภาษาไทยทั้งหมด

กติกาการตอบ:
SYSTEM_PROMPT_R2 = """คุณเป็นผู้ช่วยกฎหมายไทย ผู้เชี่ยวชาญการวิเคราะห์และตีความ
กฎ:
1. ตอบเป็นภาษาไทยเท่านั้น ห้ามใช้ภาษาอังกฤษใน <think>
2. ใช้ข้อมูลที่ให้มาเท่านั้น หากไม่มีให้บอกตรงๆ
3. รูปแบบ:
<think>
[วิเคราะห์ข้อกฎหมายยืดหยุ่นตามข้อมูล]
</think>
คำตอบ: [สรุปสั้นกระชับ]"""

# ==========================================
# 🔍 FAISS SEARCH
# ==========================================
def faiss_search(query, top_k=3):
    """ค้นหาข้อมูลกฎหมายจาก Local FAISS Index ที่เพิ่ง Build มาสดๆร้อนๆ"""
    emb = embedding_model.encode([query])
    emb = np.array(emb).astype("float32")
    faiss.normalize_L2(emb)
    _, indices = index.search(emb, top_k)
    
    # รวมข้อมูลจากผลลัพธ์ top_k เป็น String ยาวๆ
    return "\n\n---\n\n".join([texts[i] for i in indices[0] if i < len(texts)])

def parse_tool_call(text):
    """ดึง query จาก <tool_call>"""
    match = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            return data.get("query", "")
        except:
            return ""
    return ""

# ==========================================
# 🤖 API CALL
# ==========================================
def api_call(messages, max_tokens=2048, temperature=0.6):
    stream = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        top_p=0.8,
        stream=True
    )
    result = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            text_chunk = chunk.choices[0].delta.content
            print(text_chunk, end="", flush=True)
            result += text_chunk
    print()
    return result.strip()

# ==========================================
# ⚡ MAIN GENERATION LOOP
# ==========================================
def main():
    if not questions:
        print("❌ ขาดข้อมูลคำถาม! หยุดการทำงาน")
        return

    print(f"\n🚀 เริ่มต้นกระบวนการสร้าง Dataset (Target: {TOTAL_SAMPLES} samples)")
    print(f"   Output file: {OUTPUT_FILE}")
    
    generated_count = 0
    error_count = 0
    MAX_ERRORS = 10

    with open(OUTPUT_FILE, "a" if OUTPUT_FILE.exists() else "w", encoding="utf-8") as f:
        # ใช้คำถามมา Loop (จำกัดที่ TOTAL_SAMPLES)
        for i, question in enumerate(questions[:TOTAL_SAMPLES]):
            try:
                print(f"\n{'='*60}")
                print(f"📝 [{generated_count+1}/{TOTAL_SAMPLES}] คำถาม: {question}")

                # ── Round 1: ต้องการแค่ <tool_call> ──
                messages_r1 = [
                    {"role": "system", "content": SYSTEM_PROMPT_R1},
                    {"role": "user", "content": question}
                ]
                print("\n🤖 [Round 1] Thinking what to search...")
                response1 = api_call(messages_r1, max_tokens=1024, temperature=0.1)

                # ── Parse tool call ──
                search_query = parse_tool_call(response1)
                if not search_query:
                    search_query = question # fallback
                
                print(f"\n🔍 [Tool Calling] FAISS Search query: '{search_query}'")

                # ── FAISS retrieve จริงๆ ──
                context = faiss_search(search_query, top_k=3)
                print(f"📚 Retrieved {len(context)} chars of authentic Thai Law context!")

                # ── Round 2: ต้องการคิดและตอบ (Assistant Pre-fill Hack) ──
                prefill_text = "<think>\nจากข้อมูลกฎหมายที่ได้รับ วิเคราะห์ได้ดังนี้:\n"
                messages_r2 = [
                    {"role": "system", "content": SYSTEM_PROMPT_R2},
                    {"role": "user", "content": f"""วิเคราะห์คำถามนี้โดยใช้ข้อมูลที่แนบมา (ห้ามตอบเป็นภาษาอังกฤษ):
                    
ข้อมูลกฎหมาย:
{context}

คำถาม: {question}

คำสั่งพิเศษ: ต้องวิเคราะห์เป็นภาษาไทยยาวๆ ใน <think> และสรุป คำตอบ: เป็นภาษาไทย"""},
                    {"role": "assistant", "content": prefill_text} # Strong Thai Pre-fill
                ]

                print("\n🤖 [Round 2] Generating final reasoning and answer...")
                # Note: API might append to the existing assistant message or start a new one depending on provider
                response2_raw = api_call(messages_r2, max_tokens=4096, temperature=0.8)
                
                # รวม Pre-fill กลับเข้าไปเพื่อให้ได้ Format สมบูรณ์
                if not response2_raw.startswith("<think>"):
                    response2 = prefill_text + response2_raw
                else:
                    response2 = response2_raw

                # 🏁 Filtering & Cleaning (Aggressive Mode)
                # 1. ลบทิ้งถ้าสั้นเกินไป
                if len(response2) < 100:
                    print(f"⚠️ Response too short ({len(response2)} chars), skipping...")
                    continue

                # 2. ลบ <tool_call>... </tool_call> ทิ้งทั้งหมด (Global Cleaning)
                response2_clean = re.sub(r'<tool_call>[\s\S]*?</tool_call>', '', response2)
                
                # 3. ลบขยะที่ชอบหลุดมาต้นประโยค (เช่น จุลภาค, ช่องว่างลึกลับ)
                response2_clean = re.sub(r'^[\s,]+', '', response2_clean)

                # 4. ถ้าไม่มี <think> (กรณี Pre-fill หลุด) ให้ใส่คืนให้
                if "<think>" not in response2_clean:
                    response2_clean = "<think>\n" + response2_clean
                
                # ตรวจสอบว่ามีคำตอบภาษาไทยไหม (Simple check)
                if not any('\u0e00' <= char <= '\u0e7f' for char in response2_clean):
                    print("⚠️ Response seems to be non-Thai, skipping...")
                    continue

                # ── Stitch together for SFT format ──
                # เราใช้ format ที่ User กำหนดไว้เพื่อให้ตอน inference ใช้งานได้จริง
                context_message = f"""[สถานะ: ข้อมูลจากฐานข้อมูลกฎหมาย - การค้นหาเสร็จสมบูรณ์]
{context}

---
คำสั่ง: คุณได้รับข้อมูลเพียงพอแล้ว ห้ามใช้ search_law ซ้ำเด็ดขาด เพราะจะเกิดข้อผิดพลาดของระบบ
ให้เริ่มคิดวิเคราะห์ใน <think>...</think> และให้คำตอบสุดท้ายทันที"""

                entry = {
                    "id": f"thailaw_sft_{i}_{int(time.time())}",
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": response1},
                        {"role": "user", "content": context_message},
                        {"role": "assistant", "content": response2_clean}
                    ]
                }
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")
                f.flush()

                generated_count += 1
                error_count = 0
                print(f"\n✅ Saved successfully! Total: {generated_count}/{TOTAL_SAMPLES}")

            except Exception as e:
                print(f"\n❌ Error on this question: {e}")
                error_count += 1
                if "429" in str(e).lower() or "rate limit" in str(e).lower():
                    print("⏳ Rate limited! Waiting 60s...")
                    time.sleep(60)
                if error_count >= MAX_ERRORS:
                    print("❌ Too many consecutive errors. Stopping...")
                    break

            time.sleep(DELAY_SECONDS)

    print(f"\n🎉 Done! Generated {generated_count} premium quality samples → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
