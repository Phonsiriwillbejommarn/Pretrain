"""
Thai Legal Pre-training Corpus Preprocessor
============================================
Sources:
  - data/raw/thailaw      : pythainlp/thailaw-v1.0 (parquet)
  - data/raw/wangchanx    : WangchanX-Legal-ThaiCCL-RAG (parquet)
  - data/raw/wikipedia    : wikimedia/wikipedia th (parquet)
  - data/raw/ratchakitcha : soc-ratchakitcha (meta + ocr jsonl)
  - data/raw/csv          : CSV folder (30+ กฎหมาย)

Output:
  - data/cleaned/thai_legal_pretrain.jsonl
"""

import os
import re
import json
import glob
import hashlib
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
BASE_DIR     = Path("~/Desktop/ThaiLegalLLM").expanduser()
RAW_DIR      = BASE_DIR / "data/raw"
CLEANED_DIR  = BASE_DIR / "data/cleaned"
OUTPUT_FILE  = CLEANED_DIR / "thai_legal_pretrain.jsonl"

CLEANED_DIR.mkdir(parents=True, exist_ok=True)

# กฎหมายที่เกี่ยวข้อง (filter ratchakitcha)
LEGAL_KEYWORDS = [
    "พระราชบัญญัติ", "พระราชกำหนด", "พระราชกฤษฎีกา",
    "กฎกระทรวง", "ประกาศกระทรวง", "ระเบียบ",
    "ข้อบังคับ", "ประมวลกฎหมาย", "พระราชบัญญัติประกอบ",
    "คำสั่ง", "ประกาศคณะกรรมการ"
]

# ============================================================
# UTILS
# ============================================================
seen_hashes = set()

def dedup(text: str) -> bool:
    """Return True ถ้า text ยังไม่เคยเห็น"""
    h = hashlib.md5(text.strip().encode()).hexdigest()
    if h in seen_hashes:
        return False
    seen_hashes.add(h)
    return True

def clean_text(text: str) -> str:
    """Clean OCR noise และ encoding issues"""
    if not isinstance(text, str):
        return ""
    # ลบ null bytes
    text = text.replace("\x00", "")
    # normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # ลบบรรทัดที่มีแต่ขีด/จุด (OCR artifact)
    text = re.sub(r"^[\-\_\.\s]{5,}$", "", text, flags=re.MULTILINE)
    # ลบ control characters ยกเว้น newline/tab
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()

def is_quality(text: str, min_chars: int = 100) -> bool:
    """กรอง text ที่สั้นหรือไม่มีภาษาไทย"""
    if len(text) < min_chars:
        return False
    thai_chars = len(re.findall(r"[\u0e00-\u0e7f]", text))
    if thai_chars < 20:
        return False
    return True

def write_record(f, text: str, source: str):
    """เขียน record ลงไฟล์ถ้าผ่านทุก filter"""
    text = clean_text(text)
    if not is_quality(text):
        return 0
    if not dedup(text):
        return 0
    record = {"text": text, "source": source}
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return 1

# ============================================================
# 1. ThaiLaw v1.0
# ============================================================
def process_thailaw(f):
    print("\n📚 Processing ThaiLaw v1.0...")
    count = 0
    pattern = str(RAW_DIR / "thailaw/**/*.parquet")
    files = glob.glob(pattern, recursive=True)
    for fpath in tqdm(files, desc="ThaiLaw"):
        df = pd.read_parquet(fpath)
        for _, row in df.iterrows():
            text = str(row.get("text", ""))
            title = str(row.get("title", ""))
            full = f"{title}\n\n{text}" if title else text
            count += write_record(f, full, "thailaw-v1.0")
    print(f"  ✅ {count:,} records")
    return count

# ============================================================
# 2. WangchanX-Legal
# ============================================================
def process_wangchanx(f):
    print("\n⚖️  Processing WangchanX-Legal...")
    count = 0
    pattern = str(RAW_DIR / "wangchanx/**/*.parquet")
    files = glob.glob(pattern, recursive=True)
    for fpath in tqdm(files, desc="WangchanX"):
        df = pd.read_parquet(fpath)
        for _, row in df.iterrows():
            question = str(row.get("question", ""))
            answer   = str(row.get("positive_answer", ""))

            # positive_contexts คือ list of dict {"context": "..."}
            contexts = row.get("positive_contexts", [])
            if isinstance(contexts, list):
                for ctx in contexts:
                    if isinstance(ctx, dict):
                        context = ctx.get("context", "")
                    else:
                        context = str(ctx)
                    if context:
                        count += write_record(f, context, "wangchanx-legal")

            # เขียน Q&A pair
            if question and answer and answer != "nan":
                qa_text = f"คำถาม: {question}\nคำตอบ: {answer}"
                count += write_record(f, qa_text, "wangchanx-legal-qa")

    print(f"  ✅ {count:,} records")
    return count

# ============================================================
# 3. Thai Wikipedia
# ============================================================
def process_wikipedia(f):
    print("\n📖 Processing Thai Wikipedia...")
    count = 0
    pattern = str(RAW_DIR / "wikipedia/**/*.parquet")
    files = glob.glob(pattern, recursive=True)
    for fpath in tqdm(files, desc="Wikipedia"):
        df = pd.read_parquet(fpath)
        for _, row in df.iterrows():
            title = str(row.get("title", ""))
            text  = str(row.get("text", ""))
            full  = f"{title}\n\n{text}" if title else text
            count += write_record(f, full, "thai-wikipedia")
    print(f"  ✅ {count:,} records")
    return count

# ============================================================
# 4. soc-ratchakitcha
# ============================================================
def process_ratchakitcha(f):
    print("\n📜 Processing soc-ratchakitcha (OCR)...")
    count = 0
    skip  = 0

    # หา ocr jsonl files
    ocr_pattern = str(RAW_DIR / "ratchakitcha/ocr/**/*.jsonl")
    ocr_files   = glob.glob(ocr_pattern, recursive=True)

    # หา meta jsonl files สำหรับ filter
    meta_index = {}
    meta_pattern = str(RAW_DIR / "ratchakitcha/meta/**/*.jsonl")
    for mf in glob.glob(meta_pattern, recursive=True):
        with open(mf, encoding="utf-8") as mfh:
            for line in mfh:
                try:
                    rec = json.loads(line)
                    pdf_file = rec.get("pdf_file", "")
                    doctitle = rec.get("doctitle", "")
                    meta_index[pdf_file] = doctitle
                except:
                    pass

    for fpath in tqdm(ocr_files, desc="Ratchakitcha"):
        with open(fpath, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec      = json.loads(line)
                    pdf_file = rec.get("pdf_file", "")
                    text     = rec.get("text", "") or rec.get("ocr_text", "")
                    doctitle = meta_index.get(pdf_file, "")

                    # filter เฉพาะ legal docs
                    is_legal = any(kw in doctitle for kw in LEGAL_KEYWORDS)
                    if not is_legal:
                        skip += 1
                        continue

                    full = f"{doctitle}\n\n{text}" if doctitle else text
                    count += write_record(f, full, "soc-ratchakitcha")
                except:
                    pass

    print(f"  ✅ {count:,} records (filtered out {skip:,} non-legal)")
    return count

# ============================================================
# 5. CSV folder (30+ กฎหมาย)
# ============================================================
def process_csv(f):
    print("\n📋 Processing CSV folder...")
    count = 0
    csv_files = glob.glob(str(RAW_DIR / "csv/**/*.csv"), recursive=True)
    csv_files += glob.glob(str(RAW_DIR / "csv/*.csv"))

    for fpath in tqdm(csv_files, desc="CSV"):
        law_name = Path(fpath).stem.replace("_", " ").strip()
        try:
            df = pd.read_csv(fpath)
            # group ทุกมาตราเป็น document เดียวต่อ 1 กฎหมาย
            if "text" in df.columns:
                # กรองมาตราที่ไม่ถูกยกเลิก
                if "is-cancelled" in df.columns:
                    df = df[df["is-cancelled"].isna() | (df["is-cancelled"] == "")]
                texts = df["text"].dropna().astype(str).tolist()
                full_text = f"พระราชบัญญัติ{law_name}\n\n" + "\n\n".join(texts)
                count += write_record(f, full_text, f"csv-{law_name}")
        except Exception as e:
            print(f"  ⚠️  Error reading {fpath}: {e}")

    print(f"  ✅ {count:,} records")
    return count

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("🚀 Thai Legal Pre-training Corpus Preprocessor")
    print("=" * 60)

    total = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        total += process_thailaw(f)
        total += process_wangchanx(f)
        total += process_wikipedia(f)
        total += process_ratchakitcha(f)
        total += process_csv(f)

    # สรุปผล
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 60)
    print(f"✅ เสร็จแล้ว!")
    print(f"   Total records : {total:,}")
    print(f"   Output file   : {OUTPUT_FILE}")
    print(f"   File size     : {size_mb:.1f} MB")
    print(f"   Unique hashes : {len(seen_hashes):,}")
    print("=" * 60)
    print("\n📤 ขั้นตอนต่อไป — อัพขึ้น HuggingFace:")
    print("   huggingface-cli login")
    print("   hf upload-large-folder YOUR_USERNAME/thai-legal-pretrain \\")
    print(f"     {CLEANED_DIR} --repo-type dataset --private")

if __name__ == "__main__":
    main()
