"""
Thai Legal RAG Builder (Memory-Efficient Version)
==================================================
Embedding : BAAI/bge-m3
Vector DB : FAISS
Sources   : ทุก source เดิม (thailaw, wangchanx, ratchakitcha, csv)

แก้ไข: ใช้ streaming + incremental FAISS เพื่อลด RAM ใน Mac

Output:
  - data/rag/faiss_index/         ← FAISS index
  - data/rag/chunks_metadata.jsonl ← metadata ของทุก chunk
"""

# ต้อง set ก่อน import ทุกอย่าง — ป้องกัน segfault ใน Mac
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


import re
import json
import glob
import gc
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Generator

# ============================================================
# CONFIG
# ============================================================
BASE_DIR      = Path("~/Desktop/ThaiLegalLLM").expanduser()
RAW_DIR       = BASE_DIR / "data/raw"
RAG_DIR       = BASE_DIR / "data/rag"
INDEX_DIR     = RAG_DIR / "faiss_index"
METADATA_FILE = RAG_DIR / "chunks_metadata.jsonl"

RAG_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else " (CPU — slow)"))

CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64
# H100/GPU: batch ใหญ่ได้, Mac CPU: เล็กเพื่อป้องกัน segfault
EMBED_BATCH   = 256 if DEVICE == "cuda" else 8
ADD_BATCH     = 5000 if DEVICE == "cuda" else 200

LEGAL_KEYWORDS = [
    "พระราชบัญญัติ", "พระราชกำหนด", "พระราชกฤษฎีกา",
    "กฎกระทรวง", "ประกาศกระทรวง", "ระเบียบ",
    "ข้อบังคับ", "ประมวลกฎหมาย", "คำสั่ง", "ประกาศคณะกรรมการ"
]

# ============================================================
# CHUNKER
# ============================================================
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """แบ่ง text เป็น chunks โดย split ที่ขอบประโยค"""
    sentences = re.split(r"(?<=[।।\.\n])\s*", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks, current, current_len = [], [], 0
    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len > chunk_size and current:
            chunks.append(" ".join(current))
            overlap_sents, overlap_len = [], 0
            for s in reversed(current):
                if overlap_len + len(s) > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_len += len(s)
            current = overlap_sents
            current_len = overlap_len
        current.append(sent)
        current_len += sent_len

    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c) > 50]


# ============================================================
# STREAMING LOADERS — yield ทีละ doc ไม่โหลดทั้งหมดพร้อมกัน
# ============================================================
def stream_all_docs() -> Generator[Dict, None, None]:
    """Stream documents ทีละอัน เพื่อประหยัด RAM"""

    # --- 1. ThaiLaw ---
    print("📚 Streaming ThaiLaw...")
    for fpath in glob.glob(str(RAW_DIR / "thailaw/**/*.parquet"), recursive=True):
        df = pd.read_parquet(fpath, columns=["text", "title"])
        for _, row in df.iterrows():
            text  = str(row.get("text", ""))
            title = str(row.get("title", ""))
            if len(text) < 50:
                continue
            yield {
                "text": f"{title}\n\n{text}" if title else text,
                "source": "thailaw-v1.0",
                "law_name": title,
                "article": "",
                "publish_date": ""
            }
        del df
        gc.collect()

    # --- 2. WangchanX ---
    print("⚖️  Streaming WangchanX...")
    for fpath in glob.glob(str(RAW_DIR / "wangchanx/**/*.parquet"), recursive=True):
        df = pd.read_parquet(fpath, columns=["context", "question", "answer", "law_name", "article"])
        for _, row in df.iterrows():
            context  = str(row.get("context", ""))
            law_name = str(row.get("law_name", ""))
            if context and len(context) > 50:
                yield {
                    "text": context,
                    "source": "wangchanx-legal",
                    "law_name": law_name,
                    "article": str(row.get("article", "")),
                    "publish_date": ""
                }
        del df
        gc.collect()

    # --- 3. Ratchakitcha ---
    print("📜 Streaming Ratchakitcha OCR...")
    meta_index = {}
    for mf in glob.glob(str(RAW_DIR / "ratchakitcha/meta/**/*.jsonl"), recursive=True):
        with open(mf, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    meta_index[rec.get("pdf_file", "")] = rec
                except:
                    pass

    for fpath in glob.glob(str(RAW_DIR / "ratchakitcha/ocr/**/*.jsonl"), recursive=True):
        with open(fpath, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec      = json.loads(line)
                    pdf_file = rec.get("pdf_file", "")
                    text     = rec.get("text", "") or rec.get("ocr_text", "")
                    meta     = meta_index.get(pdf_file, {})
                    doctitle = meta.get("doctitle", "")

                    if not any(kw in doctitle for kw in LEGAL_KEYWORDS):
                        continue
                    if len(text) < 100:
                        continue

                    yield {
                        "text": f"{doctitle}\n\n{text}",
                        "source": "soc-ratchakitcha",
                        "law_name": doctitle,
                        "article": "",
                        "publish_date": str(meta.get("publishDate", ""))
                    }
                except:
                    pass

    # --- 4. CSV ---
    print("📋 Streaming CSV folder...")
    csv_files = glob.glob(str(RAW_DIR / "csv/**/*.csv"), recursive=True)
    csv_files += glob.glob(str(RAW_DIR / "csv/*.csv"))
    for fpath in csv_files:
        law_name = Path(fpath).stem.replace("_", " ").strip()
        try:
            df = pd.read_csv(fpath)
            if "text" not in df.columns:
                continue
            if "is-cancelled" in df.columns:
                df = df[df["is-cancelled"].isna() | (df["is-cancelled"] == "")]
            for _, row in df.iterrows():
                text    = str(row.get("text", ""))
                article = str(row.get("article", ""))
                if len(text) < 20:
                    continue
                yield {
                    "text": text,
                    "source": "csv-law",
                    "law_name": law_name,
                    "article": article,
                    "publish_date": ""
                }
            del df
            gc.collect()
        except Exception as e:
            print(f"  ⚠️  {fpath}: {e}")


# ============================================================
# BUILD FAISS INDEX — Incremental (ประหยัด RAM)
# ============================================================
def build_index():
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ ติดตั้ง dependencies ก่อน:")
        print("   pip install faiss-cpu sentence-transformers")
        return

    print("\n🤖 Loading BGE-M3 model...")
    model = SentenceTransformer("BAAI/bge-m3", device=DEVICE)
    model.max_seq_length = 512

    index = None   # สร้างตอนรู้ dimension จาก batch แรก
    chunk_id = 0
    chunk_buffer = []
    meta_buffer  = []

    meta_fh = open(METADATA_FILE, "w", encoding="utf-8")

    def flush_buffer():
        """Embed + add chunk_buffer เข้า FAISS แล้วล้าง buffer"""
        nonlocal index

        if not chunk_buffer:
            return

        texts = [m["text"] for m in meta_buffer]
        print(f"  🔢 Embedding {len(texts):,} chunks...", flush=True)

        if DEVICE == "cuda":
            # GPU: encode ทีเดียวได้เลย เร็วมาก
            embeddings = model.encode(
                texts,
                batch_size=EMBED_BATCH,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True
            ).astype("float32")
        else:
            # Mac CPU: encode ทีละ EMBED_BATCH เพื่อป้องกัน segfault
            all_embs = []
            for i in range(0, len(texts), EMBED_BATCH):
                sub = texts[i:i + EMBED_BATCH]
                emb = model.encode(
                    sub,
                    batch_size=EMBED_BATCH,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                ).astype("float32")
                all_embs.append(emb)
                del emb
                gc.collect()
            embeddings = np.concatenate(all_embs, axis=0)
            del all_embs

        if index is None:
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            print(f"  📐 FAISS index created (dim={dim})")

        index.add(embeddings)

        for meta in meta_buffer:
            meta_fh.write(json.dumps(meta, ensure_ascii=False) + "\n")

        del embeddings
        gc.collect()
        chunk_buffer.clear()
        meta_buffer.clear()

    print("\n🔪 Chunking + Embedding (streaming)...")
    doc_count = 0
    for doc in tqdm(stream_all_docs(), desc="Documents"):
        doc_count += 1
        doc_chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(doc_chunks):
            chunk_buffer.append(chunk)
            meta_buffer.append({
                "chunk_id"    : chunk_id,
                "text"        : chunk,
                "source"      : doc["source"],
                "law_name"    : doc["law_name"],
                "article"     : doc["article"],
                "publish_date": doc["publish_date"],
                "chunk_index" : i
            })
            chunk_id += 1

            # Flush ทุก ADD_BATCH chunks
            if len(chunk_buffer) >= ADD_BATCH:
                flush_buffer()

    # flush ที่เหลือ
    flush_buffer()
    meta_fh.close()

    if index is None:
        print("❌ ไม่มีข้อมูล — ตรวจสอบ RAW_DIR")
        return

    # save index
    faiss.write_index(index, str(INDEX_DIR / "legal.index"))
    print(f"\n✅ FAISS index saved → {INDEX_DIR / 'legal.index'}")
    print(f"✅ Metadata saved → {METADATA_FILE}")
    print(f"\n📊 Index stats:")
    print(f"   Documents : {doc_count:,}")
    print(f"   Vectors   : {index.ntotal:,}")
    print(f"   Size      : {os.path.getsize(INDEX_DIR / 'legal.index') / 1024 / 1024:.1f} MB")


# ============================================================
# TEST RETRIEVAL
# ============================================================
def test_retrieval():
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except:
        return

    print("\n🧪 Testing retrieval...")
    model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    index = faiss.read_index(str(INDEX_DIR / "legal.index"))

    metas = []
    with open(METADATA_FILE, encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))

    test_queries = [
        "นายจ้างไม่จ่ายเงินเดือน ทำยังไงได้บ้าง",
        "ซื้อของออนไลน์โดนโกง ร้องเรียนที่ไหน",
        "สัญญาเช่าบ้านต้องมีอะไรบ้าง"
    ]

    for query in test_queries:
        print(f"\n❓ Query: {query}")
        emb = model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")

        scores, indices = index.search(emb, k=3)
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            meta = metas[idx]
            print(f"  [{rank+1}] score={score:.3f} | {meta['law_name']} | {meta['text'][:80]}...")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Thai Legal RAG Builder (Memory-Efficient)")
    print("=" * 60)
    print("\n📦 Install dependencies ก่อนถ้ายังไม่ได้ติดตั้ง:")
    print("   pip install faiss-cpu sentence-transformers")
    print()

    build_index()
    test_retrieval()

    print("\n" + "=" * 60)
    print("✅ RAG index พร้อมใช้แล้ว!")
    print(f"   Index    : {INDEX_DIR / 'legal.index'}")
    print(f"   Metadata : {METADATA_FILE}")
    print("=" * 60)
