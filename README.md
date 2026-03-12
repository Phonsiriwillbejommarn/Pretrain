# ThaiLegalLLM — Pretrain

Thai Legal Language Model — Continued Pretraining & RAG Pipeline

## Scripts

| Script | Description |
|--|--|
| `scripts/build_rag.py` | Build FAISS RAG index from Thai legal sources (BGE-M3) |

## Data Sources
- **ThaiLaw-v1.0** — pythainlp/thailaw-v1.0
- **WangchanX Legal** — wangchanx-legal
- **Ratchakitcha OCR** — สำนักงานราชกิจจานุเบกษา
- **CSV Law** — ประมวลกฎหมายจากหน่วยงานต่างๆ

## Setup

```bash
pip install faiss-cpu sentence-transformers
# or for GPU:
pip install faiss-gpu sentence-transformers

python scripts/build_rag.py
```

> Data files are not included in this repo (see `.gitignore`)
