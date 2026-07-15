# 🔌 vLLM Endpoints Configuration Guide

This document describes all the vLLM endpoints your project needs and how to configure them.

---

## Architecture Overview

vLLM serves **one model per process**, so you need separate server instances for each model type. Each instance exposes an **OpenAI-compatible API** (on `/v1/...`).

```
┌────────────────┐     ┌──────────────────────────────────────┐
│  project_agv   │     │           vLLM Servers                │
│                │     │                                      │
│  rag/llm.py    │────▶│  :8000/v1  ─ LLM (text generation)  │
│                │     │                                      │
│  rag/          │     │                                      │
│  embeddings.py │────▶│  :8001/v1  ─ Embedding model         │
│                │     │                                      │
│  query/        │     │                                      │
│  query_runner   │────▶│  :8002/v1  ─ Vision model (optional)│
│  + rag/llm.py  │     │                                      │
└────────────────┘     └──────────────────────────────────────┘
```

---

## Required Endpoints

### 1. 🧠 LLM Server (Text Generation)

| Setting | Value |
|---------|-------|
| **Env Var** | `LLM_BINDING_HOST` |
| **Default** | `http://127.0.0.1:8000/v1` |
| **API Key Env Var** | `LLM_BINDING_API_KEY` (default: `not_needed`) |
| **Model Env Var** | `LLM_MODEL` |
| **Default Model** | `Qwen/Qwen2.5-Coder-14B-Instruct` |

**Start command:**
```bash
vllm serve Qwen/Qwen2.5-Coder-14B-Instruct \
    --port 8000 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85
```

**Used by:**
- `rag/llm.py` → `custom_llm_func()` — all text LLM calls (entity extraction, query synthesis)
- `rag/lightrag_init.py` → `llm_model_kwargs` passed to LightRAG

**API endpoints consumed:**
- `POST /v1/chat/completions`

---

### 2. 📐 Embedding Server

| Setting | Value |
|---------|-------|
| **Env Var** | `EMBEDDING_BINDING_HOST` |
| **Default** | `http://127.0.0.1:8001/v1` |
| **API Key Env Var** | `EMBEDDING_BINDING_API_KEY` (default: `not_needed`) |
| **Model Env Var** | `EMBEDDING_MODEL` |
| **Default Model** | `nomic-ai/nomic-embed-text-v1.5` |
| **Dimension Env Var** | `EMBEDDING_DIM` (default: `768`) |

**Start command:**
```bash
vllm serve nomic-ai/nomic-embed-text-v1.5 \
    --port 8001 \
    --task embedding \
    --gpu-memory-utilization 0.3
```

**Used by:**
- `rag/embeddings.py` → `get_embedding_func()` — all vector embedding calls

**API endpoints consumed:**
- `POST /v1/embeddings`

> [!IMPORTANT]
> Make sure `EMBEDDING_DIM` in your `.env` matches the actual output dimension of the model you serve. For `nomic-ai/nomic-embed-text-v1.5` it is `768`. For `BAAI/bge-m3` it would be `1024`.

---

### 3. 👁️ Vision Model Server (Optional)

| Setting | Value |
|---------|-------|
| **Env Var** | Uses `LLM_BINDING_HOST` (same port as LLM, or separate) |
| **Model Env Var** | `VISION_MODEL` |
| **Default Model** | `Qwen/Qwen2.5-VL-7B-Instruct` |

**Start command (separate port):**
```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --port 8002 \
    --gpu-memory-utilization 0.5 \
    --max-model-len 4096
```

**Used by:**
- `rag/llm.py` → `vision_llm_func()` — multimodal image+text queries during ingestion
- `query/query_runner.py` → `_vlm_stream()` — streaming VLM-enhanced queries in the dashboard

**API endpoints consumed:**
- `POST /v1/chat/completions` (with multimodal `image_url` content)

> [!NOTE]
> If your LLM and vision model are the **same model** (e.g., a VLM that handles both text and vision), you can point `VISION_MODEL` at the same `LLM_BINDING_HOST:8000` and skip the separate server. The code reads `LLM_BINDING_HOST` for vision calls by default.

> [!WARNING]
> If you run the vision model on a **separate port** (e.g., `:8002`), you'll need to either:
> 1. Update `LLM_BINDING_HOST` in the code paths that call vision (not ideal — it would break text LLM), **or**
> 2. Add a `VISION_BINDING_HOST` env var and update `rag/llm.py` and `query/query_runner.py` to read it. This is a small 4-line change if needed.

---

## `.env` File Template

```env
# ── Models ────────────────────────────────────────────────────
LLM_MODEL=Qwen/Qwen2.5-Coder-14B-Instruct
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
EMBEDDING_DIM=768
MAX_EMBED_TOKENS=8192
VISION_MODEL=Qwen/Qwen2.5-VL-7B-Instruct

# ── vLLM Endpoints ───────────────────────────────────────────
LLM_BINDING_HOST=http://127.0.0.1:8000/v1
EMBEDDING_BINDING_HOST=http://127.0.0.1:8001/v1
LLM_BINDING_API_KEY=not_needed
EMBEDDING_BINDING_API_KEY=not_needed

# ── Timeouts ──────────────────────────────────────────────────
TIMEOUT=900
```

---

## Verifying Endpoints

After starting each vLLM server, verify it's running:

```bash
# Check LLM server
curl http://127.0.0.1:8000/v1/models

# Check Embedding server
curl http://127.0.0.1:8001/v1/models

# Check Vision server (if separate)
curl http://127.0.0.1:8002/v1/models
```

Each should return a JSON response like:
```json
{
  "data": [
    {
      "id": "Qwen/Qwen2.5-Coder-14B-Instruct",
      "object": "model",
      ...
    }
  ]
}
```

---

## Common Alternative Models

| Role | Model | Dimension | Notes |
|------|-------|-----------|-------|
| **LLM** | `Qwen/Qwen2.5-Coder-14B-Instruct` | — | Good for code-heavy content |
| **LLM** | `Qwen/Qwen3-14B-AWQ` | — | Quantized, lower VRAM |
| **LLM** | `meta-llama/Llama-3.1-8B-Instruct` | — | Smaller, faster |
| **Embedding** | `nomic-ai/nomic-embed-text-v1.5` | 768 | Solid general-purpose |
| **Embedding** | `BAAI/bge-m3` | 1024 | Multilingual |
| **Embedding** | `Qwen/Qwen3-Embedding-0.6B` | 1024 | Lightweight |
| **Vision** | `Qwen/Qwen2.5-VL-7B-Instruct` | — | Strong VLM |

> [!TIP]
> When changing embedding models, always update `EMBEDDING_DIM` in `.env` to match the model's output dimension.

---

## GPU Memory Planning

Running all 3 models simultaneously requires careful GPU memory management:

| Model | Approx VRAM | Suggested `--gpu-memory-utilization` |
|-------|-------------|--------------------------------------|
| LLM (14B params) | ~16-20 GB | `0.85` |
| Embedding (small) | ~2-4 GB | `0.3` |
| Vision (7B params) | ~8-12 GB | `0.5` |

For a single GPU with 24 GB VRAM, you may need to:
- Use quantized models (AWQ/GPTQ)
- Run embedding on CPU
- Share the LLM port for vision if the model supports it

For multi-GPU setups, assign models to different GPUs with `CUDA_VISIBLE_DEVICES`:
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-Coder-14B-Instruct --port 8000
CUDA_VISIBLE_DEVICES=1 vllm serve nomic-ai/nomic-embed-text-v1.5 --port 8001
```

---

## Code Files That Reference Endpoints

| File | Endpoint Used | What It Does |
|------|--------------|--------------|
| `rag/llm.py` | `LLM_BINDING_HOST` | Text LLM calls + Vision LLM calls |
| `rag/embeddings.py` | `EMBEDDING_BINDING_HOST` | All embedding calls |
| `rag/lightrag_init.py` | `LLM_BINDING_HOST` | Passes base_url in `llm_model_kwargs` |
| `query/query_runner.py` | `LLM_BINDING_HOST` | VLM streaming in dashboard |
| `config/preflight.py` | Both hosts | Health checks at startup |
| `main.py` | Both (via preflight) | Startup connectivity verification |
| `app.py` | Both (via preflight) | Sidebar status display |
