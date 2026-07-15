# 🧬 Advanced Graph RAG Dashboard

![alt text](./app_snapshots/image.png)

An integrated Graph RAG and Neo4j Streamlit dashboard combining the capabilities of **HKUDS/LightRAG** and **HKUDS/RAG-Anything** using a local Neo4j database and local LLMs served by vLLM.

> [!TIP]
> **New to the project?** Read the [Architecture & Components Guide (COMPONENTS.md)](COMPONENTS.md) to understand the codebase layout, data flows, and design details.

---

## 🚀 Quick Start Guide

### 1. Clone this Repository
```bash
git clone https://github.com/Muntaha15/multimodal_LightRag_application.git
cd project_agv
```

### 2. Clone the Required Libraries
Since `LightRAG` and `RAG-Anything` are excluded from Git tracking, you must clone them directly into the root folder:
```bash
git clone https://github.com/HKUDS/LightRAG.git
git clone https://github.com/HKUDS/RAG-Anything.git
```

### 3. Install Dependencies
Activate your python/conda environment and install the requirements:
```bash
pip install -r requirements.txt
pip install -e ./LightRAG
pip install -e ./RAG-Anything
pip install vllm
```

### 4. Start Infrastructure
* **vLLM:** Start vLLM servers for each model (see [VLLM_ENDPOINTS.md](VLLM_ENDPOINTS.md) for full details):
  ```bash
  # Terminal 1 — LLM
  vllm serve Qwen/Qwen2.5-Coder-14B-Instruct --port 8000

  # Terminal 2 — Embedding model
  vllm serve nomic-ai/nomic-embed-text-v1.5 --port 8001

  # Terminal 3 — Vision model (if needed)
  vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8002
  ```
* **Neo4j Database:** Spin up the database docker container:
  ```bash
  docker compose up -d
  ```

### 5. Configure Environment
Create a `.env` file in the root folder with the following configuration:
```env
# Models
LLM_MODEL=Qwen/Qwen2.5-Coder-14B-Instruct
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
VISION_MODEL=Qwen/Qwen2.5-VL-7B-Instruct

# vLLM Binding Hosts (OpenAI-compatible endpoints)
LLM_BINDING_HOST=http://127.0.0.1:8000/v1
EMBEDDING_BINDING_HOST=http://127.0.0.1:8001/v1
LLM_BINDING_API_KEY=not_needed
EMBEDDING_BINDING_API_KEY=not_needed

# Storage
WORKING_DIR=./storage/dickens_v1
GRAPH_STORAGE=Neo4JStorage

# Neo4j Details
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j
```

### 6. Index Content & Start Dashboard
* **Initial Indexing:**
  ```bash
  python main.py --fresh
  ```
* **Start UI Dashboard:**
  ```bash
  streamlit run app.py --server.port 8501 --server.address 0.0.0.0
  ```

For more detailed verification and step-by-step instructions, please read the manual [guide.md](guide.md). For codebase layout and data flows, check the [COMPONENTS.md](COMPONENTS.md) onboarding guide.
