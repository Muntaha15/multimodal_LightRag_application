# 🧬 Graph RAG & Neo4j Manual Execution Guide

This guide describes how to run and test the integrated Graph RAG and Neo4j Streamlit application manually on your GPU machine.

> [!NOTE]
> For a detailed code breakdown, directory structure details, and data flows, please refer to the [Architecture & Components Guide (COMPONENTS.md)](COMPONENTS.md).

---

## 🛠️ Prerequisites

Ensure you are using the correct Python environment:
- **Conda Environment:** `cloudspace`
- **Python Executable:** `/home/zeus/miniconda3/envs/cloudspace/bin/python`

---

## 🚀 Step 1: Start vLLM (LLM & Embeddings Server)

vLLM serves the local open-source models on your machine via an OpenAI-compatible API.

> [!IMPORTANT]
> vLLM serves **one model per process**. You need separate terminals for each model.
> See [VLLM_ENDPOINTS.md](VLLM_ENDPOINTS.md) for full endpoint configuration details.

1. Open a new terminal and start the LLM server:
   ```bash
   vllm serve Qwen/Qwen2.5-Coder-14B-Instruct --port 8000
   ```
2. Open another terminal and start the embedding server:
   ```bash
   vllm serve nomic-ai/nomic-embed-text-v1.5 --port 8001
   ```
3. (Optional) Open another terminal for the vision model:
   ```bash
   vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8002
   ```
4. Verify models are loaded:
   ```bash
   curl http://127.0.0.1:8000/v1/models
   curl http://127.0.0.1:8001/v1/models
   ```

---

## 🗄️ Step 2: Spin up the Neo4j Database

In another terminal, navigate to your project directory and start the Neo4j Community container via Docker Compose:

```bash
cd /teamspace/studios/this_studio/project_agv
docker compose up -d
```

Verify that the Neo4j container is running:
```bash
docker ps
```
*Neo4j should be accessible on ports `7474` (HTTP console) and `7687` (Bolt API).*

---

## 📥 Step 3: Run the Ingestion Pipeline

To run the initial clean indexing of all documents in the configured input folder (by default, `./docs/` which now contains `book.txt`):

1. Activate your Conda environment:
   ```bash
   conda activate cloudspace
   ```
2. Run the ingestion command with the `--fresh` flag to clear any stale local files and synchronize with a clean Neo4j DB:
   ```bash
   cd /teamspace/studios/this_studio/project_agv
   python main.py --fresh
   ```
3. Watch the progress logs directly in your terminal. Since you are on a GPU machine, the layout prediction and entity extraction will run efficiently.

---

## 🖥️ Step 4: Run the Streamlit Dashboard

Start the web application dashboard:

1. In your activated conda terminal, run:
   ```bash
   cd /teamspace/studios/this_studio/project_agv
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```
2. Open the URL displayed in the terminal in your browser (usually `http://localhost:8501`).

---

## 🧬 Step 5: Verify the Dashboard Tabs

Once the Streamlit UI is open, you can test the following three integrated tabs:

### 1. 🔍 RAG & Graph Retrieval (Tab 1)
* **Querying:** Type a question in the query input box (e.g. *"What are the main events in the story?"*).
* **Streaming Response:** Choose a search mode (e.g. `hybrid`, `local`, or `global`) and click **Retrieve & Generate**. Watch the answer stream in real-time.
* **Subgraph Visualization:** Under the answer, look at the interactive, physics-based network graph rendered dynamically from the retrieved Neo4j entities/relationships.

### 2. 🗄️ Neo4j Graph Manager (Tab 2)
* **Database Metrics:** Look at the real-time node count, relationship count, node label distributions, and relationship type distributions.
* **Cypher Playground:** Write custom Cypher queries (e.g., `MATCH (n) RETURN labels(n)[0] as label, n.entity_name as name LIMIT 10`) and execute them to inspect the database contents in tabular format.
* **Danger Zone:** Check the confirmation box and click **Purge Database & Index** to completely wipe Neo4j and reset local index files for a fresh start.

### 3. 📤 Document Ingestion (Tab 3)
* **File Upload:** Drag and drop or browse files (PDFs, TXT, DOCX, etc.) to index them into your Graph RAG database.
* **Real-time Logs:** Click **Start Ingestion Process** and watch the real-time activity log displaying logs from MinerU parsing and LightRAG extraction as it processes the document.

### 4. 🌐 Complete Graph Explorer (Tab 4)
* **Reload Graph Data:** Click **Reload Graph Data** to fetch the complete knowledge graph directly from your active storage backend (Neo4j or local NetworkXStorage).
* **Search & Neighbors Filter:** Search for a specific node by name and adjust the **Search Neighbor Hops** (1 or 2 hops) to explore its immediate neighborhood.
* **Entity Type Filtering:** Select one or more entity types from the dynamically populated multiselect dropdown to focus on specific categories of nodes.
* **Density & Performance Controls:** Adjust the **Minimum Connection Degree** slider to filter out leaf nodes and isolate key hubs, and use the **Max Nodes to Render** slider to optimize browser rendering performance.
