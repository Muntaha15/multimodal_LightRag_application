# Walkthrough - Graph RAG & Neo4j Streamlit Application

This document provides a summary of the completed work, files created or modified, and details on how to test the application manually.

## Changes Made

1. **Streamlit UI Application:**
   * Created [app.py](file:///teamspace/studios/this_studio/project_agv/app.py) featuring:
     * **Tab 1: RAG Querying & Retrieval Visualization:** Allows querying the knowledge graph, displays a streaming response from the local LLM, and builds/renders a query-specific interactive network diagram using `pyvis` (embedded as an iframe).
     * **Tab 2: Neo4j Graph Manager:** Displays connected database status, statistics (total node and relationship counts, distribution tables), a Cypher query execution console, and a cache database reset option.
     * **Tab 3: Document Ingestion:** Hosts a file uploader supporting PDFs, TXT, DOCX, etc., with a real-time tailing logs console rendering MinerU parsing and LightRAG indexing outputs.
     * **Tab 4: Complete Graph Explorer:** Allows visual exploration of the entire knowledge graph using an interactive physics-based PyVis canvas. Includes controls to reload graph data, search nodes by name (with adjustable 1 or 2-hop neighborhood views), filter dynamically by entity types, and adjust the rendering density (via connection degree and node capacity limits) to prevent browser slowdowns.
     * **Premium Styling:** Applied dark-mode aesthetics, custom fonts (Outfit and Plus Jakarta Sans), gradient headers, glassmorphism cards, and interactive status indicators.

2. **Graph Visualization Module:**
   * Refactored [viz/graph_viz.py](file:///teamspace/studios/this_studio/project_agv/viz/graph_viz.py) to export the core function `render_networkx_graph(G, output_path)`. This module creates HTML pages dynamically with interactive physics layouts and dark theme rendering.

3. **Manual Guide:**
   * Created [guide.md](file:///teamspace/studios/this_studio/project_agv/guide.md) to serve as a reference for starting the database, launching Ollama, pulling models, running the ingestion pipeline, and starting Streamlit.

4. **Wrapper-Level Plain Text/MD Ingestion Bypass:**
   * Keeps the `LightRAG` and `RAG-Anything` git submodules completely clean of any parser-level changes.
   * Handles text (`.txt`) and markdown (`.md`) bypassing directly in the wrapper layer:
     * In [file_ingest.py](file:///teamspace/studios/this_studio/project_agv/ingest/file_ingest.py): Gated the direct `insert_text_file` call behind the `BYPASS_TEXT_MINERU` environment variable. Added `.md` support to `SUPPORTED_EXTENSIONS`.
     * In [app.py](file:///teamspace/studios/this_studio/project_agv/app.py): Checked the `BYPASS_TEXT_MINERU` boolean in the background ingestion loop before routing text/markdown files.
     * In [main.py](file:///teamspace/studios/this_studio/project_agv/main.py): Gated the startup plain-text ingestion flow using the `BYPASS_TEXT_MINERU` check, falling back to RAG-Anything's default `process_document_complete` if bypass is disabled.
     * Configured `BYPASS_TEXT_MINERU=True` as the default in [.env](file:///teamspace/studios/this_studio/project_agv/.env).


## What Was Tested

1. **Environment Initialization:**
   * Verified connectivity to Ollama and pulled model readiness check (`qwen2.5-coder:14b`, `qwen2.5vl:7b`, `nomic-embed-text:latest`).
   * Smoke tested embedding generation using the `nomic-embed-text` model.
2. **MinerU Document Parser Verification:**
   * Tested document parsing workflow using MinerU layout analyzer with GPU acceleration on the A100 environment.
3. **Native Text Parser Verification:**
   * Verified that `MineruParser` and `DoclingParser` now natively parse `.txt` files directly (e.g. `book.txt`) without converting to PDF or launching external CLI tools, completing in milliseconds instead of minutes.
3. **Task Clean Up:**
   * Successfully terminated all background processes (Ollama, Python pipeline, Streamlit server) and confirmed that the GPU VRAM has been fully released (0MiB memory usage).

## Event Loop Mismatch Resolution (Bug Fix)

* **Problem:** In Streamlit's architecture, the application script executes from top to bottom on every user interaction (e.g., clicking a button, selecting a tab, or inputting text). The previous implementation recreated the background thread and `asyncio` event loop on every script execution while caching the LightRAG instance in `st.session_state`. This resulted in a cross-loop mismatch: queries submitted to the newly created loop failed because LightRAG's internal embedding workers and HTTP clients remained bound to the old loop, throwing `Event loop is closed` and `TCPTransport closed` errors.
* **Solution:** Refactored [app.py](file:///teamspace/studios/this_studio/project_agv/app.py) to globally cache the background loop, thread, and initialized RAG instances using Streamlit's `@st.cache_resource` decorator. This guarantees that they are created exactly once globally and shared across all subsequent reruns and sessions. In the Neo4j Graph Manager's danger zone, we replaced manual session state deletion with `get_rag_instances_global.clear()` to force clean re-initialization.
* **Verification:** Validated the fix using a multi-query simulation script that executed sequential queries in `hybrid`, `local`, and `global` search modes. All queries finished successfully, confirming that the loop-consistency issue is fully resolved.

## Manual Verification Steps

Please refer to the detailed [guide.md](file:///teamspace/studios/this_studio/project_agv/guide.md) in your project directory to run the components and verify the Streamlit dashboard manually.


## 🧬 Knowledge Graph Optimizations (Expanded Entity Types & Guardrails)

We updated the Graph RAG configuration to fully utilize the system's capabilities in building a rich, context-rich semantic network:

1. **Expanded Entity Schema:**
   * In [lightrag_init.py](file:///teamspace/studios/this_studio/project_agv/rag/lightrag_init.py): Expanded the permitted `entity_types` in LightRAG from the restrictive `["person", "location", "organization"]` list to the more comprehensive list: `["Person", "Creature", "Organization", "Location", "Event", "Concept", "Method", "Artifact", "NaturalObject"]`.
   * This allows the knowledge graph to ingest and relate abstract concepts, technical methods, historical events, and artifacts, which are crucial for high-level semantic retrieval.

2. **Relaxed Abstraction Constraints:**
   * In [llm.py](file:///teamspace/studios/this_studio/project_agv/rag/llm.py): Modified [ENTITY_EXTRACTION_ADDENDUM](file:///teamspace/studios/this_studio/project_agv/rag/llm.py#L33) to allow extracting significant thematic concepts, methods, or processes (e.g., *"Industrialization"*, *"Poverty"*, *"Binary Search"*).
   * Maintained strong negative examples (e.g., *"Mashed Potatoes"*, *"Wicker Baskets"*) to filter out trivial everyday objects and descriptive noise while preserving meaningful abstract connections.

## 📖 Developer Onboarding & Architecture Guide (COMPONENTS.md)

We have created a new architecture and onboarding guide to help developers quickly understand the codebase structure and internal components:
* Created [COMPONENTS.md](file:///teamspace/studios/this_studio/project_agv/COMPONENTS.md) which includes:
  * A file-by-file directory tree map.
  * Comprehensive breakdowns of all modules, including Streamlit's event-loop thread caching implementation.
  * Inline Mermaid diagrams mapping out the Document Ingestion and Retrieval flows.
  * Key development and optimization tips for contributors.
* Modified [README.md](file:///teamspace/studios/this_studio/project_agv/README.md) and [guide.md](file:///teamspace/studios/this_studio/project_agv/guide.md) to prominently display links to the onboarding guide.

