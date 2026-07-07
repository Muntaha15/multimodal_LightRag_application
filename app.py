import os
import sys
import time
import asyncio
import threading
import pandas as pd
import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# Ensure the project directory is in the system path for local imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Load environment variables
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"), override=True)

# Import local modules
from config.logging_config import configure_logging
# Initialize logging before any deep lightrag imports
configure_logging(log_filename="rag_pipeline.log")

from config.preflight import check_ollama_connectivity, verify_required_models
from rag.lightrag_init import initialize_lightrag
from rag.raganything_init import initialize_raganything
from viz.graph_viz import render_networkx_graph
from lightrag import QueryParam
from rag.reranker import ENABLE_RERANK
from query.query_runner import aquery_with_vlm_stream
from neo4j import GraphDatabase

# Set page config with dark/premium default
st.set_page_config(
    page_title="Graph RAG & Neo4j Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Custom CSS for Premium Glassmorphism & Aesthetics
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* Font overrides */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header style with gradients */
.main-header {
    background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    margin-bottom: 0.2rem;
    text-align: center;
}

.main-subtitle {
    color: #9ca3af;
    font-size: 1.1rem;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 400;
}

/* Glassmorphism cards */
.glass-card {
    background: rgba(17, 24, 39, 0.65);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.glass-card:hover {
    transform: translateY(-4px);
    border-color: rgba(139, 92, 246, 0.4);
    box-shadow: 0 12px 40px 0 rgba(139, 92, 246, 0.15);
}

/* Status dots */
.status-dot {
    height: 10px;
    width: 10px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
    box-shadow: 0 0 8px currentColor;
}
.status-green {
    background-color: #10b981;
    color: #10b981;
}
.status-red {
    background-color: #ef4444;
    color: #ef4444;
}
.status-yellow {
    background-color: #f59e0b;
    color: #f59e0b;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #0b0f19;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Buttons custom style */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
}

div.stButton > button:first-child:hover {
    background: linear-gradient(135deg, #a78bfa 0%, #7c3aed 100%);
    box-shadow: 0 6px 16px rgba(139, 92, 246, 0.5);
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)

# Title Header
st.markdown('<div class="main-header">🧬 Advanced Graph RAG Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">LightRAG & RAG-Anything Integration with Neo4j Backend</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Persistent Background Event Loop & Async Helpers
# ---------------------------------------------------------------------------
import queue

# Create a single persistent event loop in a background daemon thread
@st.cache_resource
def get_background_loop():
    """Create a single persistent event loop in a background daemon thread."""
    loop = asyncio.new_event_loop()
    def start_background_loop(l):
        asyncio.set_event_loop(l)
        l.run_forever()
    t = threading.Thread(target=start_background_loop, args=(loop,), daemon=True)
    t.start()
    return loop

# Retrieve the persistent background event loop
_loop = get_background_loop()

def run_async(coro):
    """Run an async coroutine synchronously using the persistent background event loop."""
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result()

def sync_generator_adapter(async_generator):
    """Adapt an async generator into a sync generator using a thread-safe Queue."""
    q = queue.Queue()
    
    async def consume():
        try:
            async for chunk in async_generator:
                q.put(chunk)
        except Exception as e:
            q.put(e)
        finally:
            q.put(None)
            
    asyncio.run_coroutine_threadsafe(consume(), _loop)
    
    while True:
        item = q.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item

# Helper to construct metric cards
def make_metric_card(title, value, icon=""):
    return f"""
    <div class="glass-card" style="padding: 16px; text-align: center; margin-bottom: 0px;">
        <div style="font-size: 1.5rem; margin-bottom: 4px;">{icon}</div>
        <div style="font-size: 0.85rem; color: #9ca3af; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;">{title}</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: #ffffff; margin-top: 8px; font-family: 'Outfit', sans-serif;">{value}</div>
    </div>
    """

# ---------------------------------------------------------------------------
# Preflight & Initialization Checks (Cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def run_preflight_checks():
    """Contact Ollama and verify model lists."""
    try:
        res = run_async(check_ollama_connectivity())
        return {"ok": True, "models": res["models"], "host": res["host"]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---------------------------------------------------------------------------
# RAG Instance Cache
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_rag_instances_global():
    """Initialize LightRAG and RAGAnything instances exactly once globally."""
    # Initialize LightRAG on the background loop
    lightrag_inst = run_async(initialize_lightrag())
    # Initialize RAGAnything (Option B wrapper is synchronous)
    rag_inst = initialize_raganything(lightrag_inst)
    return rag_inst, lightrag_inst

def get_rag_instances_sync():
    """Retrieve or initialize LightRAG and RAGAnything instances on the Streamlit thread."""
    rag_inst, lightrag_inst = get_rag_instances_global()
    return rag_inst, lightrag_inst

# ---------------------------------------------------------------------------
# Neo4j Operations
# ---------------------------------------------------------------------------
def check_neo4j_connection(uri, user, password):
    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            driver.verify_connectivity()
            return True, "Connected successfully"
    except Exception as e:
        return False, str(e)

def get_neo4j_stats(uri, user, password):
    stats = {}
    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session() as session:
                # Count total nodes
                node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
                stats["node_count"] = node_count

                # Count total relationships
                rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
                stats["relationship_count"] = rel_count

                # Count by label
                labels_res = session.run("MATCH (n) RETURN labels(n)[0] as label, count(n) as count")
                stats["labels"] = {r["label"] or "No Label": r["count"] for r in labels_res}

                # Count by type
                types_res = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count")
                stats["types"] = {r["type"]: r["count"] for r in types_res}
    except Exception as e:
        stats["error"] = str(e)
    return stats

def run_cypher_query(uri, user, password, query):
    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session() as session:
                result = session.run(query)
                records = [dict(r) for r in result]
                return True, records
    except Exception as e:
        return False, str(e)

def clear_neo4j_database(uri, user, password):
    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
        return True, "Database cleared successfully"
    except Exception as e:
        return False, str(e)

def purge_local_directory():
    import shutil
    working_dir = os.getenv("WORKING_DIR", "./storage/dickens_v1")
    if os.path.exists(working_dir):
        for item in os.listdir(working_dir):
            path = os.path.join(working_dir, item)
            try:
                if os.path.isfile(path) or os.path.islink(path):
                    os.unlink(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except Exception as e:
                return False, f"Failed to delete {path}: {e}"
        return True, f"Purged local files in {working_dir}"
    return False, "Working directory does not exist"

def load_complete_graph(storage_type, uri, user, password, working_dir):
    """Load the complete knowledge graph based on current storage backend.
    
    Returns a tuple (success, nx.Graph or error_message).
    """
    if storage_type == "Neo4JStorage":
        query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN n.entity_id AS src_id, 
               n.entity_type AS src_type, 
               n.description AS src_desc,
               n.source_id AS src_source_id,
               type(r) AS rel_type, 
               r.keywords AS keywords, 
               r.description AS rel_desc, 
               r.weight AS weight,
               m.entity_id AS tgt_id
        """
        success, records = run_cypher_query(uri, user, password, query)
        if not success:
            return False, f"Neo4j Query Error: {records}"
        
        G = nx.Graph()
        for rec in records:
            src = rec.get("src_id")
            if not src:
                continue
            
            # Add or update source node
            if not G.has_node(src):
                G.add_node(
                    src,
                    entity_type=rec.get("src_type", "UNKNOWN"),
                    description=rec.get("src_desc", ""),
                    source_id=rec.get("src_source_id", "")
                )
            else:
                # If we initialized it earlier as target, update properties
                if rec.get("src_type"):
                    G.nodes[src]["entity_type"] = rec.get("src_type")
                if rec.get("src_desc"):
                    G.nodes[src]["description"] = rec.get("src_desc")
                if rec.get("src_source_id"):
                    G.nodes[src]["source_id"] = rec.get("src_source_id")
            
            # If relationship exists, add edge and target node
            tgt = rec.get("tgt_id")
            if tgt:
                if not G.has_node(tgt):
                    G.add_node(
                        tgt,
                        entity_type="UNKNOWN",
                        description=""
                    )
                
                weight_val = rec.get("weight")
                weight = float(weight_val) if weight_val is not None else 1.0
                
                G.add_edge(
                    src,
                    tgt,
                    keywords=rec.get("keywords", ""),
                    description=rec.get("rel_desc", ""),
                    weight=weight,
                    relation_type=rec.get("rel_type", "")
                )
        return True, G
    else:
        # Fall back to local GraphML file loader
        graphml_path = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")
        if not os.path.exists(graphml_path):
            return False, f"Local index file not found at: {graphml_path}. Please run ingestion first."
        try:
            G = nx.read_graphml(graphml_path)
            return True, G
        except Exception as e:
            return False, f"Failed to load local GraphML file: {e}"

# ---------------------------------------------------------------------------
# Sidebar - System Status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="glass-card" style="padding:16px;">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top:0; font-family:\'Outfit\';">⚙️ System Status</h3>', unsafe_allow_html=True)
    
    # 1. Ollama Connectivity Check
    ollama_check = run_preflight_checks()
    if ollama_check["ok"]:
        st.markdown(
            f'<div style="margin-bottom:8px;"><span class="status-dot status-green"></span>'
            f'<b>Ollama:</b> Connected ({ollama_check["host"]})</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="margin-bottom:8px;"><span class="status-dot status-red"></span>'
            '<b>Ollama:</b> Disconnected</div>',
            unsafe_allow_html=True
        )
        st.error(f"Cannot reach Ollama: {ollama_check.get('error')}")

    # 2. Neo4j Status Check
    n4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    n4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
    n4j_pass = os.getenv("NEO4J_PASSWORD", "password")
    
    n4j_ok, n4j_msg = check_neo4j_connection(n4j_uri, n4j_user, n4j_pass)
    if n4j_ok:
        st.markdown(
            f'<div style="margin-bottom:8px;"><span class="status-dot status-green"></span>'
            f'<b>Neo4j DB:</b> Connected</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div style="margin-bottom:8px;"><span class="status-dot status-red"></span>'
            f'<b>Neo4j DB:</b> Disconnected</div>',
            unsafe_allow_html=True
        )
        st.warning(f"Neo4j details: {n4j_msg}")

    # 3. Model configurations
    st.markdown("<hr style='margin:12px 0; border:0; border-top:1px solid rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
    st.markdown("<b>Active Config</b>", unsafe_allow_html=True)
    
    llm_model = os.getenv("LLM_MODEL", "qwen2.5-coder:14b")
    embed_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
    vision_model = os.getenv("VISION_MODEL", "qwen2.5vl:7b")
    graph_storage = os.getenv("GRAPH_STORAGE", "NetworkXStorage")
    
    # Simple check if pulled
    available_models_lower = {m.lower() for m in ollama_check.get("models", [])}
    
    def model_status_html(name, label):
        is_pulled = name.lower() in available_models_lower or any(name.lower() in am for am in available_models_lower)
        dot_class = "status-green" if is_pulled else "status-yellow"
        tooltip = "Available" if is_pulled else "Not found in Ollama list (please pull it)"
        return f'<div style="font-size:0.85rem; margin-bottom:4px;"><span class="status-dot {dot_class}" title="{tooltip}"></span><b>{label}:</b> {name}</div>'

    st.markdown(model_status_html(llm_model, "LLM"), unsafe_allow_html=True)
    st.markdown(model_status_html(embed_model, "Embedding"), unsafe_allow_html=True)
    st.markdown(model_status_html(vision_model, "Vision"), unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.85rem;"><span class="status-dot status-green"></span><b>Graph Store:</b> {graph_storage}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Sidebar Footer
    st.markdown(
        "<div style='text-align:center; font-size:0.75rem; color:#6b7280; margin-top:20px;'>"
        "HKUDS/LightRAG & RAG-Anything Integration"
        "</div>",
        unsafe_allow_html=True
    )

# ---------------------------------------------------------------------------
# Tabs Scaffolding
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 RAG & Graph Retrieval",
    "🗄️ Neo4j Graph Manager",
    "📤 Document Ingestion",
    "🌐 Complete Graph Explorer"
])

# ---------------------------------------------------------------------------
# Tab 1: RAG Querying & Retrieval Visualization
# ---------------------------------------------------------------------------
with tab1:
    st.markdown("### Query the Knowledge Graph")
    st.markdown("Submit queries to retrieve context from the vector index and graph structures. The query-specific subgraph will be displayed in real time.")
    
    col_q, col_opt = st.columns([3, 1])
    
    with col_q:
        query_input = st.text_input("Ask a question about the indexed content:", value="Who is Ebenezer Scrooge and what are the main themes of his story?")
    with col_opt:
        query_mode = st.selectbox("Search Mode", options=["hybrid", "local", "global", "naive"], index=0)
        vlm_enabled = st.checkbox(
            "🖼️ VLM Enhanced",
            value=False,
            help="When enabled, the vision model processes images found in retrieved context for richer answers. Falls back to text-only if no images are present.",
        )
        
    if st.button("Retrieve & Generate", type="primary", use_container_width=True):
        if query_input.strip() == "":
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Initializing models and processing query..."):
                try:
                    # Get RAG instances
                    rag, lightrag = get_rag_instances_sync()
                    
                    if vlm_enabled:
                        # VLM-enhanced streaming: processes images in retrieved
                        # context and streams VLM answers with subgraph data.
                        result = run_async(aquery_with_vlm_stream(
                            rag, lightrag, query_input,
                            mode=query_mode,
                            enable_rerank=ENABLE_RERANK,
                        ))
                    else:
                        # Standard text-only streaming query
                        param = QueryParam(
                            mode=query_mode,
                            stream=True,
                            enable_rerank=ENABLE_RERANK
                        )
                        result = run_async(lightrag.aquery_llm(query_input, param=param))
                except Exception as ex:
                    st.error(f"Failed to run RAG query: {ex}")
                    result = None
            
            if result and result.get("status") == "success":
                llm_resp = result.get("llm_response", {})
                is_streaming = llm_resp.get("is_streaming", False)
                
                # Streaming Response Container
                st.markdown("#### Generated Answer")
                answer_placeholder = st.empty()
                
                if is_streaming and llm_resp.get("response_iterator"):
                    try:
                        iterator = llm_resp.get("response_iterator")
                        full_response = answer_placeholder.write_stream(sync_generator_adapter(iterator))
                    except Exception as stream_ex:
                        st.error(f"Error streaming response: {stream_ex}")
                        full_response = ""
                else:
                    full_response = llm_resp.get("content", "No content returned.")
                    st.write(full_response)
                
                # Query Subgraph Visualization
                st.markdown("#### Retrieved Subgraph")
                entities = result.get("data", {}).get("entities", [])
                relationships = result.get("data", {}).get("relationships", [])
                
                if query_mode == "naive":
                    st.info("💡 **Naive Mode** retrieves text chunks purely via vector similarity. It does not traverse graph entities or relationships. Therefore, no subgraph is displayed.")
                elif not entities and not relationships:
                    st.info("No nodes or relationships were retrieved for this query. The graph may be empty or matching entities were not found.")
                else:
                    # Build and render the retrieved subgraph
                    G_sub = nx.Graph()
                    for ent in entities:
                        name = ent.get("entity_name")
                        if name:
                            G_sub.add_node(
                                name,
                                entity_type=ent.get("entity_type", "UNKNOWN"),
                                description=ent.get("description", ""),
                                source_id=ent.get("source_id", ""),
                                file_path=ent.get("file_path", "")
                            )
                    
                    for rel in relationships:
                        src = rel.get("src_id")
                        tgt = rel.get("tgt_id")
                        if src and tgt:
                            G_sub.add_edge(
                                src,
                                tgt,
                                description=rel.get("description", ""),
                                keywords=rel.get("keywords", ""),
                                weight=float(rel.get("weight", 1.0)),
                                relation_type=rel.get("keywords", "") # fall back for rendering helper
                            )
                            
                    try:
                        temp_viz_path = os.path.join(PROJECT_ROOT, "query_subgraph.html")
                        render_networkx_graph(G_sub, temp_viz_path)
                        
                        if os.path.exists(temp_viz_path):
                            with open(temp_viz_path, "r", encoding="utf-8") as f:
                                html_data = f.read()
                            components.html(html_data, height=600, scrolling=True)
                        else:
                            st.error("Failed to generate the interactive graph visualization HTML file.")
                    except Exception as viz_ex:
                        st.error(f"Visualization rendering error: {viz_ex}")

# ---------------------------------------------------------------------------
# Tab 2: Neo4j Graph Manager
# ---------------------------------------------------------------------------
with tab2:
    st.markdown("### Graph Database Management & Inspector")
    st.markdown("Monitor and interact with the Neo4j instance directly, view schema statistics, run Cypher queries, and manage indexing cache.")
    
    # 1. Connection Status Card
    if n4j_ok:
        n4j_database = os.getenv('NEO4J_DATABASE', 'neo4j')
        st.success(f"Connected to Neo4j instance at `{n4j_uri}` (Database: `{n4j_database}`).")
        
        # Pull stats
        stats = get_neo4j_stats(n4j_uri, n4j_user, n4j_pass)
        
        if "error" in stats:
            st.error(f"Error fetching stats from Neo4j: {stats['error']}")
        else:
            # Metrics Row (Glassmorphic Cards)
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                st.markdown(
                    make_metric_card("Total Graph Nodes", f"{stats['node_count']:,}", "⚪"),
                    unsafe_allow_html=True
                )
            with col_m2:
                st.markdown(
                    make_metric_card("Total Relationships", f"{stats['relationship_count']:,}", "⛓️"),
                    unsafe_allow_html=True
                )
            
            # Show Labels and Relationship Types Distributions side-by-side
            st.markdown("<br>", unsafe_allow_html=True)
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                st.markdown("#### Node Labels Distribution")
                if stats["labels"]:
                    df_labels = pd.DataFrame(
                        list(stats["labels"].items()), 
                        columns=["Node Label", "Node Count"]
                    ).sort_values(by="Node Count", ascending=False)
                    st.dataframe(df_labels, use_container_width=True, hide_index=True)
                else:
                    st.info("No nodes exist in the graph yet.")
                    
            with col_d2:
                st.markdown("#### Relationship Types Distribution")
                if stats["types"]:
                    df_types = pd.DataFrame(
                        list(stats["types"].items()), 
                        columns=["Relationship Type", "Relationship Count"]
                    ).sort_values(by="Relationship Count", ascending=False)
                    st.dataframe(df_types, use_container_width=True, hide_index=True)
                else:
                    st.info("No relationships exist in the graph yet.")
            
            # Cypher Query playground
            st.markdown("<br><hr style='border:0; border-top:1px solid rgba(255,255,255,0.1);'><br>", unsafe_allow_html=True)
            st.markdown("#### ⚡ Cypher Playground")
            st.markdown("Execute read-only Cypher queries directly on your Neo4j database.")
            
            cypher_q = st.text_area(
                "Enter Cypher Query:",
                value="MATCH (n) RETURN labels(n)[0] as label, n.entity_name as name, n.description as description LIMIT 10",
                height=100
            )
            
            if st.button("Run Cypher Query"):
                if cypher_q.strip() == "":
                    st.warning("Please enter a Cypher query.")
                else:
                    with st.spinner("Executing Cypher query..."):
                        success, query_results = run_cypher_query(n4j_uri, n4j_user, n4j_pass, cypher_q)
                        
                    if success:
                        if query_results:
                            st.markdown(f"**Results ({len(query_results)} rows):**")
                            # Convert records (dicts) to flat table
                            df_res = pd.DataFrame(query_results)
                            st.dataframe(df_res, use_container_width=True)
                        else:
                            st.success("Query executed successfully. Returned 0 rows.")
                    else:
                        st.error(f"Cypher Error: {query_results}")
                        
            # Graph Editor Section
            st.markdown("<br><hr style='border:0; border-top:1px solid rgba(255,255,255,0.1);'><br>", unsafe_allow_html=True)
            st.markdown("### ✏️ Graph Editor")
            st.markdown("Directly add, update, or remove nodes and relationships in the Neo4j database.")
            
            # Fetch storage instance
            storage = None
            try:
                rag, lightrag = get_rag_instances_sync()
                storage = lightrag.chunk_entity_relation_graph
            except Exception as e:
                st.error(f"Failed to initialize RAG storage for editing: {e}")
                
            if storage is not None:
                # Render editor messages first if any (survives st.rerun)
                if "editor_message" in st.session_state:
                    msg_type, msg_text = st.session_state.pop("editor_message")
                    if msg_type == "success":
                        st.success(msg_text)
                    elif msg_type == "error":
                        st.error(msg_text)
                    elif msg_type == "warning":
                        st.warning(msg_text)
                    elif msg_type == "info":
                        st.info(msg_text)

                editor_tabs = st.tabs(["🟢 Node Editor", "⛓️ Relationship Editor"])
                
                with editor_tabs[0]:
                    st.markdown("#### Node Editor")
                    col_node_add, col_node_del = st.columns(2)
                    
                    with col_node_add:
                        st.markdown("##### Add / Update Node")
                        node_name = st.text_input("Node Name / ID:", key="node_name_input", placeholder="e.g. Ebenezer Scrooge")
                        node_type = st.text_input("Node Label / Type:", key="node_type_input", placeholder="e.g. person, location")
                        node_desc = st.text_area("Node Description:", key="node_desc_input", placeholder="e.g. The protagonist of the novel...", height=100)
                        
                        if st.button("Add/Update Node", type="primary", key="node_add_btn"):
                            if not node_name.strip():
                                st.warning("Node Name / ID cannot be empty.")
                            elif not node_type.strip():
                                st.warning("Node Label / Type cannot be empty.")
                            else:
                                with st.spinner(f"Upserting node '{node_name}'..."):
                                    try:
                                        node_data = {
                                            "entity_id": node_name.strip(),
                                            "entity_type": node_type.strip(),
                                            "description": node_desc.strip()
                                        }
                                        run_async(storage.upsert_node(node_name.strip(), node_data))
                                        st.session_state["editor_message"] = ("success", f"Node '{node_name}' successfully added/updated!")
                                        st.rerun()
                                    except Exception as ex:
                                        st.session_state["editor_message"] = ("error", f"Failed to upsert node: {ex}")
                                        st.rerun()
                                        
                    with col_node_del:
                        st.markdown("##### Delete Node")
                        node_del_name = st.text_input("Node Name to Delete:", key="node_del_input", placeholder="e.g. Ebenezer Scrooge")
                        confirm_node_del = st.checkbox("Confirm node deletion", key="confirm_node_del_cb", help="This will permanently delete the node and all of its connected relationships in Neo4j.")
                        
                        if st.button("Delete Node", type="secondary", key="node_del_btn"):
                            if not node_del_name.strip():
                                st.warning("Please specify a Node Name to delete.")
                            elif not confirm_node_del:
                                st.error("Please check the confirmation box to proceed.")
                            else:
                                node_to_delete = node_del_name.strip()
                                with st.spinner(f"Deleting node '{node_to_delete}'..."):
                                    try:
                                        node_exists = run_async(storage.has_node(node_to_delete))
                                        if not node_exists:
                                            # Look for case-insensitive matches using Cypher query
                                            escaped_node = node_to_delete.replace("'", "\\'")
                                            chk_query = f"MATCH (n) WHERE toLower(n.entity_id) = toLower('{escaped_node}') RETURN n.entity_id as entity_id LIMIT 3"
                                            ok, res = run_cypher_query(n4j_uri, n4j_user, n4j_pass, chk_query)
                                            
                                            suggestions = []
                                            if ok and isinstance(res, list):
                                                suggestions = [r["entity_id"] for r in res if "entity_id" in r]
                                            
                                            if suggestions:
                                                sug_str = ", ".join([f"'{s}'" for s in suggestions])
                                                st.session_state["editor_message"] = ("error", f"Node '{node_to_delete}' not found in the graph. Did you mean: {sug_str}?")
                                            else:
                                                st.session_state["editor_message"] = ("error", f"Node '{node_to_delete}' not found in the graph.")
                                            st.rerun()
                                        else:
                                            run_async(storage.delete_node(node_to_delete))
                                            st.session_state["editor_message"] = ("success", f"Node '{node_to_delete}' and all its connected relationships deleted!")
                                            st.rerun()
                                    except Exception as ex:
                                        st.session_state["editor_message"] = ("error", f"Failed to delete node: {ex}")
                                        st.rerun()
                                        
                with editor_tabs[1]:
                    st.markdown("#### Relationship Editor")
                    col_rel_add, col_rel_del = st.columns(2)
                    
                    with col_rel_add:
                        st.markdown("##### Add / Update Relationship")
                        rel_source = st.text_input("Source Node Name / ID:", key="rel_src_input", placeholder="e.g. Ebenezer Scrooge")
                        rel_target = st.text_input("Target Node Name / ID:", key="rel_tgt_input", placeholder="e.g. London")
                        rel_keywords = st.text_input("Relationship Keywords / Type:", key="rel_kw_input", placeholder="e.g. LIVES_IN, PARTNER_OF")
                        rel_desc = st.text_area("Relationship Description:", key="rel_desc_input", placeholder="e.g. Scrooge has lived in London for his entire business career.", height=100)
                        rel_weight = st.number_input("Weight:", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="rel_weight_input")
                        
                        if st.button("Add/Update Relationship", type="primary", key="rel_add_btn"):
                            if not rel_source.strip():
                                st.warning("Source Node Name cannot be empty.")
                            elif not rel_target.strip():
                                st.warning("Target Node Name cannot be empty.")
                            elif not rel_keywords.strip():
                                st.warning("Relationship Keywords / Type cannot be empty.")
                            else:
                                with st.spinner(f"Upserting relationship between '{rel_source}' and '{rel_target}'..."):
                                    try:
                                        # Verify and auto-create source node if missing
                                        src_exists = run_async(storage.has_node(rel_source.strip()))
                                        if not src_exists:
                                            st.info(f"Source node '{rel_source}' not found. Auto-creating with type 'unknown'...")
                                            run_async(storage.upsert_node(rel_source.strip(), {
                                                "entity_id": rel_source.strip(),
                                                "entity_type": "unknown",
                                                "description": f"Automatically created as source of relationship to {rel_target.strip()}."
                                            }))
                                            
                                        # Verify and auto-create target node if missing
                                        tgt_exists = run_async(storage.has_node(rel_target.strip()))
                                        if not tgt_exists:
                                            st.info(f"Target node '{rel_target}' not found. Auto-creating with type 'unknown'...")
                                            run_async(storage.upsert_node(rel_target.strip(), {
                                                "entity_id": rel_target.strip(),
                                                "entity_type": "unknown",
                                                "description": f"Automatically created as target of relationship from {rel_source.strip()}."
                                            }))
                                            
                                        # Upsert edge
                                        edge_data = {
                                            "weight": float(rel_weight),
                                            "description": rel_desc.strip(),
                                            "keywords": rel_keywords.strip(),
                                            "source_id": f"{rel_source.strip()}_{rel_target.strip()}"
                                        }
                                        run_async(storage.upsert_edge(rel_source.strip(), rel_target.strip(), edge_data))
                                        st.session_state["editor_message"] = ("success", f"Relationship from '{rel_source}' to '{rel_target}' successfully added/updated!")
                                        st.rerun()
                                    except Exception as ex:
                                        st.session_state["editor_message"] = ("error", f"Failed to upsert relationship: {ex}")
                                        st.rerun()
                                        
                    with col_rel_del:
                        st.markdown("##### Delete Relationship")
                        rel_del_src = st.text_input("Source Node Name / ID:", key="rel_del_src_input", placeholder="e.g. Ebenezer Scrooge")
                        rel_del_tgt = st.text_input("Target Node Name / ID:", key="rel_del_tgt_input", placeholder="e.g. London")
                        confirm_rel_del = st.checkbox("Confirm relationship deletion", key="confirm_rel_del_cb", help="This will permanently delete the relationship edge between these two nodes in Neo4j (the nodes themselves will remain).")
                        
                        if st.button("Delete Relationship", type="secondary", key="rel_del_btn"):
                            if not rel_del_src.strip() or not rel_del_tgt.strip():
                                  st.warning("Please specify both Source and Target node names to delete the relationship.")
                            elif not confirm_rel_del:
                                  st.error("Please check the confirmation box to proceed.")
                            else:
                                with st.spinner(f"Deleting relationship from '{rel_del_src}' to '{rel_del_tgt}'..."):
                                    try:
                                        edge_exists = run_async(storage.has_edge(rel_del_src.strip(), rel_del_tgt.strip()))
                                        if not edge_exists:
                                            # Check reverse direction
                                            reverse_exists = run_async(storage.has_edge(rel_del_tgt.strip(), rel_del_src.strip()))
                                            if reverse_exists:
                                                st.session_state["editor_message"] = ("error", f"Relationship does not exist from '{rel_del_src}' to '{rel_del_tgt}'. Did you mean from '{rel_del_tgt}' to '{rel_del_src}'?")
                                            else:
                                                st.session_state["editor_message"] = ("error", f"Relationship from '{rel_del_src}' to '{rel_del_tgt}' not found.")
                                            st.rerun()
                                        else:
                                            # remove_edges expects a list of (source, target) tuples
                                            run_async(storage.remove_edges([(rel_del_src.strip(), rel_del_tgt.strip())]))
                                            st.session_state["editor_message"] = ("success", f"Relationship from '{rel_del_src}' to '{rel_del_tgt}' deleted!")
                                            st.rerun()
                                    except Exception as ex:
                                        st.session_state["editor_message"] = ("error", f"Failed to delete relationship: {ex}")
                                        st.rerun()
            else:
                st.warning("RAG storage connection is not active or could not be loaded. Please ensure the backend is connected.")
                        
            # Reset Database section
            st.markdown("<br><hr style='border:0; border-top:1px solid rgba(255,255,255,0.1);'><br>", unsafe_allow_html=True)
            st.markdown("#### ⚠️ Danger Zone")
            st.markdown("Use this to clear all database entities and local indices if you want to start a fresh ingest.")
            
            col_clear_btn, col_clear_info = st.columns([1, 2])
            
            with col_clear_btn:
                confirm_clear = st.checkbox("I understand this will delete all data permanently")
                if st.button("Purge Database & Index", type="secondary"):
                    if not confirm_clear:
                        st.error("Please check the confirmation box to proceed.")
                    else:
                        with st.spinner("Purging Neo4j DB and local working directory..."):
                            # Clear Neo4j
                            n4j_clear_ok, n4j_clear_msg = clear_neo4j_database(n4j_uri, n4j_user, n4j_pass)
                            # Purge files
                            local_clear_ok, local_clear_msg = purge_local_directory()
                            
                            # Clear the cached RAG instances to force clean re-initialization
                            get_rag_instances_global.clear()
                                
                        if n4j_clear_ok and local_clear_ok:
                            st.success("Successfully cleared Neo4j database and local storage. RAG instances have been reset!")
                            st.info("Refresh the page or submit a query/ingest to re-initialize.")
                        else:
                            st.error(f"Purge completed with errors.\nNeo4j: {n4j_clear_msg}\nLocal files: {local_clear_msg}")
                            
            with col_clear_info:
                st.markdown(
                    "<div style='font-size:0.9rem; color:#9ca3af; padding-left:10px; border-left:2px solid #ef4444;'>"
                    "<b>Actions performed:</b><br>"
                    "1. Executes <code>MATCH (n) DETACH DELETE n</code> in Neo4j.<br>"
                    "2. Purges all <code>.json</code> and <code>.graphml</code> index files inside the active local working directory.<br>"
                    "3. Resets the cached LightRAG/RAGAnything instances in session state."
                    "</div>",
                    unsafe_allow_html=True
                )
    else:
        st.error(f"Cannot connect to the Neo4j instance at `{n4j_uri}`. Verify credentials and ensure the Docker container is running.")
        st.info("Check status instructions in the sidebar.")



# ---------------------------------------------------------------------------
# Tab 3: Document Ingestion (Uploader)
# ---------------------------------------------------------------------------
with tab3:
    st.markdown("### Document Indexing Engine")
    st.markdown("Upload new files to parse, chunk, extract entity-relations, and index them into the RAG database. Supports plain text, PDFs, Office documents, and images.")
    
    # Ingestion logs monitor setup
    log_file_path = os.path.abspath(os.path.join(os.getenv("LOG_DIR", os.getcwd()), "rag_pipeline.log"))
    
    uploaded_files = st.file_uploader(
        "Upload files for indexing:",
        type=["txt", "pdf", "docx", "pptx", "xlsx", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )
    
    if st.button("Start Ingestion Process", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one file first.")
        else:
            # Setup temp folder
            temp_upload_dir = os.path.join(PROJECT_ROOT, "temp_uploads")
            os.makedirs(temp_upload_dir, exist_ok=True)
            
            # Save uploaded files to temp folder
            saved_paths = []
            for uploaded_file in uploaded_files:
                temp_path = os.path.join(temp_upload_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_paths.append(temp_path)
            
            st.info(f"Saved {len(saved_paths)} file(s) temporarily. Starting background ingestion...")
            
            # Setup Log Tailing UI
            log_header = st.markdown("#### Ingestion Activity Log")
            log_code_box = st.empty()
            
            # Track state for background coroutine
            ingest_state = {"done": False, "error": None}
            
            # Coroutine to run on the persistent background loop
            async def run_ingestion_coro(rag_instance, file_paths):
                from lightrag.utils import compute_mdhash_id, sanitize_text_for_encoding
                try:
                    bypass = os.getenv("BYPASS_TEXT_MINERU", "True").lower() == "true"
                    for fp in file_paths:
                        if bypass and fp.lower().endswith(('.txt', '.md')):
                            logger.info(f"Ingesting plain text directly via LightRAG (bypassing MinerU): {fp}")
                            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                                content = f.read()
                            # Pre-compute the same doc ID that LightRAG generates
                            # internally (sanitize → MD5 hash) so both layers agree.
                            cleaned = sanitize_text_for_encoding(content)
                            doc_id = compute_mdhash_id(cleaned, prefix="doc-")
                            # Insert directly into LightRAG with explicit ID
                            await rag_instance.lightrag.ainsert(
                                input=content,
                                ids=doc_id,
                                file_paths=os.path.basename(fp),
                            )
                            # Register document status in RAGAnything using the
                            # same content-hash doc_id (positional: doc_id, file_path).
                            await rag_instance._upsert_doc_status(
                                doc_id,
                                fp,
                                status="processed",
                                error_msg="",
                            )
                        else:
                            await rag_instance.process_document_complete(
                                file_path=fp,
                                output_dir=os.path.join(PROJECT_ROOT, "rag_storage/output"),
                                parse_method="auto"
                            )
                        # Remove temp file
                        if os.path.exists(fp):
                            os.remove(fp)
                    ingest_state["done"] = True
                except Exception as e:
                    ingest_state["error"] = str(e)
                    ingest_state["done"] = True
                    # Clean up remaining temp files on failure
                    for fp in file_paths:
                        if os.path.exists(fp):
                            try: os.remove(fp)
                            except: pass

            
            # Obtain RAG instances
            rag, _ = get_rag_instances_sync()
            
            # Clear log file or open it to tail from end
            log_content = ""
            if os.path.exists(log_file_path):
                with open(log_file_path, "r", encoding="utf-8") as lf:
                    lf.seek(0, os.SEEK_END)
                    
                    # Submit coroutine to the background loop
                    future = asyncio.run_coroutine_threadsafe(
                        run_ingestion_coro(rag, saved_paths),
                        _loop
                    )
                    
                    # Read logs dynamically while background task runs
                    while not ingest_state["done"] or not future.done():
                        line = lf.readline()
                        if line:
                            log_content += line
                            # Limit display length to last 60 lines for performance
                            lines_list = log_content.splitlines()
                            if len(lines_list) > 60:
                                log_content = "\n".join(lines_list[-60:])
                            log_code_box.code(log_content)
                        else:
                            time.sleep(0.3)
                    
                    # Read any remaining logs
                    remaining = lf.read()
                    if remaining:
                        log_content += remaining
                        lines_list = log_content.splitlines()
                        if len(lines_list) > 60:
                            log_content = "\n".join(lines_list[-60:])
                        log_code_box.code(log_content)
            else:
                # If log file doesn't exist, just run and wait
                st.warning("Log file not found. Ingestion running silently...")
                future = asyncio.run_coroutine_threadsafe(
                    run_ingestion_coro(rag, saved_paths),
                    _loop
                )
                while not ingest_state["done"] or not future.done():
                    time.sleep(0.3)
                
            if ingest_state["error"]:
                st.error(f"Ingestion failed: {ingest_state['error']}")
            else:
                st.success("Successfully ingested all documents and updated the Knowledge Graph!")

# ---------------------------------------------------------------------------
# Tab 4: Complete Graph Explorer
# ---------------------------------------------------------------------------
with tab4:
    st.markdown("### 🌐 Complete Knowledge Graph Explorer")
    st.markdown("Visualize and explore the entire knowledge graph. Use the controls below to filter and search nodes for optimal browser rendering performance.")

    # 1. Fetch current configuration
    active_storage = os.getenv("GRAPH_STORAGE", "NetworkXStorage")
    n4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    n4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
    n4j_pass = os.getenv("NEO4J_PASSWORD", "password")
    working_dir = os.getenv("WORKING_DIR", "./storage/dickens_v1")

    # 2. Add Controls inside a column layout
    col_c1, col_c2, col_c3 = st.columns([1, 1, 1])
    
    with col_c1:
        max_nodes = st.slider(
            "Max Nodes to Render:",
            min_value=10,
            max_value=500,
            value=150,
            step=10,
            help="Cap the number of nodes to render. Higher numbers (>300) might slow down your browser."
        )
        min_degree = st.slider(
            "Minimum Connection Degree:",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
            help="Only show nodes with at least this number of relationships. Helps filter out leaf nodes."
        )
    with col_c2:
        search_query = st.text_input(
            "Search Node by Name:",
            placeholder="e.g. Scrooge",
            help="Filters the graph to only show this node and its immediate neighbors (up to 2-hops)."
        )
        hop_limit = st.selectbox(
            "Search Neighbor Hops:",
            options=[1, 2],
            index=0,
            help="How many steps away from the searched node to display."
        )
    with col_c3:
        refresh_btn = st.button("🔄 Reload Graph Data", use_container_width=True)
        if max_nodes > 300:
            st.warning("⚠️ High node limit (>300) may cause UI latency.")

    # Load the full graph into session state to avoid reloading on every filter move
    if "full_graph" not in st.session_state or refresh_btn:
        with st.spinner("Loading complete graph from storage backend..."):
            if refresh_btn:
                # Add a brief artificial delay so the user can perceive the loading spinner
                time.sleep(0.5)
            success, result = load_complete_graph(active_storage, n4j_uri, n4j_user, n4j_pass, working_dir)
            if success:
                st.session_state["full_graph"] = result
                st.session_state["graph_load_error"] = None
                if refresh_btn:
                    st.toast(f"Graph reloaded: {result.number_of_nodes()} nodes, {result.number_of_edges()} relationships!", icon="🔄")
            else:
                st.session_state["full_graph"] = None
                st.session_state["graph_load_error"] = result

    if st.session_state.get("graph_load_error"):
        st.warning(st.session_state["graph_load_error"])
    elif st.session_state.get("full_graph") is not None:
        G_full = st.session_state["full_graph"]
        
        # Get all unique entity types in the loaded graph dynamically for the multiselect filter
        all_types = sorted(list(set(
            data.get("entity_type", "UNKNOWN") 
            for _, data in G_full.nodes(data=True)
        )))
        
        # Place the multiselect filter dynamically
        with col_c2:
            selected_types = st.multiselect(
                "Filter by Entity Type(s):",
                options=all_types,
                default=[],
                help="Only show nodes of these entity types. Leave empty to show all."
            )

        # Apply Filtering logic
        G_filtered = G_full.copy()
        
        # A. Filter by Search Query (Hop-based pruning)
        if search_query.strip():
            query_node = search_query.strip()
            # Try case-insensitive matching
            found_node = None
            for node in G_filtered.nodes:
                if node.lower() == query_node.lower():
                    found_node = node
                    break
            
            if found_node:
                nodes_to_keep = {found_node}
                current_frontier = {found_node}
                for _ in range(hop_limit):
                    next_frontier = set()
                    for node in current_frontier:
                        neighbors = set(G_filtered.neighbors(node))
                        next_frontier.update(neighbors)
                    nodes_to_keep.update(next_frontier)
                    current_frontier = next_frontier
                
                G_filtered = G_filtered.subgraph(nodes_to_keep).copy()
            else:
                st.info(f"Node '{query_node}' not found in the graph.")
                G_filtered = nx.Graph()  # Empty graph

        # B. Filter by Entity Type
        if selected_types and G_filtered.number_of_nodes() > 0:
            nodes_to_keep = [
                n for n, data in G_filtered.nodes(data=True)
                if data.get("entity_type") in selected_types
            ]
            G_filtered = G_filtered.subgraph(nodes_to_keep).copy()

        # C. Filter by connection degree
        if min_degree > 1 and G_filtered.number_of_nodes() > 0:
            nodes_to_keep = [
                n for n in G_filtered.nodes()
                if G_filtered.degree(n) >= min_degree
            ]
            G_filtered = G_filtered.subgraph(nodes_to_keep).copy()

        # D. Limit max nodes (Sort by degree descending to keep high-value hubs)
        num_nodes = G_filtered.number_of_nodes()
        if num_nodes > max_nodes:
            sorted_nodes = sorted(
                G_filtered.nodes(),
                key=lambda n: G_filtered.degree(n),
                reverse=True
            )
            nodes_to_keep = sorted_nodes[:max_nodes]
            G_filtered = G_filtered.subgraph(nodes_to_keep).copy()

        # Render Graph Visuals
        num_nodes_final = G_filtered.number_of_nodes()
        num_edges_final = G_filtered.number_of_edges()
        
        st.markdown(f"**Showing {num_nodes_final} nodes and {num_edges_final} relationships** (out of {G_full.number_of_nodes()} total nodes)")

        if num_nodes_final == 0:
            st.info("No nodes match the current filter criteria.")
        else:
            with st.spinner("Rendering visualization..."):
                try:
                    temp_viz_path = os.path.join(PROJECT_ROOT, "lightrag_graph.html")
                    render_networkx_graph(G_filtered, temp_viz_path)
                    
                    if os.path.exists(temp_viz_path):
                        with open(temp_viz_path, "r", encoding="utf-8") as f:
                            html_data = f.read()
                        components.html(html_data, height=700, scrolling=True)
                    else:
                        st.error("Failed to generate the interactive graph HTML.")
                except Exception as render_ex:
                    st.error(f"Failed to render graph visualization: {render_ex}")
