"""
LocalGPT - Main Application (Phase 2 - Step 4: Add Conversation History).
Clean, modern ChatGPT-style conversational interface powered by local Qwen/Qwen2.5-1.5B-Instruct.
Includes real-time token streaming, multi-turn memory, system prompt controls, action toolbar, persistent conversation history sidebar, and optional LLM X-Ray inspection.
"""

import os
import html
import json
from datetime import datetime
from typing import Optional
import streamlit as st
import streamlit.components.v1 as components

from model import load_model_and_tokenizer, MODEL_NAME
import database
from memory import (
    init_memory,
    get_messages,
    add_user_message,
    edit_user_message,
    add_assistant_message,
    remove_last_assistant_message,
    clear_messages,
    get_qwen_chat_messages,
    get_active_conversation_id,
    switch_conversation,
    start_new_conversation,
    rename_conversation,
    delete_conversation_and_switch,
    DEFAULT_SYSTEM_PROMPT,
)
from chat import stream_chat_response, generate_chat_response
from xray import render_xray_analysis
from document_loader import save_uploaded_file, load_document
from chunking import chunk_document, chunk_documents
from embeddings import load_embedding_model, embed_text, embed_chunks, get_embeddings_matrix, EMBEDDING_DIM
from vector_store import (
    create_vector_store,
    search_vector_store,
    save_vector_store,
    load_vector_store
)
from rag import retrieve_and_build_context, should_use_rag
from visualization import render_system_architecture_html

# Page configuration
st.set_page_config(
    page_title="LocalGPT",
    page_icon="🔍",
    layout="centered"
)

# Custom CSS for modern, premium ChatGPT-style interface
st.markdown("""
<style>
    /* Google Font imports / Clean Modern Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Fira+Code:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Global layout & typography */
    .block-container {
        padding-top: 1.0rem;
        padding-bottom: 3.5rem;
        max-width: 840px;
    }
    
    /* 1. Header styling - Modern ChatGPT Top Bar */
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 12px;
        margin-bottom: 16px;
        flex-wrap: wrap;
        gap: 10px;
    }
    .header-title {
        font-size: 1.3rem;
        font-weight: 800;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 8px;
        letter-spacing: -0.02em;
    }
    .model-badge {
        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        color: #334155;
        border: 1px solid #cbd5e1;
        font-size: 0.76rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        font-family: 'Fira Code', monospace;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
    }

    /* 2. ChatGPT-Style Hero / Empty State */
    .chatgpt-hero-container {
        text-align: center;
        padding: 36px 20px 24px 20px;
        margin: 10px 0 24px 0;
    }
    .hero-icon {
        font-size: 2.4rem;
        margin-bottom: 8px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 60px;
        height: 60px;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08);
    }
    .hero-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 0.88rem;
        color: #64748b;
        max-width: 540px;
        margin: 0 auto 20px auto;
        line-height: 1.55;
    }
    .hero-cards-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 10px;
        max-width: 700px;
        margin: 0 auto;
        text-align: left;
    }
    .hero-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 12px 14px;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .hero-card:hover {
        border-color: #93c5fd;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08);
        transform: translateY(-1px);
    }
    .hero-card-icon {
        font-size: 1.1rem;
        margin-bottom: 4px;
    }
    .hero-card-title {
        font-weight: 700;
        font-size: 0.84rem;
        color: #1e293b;
    }
    .hero-card-desc {
        font-size: 0.74rem;
        color: #64748b;
        margin-top: 2px;
        line-height: 1.4;
    }

    /* 3. ChatGPT-style Chat Bubbles */
    .user-row {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 14px;
        margin-top: 8px;
    }
    .user-bubble {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: #ffffff;
        border-radius: 18px 18px 4px 18px;
        padding: 10px 16px;
        max-width: 82%;
        font-size: 0.94rem;
        line-height: 1.55;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.16);
        word-break: break-word;
    }
    .assistant-row {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 4px;
        margin-top: 6px;
    }
    .assistant-bubble {
        background-color: #ffffff;
        color: #0f172a;
        border: 1px solid #e2e8f0;
        border-radius: 4px 18px 18px 18px;
        padding: 12px 16px;
        max-width: 96%;
        font-size: 0.95rem;
        line-height: 1.6;
        white-space: pre-wrap;
        word-break: break-word;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .avatar-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
        color: #64748b;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    /* 4. Small, subtle ChatGPT-style action buttons */
    .actions-bar {
        display: flex;
        gap: 6px;
        align-items: center;
        margin-top: 4px;
        margin-bottom: 8px;
    }
    .action-btn {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 12px;
        font-weight: 500;
        cursor: pointer;
        color: #64748b;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 4px;
        transition: all 0.15s ease;
        height: 32px;
        box-sizing: border-box;
    }
    .action-btn:hover {
        background: #f8fafc;
        border-color: #cbd5e1;
        color: #0f172a;
    }
    div[data-testid="stHorizontalBlock"] button {
        padding: 2px 8px !important;
        font-size: 12px !important;
        border-radius: 6px !important;
        border: 1px solid #e2e8f0 !important;
        background: #ffffff !important;
        color: #64748b !important;
        min-height: 32px !important;
        height: 32px !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="stHorizontalBlock"] button:hover {
        background: #f8fafc !important;
        border-color: #cbd5e1 !important;
        color: #0f172a !important;
    }

    /* 5. Bottom Composer styles */
    .composer-container {
        margin-top: 14px;
        padding-top: 8px;
    }
    .composer-icon-btn {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        font-size: 15px;
        cursor: pointer;
        color: #334155;
        display: flex;
        align-items: center;
        justify-content: center;
        height: 42px;
        width: 100%;
        transition: all 0.15s ease;
        margin-top: 1px;
    }
    .composer-icon-btn:hover {
        background: #e2e8f0;
        border-color: #94a3b8;
    }

    /* 6. Cards & Visualizations inside X-Ray expander */
    .tokenizer-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin-top: 6px;
        margin-bottom: 12px;
    }
    .arrow-divider {
        text-align: center;
        font-weight: 700;
        font-size: 0.92rem;
        color: #4f46e5;
        margin: 6px 0;
    }
    .token-box {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 8px 12px;
        font-family: 'Fira Code', monospace;
        font-size: 0.86rem;
        color: #0f172a;
        word-break: break-word;
        margin-bottom: 6px;
    }
    .arch-flow {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px;
        font-family: 'Fira Code', monospace;
        font-size: 0.86rem;
        line-height: 1.5;
        color: #1e293b;
        margin-bottom: 10px;
    }
    .layer-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin-top: 6px;
        margin-bottom: 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }

    /* 7. Sidebar Conversation History Cards */
    .conv-item-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 8px 10px;
        margin-bottom: 8px;
        transition: all 0.15s ease;
    }
    .conv-item-card.active-card {
        background: #f8fafc;
        border-color: #93c5fd;
        box-shadow: 0 1px 3px rgba(37, 99, 235, 0.08);
    }
    .chat-meta-box {
        font-size: 0.76rem;
        color: #64748b;
        line-height: 1.45;
        padding: 4px 8px 4px 8px;
        margin-top: -2px;
        margin-bottom: 6px;
        border-left: 2px solid #e2e8f0;
        margin-left: 4px;
        background: rgba(248, 250, 252, 0.6);
        border-radius: 0 4px 4px 0;
    }
    .chat-meta-box.active-meta {
        border-left-color: #2563eb;
        background: rgba(239, 246, 255, 0.8);
    }
    .meta-label {
        font-weight: 600;
        color: #475569;
    }
    .meta-code {
        font-family: 'Fira Code', monospace;
        font-size: 0.74rem;
        background: #f1f5f9;
        padding: 1px 4px;
        border-radius: 4px;
        color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SECTION HEADING (Reference Target) -----------------
st.markdown("""
<div style="margin-bottom: 18px;">
    <div style="font-size: 1.5rem; font-weight: 700; color: #1e3a8a; margin-bottom: 6px; letter-spacing: -0.01em;">
        Final Interface
    </div>
    <div style="height: 2px; background: #2563eb; width: 100%; border-radius: 1px; margin-bottom: 16px;"></div>
</div>
""", unsafe_allow_html=True)

# ----------------- 1. APPLICATION HEADER -----------------
col_head_left, col_head_right = st.columns([5.8, 4.2], vertical_alignment="center")
with col_head_left:
    st.markdown(f"""
    <div class="header-title">
        <span>🔍</span> LocalGPT
        <span class="model-badge"><span class="status-dot"></span> ⚡ {MODEL_NAME}</span>
    </div>
    """, unsafe_allow_html=True)
with col_head_right:
    enable_llm_xray = st.toggle(
        "🔬 Enable LLM X-Ray",
        value=st.session_state.get("enable_llm_xray", False),
        key="enable_llm_xray",
        help="Toggle between Normal mode (User → AI Response) and X-Ray mode (User → Tokens → Token IDs → Embeddings → Layers → Attention → Hidden States → Logits → Probabilities → Generated Tokens → Response)"
    )

if enable_llm_xray:
    st.markdown("""
    <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 8px 12px; margin-bottom: 14px; font-size: 0.84rem; color: #1e40af; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 6px;">
        <div><strong>🔬 LLM X-Ray Mode: ON</strong> — Internal Generation Stages Active</div>
        <div style="font-size: 0.76rem; font-family: monospace; color: #2563eb;">User ➔ Tokens ➔ Embeddings ➔ Layers ➔ Attention ➔ Logits ➔ Response</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='border-bottom: 1px solid #f1f5f9; margin-bottom: 14px;'></div>", unsafe_allow_html=True)

# ----------------- SYSTEM ARCHITECTURE (Lightweight & Informational) -----------------
with st.expander("🏗️ LocalGPT System Architecture", expanded=False):
    st.markdown(render_system_architecture_html(), unsafe_allow_html=True)

# Initialize conversation memory
init_memory()

# Cache model and tokenizer loading
@st.cache_resource(show_spinner="Loading Qwen2.5-1.5B-Instruct model and tokenizer...")
def get_model():
    return load_model_and_tokenizer()

model, tokenizer = get_model()

# Model parameters
num_layers = getattr(model.config, "num_hidden_layers", 28)
num_heads = getattr(model.config, "num_attention_heads", 12)

def format_datetime_str(dt_str: Optional[str]) -> str:
    """
    Format SQLite timestamp (e.g. '2026-08-31 12:10:00') into 'Aug 31, 2026 12:10 PM'.
    """
    if not dt_str:
        return "N/A"
    try:
        clean_str = str(dt_str).replace("T", " ")
        if "." in clean_str:
            clean_str = clean_str.split(".")[0]
        dt = datetime.strptime(clean_str, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%b %d, %Y %I:%M %p")
    except Exception:
        return str(dt_str)

# ----------------- 7. SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.markdown("### 💬 LocalGPT Controls")
    
    # 1. ➕ New Chat button
    if st.button("➕ New Chat", key="new_chat_btn", use_container_width=True, type="primary"):
        start_new_conversation("New Chat")
        st.session_state["last_response"] = None
        st.session_state["clear_chat_input"] = True
        st.rerun()

    # 2. 📜 Recent Chats list
    st.markdown("##### 📜 Recent Chats")
    all_conversations = database.get_conversations()
    active_conv_id = get_active_conversation_id()

    if all_conversations:
        for conv in all_conversations[:25]:
            c_id = conv["id"]
            c_title = conv.get("title", "New Chat") or "New Chat"
            is_active = (c_id == active_conv_id)
            c_created = format_datetime_str(conv.get("created_at"))
            c_updated = format_datetime_str(conv.get("updated_at"))

            prefix = "🟢 " if is_active else ""
            display_title = c_title if len(c_title) <= 22 else c_title[:20] + "..."

            # Header row: Select / Switch button + Options Popover (⋮)
            col_item, col_menu = st.columns([5, 1], vertical_alignment="center")
            with col_item:
                if st.button(
                    f"{prefix}{display_title}",
                    key=f"conv_select_{c_id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                    help=f"{c_title}\nID: {c_id}\nCreated: {c_created}\nUpdated: {c_updated}"
                ):
                    if not is_active:
                        switch_conversation(c_id)
                        st.session_state["last_response"] = None
                        st.session_state["clear_chat_input"] = True
                        st.rerun()

            with col_menu:
                with st.popover("⋮", help=f"Options for '{c_title}'"):
                    st.markdown(f"**Chat Options**")
                    st.caption(f"ID: `{c_id}`")
                    
                    # Rename Form
                    with st.form(key=f"rename_form_{c_id}"):
                        st.markdown("✏️ **Rename Chat**")
                        new_title_val = st.text_input(
                            "New Title",
                            value=c_title,
                            key=f"rename_title_input_{c_id}",
                            label_visibility="collapsed"
                        )
                        save_title_btn = st.form_submit_button("Save Title", use_container_width=True)
                        if save_title_btn:
                            clean_t = new_title_val.strip()
                            if clean_t and clean_t != c_title:
                                rename_conversation(c_id, clean_t)
                                st.toast("Chat title updated!", icon="✏️")
                                st.rerun()
                            elif not clean_t:
                                st.warning("Title cannot be empty.")

                    st.divider()
                    # Delete Chat Action
                    if st.button("🗑️ Delete Chat", key=f"del_conv_btn_{c_id}", type="secondary", use_container_width=True):
                        delete_conversation_and_switch(c_id)
                        st.toast("Conversation deleted.", icon="🗑️")
                        st.session_state["last_response"] = None
                        st.rerun()

            # Metadata Display Box
            active_class = " active-meta" if is_active else ""
            st.markdown(f"""
            <div class="chat-meta-box{active_class}">
                <div><span class="meta-label">ID:</span> <code class="meta-code" title="{c_id}">{c_id[:8]}</code></div>
                <div><span class="meta-label">Created:</span> {c_created}</div>
                <div><span class="meta-label">Updated:</span> {c_updated}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("No conversations yet. Click ➕ New Chat to begin.")

    # 3. Clear Active Conversation messages
    if st.button("🗑️ Clear Active Chat", key="clear_active_conv_btn", use_container_width=True):
        clear_messages()
        st.session_state["last_response"] = None
        st.session_state["clear_chat_input"] = True
        st.rerun()

    st.divider()

    # 6. System Prompt / System Instructions
    st.markdown("##### ⚙️ System Prompt")

    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT

    if "system_prompt_input_field" not in st.session_state:
        st.session_state["system_prompt_input_field"] = st.session_state.system_prompt

    if st.session_state.get("reset_system_prompt", False):
        st.session_state["system_prompt_input_field"] = DEFAULT_SYSTEM_PROMPT
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
        st.session_state["reset_system_prompt"] = False

    system_prompt_input = st.text_area(
        "System Instructions",
        height=85,
        key="system_prompt_input_field",
        help="Defines persona, behavior, instructions, and constraints for LocalGPT responses."
    )
    st.session_state.system_prompt = system_prompt_input

    col_sys_apply, col_sys_reset = st.columns([1, 1])
    with col_sys_apply:
        if st.button("Apply", key="btn_apply_system_prompt", use_container_width=True, help="Apply current system instructions"):
            st.session_state.system_prompt = system_prompt_input
            st.toast("System prompt applied!", icon="⚙️")
            st.rerun()
    with col_sys_reset:
        if st.button("Reset Default", key="btn_reset_system_prompt", use_container_width=True, help="Reset to default system instructions"):
            st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
            st.session_state["reset_system_prompt"] = True
            st.toast("Reset to default system prompt!", icon="🔄")
            st.rerun()

    st.divider()

    # Generation Controls
    st.markdown("##### ⚙️ Generation Controls")

    DEFAULT_TEMP = 0.7
    DEFAULT_TOP_K = 50
    DEFAULT_TOP_P = 0.9
    DEFAULT_MAX_TOKENS = 512

    if "temperature" not in st.session_state:
        st.session_state.temperature = DEFAULT_TEMP
    if "top_k" not in st.session_state:
        st.session_state.top_k = DEFAULT_TOP_K
    if "top_p" not in st.session_state:
        st.session_state.top_p = DEFAULT_TOP_P
    if "max_new_tokens" not in st.session_state:
        st.session_state.max_new_tokens = DEFAULT_MAX_TOKENS

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=float(st.session_state.temperature),
        step=0.1,
        key="temperature_slider",
        help="Controls randomness: 0.0 is deterministic/greedy, higher values increase creativity (0.0 – 2.0)."
    )
    st.session_state.temperature = temperature

    top_k = st.slider(
        "Top-K",
        min_value=1,
        max_value=100,
        value=int(st.session_state.top_k),
        step=1,
        key="top_k_slider",
        help="Limits sampling to top K highest probability vocabulary tokens (1 – 100)."
    )
    st.session_state.top_k = top_k

    top_p = st.slider(
        "Top-P (Nucleus Sampling)",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.top_p),
        step=0.05,
        key="top_p_slider",
        help="Limits sampling to the smallest set of tokens whose cumulative probability reaches P (0.0 – 1.0)."
    )
    st.session_state.top_p = top_p

    max_new_tokens = st.slider(
        "Max New Tokens",
        min_value=64,
        max_value=2048,
        value=int(st.session_state.max_new_tokens),
        step=64,
        key="max_tokens_slider",
        help="Maximum number of new tokens to generate (64 – 2048)."
    )
    st.session_state.max_new_tokens = max_new_tokens

    if st.button("Reset Controls", key="btn_reset_generation_controls", use_container_width=True, help="Reset generation parameters to defaults"):
        st.session_state.temperature = DEFAULT_TEMP
        st.session_state.top_k = DEFAULT_TOP_K
        st.session_state.top_p = DEFAULT_TOP_P
        st.session_state.max_new_tokens = DEFAULT_MAX_TOKENS
        st.session_state["temperature_slider"] = DEFAULT_TEMP
        st.session_state["top_k_slider"] = DEFAULT_TOP_K
        st.session_state["top_p_slider"] = DEFAULT_TOP_P
        st.session_state["max_tokens_slider"] = DEFAULT_MAX_TOKENS
        st.toast("Reset generation controls to defaults!", icon="⚙️")
        st.rerun()

    # ----------------- DOCUMENT UPLOAD & VECTOR STORE (RAG Foundation) -----------------
    st.divider()
    st.markdown("##### 📁 Document Knowledge (RAG Prep)")
    
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "document_chunks" not in st.session_state:
        st.session_state.document_chunks = []
    if "document_embeddings" not in st.session_state:
        st.session_state.document_embeddings = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = load_vector_store()

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT documents",
        type=["pdf", "docx", "doc", "txt", "md"],
        accept_multiple_files=True,
        key="doc_file_uploader",
        help="Upload documents to extract text, create chunks, generate sentence embeddings, and build a FAISS index."
    )

    if uploaded_files:
        existing_filenames = {d.get("filename") for d in st.session_state.documents}
        new_docs_count = 0
        new_chunks_count = 0
        
        for uf in uploaded_files:
            if uf.name not in existing_filenames:
                try:
                    # 1. Save uploaded file safely
                    saved_path = save_uploaded_file(uf)
                    
                    # 2. Extract structured document
                    doc_data = load_document(saved_path)
                    st.session_state.documents.append(doc_data)
                    existing_filenames.add(uf.name)
                    new_docs_count += 1
                    
                    # 3. Chunk document text
                    if doc_data.get("text", "").strip():
                        current_chunk_offset = len(st.session_state.document_chunks)
                        doc_chunks = chunk_document(doc_data, start_chunk_id=current_chunk_offset)
                        
                        # 4. Generate Sentence Transformer Embeddings
                        if doc_chunks:
                            emb_model = load_embedding_model()
                            embedded_chunks = embed_chunks(doc_chunks, model=emb_model)
                            st.session_state.document_chunks.extend(embedded_chunks)
                            st.session_state.document_embeddings = get_embeddings_matrix(st.session_state.document_chunks)
                            new_chunks_count += len(embedded_chunks)
                except Exception as e:
                    st.error(f"Error processing {uf.name}: {e}")
        
        # 5. Build and save FAISS Vector Store
        if new_chunks_count > 0 and st.session_state.document_chunks:
            try:
                st.session_state.vector_store = create_vector_store(
                    st.session_state.document_chunks,
                    st.session_state.document_embeddings
                )
                save_vector_store(st.session_state.vector_store)
                st.toast(f"Indexed {new_chunks_count} chunk(s) into FAISS Vector Store!", icon="🔎")
            except Exception as e:
                st.error(f"FAISS indexing error: {e}")

    # RAG Retrieval Controls
    st.markdown("##### 🔎 RAG Document Retrieval")
    enable_rag = st.checkbox(
        "Enable RAG Context Injection",
        value=True,
        key="enable_rag_checkbox",
        help="When enabled, relevant document chunks are retrieved from FAISS and injected into the prompt."
    )
    rag_top_k = st.slider(
        "RAG Top-K Chunks",
        min_value=1,
        max_value=10,
        value=1,
        key="rag_top_k_slider",
        help="Number of most relevant document chunks to inject."
    )
    rag_score_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.00,
        step=0.05,
        key="rag_threshold_slider",
        help="Minimum cosine similarity required to inject a chunk."
    )

    # Document & Vector Store Summary
    total_docs = len(st.session_state.documents)
    total_chunks = len(st.session_state.document_chunks)
    emb_shape_str = f"{total_chunks} × {EMBEDDING_DIM}" if total_chunks > 0 else f"0 × {EMBEDDING_DIM}"
    vs_ready = (st.session_state.get("vector_store") is not None 
                and st.session_state.vector_store.get("index") is not None 
                and st.session_state.vector_store["index"].ntotal > 0)
    vs_status_badge = "✅ Ready" if vs_ready else "⚪ Empty"
    rag_status_badge = "✅ Ready" if (vs_ready and enable_rag) else ("⚪ Disabled" if not enable_rag else "⚪ No documents")

    st.markdown(f"""
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px 12px; margin-top: 6px; font-size: 0.84rem; line-height: 1.6;">
        <div>📄 <strong>Documents:</strong> {total_docs}</div>
        <div>📦 <strong>Chunks:</strong> {total_chunks}</div>
        <div>🧠 <strong>Embeddings:</strong> <code>{emb_shape_str}</code></div>
        <div>🔎 <strong>Vector Store:</strong> <code>{vs_status_badge}</code></div>
        <div>📚 <strong>RAG Retrieval:</strong> <code>{rag_status_badge}</code></div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Document & Chunk Details Expander
    if st.session_state.documents:
        with st.expander(f"📚 Document Processing ({total_docs} Docs • {total_chunks} Chunks)", expanded=False):
            st.markdown("###### 📄 Uploaded Documents")
            for doc_idx, doc in enumerate(st.session_state.documents):
                status = doc.get("status", "success")
                status_icon = "✅" if status == "success" else ("⚠️" if status == "warning" else "❌")
                st.markdown(f"**{status_icon} {doc.get('filename', 'Unknown')}**")
                st.caption(f"Type: `{doc.get('file_type', 'N/A')}` | Pages: {doc.get('num_pages', 1)} | Words: {doc.get('word_count', 0):,} | Chars: {doc.get('char_count', 0):,}")
            
            st.divider()
            st.markdown("###### 📦 Processed Chunks & Embeddings")
            for c_idx, ch in enumerate(st.session_state.document_chunks[:8]):
                p_info = f"Page {ch['page_start']}" if ch['page_start'] == ch['page_end'] else f"Pages {ch['page_start']}–{ch['page_end']}"
                has_emb = "✅ 384-dim" if "embedding" in ch and ch["embedding"] is not None else "❌"
                st.markdown(f"**Chunk #{ch['chunk_id']}** — `{ch['filename']}` ({p_info})")
                st.caption(f"Words: {ch['word_count']} | Chars: {ch['char_count']} | Embedding: {has_emb}")
                preview_chunk = ch["text"][:240] + ("..." if len(ch["text"]) > 240 else "")
                st.text_area(
                    f"Chunk #{ch['chunk_id']} Text",
                    value=preview_chunk,
                    height=70,
                    disabled=True,
                    key=f"chunk_preview_{c_idx}"
                )
                st.markdown("<hr style='margin: 4px 0; border: none; border-top: 1px dashed #e2e8f0;'>", unsafe_allow_html=True)
            
            if total_chunks > 8:
                st.caption(f"... and {total_chunks - 8} more chunks.")

    # 2. Vector Store Details Expander
    if vs_ready:
        with st.expander("🔎 Vector Store Details", expanded=False):
            vs_info = st.session_state.vector_store
            idx_total = vs_info["index"].ntotal if vs_info.get("index") else 0
            st.markdown(f"""
            - **Index Type:** `faiss.IndexFlatIP (Cosine Similarity)`
            - **Indexed Vectors:** `{idx_total:,}`
            - **Vector Dimension:** `{vs_info.get('dimension', EMBEDDING_DIM)}`
            - **Similarity Metric:** Inner Product with L2 Normalized Embeddings
            - **Source Documents:** `{total_docs}`
            """)
            st.caption("FAISS index is stored locally and ready for low-latency similarity retrieval.")

    # 3. Interactive Search Test Expander
    if vs_ready:
        with st.expander("🧪 Test Document Search", expanded=False):
            st.caption("Search vector store chunks via Sentence Transformer cosine similarity.")
            search_query_input = st.text_input(
                "Search Query",
                placeholder="e.g. artificial intelligence algorithms",
                key="rag_search_test_input_box",
                label_visibility="collapsed"
            )
            
            col_k, col_s = st.columns([1, 1], vertical_alignment="bottom")
            with col_k:
                search_top_k = st.slider("Top K", min_value=1, max_value=10, value=3, key="rag_search_topk_slider")
            with col_s:
                run_search_btn = st.button("🔍 Search", key="rag_run_search_test_btn", use_container_width=True)

            if run_search_btn and search_query_input.strip():
                try:
                    q_emb = embed_text(search_query_input.strip(), model=load_embedding_model())
                    matches = search_vector_store(st.session_state.vector_store, q_emb, top_k=search_top_k)
                    
                    if matches:
                        st.markdown(f"**Found {len(matches)} matching chunk(s):**")
                        for m in matches:
                            p_info = f"Page {m['page_start']}" if m['page_start'] == m['page_end'] else f"Pages {m['page_start']}–{m['page_end']}"
                            st.markdown(f"**Result #{m['rank']}** — `{m['filename']}` ({p_info})")
                            st.caption(f"Similarity Score: **{m['score']:.4f}** | Words: {m['word_count']} | Chars: {m['char_count']}")
                            st.text_area(
                                f"Result #{m['rank']} Content",
                                value=m["text"],
                                height=80,
                                disabled=True,
                                key=f"search_res_text_{m['rank']}"
                            )
                            st.markdown("<hr style='margin: 4px 0; border: none; border-top: 1px dashed #cbd5e1;'>", unsafe_allow_html=True)
                    else:
                        st.info("No matching chunks found for query.")
                except Exception as e:
                    st.error(f"Search error: {e}")
            elif run_search_btn and not search_query_input.strip():
                st.warning("Please enter a search query first.")

    # 4. Clear Documents and Vector Store
    if st.session_state.documents or vs_ready:
        if st.button("🗑️ Clear Documents & Vector Store", key="clear_all_docs_and_chunks_btn", use_container_width=True):
            st.session_state.documents = []
            st.session_state.document_chunks = []
            st.session_state.document_embeddings = None
            st.session_state.vector_store = None
            # Clean directory
            for f in ["data/vector_store/index.faiss", "data/vector_store/chunks.json"]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
            st.rerun()

    st.divider()
    st.caption(f"Messages in Memory: **{len(get_messages())}**")
    st.caption(f"Model: `{MODEL_NAME}`")


# ----------------- 6. TABS / NAVIGATION (Chat as default) -----------------

894
# ----------------- 6. TABS / NAVIGATION (Chat as default) -----------------
if enable_llm_xray:
    tab_chat, tab_tokens, tab_embeddings, tab_attention, tab_layers, tab_logits, tab_generation = st.tabs([
        "💬 Chat",
        "🔤 Tokens",
        "📊 Embeddings",
        "🔬 Attention",
        "🧠 Layers",
        "📈 Logits",
        "⚡ Generation"
    ])
else:
    tab_chat = st.container()
    tab_tokens = None
    tab_embeddings = None
    tab_attention = None
    tab_layers = None
    tab_logits = None
    tab_generation = None
# Retrieve conversation history
messages = get_messages()

# Find last user prompt and assistant response for standalone tabs
last_user_prompt = "Explain Machine Learning"
last_assistant_resp = ""
for m in reversed(messages):
    if m.get("role") == "assistant" and not last_assistant_resp:
        last_assistant_resp = m.get("content", "")
    if m.get("role") == "user" and last_user_prompt == "Explain Machine Learning":
        last_user_prompt = m.get("content", "")


# =========================================================================
# 2. CHATGPT-STYLE CHAT AREA (Default Tab)
# =========================================================================
with tab_chat:
    if not messages:
        st.markdown("""
        <div class="chatgpt-hero-container">
            <div class="hero-icon">🔍</div>
            <div class="hero-title">How can LocalGPT help you today?</div>
            <div class="hero-subtitle">
                Private, local AI assistant powered by <strong>Qwen2.5-1.5B-Instruct</strong> with multi-turn memory, RAG document grounding, and transparent LLM internal inspection.
            </div>
            <div class="hero-cards-grid">
                <div class="hero-card">
                    <div class="hero-card-icon">💬</div>
                    <div class="hero-card-title">Natural Conversation</div>
                    <div class="hero-card-desc">Multi-turn chat with persistent memory and zero data leakage.</div>
                </div>
                <div class="hero-card">
                    <div class="hero-card-icon">📁</div>
                    <div class="hero-card-title">Grounded RAG Ingestion</div>
                    <div class="hero-card-desc">Upload PDF, DOCX, or TXT docs for FAISS similarity retrieval.</div>
                </div>
                <div class="hero-card">
                    <div class="hero-card-icon">🔬</div>
                    <div class="hero-card-title">LLM X-Ray Inspection</div>
                    <div class="hero-card-desc">Examine tokens, embeddings, 28 layers, attention & probabilities.</div>
                </div>
                <div class="hero-card">
                    <div class="hero-card-icon">⚡</div>
                    <div class="hero-card-title">Instant Token Streaming</div>
                    <div class="hero-card-desc">Real-time CPU/GPU generation with interactive action controls.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Group messages by turn for clean conversational display
        turn_pairs = []
        curr_user = None
        curr_user_idx = None
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                curr_user = msg
                curr_user_idx = i
            elif msg["role"] == "assistant":
                turn_pairs.append({
                    "user": curr_user,
                    "user_idx": curr_user_idx,
                    "assistant": msg,
                    "assistant_idx": i
                })
                curr_user = None
                curr_user_idx = None
        
        # Also render any pending user message without response
        if curr_user is not None:
            turn_pairs.append({
                "user": curr_user,
                "user_idx": curr_user_idx,
                "assistant": None,
                "assistant_idx": None
            })

        active_editing_idx = st.session_state.get("editing_msg_idx")

        for idx, turn in enumerate(turn_pairs):
            user_msg = turn["user"]
            user_idx = turn["user_idx"]
            asst_msg = turn["assistant"]
            turn_id = user_msg["id"] if user_msg and "id" in user_msg else idx
            is_editing = (active_editing_idx is not None) and (active_editing_idx == user_idx)

            # If editing an earlier turn, skip rendering later turns
            if active_editing_idx is not None and user_idx is not None and user_idx > active_editing_idx:
                break

            # 8. User Message Bubble (Right-aligned) or Inline Edit Box
            if user_msg:
                if is_editing:
                    st.markdown("""
                    <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 12px; padding: 12px 14px 4px 14px; margin: 8px 0 10px 0;">
                        <div style="font-size: 0.76rem; font-weight: 700; text-transform: uppercase; color: #64748b; margin-bottom: 6px;">✏️ Edit Message</div>
                    </div>
                    """, unsafe_allow_html=True)
                    edited_prompt = st.text_area(
                        "Edit message",
                        value=user_msg["content"],
                        key=f"edit_prompt_text_{user_idx}",
                        height=85,
                        label_visibility="collapsed"
                    )
                    col_save, col_cancel, _ = st.columns([1.6, 1.2, 5.2], vertical_alignment="center")
                    with col_save:
                        if st.button("Save & Submit", key=f"btn_save_edit_{user_idx}", type="primary", use_container_width=True):
                            clean_edited = edited_prompt.strip()
                            if clean_edited:
                                # 1. Truncate & update in memory and SQLite
                                edit_user_message(user_idx, clean_edited)
                                st.session_state.pop("editing_msg_idx", None)

                                # 2. RAG Retrieval if enabled
                                rag_prompt, retrieved_sources = clean_edited, []
                                if should_use_rag(st.session_state.get("vector_store"), enable_rag=enable_rag):
                                    rag_prompt, retrieved_sources = retrieve_and_build_context(
                                        query=clean_edited,
                                        vector_store=st.session_state.vector_store,
                                        embedding_model=load_embedding_model(),
                                        top_k=rag_top_k,
                                        score_threshold=rag_score_threshold
                                    )

                                # 3. Prepare multi-turn context
                                chat_context = get_qwen_chat_messages(
                                    system_prompt_input,
                                    latest_user_override=rag_prompt
                                )
                                edit_stream_placeholder = st.empty()
                                accumulated_edit_resp = ""

                                try:
                                    for chunk in stream_chat_response(
                                        model,
                                        tokenizer,
                                        chat_context,
                                        max_new_tokens=max_new_tokens,
                                        temperature=temperature,
                                        top_k=top_k,
                                        top_p=top_p,
                                    ):
                                        accumulated_edit_resp += chunk
                                        edit_stream_placeholder.markdown(f"""
                                        <div class="assistant-row">
                                            <div style="display: flex; flex-direction: column; align-items: flex-start; max-width: 96%;">
                                                <div class="avatar-label">LocalGPT</div>
                                                <div class="assistant-bubble">{html.escape(accumulated_edit_resp)}▌</div>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)

                                    add_assistant_message(accumulated_edit_resp, sources=retrieved_sources)
                                    st.session_state["last_response"] = accumulated_edit_resp
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Generation error: {e}")
                            else:
                                st.warning("Please enter a message before submitting.")
                    with col_cancel:
                        if st.button("Cancel", key=f"btn_cancel_edit_{user_idx}", use_container_width=True):
                            st.session_state.pop("editing_msg_idx", None)
                            st.rerun()
                else:
                    st.markdown(f"""
                    <div class="user-row">
                        <div style="display: flex; flex-direction: column; align-items: flex-end; max-width: 82%;">
                            <div class="avatar-label">You</div>
                            <div class="user-bubble">{html.escape(user_msg["content"])}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # User Action Toolbar (Edit button aligned right)
                    col_u_space, col_u_edit = st.columns([8.8, 1.2], vertical_alignment="center")
                    with col_u_edit:
                        if st.button("✏️ Edit", key=f"btn_edit_user_{user_idx}", help="Edit this message and regenerate"):
                            st.session_state["editing_msg_idx"] = user_idx
                            st.rerun()

            # 8. Assistant Message (Left-aligned)
            if asst_msg and not is_editing:
                asst_id = asst_msg["id"]
                resp_text = asst_msg["content"]
                sources = asst_msg.get("sources")

                st.markdown(f"""
                <div class="assistant-row">
                    <div style="display: flex; flex-direction: column; align-items: flex-start; max-width: 96%;">
                        <div class="avatar-label">LocalGPT</div>
                        <div class="assistant-bubble">{html.escape(resp_text)}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # 8. Retrieved Sources Expandable Inspection (if sources were used)
                if sources:
                    num_chunks = len(sources)
                    with st.expander(f"📚 Sources ({num_chunks} chunk{'s' if num_chunks != 1 else ''} used)", expanded=False):
                        for s_idx, s in enumerate(sources, start=1):
                            p_start = s.get("page_start", 1)
                            p_end = s.get("page_end", 1)
                            p_info = f"Page {p_start}" if p_start == p_end else f"Pages {p_start}–{p_end}"
                            score_val = s.get("score")
                            score_str = f" — Relevance: `{score_val:.2f}`" if score_val is not None else ""
                            rank_val = s.get("rank", s_idx)
                            filename = s.get("filename", "document")
                            st.markdown(f"**[Source {rank_val}] `{filename}`** — {p_info}{score_str}")
                            excerpt = s.get("text", "").strip()
                            if excerpt:
                                st.caption(excerpt[:280] + ("..." if len(excerpt) > 280 else ""))
                            if s_idx < len(sources):
                                st.markdown("<hr style='margin: 4px 0; border: none; border-top: 1px dashed #e2e8f0;'>", unsafe_allow_html=True)

                # 9. Small, subtle ChatGPT-style action toolbar
                col_actions, _ = st.columns([8, 2])
                with col_actions:
                    col_c1, col_c2, col_c3, col_c4, col_c5, col_c6 = st.columns([1.0, 0.8, 0.8, 1.4, 0.7, 1.1], vertical_alignment="center")
                    
                    # 1. Copy
                    with col_c1:
                        esc_resp = json.dumps(resp_text)
                        st.html(f"""
                        <button id="copy-btn-{asst_id}" class="action-btn" title="Copy response to clipboard" onclick="copyText_{asst_id}()">
                            📋 Copy
                        </button>
                        <script>
                        function copyText_{asst_id}() {{
                            const text = {esc_resp};
                            const btn = document.getElementById("copy-btn-{asst_id}");
                            if (navigator.clipboard && navigator.clipboard.writeText) {{
                                navigator.clipboard.writeText(text).then(() => {{
                                    if (btn) {{
                                        const old = btn.innerHTML;
                                        btn.innerHTML = "✓ Copied!";
                                        setTimeout(() => {{ btn.innerHTML = old; }}, 2000);
                                    }}
                                }}).catch(() => fallbackCopy_{asst_id}(text));
                            }} else {{
                                fallbackCopy_{asst_id}(text);
                            }}
                        }}
                        function fallbackCopy_{asst_id}(text) {{
                            const ta = document.createElement('textarea');
                            ta.value = text;
                            document.body.appendChild(ta);
                            ta.select();
                            document.execCommand('copy');
                            document.body.removeChild(ta);
                            const btn = document.getElementById("copy-btn-{asst_id}");
                            if (btn) {{
                                btn.innerHTML = "✓ Copied!";
                                setTimeout(() => {{ btn.innerHTML = "📋 Copy"; }}, 2000);
                            }}
                        }}
                        </script>
                        """)
                    
                    # 2. Like
                    with col_c2:
                        liked = asst_msg.get("feedback") == "like"
                        if st.button("👍 Liked" if liked else "👍", key=f"btn_like_{asst_id}", help="Good response"):
                            asst_msg["feedback"] = "like" if not liked else None
                            st.rerun()
                    
                    # 3. Dislike
                    with col_c3:
                        disliked = asst_msg.get("feedback") == "dislike"
                        if st.button("👎 Disliked" if disliked else "👎", key=f"btn_dislike_{asst_id}", help="Bad response"):
                            asst_msg["feedback"] = "dislike" if not disliked else None
                            st.rerun()
                    
                    # 4. Regenerate (Streaming with RAG)
                    with col_c4:
                        if st.button("🔄 Regenerate", key=f"btn_regen_{asst_id}", help="Regenerate this response with live streaming"):
                            # Remove the previous assistant response from memory
                            remove_last_assistant_message()
                            
                            # Prepare RAG context if enabled
                            user_prompt_text = user_msg["content"] if user_msg else ""
                            rag_prompt, retrieved_sources = user_prompt_text, []
                            if should_use_rag(st.session_state.get("vector_store"), enable_rag=enable_rag):
                                rag_prompt, retrieved_sources = retrieve_and_build_context(
                                    query=user_prompt_text,
                                    vector_store=st.session_state.vector_store,
                                    embedding_model=load_embedding_model(),
                                    top_k=rag_top_k,
                                    score_threshold=rag_score_threshold
                                )
                            
                            # Prepare context and stream new response
                            chat_context = get_qwen_chat_messages(
                                system_prompt_input,
                                latest_user_override=rag_prompt
                            )
                            regen_stream_placeholder = st.empty()
                            accumulated_regen = ""
                            
                            try:
                                for chunk in stream_chat_response(
                                    model,
                                    tokenizer,
                                    chat_context,
                                    max_new_tokens=max_new_tokens,
                                    temperature=temperature,
                                    top_k=top_k,
                                    top_p=top_p,
                                ):
                                    accumulated_regen += chunk
                                    regen_stream_placeholder.markdown(f"""
                                    <div class="assistant-row">
                                        <div style="display: flex; flex-direction: column; align-items: flex-start; max-width: 96%;">
                                            <div class="avatar-label">LocalGPT</div>
                                            <div class="assistant-bubble">{html.escape(accumulated_regen)}▌</div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Store the single complete regenerated response
                                add_assistant_message(accumulated_regen, sources=retrieved_sources)
                                st.session_state["last_response"] = accumulated_regen
                                st.rerun()
                            except Exception as e:
                                st.error(f"Regeneration error: {e}")
                    
                    # 5. More (Word count, Character count, Download .txt, Read Aloud)
                    with col_c5:
                        with st.popover("⋯", help="More options"):
                            st.markdown("**Response Details**")
                            word_count = len(resp_text.split())
                            char_count = len(resp_text)
                            st.caption(f"Words: {word_count} | Characters: {char_count}")
                            st.divider()
                            
                            st.html(f"""
                            <button id="tts-btn-{asst_id}" class="action-btn" style="width: 100%; margin-bottom: 8px; justify-content: center;" onclick="toggleTTS_{asst_id}()">
                                🔊 Read Aloud
                            </button>
                            <script>
                            function toggleTTS_{asst_id}() {{
                                const text = {esc_resp};
                                const btn = document.getElementById("tts-btn-{asst_id}");
                                if (!('speechSynthesis' in window)) {{
                                    alert("Speech synthesis is not supported in this browser.");
                                    return;
                                }}
                                if (window.speechSynthesis.speaking) {{
                                    window.speechSynthesis.cancel();
                                    if (btn) btn.innerHTML = "🔊 Read Aloud";
                                }} else {{
                                    const utterance = new SpeechSynthesisUtterance(text);
                                    utterance.onend = function() {{
                                        if (btn) btn.innerHTML = "🔊 Read Aloud";
                                    }};
                                    utterance.onerror = function() {{
                                        if (btn) btn.innerHTML = "🔊 Read Aloud";
                                    }};
                                    window.speechSynthesis.speak(utterance);
                                    if (btn) btn.innerHTML = "⏹️ Stop";
                                }}
                            }}
                            </script>
                            """)
                            
                            st.download_button(
                                label="💾 Download .txt",
                                data=resp_text,
                                file_name=f"localgpt_response_{idx+1}.txt",
                                mime="text/plain",
                                key=f"dl_{asst_id}",
                                use_container_width=True
                            )
                    
                    # 6. Sources Popover
                    with col_c6:
                        with st.popover("Sources", help="View knowledge sources"):
                            if sources:
                                st.markdown("**Document Sources (FAISS Retrieval)**")
                                for s_idx, s in enumerate(sources, start=1):
                                    p_start = s.get("page_start", 1)
                                    p_end = s.get("page_end", 1)
                                    p_info = f"Page {p_start}" if p_start == p_end else f"Pages {p_start}–{p_end}"
                                    score_val = s.get("score")
                                    score_str = f" — Relevance: `{score_val:.2f}`" if score_val is not None else ""
                                    filename = s.get("filename", "document")
                                    st.markdown(f"• **`{filename}`** — {p_info}{score_str}")
                            else:
                                st.markdown("**Generation Sources**")
                                st.caption(f"**Model:** `{MODEL_NAME}`")
                                st.caption("Generated directly by local Qwen2.5-1.5B-Instruct weights. No document sources were retrieved for this response.")

                # 10. LLM X-Ray Analysis (Automatic in X-Ray mode, On-Demand in Normal mode)
                xray_turn_key = f"show_xray_turn_{turn_id}"
                xray_expanded = bool(enable_llm_xray or st.session_state.get(xray_turn_key, False))
                with st.expander("🔬 LLM X-Ray Inspection (Internal Generation Pipeline)", expanded=xray_expanded):
                    if enable_llm_xray or st.checkbox("Enable LLM X-Ray Deep Inspection (Steps 4–11)", key=xray_turn_key, value=xray_expanded, help="Click to run real-time layer-by-layer forward pass and attention inspection on this turn."):
                        prompt_for_xray = user_msg["content"] if user_msg else "Prompt"
                        render_xray_analysis(
                            model,
                            tokenizer,
                            prompt_for_xray,
                            section_key_prefix=f"chat_turn_{turn_id}",
                            response_text=resp_text
                        )
                    else:
                        st.caption("Check the box above or enable '🔬 Enable LLM X-Ray' in the top header to inspect internal stages.")
                
                st.markdown("<div style='margin-bottom: 14px;'></div>", unsafe_allow_html=True)

    # 12. ChatGPT-Style Bottom Composer
    st.markdown("<div class='composer-container'></div>", unsafe_allow_html=True)
    
    # Reset chat input if marked for clearing (after message submission, new chat, or chat switch)
    if st.session_state.get("clear_chat_input", False):
        st.session_state["user_prompt_composer"] = ""
        st.session_state["clear_chat_input"] = False

    col_input, col_mic, col_send = st.columns([8, 1, 1], vertical_alignment="bottom")
    
    with col_input:
        user_input_prompt = st.text_area(
            "Message",
            key="user_prompt_composer",
            height=60,
            placeholder="Message LocalGPT...",
            label_visibility="collapsed"
        )
    
    with col_mic:
        components.html(
            """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <style>
                    * {
                        box-sizing: border-box;
                    }
                    html, body {
                        margin: 0;
                        padding: 0;
                        background: transparent;
                        overflow: hidden;
                        height: 100%;
                        width: 100%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    }
                    .composer-icon-btn {
                        background: #f8fafc;
                        border: 1px solid #cbd5e1;
                        border-radius: 8px;
                        font-size: 16px;
                        cursor: pointer;
                        color: #334155;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        height: 42px;
                        width: 100%;
                        transition: all 0.15s ease;
                        user-select: none;
                        outline: none;
                    }
                    .composer-icon-btn:hover {
                        background: #e2e8f0;
                        border-color: #94a3b8;
                    }
                    .composer-icon-btn.recording {
                        background: #fef2f2;
                        border-color: #ef4444;
                        color: #ef4444;
                        animation: pulse 1.2s infinite;
                    }
                    @keyframes pulse {
                        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
                        70% { box-shadow: 0 0 0 6px rgba(239, 68, 68, 0); }
                        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
                    }
                </style>
            </head>
            <body>
                <button id="mic-trigger-btn" class="composer-icon-btn" type="button" title="Click to speak (Voice-to-Text)">
                    🎤
                </button>
                <script>
                let recognitionInstance = null;
                let isRecording = false;
                let baseText = "";

                function getSpeechRecognition() {
                    return window.SpeechRecognition || 
                           window.webkitSpeechRecognition || 
                           (window.parent && window.parent.SpeechRecognition) || 
                           (window.parent && window.parent.webkitSpeechRecognition) || 
                           null;
                }

                function findTextarea() {
                    const parentDoc = (window.parent && window.parent.document) ? window.parent.document : document;

                    // Strategy 1: Find the textarea in the same horizontal columns container (sibling column in composer row)
                    try {
                        const frame = window.frameElement;
                        if (frame) {
                            const row = frame.closest('[data-testid="stHorizontalBlock"]') ||
                                        frame.closest('[data-testid="column"]')?.parentElement ||
                                        frame.closest('.stHorizontalBlock');
                            if (row) {
                                const rowTa = row.querySelector('textarea');
                                if (rowTa && !rowTa.disabled) return rowTa;
                            }
                        }
                    } catch (e) {}

                    // Strategy 2: Specific placeholder and aria-label of the main chat composer
                    try {
                        const byPlaceholder = parentDoc.querySelector('textarea[placeholder*="Message LocalGPT"]');
                        if (byPlaceholder && !byPlaceholder.disabled) return byPlaceholder;

                        const byAria = parentDoc.querySelector('textarea[aria-label="Message"]');
                        if (byAria && !byAria.disabled && !byAria.closest('[data-testid="stSidebar"]')) return byAria;
                    } catch (e) {}

                    // Strategy 3: Iterate all textareas, strictly ignoring sidebar and non-message fields
                    try {
                        const allTextareas = parentDoc.querySelectorAll('textarea');
                        for (const ta of allTextareas) {
                            if (ta.disabled || ta.readOnly) continue;
                            if (ta.closest('[data-testid="stSidebar"]') || ta.closest('section[aria-label="sidebar"]')) continue;
                            const aria = ta.getAttribute('aria-label') || '';
                            const ph = ta.getAttribute('placeholder') || '';
                            if (aria === 'System Instructions' || ph.includes('System')) continue;
                            if (ph.includes('Message') || aria === 'Message') {
                                return ta;
                            }
                        }

                        // Strategy 4: Bottom-most textarea outside sidebar in main chat area
                        for (let i = allTextareas.length - 1; i >= 0; i--) {
                            const ta = allTextareas[i];
                            if (ta.disabled || ta.readOnly) continue;
                            if (ta.closest('[data-testid="stSidebar"]')) continue;
                            if ((ta.getAttribute('aria-label') || '') === 'System Instructions') continue;
                            return ta;
                        }
                    } catch (e) {}

                    return null;
                }

                function updateInputValue(text) {
                    const ta = findTextarea();
                    if (!ta) return;
                    ta.focus();
                    const win = (ta.ownerDocument && ta.ownerDocument.defaultView) || window.parent || window;
                    const nativeSetter = Object.getOwnPropertyDescriptor(win.HTMLTextAreaElement.prototype, 'value')?.set ||
                                         Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value')?.set;
                    if (nativeSetter) {
                        nativeSetter.call(ta, text);
                    } else {
                        ta.value = text;
                    }
                    ta.dispatchEvent(new Event('input', { bubbles: true }));
                    ta.dispatchEvent(new Event('change', { bubbles: true }));
                }

                function stopRecordingUI() {
                    isRecording = false;
                    const btn = document.getElementById("mic-trigger-btn");
                    if (btn) {
                        btn.innerHTML = "🎤";
                        btn.title = "Click to speak (Voice-to-Text)";
                        btn.classList.remove("recording");
                    }
                }

                function startRecordingUI() {
                    isRecording = true;
                    const btn = document.getElementById("mic-trigger-btn");
                    if (btn) {
                        btn.innerHTML = "🔴";
                        btn.title = "Listening... Speak now. Click to stop.";
                        btn.classList.add("recording");
                    }
                }

                function toggleSpeechRecognition() {
                    const SpeechRecognitionClass = getSpeechRecognition();
                    if (!SpeechRecognitionClass) {
                        alert("Speech Recognition is not supported in this browser. Please use Google Chrome, Microsoft Edge, or Safari.");
                        return;
                    }

                    if (isRecording && recognitionInstance) {
                        try { recognitionInstance.stop(); } catch(e) {}
                        stopRecordingUI();
                        return;
                    }

                    try {
                        recognitionInstance = new SpeechRecognitionClass();
                        recognitionInstance.continuous = true;
                        recognitionInstance.interimResults = true;
                        recognitionInstance.lang = 'en-US';

                        recognitionInstance.onstart = function() {
                            startRecordingUI();
                            const ta = findTextarea();
                            let currentVal = (ta && ta.value) ? ta.value : "";
                            if (currentVal.length > 0 && !currentVal.endsWith(" ")) {
                                baseText = currentVal + " ";
                            } else {
                                baseText = currentVal;
                            }
                        };

                        recognitionInstance.onresult = function(event) {
                            let transcript = '';
                            for (let i = 0; i < event.results.length; ++i) {
                                transcript += event.results[i][0].transcript;
                            }
                            if (transcript) {
                                updateInputValue(baseText + transcript);
                            }
                        };

                        recognitionInstance.onerror = function(event) {
                            stopRecordingUI();
                            if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
                                alert("Microphone permission was denied. Please allow microphone access in your browser settings.");
                            } else if (event.error === 'audio-capture') {
                                alert("No microphone found on your device.");
                            } else if (event.error !== 'no-speech') {
                                console.error("Speech recognition error:", event.error);
                            }
                        };

                        recognitionInstance.onend = function() {
                            stopRecordingUI();
                        };

                        recognitionInstance.start();
                    } catch (err) {
                        stopRecordingUI();
                        console.error("Speech recognition start error:", err);
                    }
                }

                document.getElementById("mic-trigger-btn").addEventListener("click", toggleSpeechRecognition);
                </script>
            </body>
            </html>
            """,
            height=44
        )
        
    with col_send:
        send_clicked = st.button("➤", key="generate_chat_send_btn", type="primary", use_container_width=True, help="Send message")

    if send_clicked:
        if user_input_prompt.strip():
            # 1. Add user message to conversation memory
            user_text = user_input_prompt.strip()
            add_user_message(user_text)
            
            # 2. Render user message bubble immediately
            st.markdown(f"""
            <div class="user-row">
                <div style="display: flex; flex-direction: column; align-items: flex-end; max-width: 82%;">
                    <div class="avatar-label">You</div>
                    <div class="user-bubble">{html.escape(user_text)}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 3. Live Streaming Assistant placeholder
            stream_placeholder = st.empty()
            accumulated_text = ""
            
            # 4. Check RAG Retrieval and build prompt
            rag_prompt, retrieved_sources = user_text, []
            if should_use_rag(st.session_state.get("vector_store"), enable_rag=enable_rag):
                rag_prompt, retrieved_sources = retrieve_and_build_context(
                    query=user_text,
                    vector_store=st.session_state.vector_store,
                    embedding_model=load_embedding_model(),
                    top_k=rag_top_k,
                    score_threshold=rag_score_threshold
                )

            # 5. Build multi-turn context with system prompt & RAG prompt override
            chat_context = get_qwen_chat_messages(
                system_prompt_input,
                latest_user_override=rag_prompt
            )
            
            try:
                # 6. Progressive Streaming Generation with cursor
                for text_chunk in stream_chat_response(
                    model,
                    tokenizer,
                    chat_context,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                ):
                    accumulated_text += text_chunk
                    stream_placeholder.markdown(f"""
                    <div class="assistant-row">
                        <div style="display: flex; flex-direction: column; align-items: flex-start; max-width: 96%;">
                            <div class="avatar-label">LocalGPT</div>
                            <div class="assistant-bubble">{html.escape(accumulated_text)}▌</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 7. Save the complete assistant message to memory with source citations
                add_assistant_message(accumulated_text, sources=retrieved_sources)
                st.session_state["last_response"] = accumulated_text
                st.session_state["clear_chat_input"] = True
                st.rerun()
            except Exception as e:
                st.error(f"Generation error: {e}")
        else:
            st.warning("Please enter a message before sending.")

    st.markdown("""
    <div style="text-align: center; font-size: 0.72rem; color: #94a3b8; margin-top: 10px; line-height: 1.4;">
        LocalGPT can make mistakes. Verify important information. Powered by local <code>Qwen2.5-1.5B-Instruct</code>.
    </div>
    """, unsafe_allow_html=True)


# =========================================================================
# STANDALONE NAVIGATION TABS (Deep dive on latest prompt/response)
# =========================================================================

if enable_llm_xray:

    with tab_tokens:
        st.markdown("### 🔤 Step 4 — Tokenization (Latest Prompt)")
        st.caption(f'Prompt: *"{last_user_prompt}"*')
        render_xray_analysis(
            model, tokenizer, last_user_prompt,
            section_key_prefix="tab_tok",
            response_text=last_assistant_resp
        )

    with tab_embeddings:
        st.markdown("### 📊 Step 5 — Embedding Visualization (Latest Prompt)")
        st.caption(f'Prompt: *"{last_user_prompt}"*')
        render_xray_analysis(
            model, tokenizer, last_user_prompt,
            section_key_prefix="tab_emb",
            response_text=last_assistant_resp
        )

    with tab_attention:
        st.markdown("### 🔬 Step 7 & 11 — Attention Visualization (Latest Prompt)")
        st.caption(f'Prompt: *"{last_user_prompt}"*')
        render_xray_analysis(
            model, tokenizer, last_user_prompt,
            section_key_prefix="tab_att",
            response_text=last_assistant_resp
        )

    with tab_layers:
        st.markdown("### 🧠 Step 6 & 8 — Transformer Layers & Hidden States (Latest Prompt)")
        st.caption(f'Prompt: *"{last_user_prompt}"*')
        render_xray_analysis(
            model, tokenizer, last_user_prompt,
            section_key_prefix="tab_lay",
            response_text=last_assistant_resp
        )

    with tab_logits:
        st.markdown("### 📈 Step 9 — Logits & Probabilities (Latest Prompt)")
        st.caption(f'Prompt: *"{last_user_prompt}"*')
        render_xray_analysis(
            model, tokenizer, last_user_prompt,
            section_key_prefix="tab_log",
            response_text=last_assistant_resp
        )

    with tab_generation:
        st.markdown("### ⚡ Step 10 — Token Generation Timeline (Latest Response)")
        st.caption(f'Prompt: *"{last_user_prompt}"*')
        render_xray_analysis(
            model, tokenizer, last_user_prompt,
            section_key_prefix="tab_gen",
            response_text=last_assistant_resp
        )
