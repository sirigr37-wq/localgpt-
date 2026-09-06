import html
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from sklearn.decomposition import PCA
from model import load_model_and_tokenizer, generate_response

st.set_page_config(
    page_title="LLM X RAY",
    page_icon="🔍",
    layout="centered"
)

# Custom styling for clean, modern AI chat interface
st.markdown("""
<style>
    .user-bubble {
        background-color: #f1f5f9;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 12px;
        color: #0f172a;
        font-size: 0.95rem;
        border-left: 4px solid #6366f1;
    }
    .assistant-bubble {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 8px;
        color: #1e293b;
        font-size: 0.97rem;
        line-height: 1.6;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        white-space: pre-wrap;
    }
    .section-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
        color: #64748b;
    }
    .tokenizer-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-top: 12px;
        margin-bottom: 16px;
    }
    .arrow-divider {
        text-align: center;
        font-weight: 700;
        font-size: 1.05rem;
        color: #6366f1;
        margin: 10px 0;
    }
    .token-box {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 10px 14px;
        font-family: monospace;
        font-size: 0.92rem;
        color: #0f172a;
        word-break: break-word;
        margin-bottom: 10px;
    }
    .action-btn {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 5px 12px;
        font-size: 13px;
        font-weight: 500;
        cursor: pointer;
        color: #334155;
        display: inline-flex;
        align-items: center;
        gap: 5px;
        transition: all 0.15s ease;
    }
    .action-btn:hover {
        background: #e2e8f0;
        border-color: #94a3b8;
    }
    .arch-flow {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        font-family: monospace;
        font-size: 0.92rem;
        line-height: 1.6;
        color: #1e293b;
        margin-bottom: 14px;
    }
    .layer-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-top: 10px;
        margin-bottom: 16px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
</style>
""", unsafe_allow_html=True)

st.title("LLM X RAY")

# Initialize session state for conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Cache model and tokenizer loading
@st.cache_resource(show_spinner="Loading model and tokenizer...")
def get_model():
    return load_model_and_tokenizer()

model, tokenizer = get_model()

# Prompt input
prompt = st.text_area("Enter your prompt", key="user_prompt", height=120, placeholder="Type your prompt here...")

# Tokenization, Embedding, and Transformer Visualization (immediately after prompt input and before Generate button)
if prompt.strip():
    encoded = tokenizer(prompt, add_special_tokens=False)
    token_ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    tokens_str = " | ".join(str(t) for t in tokens)
    token_ids_str = " | ".join(str(tid) for tid in token_ids)
    
    # Step 4: Tokenization Display Section
    st.markdown(f"""
    <div class="tokenizer-card">
        <div class="section-label">Input</div>
        <div class="token-box" style="font-family: inherit;">{html.escape(prompt)}</div>
        <div class="arrow-divider">↓ Tokenizer ↓</div>
        <div class="section-label">Tokens</div>
        <div class="token-box">{html.escape(tokens_str)}</div>
        <div class="section-label">Token IDs</div>
        <div class="token-box">{html.escape(token_ids_str)}</div>
    </div>
    """, unsafe_allow_html=True)

    # Step 5 — Embedding Visualization
    st.markdown("### Step 5 — Embedding Visualization")

    # Extract actual input embeddings from model
    input_embeddings_layer = model.get_input_embeddings()
    with torch.no_grad():
        token_tensor = torch.tensor(token_ids, device=next(model.parameters()).device)
        embeddings_tensor = input_embeddings_layer(token_tensor)
        emb_np = embeddings_tensor.detach().cpu().to(torch.float32).numpy()

    emb_dim = int(emb_np.shape[1])

    # Embedding Dimension
    st.metric(label="Embedding Dimension", value=f"{emb_dim} dimensions")

    # Vector Statistics
    stats_data = []
    for i, (tok, tid) in enumerate(zip(tokens, token_ids)):
        vec = emb_np[i]
        stats_data.append({
            "Token": tok,
            "Token ID": tid,
            "Min Value": float(np.min(vec)),
            "Max Value": float(np.max(vec)),
            "Mean": float(np.mean(vec)),
            "Std Dev": float(np.std(vec)),
        })
    stats_df = pd.DataFrame(stats_data)

    st.markdown("#### Vector Statistics")
    st.dataframe(
        stats_df.style.format({
            "Min Value": "{:.4f}",
            "Max Value": "{:.4f}",
            "Mean": "{:.4f}",
            "Std Dev": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True
    )

    # Complete Embedding Vector Values
    with st.expander("🔍 View Complete Embedding Vectors", expanded=False):
        for i, (tok, tid) in enumerate(zip(tokens, token_ids)):
            st.markdown(f"**Token:** `{tok}` | **Token ID:** `{tid}` | **Shape:** `({emb_dim},)`")
            st.code(str(emb_np[i].tolist()), language="json")

    # PCA 2D Visualization
    st.markdown("#### PCA 2D Visualization")
    if len(token_ids) >= 2:
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(emb_np)

        pca_df = pd.DataFrame({
            "Token": tokens,
            "Token ID": token_ids,
            "PCA Dim 1": pca_coords[:, 0],
            "PCA Dim 2": pca_coords[:, 1],
        })

        fig = px.scatter(
            pca_df,
            x="PCA Dim 1",
            y="PCA Dim 2",
            text="Token",
            hover_data={"Token": True, "Token ID": True, "PCA Dim 1": ":.4f", "PCA Dim 2": ":.4f"},
            title="2D PCA Projection of Token Embeddings"
        )
        fig.update_traces(
            textposition="top center",
            marker=dict(size=14, color="#6366f1", line=dict(width=2, color="#312e81"))
        )
        fig.update_layout(
            template="plotly_white",
            height=420,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Enter at least 2 tokens to display the 2D PCA projection.")

    # Step 6 — Transformer Visualization
    st.markdown("### Step 6 — Transformer Visualization")
    
    num_layers = getattr(model.config, "num_hidden_layers", 28)
    hidden_size = getattr(model.config, "hidden_size", 1536)
    num_heads = getattr(model.config, "num_attention_heads", 12)
    num_kv_heads = getattr(model.config, "num_key_value_heads", 2)
    intermediate_size = getattr(model.config, "intermediate_size", 8960)

    # 1 & 2. Display complete transformer architecture flow
    st.markdown("#### Model Architecture Overview")
    st.markdown(f"**Embeddings** → **Layer 1** → **Layer 2** → **...** → **Layer {num_layers}** → **Output**")

    with st.expander(f"🔍 View Complete Architecture Flow ({num_layers} Layers)", expanded=False):
        arch_steps = ["Embeddings"] + [f"Layer {i}" for i in range(1, num_layers + 1)] + ["Output"]
        arch_flow_text = "\n↓\n".join(arch_steps)
        st.markdown(f'<div class="arch-flow" style="white-space: pre; text-align: center;">{arch_flow_text}</div>', unsafe_allow_html=True)

    # 3 & 4. Layer Selector
    layer_options = [f"Layer {i}" for i in range(1, num_layers + 1)]
    selected_layer = st.selectbox(
        "Select Transformer Layer",
        options=layer_options,
        index=0,
        key="selected_transformer_layer"
    )

    # 5. Display Selected Layer Details from actual loaded model
    layer_num = int(selected_layer.split()[1])
    layer_idx = layer_num - 1
    actual_layer = model.model.layers[layer_idx] if hasattr(model, "model") and hasattr(model.model, "layers") else None

    st.markdown(f"""
    <div class="layer-card">
        <h4 style="margin-top: 0; color: #4f46e5;">Selected: {selected_layer}</h4>
        <p><strong>Layer Position:</strong> {layer_num} of {num_layers} Transformer Blocks</p>
        <ul>
            <li><strong>Hidden Size:</strong> {hidden_size}</li>
            <li><strong>Attention Heads (Query):</strong> {num_heads}</li>
            <li><strong>Key/Value Heads (KV / GQA):</strong> {num_kv_heads}</li>
            <li><strong>Intermediate (MLP) Size:</strong> {intermediate_size}</li>
            <li><strong>Sub-modules:</strong> Input RMSNorm, Self-Attention (Q/K/V Projections + RoPE + GQA), Post-Attention RMSNorm, MLP (Gate/Up/Down Projections + SiLU)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # 6. Display selected layer's actual module/architecture structure
    if actual_layer is not None:
        with st.expander(f"🔍 Inspect {selected_layer} Module Architecture", expanded=False):
            st.code(str(actual_layer), language="text")

# Generate button
if st.button("Generate", key="generate_btn", type="primary"):
    if prompt.strip():
        with st.spinner("Generating response..."):
            response = generate_response(model, tokenizer, prompt)
            st.session_state.history.append({
                "id": len(st.session_state.history),
                "prompt": prompt,
                "response": response,
                "feedback": None
            })
            st.session_state["last_response"] = response
    else:
        st.warning("Please enter a prompt before generating.")

# Response section
st.subheader("Response")

if not st.session_state.history:
    st.info("No prompts submitted yet. Enter a prompt above and click **Generate** to start!")
else:
    for idx, item in enumerate(st.session_state.history):
        item_id = item["id"]
        
        # User prompt block
        st.markdown(f'<div class="section-label">Prompt #{idx + 1}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="user-bubble">{item["prompt"]}</div>', unsafe_allow_html=True)
        
        # Model response block
        st.markdown('<div class="section-label">Response</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="assistant-bubble">{item["response"]}</div>', unsafe_allow_html=True)
        
        # Response Action Controls (Likes, Dislikes, More options)
        col_fb1, col_fb2, col_more, _ = st.columns([1, 1, 1, 5])
        
        with col_fb1:
            like_active = item.get("feedback") == "like"
            if st.button("👍 Liked" if like_active else "👍", key=f"like_{item_id}", help="Like response"):
                item["feedback"] = "like" if not like_active else None
                st.rerun()
                
        with col_fb2:
            dislike_active = item.get("feedback") == "dislike"
            if st.button("👎 Disliked" if dislike_active else "👎", key=f"dislike_{item_id}", help="Dislike response"):
                item["feedback"] = "dislike" if not dislike_active else None
                st.rerun()
                
        with col_more:
            with st.popover("⋮", help="More options"):
                st.markdown("**Response Details**")
                word_count = len(item["response"].split())
                char_count = len(item["response"])
                st.caption(f"Words: {word_count} | Characters: {char_count}")
                st.download_button(
                    label="💾 Download .txt",
                    data=item["response"],
                    file_name=f"response_{idx+1}.txt",
                    mime="text/plain",
                    key=f"dl_{item_id}",
                    use_container_width=True
                )
        
        # Client-side Copy and Read Aloud buttons (Web Speech API & Clipboard)
        escaped_resp = json.dumps(item["response"])
        st.html(f"""
        <div style="display: flex; gap: 8px; align-items: center; margin-top: 4px; margin-bottom: 8px;">
            <button id="copy-btn-{item_id}" class="action-btn" onclick="copyText_{item_id}()">
                📋 Copy
            </button>
            <button id="tts-btn-{item_id}" class="action-btn" onclick="toggleTTS_{item_id}()">
                🔊 Read Aloud
            </button>
        </div>
        <script>
        function copyText_{item_id}() {{
            const text = {escaped_resp};
            const btn = document.getElementById("copy-btn-{item_id}");
            if (navigator.clipboard && navigator.clipboard.writeText) {{
                navigator.clipboard.writeText(text).then(() => {{
                    if (btn) {{
                        const old = btn.innerHTML;
                        btn.innerHTML = "✓ Copied!";
                        setTimeout(() => {{ btn.innerHTML = old; }}, 2000);
                    }}
                }}).catch(() => fallbackCopy_{item_id}(text));
            }} else {{
                fallbackCopy_{item_id}(text);
            }}
        }}
        function fallbackCopy_{item_id}(text) {{
            const ta = document.createElement('textarea');
            ta.value = text;
            document.body.appendChild(ta);
            ta.select();
            document.execCommand('copy');
            document.body.removeChild(ta);
            const btn = document.getElementById("copy-btn-{item_id}");
            if (btn) {{
                btn.innerHTML = "✓ Copied!";
                setTimeout(() => {{ btn.innerHTML = "📋 Copy"; }}, 2000);
            }}
        }}
        function toggleTTS_{item_id}() {{
            const text = {escaped_resp};
            const btn = document.getElementById("tts-btn-{item_id}");
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
        
        st.divider()

# Sidebar for conversation management
with st.sidebar:
    st.subheader("Conversation")
    if st.button("🗑️ Clear History", key="clear_history_btn", use_container_width=True):
        st.session_state.history = []
        if "last_response" in st.session_state:
            del st.session_state["last_response"]
        st.rerun()
    st.caption(f"Total Exchanges: {len(st.session_state.history)}")
