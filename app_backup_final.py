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
    page_title="LLM X-Ray",
    page_icon="🔍",
    layout="centered"
)

# Custom styling for clean, compact, modern AI chat interface
st.markdown("""
<style>
    /* Global spacing adjustments */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 820px;
    }
    
    /* Header styling */
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 12px;
        margin-bottom: 16px;
        flex-wrap: wrap;
        gap: 8px;
    }
    .header-title {
        font-size: 1.32rem;
        font-weight: 700;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .header-subtitle {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 1px;
    }
    .model-badge {
        background: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        font-family: monospace;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    .status-pill {
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        font-size: 0.74rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 9999px;
    }

    /* Chat message layout */
    .user-row {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 12px;
    }
    .user-bubble {
        background-color: #2563eb;
        color: #ffffff;
        border-radius: 16px 16px 4px 16px;
        padding: 10px 16px;
        max-width: 82%;
        font-size: 0.94rem;
        line-height: 1.5;
        box-shadow: 0 1px 2px rgba(0,0,0,0.08);
        word-break: break-word;
    }
    .assistant-row {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 6px;
    }
    .assistant-bubble {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        color: #0f172a;
        border-radius: 16px 16px 16px 4px;
        padding: 14px 18px;
        max-width: 92%;
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        white-space: pre-wrap;
        word-break: break-word;
    }
    .avatar-label {
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 3px;
        color: #64748b;
    }
    
    /* Action & Composer buttons */
    .action-btn {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 12px;
        font-weight: 500;
        cursor: pointer;
        color: #334155;
        display: inline-flex;
        align-items: center;
        gap: 4px;
        transition: all 0.15s ease;
    }
    .action-btn:hover {
        background: #e2e8f0;
        border-color: #94a3b8;
    }
    .composer-icon-btn {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        font-size: 16px;
        cursor: pointer;
        color: #0f172a;
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

    /* Cards & Visualizations */
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
        font-size: 0.95rem;
        color: #4f46e5;
        margin: 6px 0;
    }
    .token-box {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 8px 12px;
        font-family: monospace;
        font-size: 0.88rem;
        color: #0f172a;
        word-break: break-word;
        margin-bottom: 6px;
    }
    .arch-flow {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px;
        font-family: monospace;
        font-size: 0.88rem;
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
    .step-divider {
        margin: 16px 0 10px 0;
        border-top: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Compact Header with visible Qwen model badge
st.markdown("""
<div class="header-container">
    <div>
        <div class="header-title">
            <span>🔍</span> LLM X-Ray
        </div>
        <div class="header-subtitle">
            Explore how the language model processes your prompt in real-time.
        </div>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
        <span class="model-badge">
            ⚡ Qwen/Qwen2.5-1.5B-Instruct
        </span>
        <span class="status-pill">
            ● Ready
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize session state for conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Cache model and tokenizer loading
@st.cache_resource(show_spinner="Loading model and tokenizer...")
def get_model():
    return load_model_and_tokenizer()

model, tokenizer = get_model()

# Sidebar: Controls & Session Settings
with st.sidebar:
    st.subheader("⚙️ Settings")
    
    # Output Length Control
    length_options = {
        "Short (64 tokens)": 64,
        "Normal (128 tokens)": 128,
        "Detailed (256 tokens)": 256,
    }
    selected_length = st.radio(
        "Output Length",
        options=list(length_options.keys()),
        index=1,  # Default: Normal (128 tokens)
        key="output_length_selector"
    )
    max_new_tokens = length_options[selected_length]

    # Sampling Parameters (Optional Controls)
    st.markdown("##### 🎛️ Sampling Controls")
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.5,
        value=0.7,
        step=0.05,
        key="temperature_slider",
        help="Controls randomness: 0.0 is deterministic/greedy, higher values increase creativity."
    )
    top_k = st.slider(
        "Top-K",
        min_value=1,
        max_value=100,
        value=50,
        step=1,
        key="top_k_slider",
        help="Limits sampling to top K highest probability vocabulary tokens."
    )
    top_p = st.slider(
        "Top-P (Nucleus Sampling)",
        min_value=0.05,
        max_value=1.0,
        value=0.9,
        step=0.05,
        key="top_p_slider",
        help="Limits sampling to the smallest set of tokens whose cumulative probability reaches P."
    )
    
    st.divider()
    if st.button("🗑️ Clear History", key="clear_history_btn", use_container_width=True):
        st.session_state.history = []
        if "last_response" in st.session_state:
            del st.session_state["last_response"]
        st.rerun()
    st.caption(f"Exchanges: {len(st.session_state.history)}")
    st.caption("Model: `Qwen/Qwen2.5-1.5B-Instruct`")

# Helper function to render complete X-Ray analysis for a prompt
def render_xray_analysis(prompt_text, section_key_prefix="", response_text=None):
    if not prompt_text.strip():
        return
        
    encoded = tokenizer(prompt_text, add_special_tokens=False)
    token_ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    tokens_str = " | ".join(str(t) for t in tokens)
    token_ids_str = " | ".join(str(tid) for tid in token_ids)
    
    # Step 4: Tokenization Section
    st.markdown("#### 🔤 Step 4 — Tokenization")
    st.markdown(f"""
    <div class="tokenizer-card">
        <div class="avatar-label">Input</div>
        <div class="token-box" style="font-family: inherit;">{html.escape(prompt_text)}</div>
        <div class="arrow-divider">↓ Tokenizer ↓</div>
        <div class="avatar-label">Tokens</div>
        <div class="token-box">{html.escape(tokens_str)}</div>
        <div class="avatar-label">Token IDs</div>
        <div class="token-box">{html.escape(token_ids_str)}</div>
    </div>
    """, unsafe_allow_html=True)

    # Step 5: Embedding Visualization
    st.markdown("#### 📊 Step 5 — Embedding Visualization")
    input_embeddings_layer = model.get_input_embeddings()
    with torch.no_grad():
        token_tensor = torch.tensor(token_ids, device=next(model.parameters()).device)
        embeddings_tensor = input_embeddings_layer(token_tensor)
        emb_np = embeddings_tensor.detach().cpu().to(torch.float32).numpy()

    emb_dim = int(emb_np.shape[1])
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

    # Complete Vectors Expander
    with st.expander("🔍 View Complete Embedding Vectors", expanded=False):
        for i, (tok, tid) in enumerate(zip(tokens, token_ids)):
            st.markdown(f"**Token:** `{tok}` | **Token ID:** `{tid}` | **Shape:** `({emb_dim},)`")
            st.code(str(emb_np[i].tolist()), language="json")

    # PCA 2D Visualization (Handles 1 token and multi-token gracefully)
    st.markdown("##### PCA 2D Visualization")
    if len(token_ids) >= 2:
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(emb_np)
    else:
        pca_coords = np.array([[0.0, 0.0]])

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
        marker=dict(size=14, color="#4f46e5", line=dict(width=2, color="#312e81"))
    )
    fig.update_layout(
        template="plotly_white",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Step 6: Transformer Visualization
    st.markdown("#### 🧠 Step 6 — Transformer Visualization")
    num_layers = getattr(model.config, "num_hidden_layers", 28)
    hidden_size = getattr(model.config, "hidden_size", 1536)
    num_heads = getattr(model.config, "num_attention_heads", 12)
    num_kv_heads = getattr(model.config, "num_key_value_heads", 2)
    intermediate_size = getattr(model.config, "intermediate_size", 8960)

    st.markdown(f"**Embeddings** → **Layer 1** → **Layer 2** → **...** → **Layer {num_layers}** → **Output**")

    with st.expander(f"🔍 View Complete Architecture Flow ({num_layers} Layers)", expanded=False):
        arch_steps = ["Embeddings"] + [f"Layer {i}" for i in range(1, num_layers + 1)] + ["Output"]
        arch_flow_text = "\n↓\n".join(arch_steps)
        st.markdown(f'<div class="arch-flow" style="white-space: pre; text-align: center;">{arch_flow_text}</div>', unsafe_allow_html=True)

    layer_options = [f"Layer {i}" for i in range(1, num_layers + 1)]
    selected_layer = st.selectbox(
        "Select Transformer Layer",
        options=layer_options,
        index=0,
        key=f"transformer_layer_select_{section_key_prefix}"
    )

    layer_num = int(selected_layer.split()[1])
    layer_idx = layer_num - 1
    actual_layer = model.model.layers[layer_idx] if hasattr(model, "model") and hasattr(model.model, "layers") else None

    st.markdown(f"""
    <div class="layer-card">
        <h5 style="margin-top: 0; color: #4f46e5;">Selected: {selected_layer}</h5>
        <p style="margin-bottom: 6px; font-size: 0.92rem;"><strong>Layer Position:</strong> {layer_num} of {num_layers} Transformer Blocks</p>
        <ul style="margin-bottom: 0; padding-left: 20px; font-size: 0.90rem;">
            <li><strong>Hidden Size:</strong> {hidden_size}</li>
            <li><strong>Attention Heads (Query):</strong> {num_heads}</li>
            <li><strong>Key/Value Heads (KV / GQA):</strong> {num_kv_heads}</li>
            <li><strong>Intermediate (MLP) Size:</strong> {intermediate_size}</li>
            <li><strong>Sub-modules:</strong> Input RMSNorm, Self-Attention (Q/K/V + RoPE + GQA), Post-Attention RMSNorm, MLP (Gate/Up/Down + SiLU)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    if actual_layer is not None:
        with st.expander(f"🔍 Inspect {selected_layer} Module Architecture", expanded=False):
            st.code(str(actual_layer), language="text")

    # Unified single inspection forward pass for Steps 7, 8, & 9
    with torch.no_grad():
        model.config._attn_implementation = "eager"
        for layer_mod in model.model.layers:
            if hasattr(layer_mod.self_attn, "config"):
                layer_mod.self_attn.config._attn_implementation = "eager"
        
        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)

    # Step 11: Attention Visualization
    st.markdown("#### 🔬 Step 11 — Attention Visualization")
    
    # Layer & Head Selection
    col_l, col_h = st.columns(2)
    with col_l:
        layer_num_options = [f"Layer {i}" for i in range(1, num_layers + 1)]
        selected_attn_layer = st.selectbox(
            "Transformer Layer",
            options=layer_num_options,
            index=0,  # Default: Layer 1 (supports Layer 1, Layer 10, Layer 20, Layer 28, etc.)
            key=f"attn_layer_select_{section_key_prefix}",
            help="Select a Transformer layer to inspect (e.g. Layer 1, Layer 10, Layer 20, Layer 28)"
        )
    with col_h:
        head_num_options = [f"Head {i}" for i in range(1, num_heads + 1)]
        selected_attn_head = st.selectbox(
            "Attention Head",
            options=head_num_options,
            index=0,  # Default: Head 1
            key=f"attn_head_select_{section_key_prefix}",
            help=f"Select an attention head dynamically ({num_heads} heads available in config)"
        )

    selected_attn_layer_idx = int(selected_attn_layer.split()[1]) - 1
    selected_attn_head_idx = int(selected_attn_head.split()[1]) - 1

    # Extract attention tensor from model output
    seq_len = len(tokens)
    layer_attn_shape_str = f"(1, {num_heads}, {seq_len}, {seq_len})"
    head_attn_shape_str = f"({seq_len}, {seq_len})"
    attn_matrix = None

    if (
        outputs is not None 
        and hasattr(outputs, "attentions") 
        and outputs.attentions is not None 
        and len(outputs.attentions) > selected_attn_layer_idx
        and outputs.attentions[selected_attn_layer_idx] is not None
    ):
        layer_attn_tensor = outputs.attentions[selected_attn_layer_idx]
        t_shape = layer_attn_tensor.shape
        layer_attn_shape_str = f"({t_shape[0]}, {t_shape[1]}, {t_shape[2]}, {t_shape[3]})"
        if t_shape[1] > selected_attn_head_idx:
            attn_matrix = layer_attn_tensor[0, selected_attn_head_idx].detach().cpu().to(torch.float32).numpy()
            head_attn_shape_str = f"({attn_matrix.shape[0]}, {attn_matrix.shape[1]})"

    if attn_matrix is None:
        st.warning("⚠️ Attention tensor was not available from the model output. Using identity fallback.")
        attn_matrix = np.eye(seq_len, dtype=np.float32)

    # Info & Tensor Shape Display Card
    st.markdown(f"""
    <div class="layer-card">
        <h5 style="margin-top: 0; color: #4f46e5;">Attention Details: {selected_attn_layer} — {selected_attn_head}</h5>
        <p style="margin-bottom: 4px; font-size: 0.92rem;"><strong>Selected Layer:</strong> {selected_attn_layer_idx + 1} of {num_layers}</p>
        <p style="margin-bottom: 4px; font-size: 0.92rem;"><strong>Selected Attention Head:</strong> {selected_attn_head_idx + 1} of {num_heads}</p>
        <p style="margin-bottom: 4px; font-size: 0.92rem;"><strong>Layer Attention Tensor Shape:</strong> <code>{layer_attn_shape_str}</code> (batch_size, num_heads, seq_len, seq_len)</p>
        <p style="margin-bottom: 0; font-size: 0.92rem;"><strong>Head Attention Matrix Shape:</strong> <code>{head_attn_shape_str}</code> (query_tokens, key_tokens)</p>
    </div>
    """, unsafe_allow_html=True)

    # Beginner-Friendly Concept Explanation
    st.markdown("""
    <div class="arch-flow" style="font-size: 0.86rem; line-height: 1.5; margin-bottom: 12px;">
        💡 <strong>How Token → Token Attention Works:</strong><br>
        • <strong>Rows (Query / Attending Token):</strong> The token looking back at context.<br>
        • <strong>Columns (Key / Attended-to Token):</strong> The token being focused on.<br>
        • <strong>Color & Values (0.0 to 1.0):</strong> Attention strength. Darker blue indicates stronger focus.<br>
        • <strong>Causal Mask:</strong> Autoregressive language models only attend to current and previous tokens (future positions are 0.00).
    </div>
    """, unsafe_allow_html=True)

    # Clean display labels for tokens
    display_tokens = []
    for i, t in enumerate(tokens):
        cleaned = str(t).replace("\n", "\\n").replace("\t", "\\t")
        if not cleaned.strip():
            cleaned = f"␣ ({repr(t)})"
        display_tokens.append(f"{cleaned} [{i}]" if len(tokens) > 10 else cleaned)

    st.markdown("##### Attention Heatmap")
    fig_attn = px.imshow(
        attn_matrix,
        x=display_tokens,
        y=display_tokens,
        labels=dict(x="Attended-to Token (Key)", y="Attending Token (Query)", color="Attention Weight"),
        color_continuous_scale="Blues",
        range_color=[0.0, 1.0],
        text_auto=".2f" if len(tokens) <= 12 else False,
        title=f"Attention Weights ({selected_attn_layer}, {selected_attn_head})"
    )
    fig_attn.update_traces(
        hoverongaps=False,
        hovertemplate="Attending Token (Query): %{y}<br>Attended Token (Key): %{x}<br>Attention Weight: %{z:.4f}<extra></extra>"
    )
    fig_attn.update_layout(
        template="plotly_white",
        height=max(380, min(650, 180 + len(tokens) * 25)),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig_attn, use_container_width=True)

    # Numerical Attention Matrix Table
    with st.expander(f"🔍 View Attention Matrix Table ({selected_attn_layer}, {selected_attn_head})", expanded=False):
        matrix_df = pd.DataFrame(
            attn_matrix,
            index=[f"{t} (row {i})" for i, t in enumerate(display_tokens)],
            columns=[f"{t} (col {i})" for i, t in enumerate(display_tokens)]
        )
        st.dataframe(
            matrix_df.style.format("{:.4f}").background_gradient(cmap="Blues", axis=None, vmin=0.0, vmax=1.0),
            use_container_width=True
        )

    # Step 8: Hidden State Visualization
    st.markdown("#### 🔬 Step 8 — Hidden State Visualization")
    
    hs_layer_options = [f"Layer {i}" for i in range(1, num_layers + 1)]
    default_hs_idx = 0  # Default Layer 1
    selected_hs_layer = st.selectbox(
        "Layer",
        options=hs_layer_options,
        index=default_hs_idx,
        key=f"hs_layer_select_{section_key_prefix}"
    )

    selected_hs_layer_num = int(selected_hs_layer.split()[1])
    
    # outputs.hidden_states has index 0 as embedding output, indices 1..28 as Transformer layer outputs
    if outputs.hidden_states and len(outputs.hidden_states) > selected_hs_layer_num:
        hs_tensor = outputs.hidden_states[selected_hs_layer_num][0]
        hs_np = hs_tensor.detach().cpu().to(torch.float32).numpy()
    else:
        hs_np = np.zeros((len(tokens), hidden_size), dtype=np.float32)

    seq_len = int(hs_np.shape[0])
    hs_dim = int(hs_np.shape[1])

    # Hidden State Information Card
    st.markdown(f"""
    <div class="layer-card">
        <h5 style="margin-top: 0; color: #4f46e5;">Hidden State Information ({selected_hs_layer})</h5>
        <p style="margin-bottom: 4px; font-size: 0.92rem;"><strong>Selected Layer:</strong> {selected_hs_layer_num} of {num_layers}</p>
        <p style="margin-bottom: 4px; font-size: 0.92rem;"><strong>Number of Tokens:</strong> {seq_len}</p>
        <p style="margin-bottom: 4px; font-size: 0.92rem;"><strong>Hidden Size:</strong> {hs_dim}</p>
        <p style="margin-bottom: 0; font-size: 0.92rem;"><strong>Hidden State Shape:</strong> <code>({seq_len}, {hs_dim})</code></p>
    </div>
    """, unsafe_allow_html=True)

    # Collapsed expander for complete vectors
    with st.expander(f"🔍 View Hidden State Vectors ({selected_hs_layer})", expanded=False):
        for i, (tok, tid) in enumerate(zip(tokens, token_ids)):
            st.markdown(f"**Token:** `{tok}` | **Token ID:** `{tid}` | **Shape:** `({hs_dim},)`")
            st.code(str(hs_np[i].tolist()), language="json")

    # Optional 2D Hidden State PCA
    show_hs_pca = st.checkbox(
        "Show 2D Hidden State PCA",
        value=True,
        key=f"show_hs_pca_{section_key_prefix}"
    )

    if show_hs_pca:
        st.markdown(f"##### Hidden State PCA — {selected_hs_layer}")
        if len(tokens) >= 2:
            hs_pca = PCA(n_components=2)
            hs_pca_coords = hs_pca.fit_transform(hs_np)

            hs_pca_df = pd.DataFrame({
                "Token": tokens,
                "Token ID": token_ids,
                "PCA Dim 1": hs_pca_coords[:, 0],
                "PCA Dim 2": hs_pca_coords[:, 1],
            })

            fig_hs_pca = px.scatter(
                hs_pca_df,
                x="PCA Dim 1",
                y="PCA Dim 2",
                text="Token",
                hover_data={"Token": True, "Token ID": True, "PCA Dim 1": ":.4f", "PCA Dim 2": ":.4f"},
                title=f"2D PCA of Hidden States ({selected_hs_layer})"
            )
            fig_hs_pca.update_traces(
                textposition="top center",
                marker=dict(size=14, color="#10b981", line=dict(width=2, color="#065f46"))
            )
            fig_hs_pca.update_layout(
                template="plotly_white",
                height=380,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_hs_pca, use_container_width=True)
        else:
            st.info("2D PCA requires at least 2 tokens. Hidden-state vector is available above.")

    # Step 9: Logits & Probabilities
    st.markdown("#### 🔬 Step 9 — Logits & Probabilities")
    
    st.markdown("""
    <div class="arch-flow" style="text-align: center; line-height: 1.6;">
        <strong>Final Hidden State</strong><br>
        ↓<br>
        <strong>LM Head</strong><br>
        ↓<br>
        <strong>Logits</strong><br>
        ↓<br>
        <strong>Softmax</strong><br>
        ↓<br>
        <strong>Probabilities</strong>
    </div>
    """, unsafe_allow_html=True)

    if outputs.logits is not None:
        last_hidden_dim = hidden_size
        vocab_size = getattr(model.config, "vocab_size", outputs.logits.shape[-1])
        
        next_token_logits = outputs.logits[0, -1, :]
        probabilities = torch.softmax(next_token_logits, dim=-1)
        prob_sum = float(torch.sum(probabilities).item())
        
        # Info Card
        st.markdown(f"""
        <div class="layer-card">
            <h5 style="margin-top: 0; color: #4f46e5;">Next-Token Distribution</h5>
            <p style="margin-bottom: 4px; font-size: 0.92rem;"><strong>Final Hidden State Shape:</strong> <code>({last_hidden_dim},)</code></p>
            <p style="margin-bottom: 4px; font-size: 0.92rem;"><strong>Vocabulary Size:</strong> <code>{vocab_size:,}</code></p>
            <p style="margin-bottom: 0; font-size: 0.92rem;"><strong>Probability Sum (Softmax):</strong> <code>{prob_sum:.4f}</code></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Top 10 Predictions Table
        top_k = 10
        top_probs, top_indices = torch.topk(probabilities, k=top_k)
        top_logits = next_token_logits[top_indices]
        
        table_rows = []
        chart_tokens = []
        chart_probs = []
        
        for rank_idx, (p_val, idx_val, logit_val) in enumerate(zip(top_probs, top_indices, top_logits), start=1):
            t_id = idx_val.item()
            decoded_tok = tokenizer.decode([t_id])
            p_float = p_val.item()
            l_float = logit_val.item()
            
            display_tok = decoded_tok if decoded_tok.strip() else f"␣ ({repr(decoded_tok)})"
            
            table_rows.append({
                "Rank": rank_idx,
                "Token": display_tok,
                "Token ID": t_id,
                "Logit": l_float,
                "Probability": f"{p_float * 100:.2f}%"
            })
            
            clean_tok_label = decoded_tok.replace("\n", "\\n").replace("\t", "\\t")
            if not clean_tok_label.strip():
                clean_tok_label = f"[{repr(decoded_tok)}]"
            chart_tokens.append(f"{clean_tok_label} (ID: {t_id})")
            chart_probs.append(p_float * 100)
            
        top_df = pd.DataFrame(table_rows)
        
        st.markdown("##### Top 10 Next-Token Predictions")
        st.dataframe(
            top_df.style.format({
                "Logit": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Horizontal Probability Bar Chart
        st.markdown("### Top 10 Next-Token Probabilities")
        bar_df = pd.DataFrame({
            "Token": chart_tokens,
            "Probability (%)": chart_probs
        })
        
        fig_bar = px.bar(
            bar_df,
            x="Probability (%)",
            y="Token",
            orientation="h",
            text="Probability (%)",
            title="Top 10 Next-Token Prediction Probabilities",
            color="Probability (%)",
            color_continuous_scale="Blues"
        )
        fig_bar.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside",
            marker=dict(line=dict(width=1, color="#1e40af"))
        )
        fig_bar.update_layout(
            template="plotly_white",
            height=400,
            margin=dict(l=20, r=40, t=40, b=20),
            xaxis_title="Probability (%)",
            yaxis_title="Token",
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Step 10: Token Generation Visualization
    st.markdown("#### 🔬 Step 10 — Token Generation Visualization")
    
    st.markdown("""
    <div class="arch-flow" style="text-align: center; line-height: 1.6;">
        <strong>Prompt</strong> → <strong>Token 1</strong> → <strong>Token 2</strong> → <strong>Token 3</strong> → ... → <strong>Final Response</strong>
    </div>
    """, unsafe_allow_html=True)
    
    if response_text and response_text.strip():
        resp_encoded = tokenizer(response_text, add_special_tokens=False)
        resp_token_ids = resp_encoded["input_ids"]
        resp_tokens = tokenizer.convert_ids_to_tokens(resp_token_ids)
        total_gen_tokens = len(resp_token_ids)
        
        # Summary Card
        st.markdown(f"""
        <div class="layer-card">
            <h5 style="margin-top: 0; color: #4f46e5;">Generation Progression Summary</h5>
            <p style="margin-bottom: 4px; font-size: 0.92rem;"><strong>Original Prompt:</strong> <em>{html.escape(prompt_text)}</em></p>
            <p style="margin-bottom: 4px; font-size: 0.92rem;"><strong>Total Generated Tokens:</strong> {total_gen_tokens}</p>
            <p style="margin-bottom: 0; font-size: 0.92rem;"><strong>Response Character Length:</strong> {len(response_text)} chars</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Step-by-Step Chronological Progression Table
        gen_steps = []
        for step_i in range(total_gen_tokens):
            t_id = resp_token_ids[step_i]
            t_tok = resp_tokens[step_i]
            t_decoded = tokenizer.decode([t_id])
            accumulated_text = tokenizer.decode(resp_token_ids[:step_i + 1])
            
            display_tok = t_decoded if t_decoded.strip() else f"␣ ({repr(t_decoded)})"
            
            gen_steps.append({
                "Step": step_i + 1,
                "Generated Token": display_tok,
                "Token ID": t_id,
                "Accumulated Text": accumulated_text
            })
            
        gen_df = pd.DataFrame(gen_steps)
        
        st.markdown("##### Chronological Token Generation Timeline")
        st.dataframe(
            gen_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Final Accumulated Response Card
        st.markdown(f"""
        <div class="layer-card" style="border-left: 4px solid #10b981;">
            <strong style="color: #065f46;">Final Accumulated Response:</strong>
            <div style="margin-top: 6px; white-space: pre-wrap; font-size: 0.93rem; color: #1e293b;">{html.escape(response_text)}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Token generation timeline will appear once a response is generated.")

# --- CONVERSATION FEED ---
if not st.session_state.history:
    st.markdown("""
    <div style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 12px; padding: 24px; text-align: center; margin: 20px 0;">
        <div style="font-size: 1.1rem; font-weight: 600; color: #1e293b; margin-bottom: 6px;">Start a Conversation</div>
        <div style="font-size: 0.88rem; color: #64748b; max-width: 480px; margin: 0 auto;">
            Type a prompt below or use the microphone. Each response includes full real-time inspection of tokens, embeddings, transformer layers, attention heatmaps, hidden states, logits, and token generation progression.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for idx, item in enumerate(st.session_state.history):
        item_id = item["id"]
        
        # User message bubble (Right-aligned)
        st.markdown(f"""
        <div class="user-row">
            <div style="display: flex; flex-direction: column; align-items: flex-end; max-width: 82%;">
                <div class="avatar-label">You</div>
                <div class="user-bubble">{html.escape(item["prompt"])}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Assistant message bubble (Left-aligned)
        st.markdown(f"""
        <div class="assistant-row">
            <div style="display: flex; flex-direction: column; align-items: flex-start; max-width: 90%;">
                <div class="avatar-label">Assistant</div>
                <div class="assistant-bubble">{html.escape(item["response"])}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Response Action Controls
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
        <div style="display: flex; gap: 6px; align-items: center; margin-top: 2px; margin-bottom: 8px;">
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

        # Organized X-Ray Analysis expander for this turn
        with st.expander(f"🔬 LLM X-Ray Analysis", expanded=False):
            render_xray_analysis(
                item["prompt"],
                section_key_prefix=f"turn_{item_id}",
                response_text=item.get("response")
            )
        
        st.divider()

# --- BOTTOM CHAT COMPOSER ---
st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)

with st.container():
    col_input, col_mic, col_send = st.columns([8, 1, 1], vertical_alignment="bottom")
    
    with col_input:
        prompt = st.text_area(
            "Message",
            key="user_prompt",
            height=65,
            placeholder="Ask anything or use voice...",
            label_visibility="collapsed"
        )
    
    with col_mic:
        st.html("""
        <button id="mic-trigger-btn" class="composer-icon-btn" type="button" title="Click to speak (Voice-to-Text)" onclick="startSpeechRecognition()">
            🎤
        </button>
        <script>
        let recognitionInstance = null;
        let isRecording = false;

        function getSpeechRecognition() {
            return window.SpeechRecognition || 
                   window.webkitSpeechRecognition || 
                   (window.parent && window.parent.SpeechRecognition) || 
                   (window.parent && window.parent.webkitSpeechRecognition) || 
                   null;
        }

        function findTextarea() {
            const docs = [document];
            if (window.parent && window.parent.document) {
                docs.unshift(window.parent.document);
            }
            for (let doc of docs) {
                try {
                    const ta = doc.querySelector('textarea[data-testid="stTextArea"]') ||
                               doc.querySelector('div[data-testid="stTextArea"] textarea') ||
                               doc.querySelector('textarea[aria-label="Message"]') ||
                               doc.querySelector('textarea');
                    if (ta) return ta;
                } catch (e) {}
            }
            return null;
        }

        function updateInputValue(text) {
            const ta = findTextarea();
            if (!ta) return;
            ta.focus();
            const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value')?.set;
            if (nativeSetter) {
                nativeSetter.call(ta, text);
            } else {
                ta.value = text;
            }
            ta.dispatchEvent(new Event('input', { bubbles: true }));
            ta.dispatchEvent(new Event('change', { bubbles: true }));
        }

        function startSpeechRecognition() {
            const SpeechRecognitionClass = getSpeechRecognition();
            if (!SpeechRecognitionClass) {
                alert("Speech Recognition is not supported in this browser. Please use Google Chrome, Microsoft Edge, or Safari.");
                return;
            }

            const btn = document.getElementById("mic-trigger-btn");

            if (isRecording && recognitionInstance) {
                try { recognitionInstance.stop(); } catch(e) {}
                isRecording = false;
                if (btn) { btn.innerHTML = "🎤"; btn.title = "Click to speak"; btn.style.borderColor = "#cbd5e1"; }
                return;
            }

            try {
                recognitionInstance = new SpeechRecognitionClass();
                recognitionInstance.continuous = false;
                recognitionInstance.interimResults = true;
                recognitionInstance.lang = 'en-US';

                recognitionInstance.onstart = function() {
                    isRecording = true;
                    if (btn) {
                        btn.innerHTML = "🔴";
                        btn.title = "Listening... Speak now. Click to stop.";
                        btn.style.borderColor = "#ef4444";
                    }
                };

                recognitionInstance.onresult = function(event) {
                    let fullText = '';
                    for (let i = 0; i < event.results.length; ++i) {
                        fullText += event.results[i][0].transcript;
                    }
                    if (fullText) {
                        updateInputValue(fullText);
                    }
                };

                recognitionInstance.onerror = function(event) {
                    isRecording = false;
                    if (btn) { btn.innerHTML = "🎤"; btn.title = "Click to speak"; btn.style.borderColor = "#cbd5e1"; }
                    if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
                        alert("Microphone permission was denied. Please allow microphone access in your browser settings.");
                    } else if (event.error === 'audio-capture') {
                        alert("No microphone found on your device.");
                    }
                };

                recognitionInstance.onend = function() {
                    isRecording = false;
                    if (btn) { btn.innerHTML = "🎤"; btn.title = "Click to speak"; btn.style.borderColor = "#cbd5e1"; }
                };

                recognitionInstance.start();
            } catch (err) {
                isRecording = false;
                if (btn) { btn.innerHTML = "🎤"; btn.title = "Click to speak"; btn.style.borderColor = "#cbd5e1"; }
                console.error("Speech recognition error:", err);
            }
        }
        </script>
        """)
        
    with col_send:
        send_clicked = st.button("➤", key="generate_btn", type="primary", use_container_width=True, help="Send message")

if send_clicked:
    if prompt.strip():
        with st.spinner("Generating response..."):
            response = generate_response(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            st.session_state.history.append({
                "id": len(st.session_state.history),
                "prompt": prompt,
                "response": response,
                "feedback": None
            })
            st.session_state["last_response"] = response
            st.rerun()
    else:
        st.warning("Please enter a prompt before sending.")
