import html
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from sklearn.decomposition import PCA
from model import load_model_and_tokenizer, generate_response, MODEL_NAME

st.set_page_config(
    page_title="LLM X-Ray",
    page_icon="🔍",
    layout="centered"
)

# Custom styling for a clean, modern ChatGPT-style conversational interface
st.markdown("""
<style>
    /* Global layout & typography */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3.5rem;
        max-width: 820px;
    }
    
    /* 1. Header styling - Compact & Clean */
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 10px;
        margin-bottom: 16px;
        flex-wrap: wrap;
        gap: 8px;
    }
    .header-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 8px;
        letter-spacing: -0.01em;
    }
    .model-badge {
        background: #f8fafc;
        color: #475569;
        border: 1px solid #e2e8f0;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 9999px;
        font-family: monospace;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }

    /* 2. ChatGPT-style Chat Area */
    .user-row {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 14px;
        margin-top: 6px;
    }
    .user-bubble {
        background-color: #2563eb;
        color: #ffffff;
        border-radius: 18px 18px 4px 18px;
        padding: 10px 16px;
        max-width: 82%;
        font-size: 0.94rem;
        line-height: 1.5;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
        word-break: break-word;
    }
    .assistant-row {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 4px;
        margin-top: 4px;
    }
    .assistant-bubble {
        background-color: #ffffff;
        color: #0f172a;
        border-radius: 4px 18px 18px 18px;
        padding: 8px 12px 10px 4px;
        max-width: 96%;
        font-size: 0.95rem;
        line-height: 1.6;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .avatar-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 3px;
        color: #94a3b8;
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
        background: transparent;
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
        background: #f1f5f9;
        border-color: #cbd5e1;
        color: #0f172a;
    }
    div[data-testid="stHorizontalBlock"] button {
        padding: 2px 8px !important;
        font-size: 12px !important;
        border-radius: 6px !important;
        border: 1px solid #e2e8f0 !important;
        background: transparent !important;
        color: #64748b !important;
        min-height: 32px !important;
        height: 32px !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="stHorizontalBlock"] button:hover {
        background: #f1f5f9 !important;
        border-color: #cbd5e1 !important;
        color: #0f172a !important;
    }

    /* 3. Bottom Composer styles */
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

    /* Cards & Visualizations inside X-Ray expander */
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
        font-family: monospace;
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
        font-family: monospace;
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

# ----------------- 1. HEADER -----------------
st.markdown(f"""
<div class="header-container">
    <div class="header-title">
        <span>🔍</span> LLM X-Ray
    </div>
    <div>
        <span class="model-badge">
            ⚡ {MODEL_NAME}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE -----------------
if "history" not in st.session_state:
    st.session_state.history = []

# Cache model and tokenizer loading
@st.cache_resource(show_spinner="Loading model and tokenizer...")
def get_model():
    return load_model_and_tokenizer()

model, tokenizer = get_model()

# Model parameters
num_layers = getattr(model.config, "num_hidden_layers", 28)
hidden_size = getattr(model.config, "hidden_size", 1536)
num_heads = getattr(model.config, "num_attention_heads", 12)
num_kv_heads = getattr(model.config, "num_key_value_heads", 2)
intermediate_size = getattr(model.config, "intermediate_size", 8960)
vocab_size = getattr(model.config, "vocab_size", 151936)

# ----------------- 7. SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.subheader("⚙️ Settings")
    
    # Output Length Control
    length_options = {
        "Short (64 tokens)": 64,
        "Normal (128 tokens)": 128,
        "Detailed (256 tokens)": 256,
        "Long (512 tokens)": 512,
    }
    selected_length = st.radio(
        "Output Length",
        options=list(length_options.keys()),
        index=1,
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

    st.markdown("##### 🔬 Default Layer & Head Target")
    sidebar_layer_options = [f"Layer {i}" for i in range(1, num_layers + 1)]
    sidebar_selected_layer = st.selectbox(
        "Transformer layer",
        options=sidebar_layer_options,
        index=0,
        key="sidebar_layer_select",
        help=f"Select Transformer layer (1 to {num_layers})"
    )
    sidebar_head_options = [f"Head {i}" for i in range(1, num_heads + 1)]
    sidebar_selected_head = st.selectbox(
        "Attention head",
        options=sidebar_head_options,
        index=0,
        key="sidebar_head_select",
        help=f"Select Attention head (1 to {num_heads})"
    )
    
    st.divider()
    if st.button("🗑️ Clear History", key="clear_history_btn", use_container_width=True):
        st.session_state.history = []
        if "last_response" in st.session_state:
            del st.session_state["last_response"]
        st.rerun()

    st.caption(f"Conversations: {len(st.session_state.history)}")
    st.caption(f"Model: `{MODEL_NAME}`")


# ----------------- 5. X-RAY ANALYSIS FUNCTION (Steps 4 to 11) -----------------
def render_xray_analysis(prompt_text, section_key_prefix="", response_text=None):
    """
    Renders complete Steps 4-11 inspection for a given prompt and response.
    Preserves all existing calculations, tables, PCA plots, heatmaps, and tensor information.
    """
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
        <div class="avatar-label">Input Text</div>
        <div class="token-box" style="font-family: inherit;">{html.escape(prompt_text)}</div>
        <div class="arrow-divider">↓ Tokenizer ↓</div>
        <div class="avatar-label">Tokens ({len(tokens)} tokens)</div>
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

    # Vector Statistics Table
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

    # PCA 2D Visualization
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
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"embedding_pca_{section_key_prefix}"
    )

    # Step 6: Transformer Visualization
    st.markdown("#### 🧠 Step 6 — Transformer Visualization")
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

    # Single inspection forward pass for Steps 7, 8, 9 & 11
    with torch.no_grad():
        model.config._attn_implementation = "eager"
        for layer_mod in model.model.layers:
            if hasattr(layer_mod.self_attn, "config"):
                layer_mod.self_attn.config._attn_implementation = "eager"
        
        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)

    # Step 7 & 11: Attention Visualization
    st.markdown("#### 🔬 Step 7 & 11 — Attention Visualization")
    
    col_l, col_h = st.columns(2)
    with col_l:
        selected_attn_layer = st.selectbox(
            "Transformer Layer",
            options=layer_options,
            index=0,
            key=f"attn_layer_select_{section_key_prefix}",
            help="Select a Transformer layer to inspect"
        )
    with col_h:
        head_num_options = [f"Head {i}" for i in range(1, num_heads + 1)]
        selected_attn_head = st.selectbox(
            "Attention Head",
            options=head_num_options,
            index=0,
            key=f"attn_head_select_{section_key_prefix}",
            help=f"Select an attention head ({num_heads} heads available)"
        )

    selected_attn_layer_idx = int(selected_attn_layer.split()[1]) - 1
    selected_attn_head_idx = int(selected_attn_head.split()[1]) - 1

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
        attn_matrix = np.eye(seq_len, dtype=np.float32)

    st.markdown(f"""
    <div class="layer-card">
        <h5 style="margin-top: 0; color: #4f46e5;">Attention Details: {selected_attn_layer} — {selected_attn_head}</h5>
        <p style="margin-bottom: 4px; font-size: 0.92rem;"><strong>Selected Layer:</strong> {selected_attn_layer_idx + 1} of {num_layers}</p>
        <p style="margin-bottom: 4px; font-size: 0.92rem;"><strong>Selected Attention Head:</strong> {selected_attn_head_idx + 1} of {num_heads}</p>
        <p style="margin-bottom: 4px; font-size: 0.92rem;"><strong>Layer Attention Tensor Shape:</strong> <code>{layer_attn_shape_str}</code></p>
        <p style="margin-bottom: 0; font-size: 0.92rem;"><strong>Head Attention Matrix Shape:</strong> <code>{head_attn_shape_str}</code></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="arch-flow" style="font-size: 0.86rem; line-height: 1.5; margin-bottom: 12px;">
        💡 <strong>Attention Overview:</strong> Rows (Query / Attending Token) attend to Columns (Key / Attended-to Token). Causal mask restricts attention to current and previous tokens.
    </div>
    """, unsafe_allow_html=True)

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
    st.plotly_chart(
        fig_attn,
        use_container_width=True,
        key=f"attention_heatmap_{section_key_prefix}"
    )

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
    
    selected_hs_layer = st.selectbox(
        "Layer",
        options=layer_options,
        index=0,
        key=f"hs_layer_select_{section_key_prefix}"
    )
    selected_hs_layer_num = int(selected_hs_layer.split()[1])
    
    if outputs.hidden_states and len(outputs.hidden_states) > selected_hs_layer_num:
        hs_tensor = outputs.hidden_states[selected_hs_layer_num][0]
        hs_np = hs_tensor.detach().cpu().to(torch.float32).numpy()
    else:
        hs_np = np.zeros((len(tokens), hidden_size), dtype=np.float32)

    hs_seq_len = int(hs_np.shape[0])
    hs_dim = int(hs_np.shape[1])

    st.markdown(f"""
    <div class="layer-card">
        <h5 style="margin-top: 0; color: #4f46e5;">Hidden State Information ({selected_hs_layer})</h5>
        <p style="margin-bottom: 4px; font-size: 0.92rem;"><strong>Selected Layer:</strong> {selected_hs_layer_num} of {num_layers}</p>
        <p style="margin-bottom: 4px; font-size: 0.92rem;"><strong>Number of Tokens:</strong> {hs_seq_len}</p>
        <p style="margin-bottom: 4px; font-size: 0.92rem;"><strong>Hidden Size:</strong> {hs_dim}</p>
        <p style="margin-bottom: 0; font-size: 0.92rem;"><strong>Hidden State Shape:</strong> <code>({hs_seq_len}, {hs_dim})</code></p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(f"🔍 View Hidden State Vectors ({selected_hs_layer})", expanded=False):
        for i, (tok, tid) in enumerate(zip(tokens, token_ids)):
            st.markdown(f"**Token:** `{tok}` | **Token ID:** `{tid}` | **Shape:** `({hs_dim},)`")
            st.code(str(hs_np[i].tolist()), language="json")

    if len(tokens) >= 2:
        st.markdown(f"##### Hidden State PCA — {selected_hs_layer}")
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
        st.plotly_chart(
            fig_hs_pca,
            use_container_width=True,
            key=f"hidden_state_pca_{section_key_prefix}"
        )

    # Step 9: Logits & Probabilities
    st.markdown("#### 🔬 Step 9 — Logits & Probabilities")
    st.markdown("""
    <div class="arch-flow" style="text-align: center; line-height: 1.6;">
        <strong>Final Hidden State</strong> → <strong>LM Head</strong> → <strong>Logits</strong> → <strong>Softmax</strong> → <strong>Probabilities</strong>
    </div>
    """, unsafe_allow_html=True)

    if outputs.logits is not None:
        last_hidden_dim = hidden_size
        v_size = getattr(model.config, "vocab_size", outputs.logits.shape[-1])
        next_token_logits = outputs.logits[0, -1, :]
        probabilities = torch.softmax(next_token_logits, dim=-1)
        prob_sum = float(torch.sum(probabilities).item())
        
        st.markdown(f"""
        <div class="layer-card">
            <h5 style="margin-top: 0; color: #4f46e5;">Next-Token Distribution</h5>
            <p style="margin-bottom: 4px; font-size: 0.92rem;"><strong>Final Hidden State Shape:</strong> <code>({last_hidden_dim},)</code></p>
            <p style="margin-bottom: 4px; font-size: 0.92rem;"><strong>Vocabulary Size:</strong> <code>{v_size:,}</code></p>
            <p style="margin-bottom: 0; font-size: 0.92rem;"><strong>Probability Sum (Softmax):</strong> <code>{prob_sum:.4f}</code></p>
        </div>
        """, unsafe_allow_html=True)
        
        top_k_count = 10
        top_probs, top_indices = torch.topk(probabilities, k=top_k_count)
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
            top_df.style.format({"Logit": "{:.4f}"}),
            use_container_width=True,
            hide_index=True
        )
        
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
        st.plotly_chart(
            fig_bar,
            use_container_width=True,
            key=f"logits_probability_{section_key_prefix}"
        )

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
        
        st.markdown(f"""
        <div class="layer-card">
            <h5 style="margin-top: 0; color: #4f46e5;">Generation Progression Summary</h5>
            <p style="margin-bottom: 4px; font-size: 0.92rem;"><strong>Original Prompt:</strong> <em>{html.escape(prompt_text)}</em></p>
            <p style="margin-bottom: 4px; font-size: 0.92rem;"><strong>Total Generated Tokens:</strong> {total_gen_tokens}</p>
            <p style="margin-bottom: 0; font-size: 0.92rem;"><strong>Response Character Length:</strong> {len(response_text)} chars</p>
        </div>
        """, unsafe_allow_html=True)
        
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
        st.dataframe(gen_df, use_container_width=True, hide_index=True)
        
        st.markdown(f"""
        <div class="layer-card" style="border-left: 4px solid #10b981;">
            <strong style="color: #065f46;">Final Accumulated Response:</strong>
            <div style="margin-top: 6px; white-space: pre-wrap; font-size: 0.93rem; color: #1e293b;">{html.escape(response_text)}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Token generation timeline will appear once a response is generated.")


# ----------------- 6. TABS / NAVIGATION (Chat as default) -----------------
tab_chat, tab_tokens, tab_embeddings, tab_attention, tab_layers, tab_logits, tab_generation = st.tabs([
    "💬 Chat",
    "🔤 Tokens",
    "📊 Embeddings",
    "🔬 Attention",
    "🧠 Layers",
    "📈 Logits",
    "⚡ Generation"
])

# Get latest prompt and response for standalone tabs
latest_prompt = st.session_state.history[-1]["prompt"] if st.session_state.history else "Explain Machine Learning"
latest_response = st.session_state.history[-1]["response"] if st.session_state.history else ""

# =========================================================================
# 2. CHATGPT-STYLE CHAT AREA (Default Tab)
# =========================================================================
with tab_chat:
    if not st.session_state.history:
        st.markdown("""
        <div style="background: #fafafa; border: 1px dashed #e2e8f0; border-radius: 12px; padding: 28px 20px; text-align: center; margin: 20px 0;">
            <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-bottom: 6px;">How can I help you today?</div>
            <div style="font-size: 0.88rem; color: #64748b; max-width: 520px; margin: 0 auto; line-height: 1.5;">
                Ask any question below. Each response comes with an expandable <strong>🔬 LLM X-Ray Analysis</strong> to inspect tokens, embeddings, attention matrices, hidden states, logits, and token generation in real-time.
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
                <div style="display: flex; flex-direction: column; align-items: flex-start; max-width: 96%;">
                    <div class="avatar-label">LLM X-Ray</div>
                    <div class="assistant-bubble">{html.escape(item["response"])}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 4. Small, subtle ChatGPT-style action toolbar
            col_actions, _ = st.columns([8, 2])
            with col_actions:
                col_c1, col_c2, col_c3, col_c4, col_c5, col_c6 = st.columns([1.0, 0.8, 0.8, 1.4, 0.7, 1.1], vertical_alignment="center")
                
                # 1. Copy
                with col_c1:
                    esc_resp = json.dumps(item["response"])
                    st.html(f"""
                    <button id="copy-btn-{item_id}" class="action-btn" title="Copy response to clipboard" onclick="copyText_{item_id}()">
                        📋 Copy
                    </button>
                    <script>
                    function copyText_{item_id}() {{
                        const text = {esc_resp};
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
                    </script>
                    """)
                
                # 2. Like
                with col_c2:
                    liked = item.get("feedback") == "like"
                    if st.button("👍 Liked" if liked else "👍", key=f"btn_like_{item_id}", help="Good response"):
                        item["feedback"] = "like" if not liked else None
                        st.rerun()
                
                # 3. Dislike
                with col_c3:
                    disliked = item.get("feedback") == "dislike"
                    if st.button("👎 Disliked" if disliked else "👎", key=f"btn_dislike_{item_id}", help="Bad response"):
                        item["feedback"] = "dislike" if not disliked else None
                        st.rerun()
                
                # 4. Regenerate
                with col_c4:
                    if st.button("🔄 Regenerate", key=f"btn_regen_{item_id}", help="Regenerate this response"):
                        with st.spinner("Regenerating response..."):
                            new_response = generate_response(
                                model,
                                tokenizer,
                                item["prompt"],
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                            )
                            item["response"] = new_response
                            st.session_state["last_response"] = new_response
                            st.rerun()
                
                # 5. More (Word count, Character count, Download .txt, Read Aloud)
                with col_c5:
                    with st.popover("⋯", help="More options"):
                        st.markdown("**Response Details**")
                        word_count = len(item["response"].split())
                        char_count = len(item["response"])
                        st.caption(f"Words: {word_count} | Characters: {char_count}")
                        st.divider()
                        
                        st.html(f"""
                        <button id="tts-btn-{item_id}" class="action-btn" style="width: 100%; margin-bottom: 8px; justify-content: center;" onclick="toggleTTS_{item_id}()">
                            🔊 Read Aloud
                        </button>
                        <script>
                        function toggleTTS_{item_id}() {{
                            const text = {esc_resp};
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
                        
                        st.download_button(
                            label="💾 Download .txt",
                            data=item["response"],
                            file_name=f"response_{idx+1}.txt",
                            mime="text/plain",
                            key=f"dl_{item_id}",
                            use_container_width=True
                        )
                
                # 6. Sources
                with col_c6:
                    with st.popover("Sources", help="View knowledge sources"):
                        st.markdown("**Generation Sources**")
                        st.caption(f"**Model:** `{MODEL_NAME}`")
                        st.caption("Generated directly by the local causal language model weights and parameters. No external web search or RAG documents were retrieved.")
            
            # 5. Expandable X-Ray Analysis section for this turn
            with st.expander(f"🔬 LLM X-Ray Analysis", expanded=False):
                render_xray_analysis(
                    item["prompt"],
                    section_key_prefix=f"chat_turn_{item_id}",
                    response_text=item.get("response")
                )
            
            st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

    # 3. ChatGPT-style Chat Composer (Fixed at the bottom of Chat view)
    st.markdown("<div class='composer-container'></div>", unsafe_allow_html=True)
    
    col_input, col_mic, col_send = st.columns([8, 1, 1], vertical_alignment="bottom")
    
    with col_input:
        prompt = st.text_area(
            "Message",
            key="user_prompt",
            height=60,
            placeholder="Message LLM X-Ray...",
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
            with st.spinner("Thinking..."):
                response = generate_response(
                    model,
                    tokenizer,
                    prompt.strip(),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                st.session_state.history.append({
                    "id": len(st.session_state.history),
                    "prompt": prompt.strip(),
                    "response": response,
                    "feedback": None
                })
                st.session_state["last_response"] = response
                st.rerun()
        else:
            st.warning("Please enter a message before sending.")


# =========================================================================
# STANDALONE NAVIGATION TABS (Deep dive on latest prompt/response)
# =========================================================================
with tab_tokens:
    st.markdown("### 🔤 Step 4 — Tokenization (Latest Prompt)")
    st.caption(f"Prompt: *\"{latest_prompt}\"*")
    render_xray_analysis(latest_prompt, section_key_prefix="tab_tok", response_text=latest_response)

with tab_embeddings:
    st.markdown("### 📊 Step 5 — Embedding Visualization (Latest Prompt)")
    st.caption(f"Prompt: *\"{latest_prompt}\"*")
    render_xray_analysis(latest_prompt, section_key_prefix="tab_emb", response_text=latest_response)

with tab_attention:
    st.markdown("### 🔬 Step 7 & 11 — Attention Visualization (Latest Prompt)")
    st.caption(f"Prompt: *\"{latest_prompt}\"*")
    render_xray_analysis(latest_prompt, section_key_prefix="tab_att", response_text=latest_response)

with tab_layers:
    st.markdown("### 🧠 Step 6 & 8 — Transformer Layers & Hidden States (Latest Prompt)")
    st.caption(f"Prompt: *\"{latest_prompt}\"*")
    render_xray_analysis(latest_prompt, section_key_prefix="tab_lay", response_text=latest_response)

with tab_logits:
    st.markdown("### 📈 Step 9 — Logits & Probabilities (Latest Prompt)")
    st.caption(f"Prompt: *\"{latest_prompt}\"*")
    render_xray_analysis(latest_prompt, section_key_prefix="tab_log", response_text=latest_response)

with tab_generation:
    st.markdown("### ⚡ Step 10 — Token Generation Timeline (Latest Response)")
    st.caption(f"Prompt: *\"{latest_prompt}\"*")
    render_xray_analysis(latest_prompt, section_key_prefix="tab_gen", response_text=latest_response)
