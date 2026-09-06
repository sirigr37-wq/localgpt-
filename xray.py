"""
X-Ray Inspection Module for LocalGPT (Phase 2 - Step 14: X-Ray Panel).
Provides clean organized tabbed inspection of internal LLM generation stages:
- Tab 1: Tokens (Tokens & Token IDs)
- Tab 2: Embeddings (Vector Statistics & 2D PCA)
- Tab 3: Layers (Transformer Structure & Sub-modules)
- Tab 4: Attention (Multi-Head Attention Heatmaps & Weight Matrices)
- Tab 5: Hidden States (Intermediate Layer Tensors & 2D PCA)
- Tab 6: Logits (Next-Token Predictions, Probabilities & Generation Timeline)
"""

import html
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from sklearn.decomposition import PCA

from tokenizer import tokenize_input, format_token_chips
from visualization import (
    plot_pca_2d,
    plot_attention_heatmap,
    plot_top_k_probabilities,
    render_pipeline_banner_html,
    render_probability_bars_html,
)


def render_xray_analysis(
    model,
    tokenizer,
    prompt_text: str,
    section_key_prefix: str = "",
    response_text: Optional[str] = None
) -> None:
    """
    Renders complete Step 14 & 15 X-Ray Panel with 6 organized tabs:
    Tokens | Embeddings | Layers | Attention | Hidden States | Logits.
    Uses section_key_prefix to guarantee unique Streamlit element IDs per turn.
    """
    if not prompt_text or not prompt_text.strip():
        st.info("No prompt text available for X-Ray analysis.")
        return

    # Extract model configuration parameters safely
    num_layers = getattr(model.config, "num_hidden_layers", 28)
    hidden_size = getattr(model.config, "hidden_size", 1536)
    num_heads = getattr(model.config, "num_attention_heads", 12)
    num_kv_heads = getattr(model.config, "num_key_value_heads", 2)
    intermediate_size = getattr(model.config, "intermediate_size", 8960)
    vocab_size = getattr(model.config, "vocab_size", 151936)

    # ----------------- INFERENCE PIPELINE BANNER -----------------
    st.markdown("""
    <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 10px; padding: 10px 14px; margin-bottom: 12px;">
        <div style="font-weight: 700; font-size: 0.85rem; color: #1e3a8a; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
            <span>🔬</span> LLM X-Ray Generation Pipeline
        </div>
        <div style="font-size: 0.76rem; color: #334155; font-family: monospace; display: flex; flex-wrap: wrap; align-items: center; gap: 4px; line-height: 1.8;">
            <span style="background:#e0e7ff; color:#3730a3; padding:1px 6px; border-radius:4px; font-weight:600;">User</span> ➔
            <span style="background:#e0e7ff; color:#3730a3; padding:1px 6px; border-radius:4px; font-weight:600;">Tokens</span> ➔
            <span style="background:#e0e7ff; color:#3730a3; padding:1px 6px; border-radius:4px; font-weight:600;">Token IDs</span> ➔
            <span style="background:#e0e7ff; color:#3730a3; padding:1px 6px; border-radius:4px; font-weight:600;">Embeddings</span> ➔
            <span style="background:#e0e7ff; color:#3730a3; padding:1px 6px; border-radius:4px; font-weight:600;">Transformer Layers</span> ➔
            <span style="background:#e0e7ff; color:#3730a3; padding:1px 6px; border-radius:4px; font-weight:600;">Attention</span> ➔
            <span style="background:#e0e7ff; color:#3730a3; padding:1px 6px; border-radius:4px; font-weight:600;">Hidden States</span> ➔
            <span style="background:#e0e7ff; color:#3730a3; padding:1px 6px; border-radius:4px; font-weight:600;">Logits</span> ➔
            <span style="background:#e0e7ff; color:#3730a3; padding:1px 6px; border-radius:4px; font-weight:600;">Probabilities</span> ➔
            <span style="background:#e0e7ff; color:#3730a3; padding:1px 6px; border-radius:4px; font-weight:600;">Generated Tokens</span> ➔
            <span style="background:#dcfce7; color:#166534; padding:1px 6px; border-radius:4px; font-weight:600;">Response</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Real Tokenizer Execution (Prompt)
    tok_data = tokenize_input(tokenizer, prompt_text, add_special_tokens=False)
    tokens = tok_data["tokens"]
    token_ids = tok_data["token_ids"]
    display_tokens = tok_data["display_tokens"]
    seq_len = len(tokens)

    # Tokenize generated response tokens if available
    resp_token_ids = []
    resp_tokens = []
    if response_text and response_text.strip():
        resp_encoded = tokenizer(response_text, add_special_tokens=False)
        resp_token_ids = resp_encoded.get("input_ids", [])
        resp_tokens = tokenizer.convert_ids_to_tokens(resp_token_ids)
    total_gen_tokens = len(resp_token_ids)

    # 2. Input Embeddings Lookup (1,536-dim)
    input_embeddings_layer = model.get_input_embeddings()
    with torch.no_grad():
        token_tensor = torch.tensor(token_ids, device=next(model.parameters()).device)
        embeddings_tensor = input_embeddings_layer(token_tensor)
        emb_np = embeddings_tensor.detach().cpu().to(torch.float32).numpy()
    emb_dim = int(emb_np.shape[1])

    # 3. Single Real Forward Pass for Attention, Hidden States & Step Logits
    full_input_ids = token_ids + resp_token_ids if total_gen_tokens > 0 else token_ids
    with torch.no_grad():
        model.config._attn_implementation = "eager"
        for layer_mod in getattr(model.model, "layers", []):
            if hasattr(layer_mod, "self_attn") and hasattr(layer_mod.self_attn, "config"):
                layer_mod.self_attn.config._attn_implementation = "eager"

        full_tensor = torch.tensor([full_input_ids], device=next(model.parameters()).device)
        outputs = model(input_ids=full_tensor, output_attentions=True, output_hidden_states=True)

    layer_options = [f"Layer {i}" for i in range(1, num_layers + 1)]

    # ----------------- 6 REQUIRED X-RAY PANEL TABS -----------------
    tab_tokens, tab_embeddings, tab_layers, tab_attention, tab_hidden, tab_logits = st.tabs([
        "🔤 Tokens",
        "📊 Embeddings",
        "🧠 Layers",
        "🔬 Attention",
        "📈 Hidden States",
        "⚡ Logits"
    ])

    # =========================================================================
    # TAB 1: TOKENS
    # =========================================================================
    with tab_tokens:
        st.markdown("##### 🔤 Tokenized Representation")
        tokens_pipe_str = " | ".join(str(t) for t in tokens)
        token_ids_str = " | ".join(str(tid) for tid in token_ids)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Total Tokens", f"{seq_len} tokens")
        with col_m2:
            st.metric("Character Length", f"{len(prompt_text)} chars")

        st.markdown(f"""
        <div class="tokenizer-card">
            <div class="avatar-label">Input Text</div>
            <div class="token-box" style="font-family: inherit;">{html.escape(prompt_text)}</div>
            <div class="arrow-divider">↓ Byte-Pair Encoding (BPE) Tokenizer ↓</div>
            <div class="avatar-label">Tokens ({seq_len} tokens)</div>
            <div class="token-box">{html.escape(tokens_pipe_str)}</div>
            <div class="avatar-label">Token IDs</div>
            <div class="token-box">{html.escape(token_ids_str)}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("###### Visual Token Chips")
        chips_html = format_token_chips(display_tokens, token_ids)
        st.markdown(f'<div style="margin-bottom: 12px;">{chips_html}</div>', unsafe_allow_html=True)

        st.markdown("###### Detailed Token Breakdown Table")
        if not tok_data["df"].empty:
            st.dataframe(tok_data["df"], use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 2: EMBEDDINGS
    # =========================================================================
    with tab_embeddings:
        st.markdown("##### 📊 Token Embedding Vectors")
        st.metric("Embedding Dimension", f"{emb_dim} dimensions")

        # Vector Statistics Table (Min, Max, Mean, Std Dev)
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
        st.markdown("###### Vector Statistics")
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

        # 2D PCA Projection Visualization
        st.markdown("###### 2D PCA Projection of Token Embeddings")
        fig_emb_pca = plot_pca_2d(
            tokens=tokens,
            token_ids=token_ids,
            vectors=emb_np,
            title="Token Embeddings in 2D PCA Space",
            marker_color="#4f46e5"
        )
        st.plotly_chart(
            fig_emb_pca,
            use_container_width=True,
            key=f"pca_emb_chart_{section_key_prefix}"
        )

        # Complete Vectors Expander
        with st.expander("🔍 View Complete Embedding Vectors", expanded=False):
            for i, (tok, tid) in enumerate(zip(tokens, token_ids)):
                st.markdown(f"**Token:** `{tok}` | **Token ID:** `{tid}` | **Shape:** `({emb_dim},)`")
                st.code(str(emb_np[i].tolist()), language="json")

    # =========================================================================
    # TAB 3: LAYERS
    # =========================================================================
    with tab_layers:
        st.markdown("##### 🧠 Transformer Architecture & Layer Flow")
        st.markdown(f"**Embeddings** → **Layer 1** → **Layer 2** → ... → **Layer {num_layers}** → **LM Head Output**")

        # Interactive Layer Selector
        selected_layer = st.selectbox(
            "Select Transformer Layer to Inspect",
            options=layer_options,
            index=0,
            key=f"layers_tab_select_{section_key_prefix}",
            help="Select one of the 28 Transformer layers to view its architecture and configuration."
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
                <li><strong>Sub-modules:</strong> Input RMSNorm, Self-Attention (Q/K/V with RoPE & GQA), Post-Attention RMSNorm, SwiGLU MLP (Gate/Up/Down + SiLU)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        if actual_layer is not None:
            with st.expander(f"🔍 Inspect {selected_layer} Module Code", expanded=False):
                st.code(str(actual_layer), language="text")

        with st.expander(f"🔍 View Architecture Sequence ({num_layers} Layers)", expanded=False):
            arch_steps = ["Input Embeddings"] + [f"Layer {i}" for i in range(1, num_layers + 1)] + ["LM Head Output"]
            st.markdown(f'<div class="arch-flow" style="white-space: pre; text-align: center;">' + "\n↓\n".join(arch_steps) + '</div>', unsafe_allow_html=True)

    # =========================================================================
    # TAB 4: ATTENTION
    # =========================================================================
    with tab_attention:
        st.markdown("##### 🔬 Multi-Head Self-Attention Visualization")

        col_attn_l, col_attn_h = st.columns(2)
        with col_attn_l:
            selected_attn_layer = st.selectbox(
                "Transformer Layer",
                options=layer_options,
                index=0,
                key=f"attn_tab_layer_select_{section_key_prefix}",
                help="Select a Transformer layer (1 to 28)"
            )
        with col_attn_h:
            head_num_options = [f"Head {i}" for i in range(1, num_heads + 1)]
            selected_attn_head = st.selectbox(
                "Attention Head",
                options=head_num_options,
                index=0,
                key=f"attn_tab_head_select_{section_key_prefix}",
                help=f"Select an attention head (1 to {num_heads})"
            )

        selected_attn_layer_idx = int(selected_attn_layer.split()[1]) - 1
        selected_attn_head_idx = int(selected_attn_head.split()[1]) - 1

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
            layer_attn_shape_str = f"({t_shape[0]}, {t_shape[1]}, {seq_len}, {seq_len})"
            if t_shape[1] > selected_attn_head_idx:
                attn_matrix = layer_attn_tensor[0, selected_attn_head_idx][:seq_len, :seq_len].detach().cpu().to(torch.float32).numpy()
                head_attn_shape_str = f"({attn_matrix.shape[0]}, {attn_matrix.shape[1]})"

        if attn_matrix is None:
            attn_matrix = np.eye(seq_len, dtype=np.float32)

        st.markdown(f"""
        <div class="layer-card">
            <h5 style="margin-top: 0; color: #4f46e5;">Attention Details: {selected_attn_layer} — {selected_attn_head}</h5>
            <p style="margin-bottom: 4px; font-size: 0.90rem;"><strong>Layer Attention Tensor Shape:</strong> <code>{layer_attn_shape_str}</code></p>
            <p style="margin-bottom: 4px; font-size: 0.90rem;"><strong>Head Attention Matrix Shape:</strong> <code>{head_attn_shape_str}</code></p>
            <p style="margin-bottom: 0; font-size: 0.86rem; color: #64748b;"><em>Rows (Query / Attending Token) attend to Columns (Key / Attended-to Token). Causal mask restricts attention to current and previous tokens.</em></p>
        </div>
        """, unsafe_allow_html=True)

        # Plotly Attention Heatmap
        clean_labels = [f"{t} [{i}]" if len(tokens) > 10 else t for i, t in enumerate(display_tokens)]
        fig_attn = plot_attention_heatmap(
            attn_matrix=attn_matrix,
            display_tokens=clean_labels,
            title=f"Attention Weights ({selected_attn_layer}, {selected_attn_head})",
            colorscale="Blues"
        )
        st.plotly_chart(
            fig_attn,
            use_container_width=True,
            key=f"attn_tab_chart_{section_key_prefix}"
        )

        # Full Attention Matrix Table Expander
        with st.expander(f"🔍 View Attention Matrix Table ({selected_attn_layer}, {selected_attn_head})", expanded=False):
            matrix_df = pd.DataFrame(
                attn_matrix,
                index=[f"{t} (row {i})" for i, t in enumerate(clean_labels)],
                columns=[f"{t} (col {i})" for i, t in enumerate(clean_labels)]
            )
            st.dataframe(
                matrix_df.style.format("{:.4f}").background_gradient(cmap="Blues", axis=None, vmin=0.0, vmax=1.0),
                use_container_width=True
            )

    # =========================================================================
    # TAB 5: HIDDEN STATES
    # =========================================================================
    with tab_hidden:
        st.markdown("##### 📈 Hidden State Representation")

        selected_hs_layer = st.selectbox(
            "Select Layer for Hidden States",
            options=layer_options,
            index=0,
            key=f"hs_tab_layer_select_{section_key_prefix}",
            help="Select which transformer layer's hidden state to inspect"
        )
        selected_hs_layer_num = int(selected_hs_layer.split()[1])

        if outputs.hidden_states and len(outputs.hidden_states) > selected_hs_layer_num:
            hs_tensor = outputs.hidden_states[selected_hs_layer_num][0][:seq_len]
            hs_np = hs_tensor.detach().cpu().to(torch.float32).numpy()
        else:
            hs_np = np.zeros((len(tokens), hidden_size), dtype=np.float32)

        hs_seq_len = int(hs_np.shape[0])
        hs_dim = int(hs_np.shape[1])

        st.markdown(f"""
        <div class="layer-card">
            <h5 style="margin-top: 0; color: #4f46e5;">Hidden State Tensor ({selected_hs_layer})</h5>
            <p style="margin-bottom: 4px; font-size: 0.90rem;"><strong>Selected Layer:</strong> {selected_hs_layer_num} of {num_layers}</p>
            <p style="margin-bottom: 4px; font-size: 0.90rem;"><strong>Sequence Length:</strong> {hs_seq_len} tokens</p>
            <p style="margin-bottom: 4px; font-size: 0.90rem;"><strong>Hidden Dimension:</strong> {hs_dim}</p>
            <p style="margin-bottom: 0; font-size: 0.90rem;"><strong>Hidden State Tensor Shape:</strong> <code>({hs_seq_len}, {hs_dim})</code></p>
        </div>
        """, unsafe_allow_html=True)

        # Hidden State 2D PCA Projection
        st.markdown(f"###### 2D PCA of Hidden States ({selected_hs_layer})")
        fig_hs_pca = plot_pca_2d(
            tokens=tokens,
            token_ids=token_ids,
            vectors=hs_np,
            title=f"2D PCA Projection of Hidden States ({selected_hs_layer})",
            marker_color="#10b981"
        )
        st.plotly_chart(
            fig_hs_pca,
            use_container_width=True,
            key=f"hs_tab_pca_chart_{section_key_prefix}"
        )

        with st.expander(f"🔍 View Hidden State Vectors ({selected_hs_layer})", expanded=False):
            for i, (tok, tid) in enumerate(zip(tokens, token_ids)):
                st.markdown(f"**Token:** `{tok}` | **Token ID:** `{tid}` | **Shape:** `({hs_dim},)`")
                st.code(str(hs_np[i].tolist()), language="json")

    # =========================================================================
    # TAB 6: LOGITS & TOKEN PROBABILITIES (Phase 2 - Step 15)
    # =========================================================================
    with tab_logits:
        st.markdown("##### ⚡ Next-Token Logits & Probability Distribution")
        st.markdown("""
        <div class="arch-flow" style="text-align: center; line-height: 1.6; margin-bottom: 12px;">
            <strong>Final Hidden State</strong> → <strong>LM Head</strong> → <strong>Logits</strong> → <strong>Softmax</strong> → <strong>Probability Distribution</strong> → <strong>Top-K Predictions</strong> → <strong>Token Probability View</strong>
        </div>
        """, unsafe_allow_html=True)

        if outputs.logits is not None:
            v_size = getattr(model.config, "vocab_size", outputs.logits.shape[-1])

            # --- 1. Generation Step Selector & Top-K Selector ---
            col_step_ctrl, col_topk_ctrl = st.columns([3, 2], vertical_alignment="center")

            if total_gen_tokens > 0:
                step_options = list(range(1, total_gen_tokens + 1))

                def format_step_choice(s_idx: int) -> str:
                    t_id = resp_token_ids[s_idx - 1]
                    t_text = tokenizer.decode([t_id])
                    clean_repr = t_text if t_text.strip() else f"␣ ({repr(t_text)})"
                    return f"Step {s_idx} / {total_gen_tokens} ➔ Token: {repr(clean_repr)} (ID: {t_id})"

                with col_step_ctrl:
                    selected_step = st.selectbox(
                        "Generation Step",
                        options=step_options,
                        index=0,
                        format_func=format_step_choice,
                        key=f"gen_step_select_{section_key_prefix}",
                        help="Select which token generation step to inspect across the autoregressive sequence (1 to N)."
                    )
            else:
                with col_step_ctrl:
                    selected_step = st.selectbox(
                        "Generation Step",
                        options=[1],
                        index=0,
                        format_func=lambda s: "Step 1 (Prompt Completion)",
                        key=f"gen_step_select_{section_key_prefix}",
                        disabled=True,
                        help="Only prompt text is available. Inspecting the next-token prediction immediately following the prompt."
                    )

            with col_topk_ctrl:
                top_k_count = st.radio(
                    "Top-K Candidates",
                    options=[5, 10, 20],
                    index=1,
                    horizontal=True,
                    key=f"topk_selector_{section_key_prefix}",
                    help="Select number of top predicted candidate tokens (Top 5, Top 10, or Top 20)."
                )

            # --- 2. Step Logits & Softmax Probability Calculation ---
            if total_gen_tokens > 0:
                safe_step = max(1, min(total_gen_tokens, int(selected_step)))
                logit_idx = seq_len - 1 + (safe_step - 1)
                actual_tok_id = resp_token_ids[safe_step - 1]
                actual_tok_raw = tokenizer.decode([actual_tok_id])
                actual_tok_display = actual_tok_raw if actual_tok_raw.strip() else f"␣ ({repr(actual_tok_raw)})"
            else:
                safe_step = 1
                logit_idx = seq_len - 1
                actual_tok_id = None
                actual_tok_raw = None
                actual_tok_display = None

            # Extract step logits at the position immediately preceding the generated token
            step_logits = outputs.logits[0, logit_idx, :].to(torch.float32)

            # Numerically stable softmax
            probabilities = torch.softmax(step_logits, dim=-1)
            prob_sum = float(torch.sum(probabilities).item())

            # Top-K candidate tokens and probabilities
            top_probs, top_indices = torch.topk(probabilities, k=int(top_k_count))
            top_logits_vals = step_logits[top_indices]

            # Actual token rank and probability across full vocabulary
            actual_rank = None
            actual_prob_val = None
            if actual_tok_id is not None:
                actual_prob_val = float(probabilities[actual_tok_id].item())
                actual_rank = int((step_logits > step_logits[actual_tok_id]).sum().item()) + 1

            # --- 3. Distribution Metrics Card ---
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Inspected Step", f"Step {safe_step}" + (f" of {total_gen_tokens}" if total_gen_tokens > 0 else ""))
            with col_m2:
                st.metric("Vocabulary Size", f"{v_size:,} tokens")
            with col_m3:
                st.metric("Softmax Prob Sum", f"{prob_sum:.4f}")

            # --- 4. Actual Generated Token Card ---
            if actual_tok_id is not None:
                rank_color = "#166534" if actual_rank == 1 else "#854d0e" if actual_rank <= 5 else "#991b1b"
                rank_bg = "#dcfce7" if actual_rank == 1 else "#fef9c3" if actual_rank <= 5 else "#fee2e2"
                rank_badge = f'<span style="background:{rank_bg}; color:{rank_color}; padding:2px 8px; border-radius:6px; font-weight:700; font-size:0.80rem;">Vocabulary Rank #{actual_rank}</span>'
                prob_badge = f'<span style="background:#e0f2fe; color:#0369a1; padding:2px 8px; border-radius:6px; font-weight:700; font-size:0.80rem;">Prob: {actual_prob_val:.4f} ({actual_prob_val*100:.2f}%)</span>'
                id_badge = f'<span style="background:#f1f5f9; color:#334155; padding:2px 8px; border-radius:6px; font-family:monospace; font-size:0.80rem;">Token ID: {actual_tok_id}</span>'

                st.markdown(f"""
                <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-left: 4px solid #2563eb; border-radius: 8px; padding: 10px 14px; margin-top: 6px; margin-bottom: 14px;">
                    <div style="font-size: 0.74rem; font-weight: 700; text-transform: uppercase; color: #64748b; margin-bottom: 4px;">
                        🎯 Actual Generated Token (Step {safe_step})
                    </div>
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                        <div style="font-size: 1.1rem; font-weight: 700; color: #0f172a; font-family: monospace;">
                            "{html.escape(actual_tok_display)}"
                        </div>
                        <div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
                            {id_badge}
                            {prob_badge}
                            {rank_badge}
                        </div>
                    </div>
                    <div style="font-size: 0.75rem; color: #64748b; margin-top: 6px; line-height: 1.4;">
                        <em>Note: The generated token is sampled based on Temperature, Top-K, and Top-P controls. It may differ from the highest-probability token (Rank #1).</em>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-left: 4px solid #94a3b8; border-radius: 8px; padding: 10px 14px; margin-top: 6px; margin-bottom: 14px;">
                    <div style="font-size: 0.74rem; font-weight: 700; text-transform: uppercase; color: #64748b; margin-bottom: 4px;">
                        🎯 Next Token Prediction (Initial Prompt)
                    </div>
                    <div style="font-size: 0.88rem; color: #334155;">
                        <strong>Actual Generated Token:</strong> <em>N/A (No response generated yet)</em>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # --- 5. Next Token Prediction Table & Charts ---
            table_rows = []
            chart_tokens = []
            chart_probs = []
            is_actual_list = []
            raw_tokens_list = []
            raw_ids_list = []
            raw_probs_list = []

            for rank_idx, (p_val, idx_val, logit_val) in enumerate(zip(top_probs, top_indices, top_logits_vals), start=1):
                t_id = idx_val.item()
                decoded_tok = tokenizer.decode([t_id])
                p_float = p_val.item()
                l_float = logit_val.item()

                is_actual = (actual_tok_id is not None and t_id == actual_tok_id)
                is_actual_list.append(is_actual)

                clean_disp = decoded_tok if decoded_tok.strip() else f"␣ ({repr(decoded_tok)})"
                tok_col_text = f'"{clean_disp}"' + (" ★ [Actual]" if is_actual else "")

                table_rows.append({
                    "Rank": rank_idx,
                    "Candidate Token": tok_col_text,
                    "Token ID": t_id,
                    "Logit": l_float,
                    "Probability": float(f"{p_float:.4f}"),
                    "Percentage": f"{p_float * 100:.2f}%"
                })

                clean_tok_label = decoded_tok.replace("\n", "\\n").replace("\t", "\\t")
                if not clean_tok_label.strip():
                    clean_tok_label = f"[{repr(decoded_tok)}]"

                chart_tokens.append(f'"{clean_tok_label}" (ID: {t_id})' + (" ★" if is_actual else ""))
                chart_probs.append(p_float * 100)

                raw_tokens_list.append(decoded_tok)
                raw_ids_list.append(t_id)
                raw_probs_list.append(p_float)

            top_df = pd.DataFrame(table_rows)

            st.markdown(f"###### 📋 Top {top_k_count} Next-Token Predictions (Step {safe_step})")
            st.dataframe(
                top_df.style.format({
                    "Logit": "{:.4f}",
                    "Probability": "{:.4f}"
                }),
                use_container_width=True,
                hide_index=True
            )

            # --- 6. Probability Bars Visualization ---
            st.markdown(f"###### 📊 Probability Distribution Bars (Top {top_k_count})")

            # Interactive Plotly horizontal bar chart
            fig_bar = plot_top_k_probabilities(
                tokens=chart_tokens,
                probabilities=chart_probs,
                title=f"Top {top_k_count} Probabilities — Step {safe_step}" + (f" (Actual: '{actual_tok_display}')" if actual_tok_id is not None else ""),
                is_actual_flags=is_actual_list
            )
            st.plotly_chart(
                fig_bar,
                use_container_width=True,
                key=f"logits_tab_bar_chart_{section_key_prefix}_{safe_step}_{top_k_count}"
            )

            # Sleek CSS horizontal progress bars
            bars_html = render_probability_bars_html(
                tokens=raw_tokens_list,
                token_ids=raw_ids_list,
                probabilities=raw_probs_list,
                actual_token_id=actual_tok_id
            )
            st.markdown(bars_html, unsafe_allow_html=True)

        # --- 7. Autoregressive Token Generation Timeline ---
        if response_text and response_text.strip() and total_gen_tokens > 0:
            st.divider()
            st.markdown("##### ⚡ Token Generation Timeline")

            st.markdown(f"""
            <div class="layer-card">
                <h5 style="margin-top: 0; color: #4f46e5;">Generation Summary</h5>
                <p style="margin-bottom: 4px; font-size: 0.90rem;"><strong>Generated Tokens:</strong> {total_gen_tokens} tokens</p>
                <p style="margin-bottom: 0; font-size: 0.90rem;"><strong>Response Length:</strong> {len(response_text)} characters</p>
            </div>
            """, unsafe_allow_html=True)

            gen_steps = []
            for step_i in range(total_gen_tokens):
                t_id = resp_token_ids[step_i]
                t_tok = resp_tokens[step_i] if step_i < len(resp_tokens) else ""
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
            st.dataframe(gen_df, use_container_width=True, hide_index=True)
