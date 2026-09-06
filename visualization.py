"""
Visualization helper module for LLM X-Ray.
Provides Plotly chart generators and HTML visual components.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA


def plot_pca_2d(
    tokens: List[str],
    token_ids: List[int],
    vectors: np.ndarray,
    title: str = "2D PCA Projection",
    marker_color: str = "#4f46e5"
):
    """
    Generate an interactive 2D PCA scatter plot of high-dimensional vectors.
    """
    if len(tokens) >= 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(vectors)
        explained_var = pca.explained_variance_ratio_
        var_text = f" (Explains {explained_var[0]*100:.1f}% + {explained_var[1]*100:.1f}% var)"
    else:
        coords = np.array([[0.0, 0.0]])
        var_text = ""

    df = pd.DataFrame({
        "Token": tokens,
        "Token ID": token_ids,
        "PCA Dim 1": coords[:, 0],
        "PCA Dim 2": coords[:, 1],
    })

    fig = px.scatter(
        df,
        x="PCA Dim 1",
        y="PCA Dim 2",
        text="Token",
        hover_data={"Token": True, "Token ID": True, "PCA Dim 1": ":.4f", "PCA Dim 2": ":.4f"},
        title=f"{title}{var_text}"
    )
    fig.update_traces(
        textposition="top center",
        marker=dict(size=14, color=marker_color, line=dict(width=2, color="#1e1b4b"))
    )
    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
    )
    return fig


def plot_attention_heatmap(
    attn_matrix: np.ndarray,
    display_tokens: List[str],
    title: str = "Attention Heatmap",
    colorscale: str = "Blues"
):
    """
    Generate an interactive Plotly heatmap for token-to-token attention weights.
    """
    seq_len = len(display_tokens)
    fig = px.imshow(
        attn_matrix,
        x=display_tokens,
        y=display_tokens,
        labels=dict(x="Attended-to Token (Key)", y="Attending Token (Query)", color="Attention Weight"),
        color_continuous_scale=colorscale,
        range_color=[0.0, 1.0],
        text_auto=".2f" if seq_len <= 12 else False,
        title=title
    )
    fig.update_traces(
        hoverongaps=False,
        hovertemplate="Attending Token (Query): %{y}<br>Attended Token (Key): %{x}<br>Attention Weight: %{z:.4f}<extra></extra>"
    )
    fig.update_layout(
        template="plotly_white",
        height=max(400, min(700, 200 + seq_len * 24)),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(side="bottom", tickangle=-45 if seq_len > 8 else 0),
        yaxis=dict(autorange="reversed")
    )
    return fig


def plot_top_k_probabilities(
    tokens: List[str],
    probabilities: List[float],
    title: str = "Top Next-Token Probabilities",
    is_actual_flags: Optional[List[bool]] = None
):
    """
    Generate a horizontal bar chart displaying top next-token probabilities.
    Optionally highlights the actual generated token.
    """
    probs_pct = [p * 100 if p <= 1.0 else p for p in probabilities]

    df = pd.DataFrame({
        "Token": tokens,
        "Probability (%)": probs_pct
    })

    if is_actual_flags and len(is_actual_flags) == len(tokens):
        df["Type"] = ["Actual Generated" if flag else "Candidate" for flag in is_actual_flags]
        color_map = {"Actual Generated": "#10b981", "Candidate": "#3b82f6"}
        fig = px.bar(
            df,
            x="Probability (%)",
            y="Token",
            orientation="h",
            text="Probability (%)",
            title=title,
            color="Type",
            color_discrete_map=color_map
        )
    else:
        fig = px.bar(
            df,
            x="Probability (%)",
            y="Token",
            orientation="h",
            text="Probability (%)",
            title=title,
            color="Probability (%)",
            color_continuous_scale="Blues"
        )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        marker=dict(line=dict(width=1, color="#1e40af"))
    )

    max_p = max(probs_pct) if probs_pct else 100
    x_max = max(10.0, min(100.0, max_p * 1.25))
    calculated_height = max(260, min(650, 60 + len(tokens) * 26))

    fig.update_layout(
        template="plotly_white",
        height=calculated_height,
        margin=dict(l=20, r=50, t=40, b=20),
        xaxis_title="Probability (%)",
        yaxis_title="Candidate Token",
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[0, x_max]),
        showlegend=bool(is_actual_flags and any(is_actual_flags))
    )
    return fig


def render_probability_bars_html(
    tokens: List[str],
    token_ids: List[int],
    probabilities: List[float],
    actual_token_id: Optional[int] = None
) -> str:
    """
    Render sleek horizontal probability progress bars in pure CSS/HTML.
    """
    import html as py_html
    rows = []

    for i, (tok, tid, prob) in enumerate(zip(tokens, token_ids, probabilities), start=1):
        p_pct = prob * 100 if prob <= 1.0 else prob
        p_val = prob if prob <= 1.0 else prob / 100.0
        bar_width = max(2.0, min(100.0, p_pct))

        is_actual = (actual_token_id is not None and tid == actual_token_id)
        clean_tok = tok if tok.strip() else f"␣ ({repr(tok)})"
        esc_tok = py_html.escape(clean_tok)

        bg_bar = "linear-gradient(90deg, #10b981, #059669)" if is_actual else "linear-gradient(90deg, #3b82f6, #2563eb)"
        badge = '<span style="background:#10b981; color:#ffffff; font-size:0.68rem; font-weight:700; padding:1px 5px; border-radius:4px; margin-left:4px;">★ ACTUAL</span>' if is_actual else ""
        border_style = "border: 1px solid #10b981; background: #f0fdf4;" if is_actual else "border: 1px solid #e2e8f0; background: #ffffff;"

        rows.append(f"""
        <div style="margin-bottom: 6px; padding: 6px 10px; border-radius: 6px; {border_style}">
            <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; margin-bottom: 3px;">
                <div style="font-family: monospace; font-weight: 600; color: #0f172a;">
                    <span style="color: #64748b; font-size: 0.76rem; margin-right: 4px;">#{i}</span>
                    "{esc_tok}"
                    <span style="color: #94a3b8; font-size: 0.74rem;">(ID: {tid})</span>
                    {badge}
                </div>
                <div style="font-weight: 700; color: {'#065f46' if is_actual else '#1e40af'}; font-size: 0.82rem;">
                    {p_pct:.2f}% <span style="font-size: 0.72rem; color: #64748b; font-weight: normal;">({p_val:.4f})</span>
                </div>
            </div>
            <div style="background: #e2e8f0; border-radius: 4px; height: 10px; width: 100%; overflow: hidden;">
                <div style="background: {bg_bar}; height: 100%; width: {bar_width}%; border-radius: 4px; transition: width 0.3s ease;"></div>
            </div>
        </div>
        """)

    return f"""
    <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px; margin-top: 8px; margin-bottom: 12px;">
        <div style="font-size: 0.76rem; font-weight: 700; text-transform: uppercase; color: #64748b; margin-bottom: 8px;">
            Horizontal Probability Bars (Real Model Distribution)
        </div>
        {"".join(rows)}
    </div>
    """


def render_pipeline_banner_html() -> str:
    """
    Render visual representation of the 9-stage Expected Output inference flow:
    Prompt -> Tokenization -> Token IDs -> Embeddings -> Transformer Layers ->
    Attention -> Hidden States -> Logits -> Probability Distribution -> Next-Token Prediction -> Final LLM Response
    """
    stages = [
        ("Prompt", "#3b82f6"),
        ("Tokenization", "#6366f1"),
        ("Token IDs", "#8b5cf6"),
        ("Embeddings", "#ec4899"),
        ("Transformer Layers", "#f43f5e"),
        ("Attention", "#0ea5e9"),
        ("Hidden States", "#10b981"),
        ("Logits", "#f59e0b"),
        ("Probabilities", "#eab308"),
        ("Next-Token", "#84cc16"),
        ("Response", "#22c55e"),
    ]

    chips = []
    for idx, (name, color) in enumerate(stages):
        chip = (
            f'<div style="display:inline-flex; align-items:center; background:#ffffff; '
            f'border:1px solid #e2e8f0; border-left:3px solid {color}; border-radius:6px; '
            f'padding:4px 8px; font-size:0.75rem; font-weight:600; color:#1e293b; box-shadow:0 1px 2px rgba(0,0,0,0.03);">'
            f'{name}'
            f'</div>'
        )
        chips.append(chip)
        if idx < len(stages) - 1:
            chips.append('<span style="color:#94a3b8; font-size:0.75rem; font-weight:700; margin:0 2px;">→</span>')

    return f"""
    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:10px 14px; margin-bottom:16px;">
        <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#64748b; margin-bottom:6px;">
            Inference Pipeline Architecture
        </div>
        <div style="display:flex; flex-wrap:wrap; align-items:center; gap:4px;">
            {"".join(chips)}
        </div>
    </div>
    """


def render_system_architecture_html() -> str:
    """
    Render a comprehensive, visual, lightweight system architecture diagram
    for LocalGPT, detailing Chat Interface, Conversation Memory, Context Manager,
    RAG Pipeline, LLM Generation, and LLM X-Ray Inspection Pipeline.
    """
    return """
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #1e293b; line-height: 1.5; margin: 4px 0 16px 0;">
        
        <!-- Top Overview Banner -->
        <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); color: #ffffff; border-radius: 10px; padding: 14px 18px; margin-bottom: 16px; box-shadow: 0 2px 4px rgba(30, 58, 138, 0.12);">
            <div style="font-size: 1.15rem; font-weight: 700; letter-spacing: -0.01em; display: flex; align-items: center; gap: 8px;">
                <span>🏗️</span> LocalGPT System Architecture & Execution Flow
            </div>
            <div style="font-size: 0.82rem; opacity: 0.92; margin-top: 4px;">
                End-to-end blueprint connecting Local Chat, Dual Memory, FAISS RAG Pipeline, Qwen2.5-1.5B Generation, and Real-Time LLM X-Ray Inspection.
            </div>
        </div>

        <!-- Section 1: Main Conversational & RAG Flow -->
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
            <div style="font-size: 0.88rem; font-weight: 700; color: #0f172a; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 12px; display: flex; align-items: center; gap: 6px;">
                <span style="background: #eff6ff; color: #2563eb; padding: 2px 8px; border-radius: 4px; font-size: 0.76rem;">PIPELINE 1</span>
                Conversational Chat & RAG Execution Flow
            </div>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; margin-bottom: 14px;">
                <!-- 1. Chat Interface -->
                <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-top: 3px solid #2563eb; border-radius: 8px; padding: 10px 12px;">
                    <div style="font-weight: 700; font-size: 0.84rem; color: #1e40af;">1. Chat Interface</div>
                    <div style="font-size: 0.76rem; color: #475569; margin-top: 4px;">
                        • User Prompt & System Instructions<br>
                        • Live typing streamer & TTS voice<br>
                        • Action Toolbar (Copy, Edit, Regen)
                    </div>
                </div>

                <!-- 2. Dual-Layer Memory -->
                <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-top: 3px solid #8b5cf6; border-radius: 8px; padding: 10px 12px;">
                    <div style="font-weight: 700; font-size: 0.84rem; color: #6d28d9;">2. Dual Memory System</div>
                    <div style="font-size: 0.76rem; color: #475569; margin-top: 4px;">
                        • <strong>Short-Term:</strong> Active Turn Memory<br>
                        • <strong>Long-Term:</strong> SQLite DB Persistence<br>
                        • Strict Multi-Chat Isolation
                    </div>
                </div>

                <!-- 3. Context Manager -->
                <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-top: 3px solid #06b6d4; border-radius: 8px; padding: 10px 12px;">
                    <div style="font-weight: 700; font-size: 0.84rem; color: #0891b2;">3. Context Manager</div>
                    <div style="font-size: 0.76rem; color: #475569; margin-top: 4px;">
                        • Multi-turn Qwen Chat Template<br>
                        • Dynamic System Prompt injection<br>
                        • RAG vs Normal Route Decision
                    </div>
                </div>

                <!-- 4. RAG Knowledge Pipeline -->
                <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-top: 3px solid #10b981; border-radius: 8px; padding: 10px 12px;">
                    <div style="font-weight: 700; font-size: 0.84rem; color: #065f46;">4. FAISS RAG Pipeline</div>
                    <div style="font-size: 0.76rem; color: #475569; margin-top: 4px;">
                        • PDF / DOCX / TXT text extraction<br>
                        • Chunking & 384-d Embeddings<br>
                        • FAISS IndexFlatIP Cosine Search<br>
                        • 180-word Context Excerpt Grounding
                    </div>
                </div>

                <!-- 5. LLM Generation Engine -->
                <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-top: 3px solid #f59e0b; border-radius: 8px; padding: 10px 12px;">
                    <div style="font-weight: 700; font-size: 0.84rem; color: #b45309;">5. LLM Generation</div>
                    <div style="font-size: 0.76rem; color: #475569; margin-top: 4px;">
                        • Local Qwen2.5-1.5B-Instruct<br>
                        • <code>torch.inference_mode()</code><br>
                        • <code>TextIteratorStreamer</code> background<br>
                        • Low latency real-time streaming
                    </div>
                </div>
            </div>

            <!-- Flow summary bar -->
            <div style="background: #f1f5f9; border-radius: 6px; padding: 8px 12px; font-family: monospace; font-size: 0.74rem; color: #334155; overflow-x: auto; white-space: nowrap;">
                <strong>Normal Flow:</strong> User ➔ Memory ➔ Context Manager ➔ Qwen2.5-1.5B ➔ TextIteratorStreamer ➔ Streaming Response ➔ SQLite Persistence<br>
                <strong>RAG Flow:</strong> User ➔ FAISS Similarity Search ➔ Top-K Excerpts (180w) ➔ Grounded Prompt ➔ Qwen2.5-1.5B ➔ Streaming Response + Source Citations
            </div>
        </div>

        <!-- Section 2: LLM X-Ray Inspection Pipeline -->
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
            <div style="font-size: 0.88rem; font-weight: 700; color: #0f172a; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 12px; display: flex; align-items: center; gap: 6px;">
                <span style="background: #fdf2f8; color: #db2777; padding: 2px 8px; border-radius: 4px; font-size: 0.76rem;">PIPELINE 2</span>
                LLM X-Ray Deep Inspection Pipeline (Internal Mechanism)
            </div>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 8px; margin-bottom: 12px;">
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 3px solid #6366f1; border-radius: 6px; padding: 8px 10px;">
                    <div style="font-weight: 700; font-size: 0.78rem; color: #4338ca;">1. BPE Tokens & IDs</div>
                    <div style="font-size: 0.72rem; color: #64748b; margin-top: 2px;">Byte-level BPE split & 151k vocab IDs</div>
                </div>
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 3px solid #ec4899; border-radius: 6px; padding: 8px 10px;">
                    <div style="font-weight: 700; font-size: 0.78rem; color: #be185d;">2. Input Embeddings</div>
                    <div style="font-size: 0.72rem; color: #64748b; margin-top: 2px;">1536-dim vectors & 2D PCA projection</div>
                </div>
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 3px solid #f43f5e; border-radius: 6px; padding: 8px 10px;">
                    <div style="font-weight: 700; font-size: 0.78rem; color: #be123c;">3. 28 Transformer Layers</div>
                    <div style="font-size: 0.72rem; color: #64748b; margin-top: 2px;">Self-Attn (12 heads) + MLP (8960-dim)</div>
                </div>
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 3px solid #0ea5e9; border-radius: 6px; padding: 8px 10px;">
                    <div style="font-weight: 700; font-size: 0.78rem; color: #0369a1;">4. Attention Heatmaps</div>
                    <div style="font-size: 0.72rem; color: #64748b; margin-top: 2px;">Query/Key weights & Causal masking</div>
                </div>
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 3px solid #10b981; border-radius: 6px; padding: 8px 10px;">
                    <div style="font-weight: 700; font-size: 0.78rem; color: #047857;">5. Hidden States</div>
                    <div style="font-size: 0.72rem; color: #64748b; margin-top: 2px;">29 layer representations [seq, 1536]</div>
                </div>
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 3px solid #f59e0b; border-radius: 6px; padding: 8px 10px;">
                    <div style="font-weight: 700; font-size: 0.78rem; color: #b45309;">6. Logits & Softmax</div>
                    <div style="font-size: 0.72rem; color: #64748b; margin-top: 2px;">151,936 vocab distribution & Top-K</div>
                </div>
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 3px solid #84cc16; border-radius: 6px; padding: 8px 10px;">
                    <div style="font-weight: 700; font-size: 0.78rem; color: #4d7c0f;">7. Actual Token Tracking</div>
                    <div style="font-size: 0.72rem; color: #64748b; margin-top: 2px;">Generated token vs sampled candidates</div>
                </div>
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 3px solid #22c55e; border-radius: 6px; padding: 8px 10px;">
                    <div style="font-weight: 700; font-size: 0.78rem; color: #15803d;">8. Generation Timeline</div>
                    <div style="font-size: 0.72rem; color: #64748b; margin-top: 2px;">Multi-step progression & final response</div>
                </div>
            </div>

            <!-- X-Ray summary bar -->
            <div style="background: #fdf4ff; border: 1px solid #f0abfc; border-radius: 6px; padding: 8px 12px; font-family: monospace; font-size: 0.74rem; color: #701a75; overflow-x: auto; white-space: nowrap;">
                <strong>X-Ray Pipeline:</strong> Prompt ➔ Tokens ➔ Token IDs ➔ Embeddings ➔ Layers (1–28) ➔ Attention ➔ Hidden States ➔ Logits ➔ Probabilities ➔ Generated Tokens ➔ Final Answer
            </div>
        </div>

        <!-- Architectural Specifications Table -->
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; font-size: 0.76rem; color: #475569;">
            <div style="font-weight: 700; color: #0f172a; margin-bottom: 4px;">⚡ System Architecture Guarantees & Constraints</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 6px;">
                <div>• <strong>Model Core:</strong> Qwen2.5-1.5B-Instruct (Untouched <code>model.py</code>)</div>
                <div>• <strong>Embeddings:</strong> all-MiniLM-L6-v2 (384 Dimensions)</div>
                <div>• <strong>Vector Index:</strong> FAISS FlatIP (L2 Normalized Cosine Similarity)</div>
                <div>• <strong>Context Limit:</strong> 180-Word Maximum RAG Excerpts</div>
                <div>• <strong>Inference Engine:</strong> <code>torch.inference_mode()</code> + <code>TextIteratorStreamer</code></div>
                <div>• <strong>Persistence:</strong> SQLite (<code>conversations</code> & <code>messages</code> schema)</div>
            </div>
        </div>

    </div>
    """
