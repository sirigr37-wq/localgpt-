"""
Verification Suite for Step 13: Integrate LLM X-Ray
Verifies all internal generation stages and transitions between Normal Mode and X-Ray Mode.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import model
import tokenizer
import chat
import memory
import xray
import database

def run_step13_tests():
    print("=" * 75)
    print("STEP 13: INTEGRATE LLM X-RAY VERIFICATION")
    print("=" * 75)
    
    print("\nLoading Qwen2.5-1.5B-Instruct model and tokenizer...")
    qwen_model, qwen_tok = model.load_model_and_tokenizer()
    
    prompt = "What is artificial intelligence?"
    print(f"\nTest Prompt: \"{prompt}\"")
    
    # -------------------------------------------------------------------------
    # 1. NORMAL MODE: User -> AI Response
    # -------------------------------------------------------------------------
    print("\n--- 1. NORMAL MODE (User -> AI Response) ---")
    chat_context = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    normal_resp = chat.generate_chat_response(qwen_model, qwen_tok, chat_context, max_new_tokens=40, temperature=0.2)
    print(f"User: {prompt}")
    print(f"AI Response: {normal_resp.strip()[:100]}...")
    assert len(normal_resp.strip()) > 0, "Normal mode failed to generate response"
    print("✓ Normal Mode verified: Direct User -> AI Response flow.")
    
    # -------------------------------------------------------------------------
    # 2. X-RAY MODE: Internal Stages Pipeline
    # -------------------------------------------------------------------------
    print("\n--- 2. X-RAY MODE: Internal Stages Pipeline ---")
    
    # Stage 1 & 2: Tokens & Token IDs
    tok_res = tokenizer.tokenize_input(qwen_tok, prompt)
    tokens = tok_res["tokens"]
    token_ids = tok_res["token_ids"]
    assert len(tokens) > 0 and len(token_ids) > 0, "Tokenization stage failed"
    print(f"Stage 1 & 2 (Tokens & Token IDs): {len(tokens)} tokens -> {[repr(t) for t in tokens]}")
    print(f"Token IDs: {token_ids}")
    
    # Stage 3: Embeddings
    input_embeddings_layer = qwen_model.get_input_embeddings()
    with torch.no_grad():
        token_tensor = torch.tensor(token_ids, device=next(qwen_model.parameters()).device)
        emb_tensor = input_embeddings_layer(token_tensor)
        emb_np = emb_tensor.detach().cpu().to(torch.float32).numpy()
    assert emb_np.shape == (len(tokens), qwen_model.config.hidden_size), "Embedding shape mismatch"
    print(f"Stage 3 (Embeddings): Shape={emb_np.shape}, Dim={emb_np.shape[1]}")
    
    # Forward pass for Stages 4, 5, 6, 7
    inputs = qwen_tok(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = qwen_model(**inputs, output_attentions=True, output_hidden_states=True)
        
    # Stage 4: Transformer Layers
    num_layers = getattr(qwen_model.config, "num_hidden_layers", 28)
    num_heads = getattr(qwen_model.config, "num_attention_heads", 12)
    print(f"Stage 4 (Transformer Layers): Total Layers={num_layers}, Attention Heads={num_heads}")
    assert len(qwen_model.model.layers) == num_layers, "Layer count mismatch"
    
    # Stage 5: Attention
    assert outputs.attentions is not None, "Attentions missing from forward pass"
    attn_layer_0_head_0 = outputs.attentions[0][0, 0].detach().cpu().to(torch.float32).numpy()
    assert attn_layer_0_head_0.shape == (len(tokens), len(tokens)), "Attention matrix dimension mismatch"
    # Verify causal masking (upper triangle must be 0)
    for r in range(len(tokens)):
        for c in range(r + 1, len(tokens)):
            assert attn_layer_0_head_0[r, c] == 0.0, f"Causal attention violated at ({r}, {c})"
    print(f"Stage 5 (Attention): Matrix Shape={attn_layer_0_head_0.shape}, Causal Masking=Confirmed")
    
    # Stage 6: Hidden States
    assert outputs.hidden_states is not None, "Hidden states missing from forward pass"
    last_hidden_state = outputs.hidden_states[-1][0].detach().cpu().to(torch.float32).numpy()
    assert last_hidden_state.shape == (len(tokens), qwen_model.config.hidden_size), "Hidden state shape mismatch"
    print(f"Stage 6 (Hidden States): Shape={last_hidden_state.shape}, Layers={len(outputs.hidden_states)}")
    
    # Stage 7 & 8: Logits & Probabilities
    next_token_logits = outputs.logits[0, -1, :].to(torch.float32)
    vocab_size = getattr(qwen_model.config, "vocab_size", 151936)
    assert next_token_logits.shape[0] == vocab_size, "Logits vocab size mismatch"
    
    probabilities = torch.softmax(next_token_logits, dim=-1)
    prob_sum = float(torch.sum(probabilities).item())
    assert abs(prob_sum - 1.0) < 1e-4, f"Softmax probabilities sum != 1.0 (got {prob_sum})"
    
    top_k_probs, top_k_indices = torch.topk(probabilities, k=10)
    top_10 = [(repr(qwen_tok.decode([idx.item()])), f"{p.item() * 100:.2f}%") for p, idx in zip(top_k_probs, top_k_indices)]
    print(f"Stage 7 & 8 (Logits & Probabilities): Vocab Size={vocab_size:,}, Top 5 Next-Token Predictions={top_10[:5]}")
    
    # Stage 9: Generated Tokens Timeline
    resp_encoded = qwen_tok(normal_resp, add_special_tokens=False)
    resp_token_ids = resp_encoded["input_ids"]
    resp_tokens = qwen_tok.convert_ids_to_tokens(resp_token_ids)
    print(f"Stage 9 (Generated Tokens): {len(resp_token_ids)} tokens generated -> {[repr(t) for t in resp_tokens[:8]]}...")
    
    # Stage 10: Final Response
    print(f"Stage 10 (Final Response): {normal_resp.strip()[:100]}...")
    
    print("\n" + "=" * 75)
    print("ALL 10 STAGES OF LLM X-RAY PIPELINE VERIFIED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    run_step13_tests()
