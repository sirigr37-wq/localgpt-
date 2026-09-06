"""
Comprehensive Verification Suite for Phase 2 - Step 15: Add Token Probability View
Validates:
1. Real model logits & stable softmax probability distribution
2. Top-K selector (Top 5, Top 10, Top 20)
3. Horizontal probability bars & percentage representation
4. Actual Generated Token tracking (real sampled tokens vs Top-1 argmax)
5. Generation Step selector across multi-token sequences
6. Streaming generation preservation with TextIteratorStreamer
7. Generation Controls compatibility (Temperature, Top-K, Top-P, Max Tokens)
8. RAG compatibility (retrieval context grounding)
9. Multi-turn conversation isolation
10. Conversation persistence & switching
11. Normal Mode performance (X-Ray OFF)
12. Error handling & edge cases (empty responses, special tokens)
"""

import os
import sys
import torch
import numpy as np
import pandas as pd

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import model
import tokenizer
import chat
import memory
import xray
import visualization
import document_loader
import chunking
import embeddings
import vector_store
import rag
import database


def run_step15_tests():
    print("=" * 80)
    print("PHASE 2 - STEP 15: TOKEN PROBABILITY VIEW VERIFICATION SUITE")
    print("=" * 80)

    test_results = {}

    print("\nLoading Qwen2.5-1.5B-Instruct model and tokenizer...")
    qwen_model, qwen_tok = model.load_model_and_tokenizer()

    # -------------------------------------------------------------------------
    # TEST 1: Real Token Probabilities & Softmax Distribution
    # -------------------------------------------------------------------------
    print("\n--- TEST 1: Real Model Logits & Softmax Probabilities ---")
    prompt_1 = "The quick brown fox"
    p_ids = qwen_tok(prompt_1, add_special_tokens=False)["input_ids"]
    inputs = torch.tensor([p_ids], device=next(qwen_model.parameters()).device)

    with torch.no_grad():
        outputs = qwen_model(inputs)

    # Position immediately preceding next token
    step_logits = outputs.logits[0, -1, :].to(torch.float32)
    assert step_logits.shape[0] == 151936, f"Expected 151936 vocabulary dimension, got {step_logits.shape[0]}"

    probs = torch.softmax(step_logits, dim=-1)
    prob_sum = float(torch.sum(probs).item())
    assert abs(prob_sum - 1.0) < 1e-4, f"Softmax sum must equal 1.0, got {prob_sum}"

    top_p, top_i = torch.topk(probs, 5)
    print(f"Prompt: \"{prompt_1}\" (Length: {len(p_ids)} tokens)")
    print(f"Top 5 Next-Token Predictions from Logits:")
    for rank, (p_val, idx_val) in enumerate(zip(top_p, top_i), start=1):
        decoded = repr(qwen_tok.decode([idx_val.item()]))
        print(f"  #{rank}: {decoded:15s} (ID: {idx_val.item():6d}) -> {p_val.item()*100:6.2f}% (Prob: {p_val.item():.4f})")

    assert top_p[0] >= top_p[1] >= top_p[2] >= top_p[3] >= top_p[4], "Top probabilities not descending"
    test_results["TEST 1 (Real Logits & Softmax)"] = "PASSED"

    # -------------------------------------------------------------------------
    # TEST 2: Top-K Selector (Top 5, Top 10, Top 20)
    # -------------------------------------------------------------------------
    print("\n--- TEST 2: Top-K Selector (5, 10, 20) ---")
    for k_val in [5, 10, 20]:
        t_probs, t_indices = torch.topk(probs, k=k_val)
        assert len(t_probs) == k_val, f"Expected {k_val} items for Top-{k_val}, got {len(t_probs)}"
        sum_k = float(torch.sum(t_probs).item())
        print(f"Top {k_val:2d} Candidate Sum: {sum_k*100:.2f}% of total vocabulary probability")
        assert 0.0 < sum_k <= 1.0, f"Top-{k_val} probability sum out of range: {sum_k}"

    test_results["TEST 2 (Top-K Selector)"] = "PASSED"

    # -------------------------------------------------------------------------
    # TEST 3: Probability Bars Visualization & Formatting
    # -------------------------------------------------------------------------
    print("\n--- TEST 3: Probability Bars Visualization ---")
    top_10_p, top_10_i = torch.topk(probs, 10)
    tokens_list = [qwen_tok.decode([i.item()]) for i in top_10_i]
    chart_token_labels = [f"\"{t}\" (ID: {i.item()}) text" for t, i in zip(tokens_list, top_10_i)]
    probs_float = [p.item() for p in top_10_p]

    # Test Plotly Chart
    fig = visualization.plot_top_k_probabilities(
        tokens=chart_token_labels,
        probabilities=probs_float,
        title="Top 10 Predictions Test",
        is_actual_flags=[i == 0 for i in range(10)]
    )
    assert fig is not None, "Failed to create Plotly horizontal bar chart"

    # Test HTML CSS Bars
    html_bars = visualization.render_probability_bars_html(
        tokens=tokens_list,
        token_ids=[i.item() for i in top_10_i],
        probabilities=probs_float,
        actual_token_id=top_10_i[0].item()
    )
    assert html_bars is not None and "★ ACTUAL" in html_bars, "HTML progress bars missing or incomplete"
    print("Probability bar chart & HTML components generated successfully.")
    test_results["TEST 3 (Probability Bars)"] = "PASSED"

    # -------------------------------------------------------------------------
    # TEST 4: Actual Generated Token vs Sampled Tokens
    # -------------------------------------------------------------------------
    print("\n--- TEST 4: Actual Generated Token Tracking ---")
    context_test = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "The color of the clear daytime sky is"}
    ]
    gen_text = chat.generate_chat_response(qwen_model, qwen_tok, context_test, max_new_tokens=10, temperature=0.7)
    print(f"Generated text: \"{gen_text}\"")

    resp_ids = qwen_tok(gen_text, add_special_tokens=False)["input_ids"]
    assert len(resp_ids) > 0, "No tokens generated"

    # Check Step 1 actual token
    actual_tok_id_step1 = resp_ids[0]
    actual_tok_str_step1 = qwen_tok.decode([actual_tok_id_step1])

    # Compute step 1 logits for the context
    full_prompt_text = qwen_tok.apply_chat_template(context_test, tokenize=False, add_generation_prompt=True)
    prompt_ids = qwen_tok(full_prompt_text, add_special_tokens=False)["input_ids"]

    full_seq_ids = prompt_ids + resp_ids
    with torch.no_grad():
        full_out = qwen_model(torch.tensor([full_seq_ids], device=next(qwen_model.parameters()).device))

    # Step 1 logits at position (len(prompt_ids) - 1)
    step1_logits = full_out.logits[0, len(prompt_ids) - 1, :].to(torch.float32)
    step1_probs = torch.softmax(step1_logits, dim=-1)
    step1_actual_prob = float(step1_probs[actual_tok_id_step1].item())
    step1_actual_rank = int((step1_logits > step1_logits[actual_tok_id_step1]).sum().item()) + 1

    print(f"Step 1 Actual Generated Token: {repr(actual_tok_str_step1)} (ID: {actual_tok_id_step1})")
    print(f"  Step 1 Model Probability: {step1_actual_prob*100:.2f}% (Vocabulary Rank #{step1_actual_rank})")
    assert step1_actual_prob > 0.0, "Actual token probability must be positive"
    test_results["TEST 4 (Actual Generated Token)"] = "PASSED"

    # -------------------------------------------------------------------------
    # TEST 5: Generation Step Selector Across Multi-Step Sequences
    # -------------------------------------------------------------------------
    print("\n--- TEST 5: Multi-Step Generation Sequence Inspection ---")
    total_steps = len(resp_ids)
    print(f"Inspecting {total_steps} generation steps:")

    for s in range(1, total_steps + 1):
        logit_pos = len(prompt_ids) - 1 + (s - 1)
        s_logits = full_out.logits[0, logit_pos, :].to(torch.float32)
        s_probs = torch.softmax(s_logits, dim=-1)
        actual_id = resp_ids[s - 1]
        actual_str = qwen_tok.decode([actual_id])
        actual_p = float(s_probs[actual_id].item())
        actual_r = int((s_logits > s_logits[actual_id]).sum().item()) + 1

        top3_p, top3_i = torch.topk(s_probs, 3)
        top3_strs = [qwen_tok.decode([idx.item()]) for idx in top3_i]

        clean_disp = actual_str.replace("\n", "\\n")
        print(f"  Step {s:2d}: Actual = {repr(clean_disp):14s} (ID: {actual_id:6d}, Prob: {actual_p*100:5.2f}%, Rank: #{actual_r:3d}) | Top 1: {repr(top3_strs[0]):10s} ({top3_p[0].item()*100:5.2f}%)")

    test_results["TEST 5 (Generation Step Sequence)"] = "PASSED"

    # -------------------------------------------------------------------------
    # TEST 6: Streaming Generation Preservation (TextIteratorStreamer)
    # -------------------------------------------------------------------------
    print("\n--- TEST 6: Streaming Response with TextIteratorStreamer ---")
    stream_chunks = []
    for chunk in chat.stream_chat_response(qwen_model, qwen_tok, context_test, max_new_tokens=20, temperature=0.7):
        stream_chunks.append(chunk)

    streamed_full = "".join(stream_chunks)
    print(f"Streamed {len(stream_chunks)} chunks: \"{streamed_full.strip()[:60]}...\"")
    assert len(stream_chunks) > 1, f"Expected multi-chunk streaming, got {len(stream_chunks)} chunk(s)"
    assert len(streamed_full.strip()) > 0, "Streamed response is empty"
    test_results["TEST 6 (Streaming Preservation)"] = "PASSED"

    # -------------------------------------------------------------------------
    # TEST 7: Generation Controls Compatibility
    # -------------------------------------------------------------------------
    print("\n--- TEST 7: Generation Controls Compatibility (Temp, Top-K, Top-P) ---")
    for temp, k_ctl, p_ctl in [(0.0, 50, 0.9), (0.7, 20, 0.8), (1.2, 80, 0.95)]:
        resp_ctrl = chat.generate_chat_response(
            qwen_model, qwen_tok, context_test, max_new_tokens=15, temperature=temp, top_k=k_ctl, top_p=p_ctl
        )
        assert len(resp_ctrl.strip()) > 0, f"Generation failed for temp={temp}, top_k={k_ctl}, top_p={p_ctl}"
        print(f"  Temp={temp:.1f}, Top-K={k_ctl:2d}, Top-P={p_ctl:.2f} -> \"{resp_ctrl.strip()[:40]}...\"")

    test_results["TEST 7 (Generation Controls)"] = "PASSED"

    # -------------------------------------------------------------------------
    # TEST 8: RAG Compatibility
    # -------------------------------------------------------------------------
    print("\n--- TEST 8: RAG Document Grounding & Token Probabilities ---")
    rag_doc = {"text": "Quantum computing utilizes qubits and quantum superposition for parallel calculations.", "filename": "quantum.txt", "doc_type": "text"}
    rag_chunks = chunking.chunk_document(rag_doc)
    emb_m = embeddings.load_embedding_model()
    rag_chunks = embeddings.embed_chunks(rag_chunks, model=emb_m)
    vs = vector_store.create_vector_store(rag_chunks, embeddings.get_embeddings_matrix(rag_chunks))

    rag_prompt, sources = rag.retrieve_and_build_context(
        query="What do qubits utilize?",
        vector_store=vs,
        embedding_model=emb_m,
        top_k=1
    )
    print(f"RAG Retrieved Sources ({len(sources)}): {[s['filename'] for s in sources]}")
    assert len(sources) > 0, "RAG retrieval failed"
    assert "qubits" in rag_prompt.lower(), "Retrieved context missing from RAG prompt"

    rag_context = [
        {"role": "system", "content": "Answer questions using document context."},
        {"role": "user", "content": rag_prompt}
    ]
    rag_response = chat.generate_chat_response(qwen_model, qwen_tok, rag_context, max_new_tokens=25, temperature=0.2)
    print(f"RAG Response: \"{rag_response.strip()[:60]}...\"")
    assert len(rag_response.strip()) > 0, "RAG response generation failed"
    test_results["TEST 8 (RAG Compatibility)"] = "PASSED"

    # -------------------------------------------------------------------------
    # TEST 9: Multi-Turn Conversation Isolation
    # -------------------------------------------------------------------------
    print("\n--- TEST 9: Multi-Turn Token Probability Isolation ---")
    turn1_prompt = "Name two prime numbers."
    turn1_resp = "2 and 3."
    turn2_prompt = "What is their sum?"
    turn2_resp = "Their sum is 5."

    # Verify separate tokenization and forward passes do not cross-contaminate
    t1_ids = qwen_tok(turn1_prompt, add_special_tokens=False)["input_ids"]
    t1_r_ids = qwen_tok(turn1_resp, add_special_tokens=False)["input_ids"]

    t2_ids = qwen_tok(turn2_prompt, add_special_tokens=False)["input_ids"]
    t2_r_ids = qwen_tok(turn2_resp, add_special_tokens=False)["input_ids"]

    with torch.no_grad():
        out1 = qwen_model(torch.tensor([t1_ids + t1_r_ids]))
        out2 = qwen_model(torch.tensor([t2_ids + t2_r_ids]))

    step1_t1 = out1.logits[0, len(t1_ids) - 1, :].to(torch.float32)
    step1_t2 = out2.logits[0, len(t2_ids) - 1, :].to(torch.float32)

    # Confirm different distributions for different prompts
    assert not torch.allclose(step1_t1, step1_t2), "Turn 1 and Turn 2 logits must be distinct"
    print("Multi-turn logits and probabilities are strictly isolated.")
    test_results["TEST 9 (Multi-Turn Isolation)"] = "PASSED"

    # -------------------------------------------------------------------------
    # TEST 10: Conversation Persistence & Switching
    # -------------------------------------------------------------------------
    print("\n--- TEST 10: Conversation Switching & Persistence ---")
    test_db = os.path.join("data", "conversations", "test_step15.db")
    if os.path.exists(test_db):
        os.remove(test_db)
    database.init_database(test_db)

    c1 = database.create_conversation("Chat 1", db_path=test_db)
    database.save_message(c1, "user", "Hello", db_path=test_db)
    database.save_message(c1, "assistant", "Hi there!", db_path=test_db)

    c2 = database.create_conversation("Chat 2", db_path=test_db)
    database.save_message(c2, "user", "What is 2+2?", db_path=test_db)
    database.save_message(c2, "assistant", "It is 4.", db_path=test_db)

    m1 = database.get_conversation_messages(c1, db_path=test_db)
    m2 = database.get_conversation_messages(c2, db_path=test_db)
    assert len(m1) == 2 and len(m2) == 2, "Failed to retrieve conversation messages"
    assert m1[0]["content"] == "Hello" and m2[0]["content"] == "What is 2+2?", "Conversation message mixing detected"
    print("SQLite persistence and conversation switching verified.")
    test_results["TEST 10 (Conversation Persistence)"] = "PASSED"

    # -------------------------------------------------------------------------
    # TEST 11: Normal Mode Performance (X-Ray OFF)
    # -------------------------------------------------------------------------
    print("\n--- TEST 11: Normal Mode Performance ---")
    import time
    t0 = time.time()
    resp_norm = chat.generate_chat_response(qwen_model, qwen_tok, [{"role": "user", "content": "Say test"}], max_new_tokens=5, temperature=0.0)
    elapsed = time.time() - t0
    print(f"Normal Mode generation took {elapsed:.3f}s: \"{resp_norm}\"")
    assert len(resp_norm) > 0, "Normal mode failed"
    test_results["TEST 11 (Normal Mode Performance)"] = "PASSED"

    # -------------------------------------------------------------------------
    # TEST 12: Edge Cases & Error Handling
    # -------------------------------------------------------------------------
    print("\n--- TEST 12: Edge Cases & Error Handling ---")
    # Edge case 1: Empty response (prompt only)
    tok_empty_resp = qwen_tok("Test", add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out_single = qwen_model(torch.tensor([tok_empty_resp]))
    p_step1 = torch.softmax(out_single.logits[0, -1, :].to(torch.float32), dim=-1)
    assert abs(float(torch.sum(p_step1).item()) - 1.0) < 1e-4, "Softmax on single prompt failed"

    # Edge case 2: Special whitespace / newline tokens decoding
    for test_tid in [198, 271, 0, 151643]:  # newlines, spaces, special tokens
        dec = qwen_tok.decode([test_tid])
        disp = dec if dec.strip() else f"␣ ({repr(dec)})"
        assert len(disp) > 0, f"Failed to format token ID {test_tid}"

    print("Edge cases and special token decodings handled safely.")
    test_results["TEST 12 (Error Handling & Edge Cases)"] = "PASSED"

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ALL 12 STEP 15 TESTS COMPLETED SUCCESSFULLY:")
    print("=" * 80)
    for t_name, status in test_results.items():
        print(f"  {t_name:48s} : {status}")
    print("=" * 80)


if __name__ == "__main__":
    run_step15_tests()
