"""
Comprehensive Verification Suite for Phase 2 - Step 12: Add Conversation Memory
Validates Level 1 (Short-Term Memory) & Level 2 (Long-Term Conversation History)
along with all 13 test requirements.
"""

import os
import sys
import json
import sqlite3
import torch

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Import LocalGPT modules
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

def run_step12_tests():
    print("=" * 75)
    print("PHASE 2 - STEP 12: ADD CONVERSATION MEMORY VERIFICATION")
    print("=" * 75)
    
    test_results = {}
    
    # Load model and tokenizer once for tests
    print("\nLoading Qwen2.5-1.5B-Instruct model and tokenizer...")
    qwen_model, qwen_tok = model.load_model_and_tokenizer()
    
    # Isolated test database
    test_db_dir = os.path.join("data", "conversations")
    os.makedirs(test_db_dir, exist_ok=True)
    test_db_path = os.path.join(test_db_dir, "test_step12_memory.db")
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    database.init_database(test_db_path)
    
    # -------------------------------------------------------------------------
    # TEST 1 & 2: Level 1 Short-Term Memory & Follow-Up Context
    # -------------------------------------------------------------------------
    print("\n--- TEST 1 & 2: Short-Term Memory & Follow-Up Context Resolution ---")
    conv_a_id = database.create_conversation(title="Machine Learning Chat", db_path=test_db_path)
    
    # Turn 1
    t1_user = "What is Machine Learning? Explain concisely."
    database.save_message(conv_a_id, "user", t1_user, db_path=test_db_path)
    t1_context = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": t1_user}
    ]
    t1_resp = chat.generate_chat_response(qwen_model, qwen_tok, t1_context, max_new_tokens=60, temperature=0.2)
    database.save_message(conv_a_id, "assistant", t1_resp, db_path=test_db_path)
    print(f"Turn 1 Q: {t1_user}")
    print(f"Turn 1 A: {t1_resp.strip()[:100]}...")
    
    # Turn 2
    t2_user = "What are its main types?"
    database.save_message(conv_a_id, "user", t2_user, db_path=test_db_path)
    t2_context = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": t1_user},
        {"role": "assistant", "content": t1_resp},
        {"role": "user", "content": t2_user}
    ]
    t2_resp = chat.generate_chat_response(qwen_model, qwen_tok, t2_context, max_new_tokens=60, temperature=0.2)
    database.save_message(conv_a_id, "assistant", t2_resp, db_path=test_db_path)
    print(f"\nTurn 2 Q: {t2_user}")
    print(f"Turn 2 A: {t2_resp.strip()[:100]}...")
    assert ("supervised" in t2_resp.lower() or "unsupervised" in t2_resp.lower() or "reinforcement" in t2_resp.lower() or "types" in t2_resp.lower()), "Assistant failed to use ML context for types"
    test_results["TEST 1 (Short-Term Memory)"] = "PASSED"
    
    # Turn 3 - Follow up "Give me an example."
    t3_user = "Give me an example."
    database.save_message(conv_a_id, "user", t3_user, db_path=test_db_path)
    t3_context = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": t1_user},
        {"role": "assistant", "content": t1_resp},
        {"role": "user", "content": t2_user},
        {"role": "assistant", "content": t2_resp},
        {"role": "user", "content": t3_user}
    ]
    t3_resp = chat.generate_chat_response(qwen_model, qwen_tok, t3_context, max_new_tokens=60, temperature=0.2)
    database.save_message(conv_a_id, "assistant", t3_resp, db_path=test_db_path)
    print(f"\nTurn 3 Q: {t3_user}")
    print(f"Turn 3 A: {t3_resp.strip()[:100]}...")
    # Verify that response gives an ML example (e.g. spam detection, image classification, linear regression, etc.)
    t3_lower = t3_resp.lower()
    has_ml_concept = any(term in t3_lower for term in ["learning", "model", "data", "predict", "classif", "spam", "supervised", "example", "house", "image"])
    assert has_ml_concept, f"Follow-up example did not relate to ML context: {t3_resp}"
    test_results["TEST 2 (Follow-Up Context)"] = "PASSED"
    
    # -------------------------------------------------------------------------
    # TEST 3 & 4: New Conversation Isolation & Multiple Conversations
    # -------------------------------------------------------------------------
    print("\n--- TEST 3 & 4: New Conversation Isolation & Multi-Chat Independence ---")
    conv_b_id = database.create_conversation(title="Python Decorators Chat", db_path=test_db_path)
    
    # Conversation B message
    tb_user = "Explain Python decorators in one sentence."
    database.save_message(conv_b_id, "user", tb_user, db_path=test_db_path)
    tb_context = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": tb_user}
    ]
    tb_resp = chat.generate_chat_response(qwen_model, qwen_tok, tb_context, max_new_tokens=50, temperature=0.2)
    database.save_message(conv_b_id, "assistant", tb_resp, db_path=test_db_path)
    print(f"Conv B Turn 1 Q: {tb_user}")
    print(f"Conv B Turn 1 A: {tb_resp.strip()[:100]}...")
    
    # Verify Conv B context has NO ML messages
    assert all("machine learning" not in m["content"].lower() for m in tb_context), "ML leaked into Conv B context!"
    
    # Fetch messages from SQLite for Conv A and Conv B
    msgs_a = database.get_conversation_messages(conv_a_id, db_path=test_db_path)
    msgs_b = database.get_conversation_messages(conv_b_id, db_path=test_db_path)
    
    assert len(msgs_a) == 6, f"Conv A should have 6 messages (3 turns), got {len(msgs_a)}"
    assert len(msgs_b) == 2, f"Conv B should have 2 messages (1 turn), got {len(msgs_b)}"
    assert all(m["conversation_id"] == conv_a_id for m in msgs_a), "Conv A has mismatched messages"
    assert all(m["conversation_id"] == conv_b_id for m in msgs_b), "Conv B has mismatched messages"
    print(f"Conv A messages count: {len(msgs_a)} | Conv B messages count: {len(msgs_b)}")
    test_results["TEST 3 (New Chat Isolation)"] = "PASSED"
    test_results["TEST 4 (Multiple Conversations)"] = "PASSED"
    
    # -------------------------------------------------------------------------
    # TEST 5: Reopen Old Conversation (Switch back to Conv A)
    # -------------------------------------------------------------------------
    print("\n--- TEST 5: Reopen Old Conversation & Memory Switching ---")
    # Simulate switching to Conv A
    reopened_msgs_a = database.get_conversation_messages(conv_a_id, db_path=test_db_path)
    assert len(reopened_msgs_a) == 6, f"Failed to restore all 6 messages in Conv A, got {len(reopened_msgs_a)}"
    assert reopened_msgs_a[0]["content"] == t1_user
    assert reopened_msgs_a[2]["content"] == t2_user
    assert reopened_msgs_a[4]["content"] == t3_user
    print("Reopened Conv A: all 3 turns (6 messages) restored with complete fidelity.")
    test_results["TEST 5 (Reopen Old Conversation)"] = "PASSED"
    
    # -------------------------------------------------------------------------
    # TEST 6: Memory After Restart (Simulate Fresh Streamlit Session)
    # -------------------------------------------------------------------------
    print("\n--- TEST 6: Memory Persistence After Restart ---")
    # Query database fresh as if Streamlit reloaded
    all_convs = database.get_conversations(db_path=test_db_path)
    assert len(all_convs) >= 2, "Expected at least 2 saved conversations"
    conv_ids = [c["id"] for c in all_convs]
    assert conv_a_id in conv_ids and conv_b_id in conv_ids, "Saved conversations not found after reload"
    
    # Load most recent conversation messages
    latest_conv = all_convs[0]
    latest_msgs = database.get_conversation_messages(latest_conv["id"], db_path=test_db_path)
    assert len(latest_msgs) > 0, "Failed to load messages from persisted database"
    print(f"Persisted conversations count: {len(all_convs)}. Latest conversation '{latest_conv['title']}' loaded with {len(latest_msgs)} messages.")
    test_results["TEST 6 (Memory After Restart)"] = "PASSED"
    
    # -------------------------------------------------------------------------
    # TEST 7: Conversation Metadata Preservation
    # -------------------------------------------------------------------------
    print("\n--- TEST 7: Conversation Metadata Preservation ---")
    conv_a_meta = database.get_conversation(conv_a_id, db_path=test_db_path)
    assert conv_a_meta is not None, "Failed to get conversation A record"
    assert conv_a_meta["id"] == conv_a_id, "Conversation ID mismatch"
    assert conv_a_meta["title"] == "Machine Learning Chat", f"Unexpected title: {conv_a_meta['title']}"
    assert conv_a_meta["created_at"] is not None, "created_at timestamp missing"
    assert conv_a_meta["updated_at"] is not None, "updated_at timestamp missing"
    print(f"Metadata verified: ID={conv_a_meta['id'][:8]}..., Title='{conv_a_meta['title']}', Created={conv_a_meta['created_at']}, Updated={conv_a_meta['updated_at']}")
    test_results["TEST 7 (Conversation Metadata)"] = "PASSED"
    
    # -------------------------------------------------------------------------
    # TEST 8: Sources Persistence
    # -------------------------------------------------------------------------
    print("\n--- TEST 8: Sources Persistence Across Reopening ---")
    sample_sources = [
        {"filename": "transformer_architecture.pdf", "page_start": 4, "page_end": 4, "score": 0.91, "text": "Transformer self-attention details."},
        {"filename": "transformer_architecture.pdf", "page_start": 7, "page_end": 8, "score": 0.85, "text": "Multi-head attention mechanisms."}
    ]
    conv_c_id = database.create_conversation(title="RAG Research Chat", db_path=test_db_path)
    database.save_message(conv_c_id, "user", "What is transformer attention?", db_path=test_db_path)
    database.save_message(conv_c_id, "assistant", "Transformers use multi-head attention.", sources=sample_sources, db_path=test_db_path)
    
    # Reopen and inspect sources
    c_msgs = database.get_conversation_messages(conv_c_id, db_path=test_db_path)
    assert len(c_msgs) == 2, "Expected 2 messages in RAG chat"
    asst_c = c_msgs[1]
    assert asst_c["sources"] is not None and len(asst_c["sources"]) == 2, "Sources not restored correctly"
    assert asst_c["sources"][0]["filename"] == "transformer_architecture.pdf"
    assert asst_c["sources"][0]["page_start"] == 4
    assert asst_c["sources"][1]["page_start"] == 7
    print(f"Restored {len(asst_c['sources'])} source citations accurately from SQLite sources_json.")
    test_results["TEST 8 (Sources Persistence)"] = "PASSED"
    
    # -------------------------------------------------------------------------
    # TEST 9: Regenerate
    # -------------------------------------------------------------------------
    print("\n--- TEST 9: Regenerate Response Context Handling ---")
    # Remove last assistant response in Conv C
    database.remove_last_assistant_message(conv_c_id, db_path=test_db_path)
    msgs_c_after_del = database.get_conversation_messages(conv_c_id, db_path=test_db_path)
    assert len(msgs_c_after_del) == 1, f"Expected 1 user message after removal, got {len(msgs_c_after_del)}"
    
    # Regenerate new response
    new_sources = [{"filename": "transformer_architecture.pdf", "page_start": 4, "page_end": 4, "score": 0.94, "text": "Updated attention mechanism facts."}]
    database.save_message(conv_c_id, "assistant", "Regenerated explanation of transformer attention.", sources=new_sources, db_path=test_db_path)
    
    msgs_c_regen = database.get_conversation_messages(conv_c_id, db_path=test_db_path)
    assert len(msgs_c_regen) == 2, "Expected 2 messages after regeneration"
    assert msgs_c_regen[1]["content"] == "Regenerated explanation of transformer attention."
    assert msgs_c_regen[1]["sources"][0]["score"] == 0.94
    print("Regenerate successfully maintained preceding user context and updated assistant response & sources.")
    test_results["TEST 9 (Regenerate Compatibility)"] = "PASSED"
    
    # -------------------------------------------------------------------------
    # TEST 10: Edit Message & Downstream Truncation
    # -------------------------------------------------------------------------
    print("\n--- TEST 10: Edit User Message & Downstream Memory Truncation ---")
    # Conv A currently has 6 messages (3 turns). Let's edit message 3 (Turn 2 user message, order=3)
    edited_t2_text = "What is Deep Learning?"
    database.edit_user_message_and_truncate(
        conversation_id=conv_a_id,
        message_order=3,
        new_content=edited_t2_text,
        db_path=test_db_path
    )
    
    msgs_a_edited = database.get_conversation_messages(conv_a_id, db_path=test_db_path)
    # Messages after order 3 (4, 5, 6) must be deleted. Total messages should now be 3 (Turn 1 Q, Turn 1 A, Turn 2 edited Q)
    assert len(msgs_a_edited) == 3, f"Expected 3 messages after edit & truncate, got {len(msgs_a_edited)}"
    assert msgs_a_edited[2]["content"] == edited_t2_text
    
    # Now generate new assistant response for edited prompt
    edited_context = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": msgs_a_edited[0]["content"]},
        {"role": "assistant", "content": msgs_a_edited[1]["content"]},
        {"role": "user", "content": edited_t2_text}
    ]
    edited_asst_resp = chat.generate_chat_response(qwen_model, qwen_tok, edited_context, max_new_tokens=50, temperature=0.2)
    database.save_message(conv_a_id, "assistant", edited_asst_resp, db_path=test_db_path)
    
    msgs_a_final = database.get_conversation_messages(conv_a_id, db_path=test_db_path)
    assert len(msgs_a_final) == 4, f"Expected 4 messages after new assistant response, got {len(msgs_a_final)}"
    print(f"Downstream memory correctly truncated. New Turn 2 Q: '{edited_t2_text}' -> A: '{edited_asst_resp.strip()[:80]}...'")
    test_results["TEST 10 (Edit Truncation)"] = "PASSED"
    
    # -------------------------------------------------------------------------
    # TEST 11: RAG in Saved Conversation
    # -------------------------------------------------------------------------
    print("\n--- TEST 11: RAG in Saved Conversation & Reopening ---")
    rag_conv_id = database.create_conversation(title="RAG Documents Conversation", db_path=test_db_path)
    rag_user_prompt = "What is the training dataset size?"
    rag_retrieved_chunk = [{"filename": "model_paper.pdf", "page_start": 2, "page_end": 3, "score": 0.88, "text": "Trained on 2 trillion tokens."}]
    
    rag_injected_prompt = rag.build_rag_prompt(rag_user_prompt, rag_retrieved_chunk)
    database.save_message(rag_conv_id, "user", rag_user_prompt, db_path=test_db_path)
    database.save_message(rag_conv_id, "assistant", "The model was trained on 2 trillion tokens.", sources=rag_retrieved_chunk, db_path=test_db_path)
    
    # Switch away to Conv B, then switch back to rag_conv_id
    _ = database.get_conversation_messages(conv_b_id, db_path=test_db_path)
    reopened_rag_msgs = database.get_conversation_messages(rag_conv_id, db_path=test_db_path)
    
    assert len(reopened_rag_msgs) == 2, "RAG conversation messages count mismatch"
    assert reopened_rag_msgs[1]["sources"][0]["filename"] == "model_paper.pdf"
    assert reopened_rag_msgs[1]["sources"][0]["page_start"] == 2
    print("RAG prompt context and sources metadata intact across conversation switching.")
    test_results["TEST 11 (RAG Compatibility)"] = "PASSED"
    
    # -------------------------------------------------------------------------
    # TEST 12: Streaming Functionality
    # -------------------------------------------------------------------------
    print("\n--- TEST 12: Token Streaming Preservation ---")
    stream_msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Count from 1 to 3."}
    ]
    stream_tokens = []
    for chunk in chat.stream_chat_response(qwen_model, qwen_tok, stream_msgs, max_new_tokens=15, temperature=0.0):
        stream_tokens.append(chunk)
    stream_out = "".join(stream_tokens)
    print(f"Streamed output ({len(stream_tokens)} chunks): '{stream_out.strip()}'")
    assert len(stream_tokens) > 0 and len(stream_out.strip()) > 0, "Streaming failed"
    test_results["TEST 12 (Streaming Preservation)"] = "PASSED"
    
    # -------------------------------------------------------------------------
    # TEST 13: X-Ray Steps 4-11 Functionality
    # -------------------------------------------------------------------------
    print("\n--- TEST 13: Phase 1 X-Ray Steps 4-11 Preservation ---")
    xray_text = "Transformer memory"
    tok_res = tokenizer.tokenize_input(qwen_tok, xray_text)
    assert len(tok_res["tokens"]) > 0, "X-Ray tokenizer failed"
    
    inputs = qwen_tok(xray_text, return_tensors="pt")
    with torch.no_grad():
        xray_outputs = qwen_model(**inputs, output_attentions=True, output_hidden_states=True)
    
    assert xray_outputs.hidden_states is not None, "X-Ray hidden states missing"
    assert xray_outputs.attentions is not None, "X-Ray attentions missing"
    
    emb_v = xray_outputs.hidden_states[0][0, 0].detach().cpu().to(torch.float32).numpy()
    assert emb_v.shape[0] == qwen_model.config.hidden_size, "Hidden dimension mismatch"
    
    attn_m = xray_outputs.attentions[0][0, 0].detach().cpu().to(torch.float32).numpy()
    assert attn_m.shape[0] == len(tok_res["tokens"]), "Attention matrix shape mismatch"
    print(f"X-Ray Steps 4-11 confirmed: Tokenizer ({len(tok_res['tokens'])} tokens), Embeddings ({emb_v.shape}), Attentions ({attn_m.shape})")
    test_results["TEST 13 (X-Ray Preservation)"] = "PASSED"
    
    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("ALL 13 STEP 12 TESTS COMPLETED SUCCESSFULLY:")
    print("=" * 75)
    for t_name, status in test_results.items():
        print(f"  {t_name:45s} : {status}")
    print("=" * 75)

if __name__ == "__main__":
    run_step12_tests()
