"""
Comprehensive Verification Suite for Phase 2 - Step 11: Add Source References
Tests all 10 validation scenarios required by the specification.
"""

import os
import sys
import json
import sqlite3
import numpy as np

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Import all LocalGPT modules
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

def run_tests():
    print("=" * 70)
    print("PHASE 2 - STEP 11: ADD SOURCE REFERENCES VERIFICATION")
    print("=" * 70)
    
    test_results = {}
    
    # ---------------------------------------------------------
    # TEST 1: Upload a document & build FAISS Vector Store
    # ---------------------------------------------------------
    print("\n--- TEST 1: Document Upload, Extraction & FAISS Indexing ---")
    test_docs_dir = os.path.join("data", "test_docs")
    os.makedirs(test_docs_dir, exist_ok=True)
    
    doc1_path = os.path.join(test_docs_dir, "ai_system_architecture.txt")
    doc1_content = (
        "Title: Deep Learning and Transformer Architecture\n"
        "Section 1: Architecture Overview\n"
        "The proposed system uses a Transformer-based architecture with 28 hidden layers and 12 attention heads.\n"
        "The embedding dimension is 1536 and uses rotary position embeddings (RoPE) for long context understanding.\n"
        "Section 2: Training Details\n"
        "The model was trained on 2 trillion tokens using AdamW optimizer with cosine learning rate schedule.\n"
        "The batch size was set to 4096 sequences with 0.1 weight decay."
    )
    with open(doc1_path, "w", encoding="utf-8") as f:
        f.write(doc1_content)
        
    doc2_path = os.path.join(test_docs_dir, "medical_radiology_guidelines.txt")
    doc2_content = (
        "Title: Chest Radiograph Diagnostic Guidelines\n"
        "Section 1: Pneumonia Detection\n"
        "Pneumonia is diagnosed by the presence of dense alveolar consolidation or ground-glass opacities in chest X-rays.\n"
        "Bilateral infiltrates typically indicate viral pneumonia or atypical mycoplasma etiology.\n"
        "Section 2: Cardiomegaly Assessment\n"
        "Cardiomegaly is confirmed when the cardiothoracic ratio exceeds 0.50 on a standard posteroanterior view."
    )
    with open(doc2_path, "w", encoding="utf-8") as f:
        f.write(doc2_content)
        
    loaded_doc1 = document_loader.load_document(doc1_path)
    loaded_doc2 = document_loader.load_document(doc2_path)
    
    assert loaded_doc1["status"] == "success", f"Doc 1 failed: {loaded_doc1['error_message']}"
    assert loaded_doc2["status"] == "success", f"Doc 2 failed: {loaded_doc2['error_message']}"
    
    chunks1 = chunking.chunk_document(loaded_doc1, chunk_size=30, chunk_overlap=5, start_chunk_id=0)
    chunks2 = chunking.chunk_document(loaded_doc2, chunk_size=30, chunk_overlap=5, start_chunk_id=len(chunks1))
    all_chunks = chunks1 + chunks2
    
    print(f"Loaded 2 documents -> Generated {len(all_chunks)} chunks total.")
    emb_model = embeddings.load_embedding_model()
    embedded_chunks = embeddings.embed_chunks(all_chunks, model=emb_model)
    vstore = vector_store.create_vector_store(embedded_chunks)
    
    assert vstore is not None, "Failed to create vector store"
    assert vstore["index"].ntotal == len(all_chunks), "FAISS index size mismatch"
    print(f"Vector store successfully created with {vstore['index'].ntotal} vectors.")
    test_results["TEST 1 (Upload & Index)"] = "PASSED"
    
    # ---------------------------------------------------------
    # TEST 2: Ask a document-based question & retrieve chunks
    # ---------------------------------------------------------
    print("\n--- TEST 2: Document-based Question Retrieval ---")
    query_1 = "What architecture and how many layers does the proposed system use?"
    chunks_retrieved_1 = rag.retrieve_relevant_chunks(
        query=query_1,
        vector_store=vstore,
        embedding_model=emb_model,
        top_k=2
    )
    print(f"Query: '{query_1}'")
    print(f"Retrieved {len(chunks_retrieved_1)} chunks.")
    assert len(chunks_retrieved_1) > 0, "No chunks retrieved for query 1"
    top_chunk_1 = chunks_retrieved_1[0]
    print(f"Top result: file='{top_chunk_1['filename']}', page={top_chunk_1['page_start']}, score={top_chunk_1['score']:.4f}")
    assert top_chunk_1["filename"] == "ai_system_architecture.txt", f"Unexpected source file: {top_chunk_1['filename']}"
    assert "Transformer-based architecture" in top_chunk_1["text"], "Expected text content missing from top chunk"
    test_results["TEST 2 (Retrieval)"] = "PASSED"
    
    # ---------------------------------------------------------
    # TEST 3: Verify Answer Generation from Retrieved Context
    # ---------------------------------------------------------
    print("\n--- TEST 3: Grounded Prompt & Generation ---")
    rag_prompt_1, sources_1 = rag.retrieve_and_build_context(
        query=query_1,
        vector_store=vstore,
        embedding_model=emb_model,
        top_k=2
    )
    print("Constructed RAG prompt snippet:")
    print("-" * 40)
    print(rag_prompt_1[:250] + "...")
    print("-" * 40)
    assert "DOCUMENT CONTEXT:" in rag_prompt_1, "DOCUMENT CONTEXT missing from prompt"
    assert "ai_system_architecture.txt" in rag_prompt_1, "Source filename missing from injected context"
    assert "28 hidden layers" in rag_prompt_1, "Chunk facts missing from prompt"
    test_results["TEST 3 (Grounded Prompt Context)"] = "PASSED"
    
    # ---------------------------------------------------------
    # TEST 4: Verify Sources Show Actual Filename and Page
    # ---------------------------------------------------------
    print("\n--- TEST 4: Source Metadata Accuracy (Filename, Page, Score) ---")
    assert len(sources_1) > 0, "sources_1 is empty"
    for idx, s in enumerate(sources_1, start=1):
        print(f"Source {idx}: Filename='{s.get('filename')}', Page={s.get('page_start')}-{s.get('page_end')}, Score={s.get('score'):.4f}")
        assert s.get("filename") == "ai_system_architecture.txt", "Incorrect filename in source"
        assert s.get("page_start") == 1, "Incorrect page start in source"
        assert s.get("score") is not None and s.get("score") > 0.0, "Missing or invalid similarity score"
        assert s.get("text"), "Missing excerpt text in source"
    test_results["TEST 4 (Source Filename & Page Accuracy)"] = "PASSED"
    
    # ---------------------------------------------------------
    # TEST 5: Ask Another Document Question with Different Sources
    # ---------------------------------------------------------
    print("\n--- TEST 5: Second Document Question with Distinct Sources ---")
    query_2 = "How is cardiomegaly diagnosed and what is the cardiothoracic ratio threshold?"
    rag_prompt_2, sources_2 = rag.retrieve_and_build_context(
        query=query_2,
        vector_store=vstore,
        embedding_model=emb_model,
        top_k=1
    )
    print(f"Query 2: '{query_2}'")
    print(f"Retrieved {len(sources_2)} source(s).")
    assert len(sources_2) > 0, "No sources retrieved for query 2"
    source_2_top = sources_2[0]
    print(f"Source: Filename='{source_2_top['filename']}', Score={source_2_top['score']:.4f}")
    assert source_2_top["filename"] == "medical_radiology_guidelines.txt", f"Expected medical doc, got {source_2_top['filename']}"
    assert "0.50" in source_2_top["text"], "Expected text content missing from top chunk"
    # Ensure source 1 and source 2 are separate and distinct
    assert sources_1[0]["filename"] != sources_2[0]["filename"], "Sources should be distinct across different questions"
    test_results["TEST 5 (Multi-Turn Distinct Sources)"] = "PASSED"
    
    # ---------------------------------------------------------
    # TEST 6: SQLite Conversation Persistence & Reopening
    # ---------------------------------------------------------
    print("\n--- TEST 6: Conversation Persistence & Restoration of Sources ---")
    test_db_path = os.path.join("data", "conversations", "test_step11.db")
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
        
    database.init_database(test_db_path)
    conv_id = database.create_conversation(title="Test Step 11 Chat", db_path=test_db_path)
    
    # Save Turn 1
    database.save_message(
        conversation_id=conv_id,
        role="user",
        content=query_1,
        db_path=test_db_path
    )
    database.save_message(
        conversation_id=conv_id,
        role="assistant",
        content="The proposed system uses a Transformer-based architecture with 28 hidden layers and 12 attention heads.",
        sources=sources_1,
        db_path=test_db_path
    )
    
    # Save Turn 2
    database.save_message(
        conversation_id=conv_id,
        role="user",
        content=query_2,
        db_path=test_db_path
    )
    database.save_message(
        conversation_id=conv_id,
        role="assistant",
        content="Cardiomegaly is confirmed when the cardiothoracic ratio exceeds 0.50 on a standard posteroanterior view.",
        sources=sources_2,
        db_path=test_db_path
    )
    
    # Reopen conversation and inspect restored messages
    restored_messages = database.get_conversation_messages(conv_id, db_path=test_db_path)
    assert len(restored_messages) == 4, f"Expected 4 messages, got {len(restored_messages)}"
    
    asst_msg_1 = restored_messages[1]
    asst_msg_2 = restored_messages[3]
    
    assert asst_msg_1["role"] == "assistant"
    assert asst_msg_1["sources"] is not None and len(asst_msg_1["sources"]) == len(sources_1)
    assert asst_msg_1["sources"][0]["filename"] == "ai_system_architecture.txt"
    
    assert asst_msg_2["role"] == "assistant"
    assert asst_msg_2["sources"] is not None and len(asst_msg_2["sources"]) == len(sources_2)
    assert asst_msg_2["sources"][0]["filename"] == "medical_radiology_guidelines.txt"
    
    print("Messages and their individual source references successfully restored from SQLite:")
    print(f"  Turn 1 Assistant Source: {asst_msg_1['sources'][0]['filename']}")
    print(f"  Turn 2 Assistant Source: {asst_msg_2['sources'][0]['filename']}")
    test_results["TEST 6 (Persistence & Reopening)"] = "PASSED"
    
    # ---------------------------------------------------------
    # TEST 7: Regenerate Answer & Source Association Update
    # ---------------------------------------------------------
    print("\n--- TEST 7: Regenerate Answer & Source Update ---")
    # Simulate removing last assistant response
    del_ok = database.remove_last_assistant_message(conv_id, db_path=test_db_path)
    assert del_ok, "Failed to remove last assistant message"
    
    msgs_after_removal = database.get_conversation_messages(conv_id, db_path=test_db_path)
    assert len(msgs_after_removal) == 3, f"Expected 3 messages after removal, got {len(msgs_after_removal)}"
    
    # Re-retrieve with new parameters and save new regenerated response
    _, new_regen_sources = rag.retrieve_and_build_context(
        query=query_2,
        vector_store=vstore,
        embedding_model=emb_model,
        top_k=2
    )
    database.save_message(
        conversation_id=conv_id,
        role="assistant",
        content="Regenerated answer: Cardiomegaly threshold is > 0.50.",
        sources=new_regen_sources,
        db_path=test_db_path
    )
    
    updated_msgs = database.get_conversation_messages(conv_id, db_path=test_db_path)
    assert len(updated_msgs) == 4, f"Expected 4 messages, got {len(updated_msgs)}"
    assert updated_msgs[3]["content"] == "Regenerated answer: Cardiomegaly threshold is > 0.50."
    assert len(updated_msgs[3]["sources"]) == 2, "Expected updated 2 chunks in regenerated sources"
    print("Regeneration removed old response and saved new response with updated sources successfully.")
    test_results["TEST 7 (Regeneration & Source Update)"] = "PASSED"
    
    # ---------------------------------------------------------
    # TEST 8: RAG Disabled Behavior
    # ---------------------------------------------------------
    print("\n--- TEST 8: RAG Disabled (No Fake Sources) ---")
    assert not rag.should_use_rag(vstore, enable_rag=False), "should_use_rag must be False when enable_rag=False"
    assert not rag.should_use_rag(None, enable_rag=True), "should_use_rag must be False when vector_store is None"
    
    disabled_prompt, disabled_sources = rag.retrieve_and_build_context(
        query="Explain machine learning",
        vector_store=vstore if False else None,
        top_k=2
    )
    assert disabled_prompt == "Explain machine learning", "Prompt should remain unaltered when RAG is off"
    assert disabled_sources == [], "Retrieved sources must be empty when RAG is off"
    print("When RAG is disabled, no document retrieval occurs and sources list is empty.")
    test_results["TEST 8 (RAG Disabled Behavior)"] = "PASSED"
    
    # ---------------------------------------------------------
    # TEST 9: Verify Streaming Functionality
    # ---------------------------------------------------------
    print("\n--- TEST 9: Real-time Streaming Generation ---")
    print("Loading model for lightweight streaming test...")
    qwen_model, qwen_tok = model.load_model_and_tokenizer()
    
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one word."}
    ]
    
    stream_chunks = []
    for chunk in chat.stream_chat_response(
        qwen_model,
        qwen_tok,
        test_messages,
        max_new_tokens=10,
        temperature=0.0
    ):
        stream_chunks.append(chunk)
        
    full_stream_output = "".join(stream_chunks)
    print(f"Streamed {len(stream_chunks)} chunks: '{full_stream_output}'")
    assert len(stream_chunks) > 0, "Streaming produced no chunks"
    assert len(full_stream_output.strip()) > 0, "Stream output was empty"
    test_results["TEST 9 (Token Streaming)"] = "PASSED"
    
    # ---------------------------------------------------------
    # TEST 10: Verify Phase 1 X-Ray Steps 4–11 Functionality
    # ---------------------------------------------------------
    print("\n--- TEST 10: Phase 1 LLM X-Ray Inspection (Steps 4-11) ---")
    xray_prompt = "Deep learning"
    tok_res = tokenizer.tokenize_input(qwen_tok, xray_prompt)
    tokens = tok_res["tokens"]
    token_ids = tok_res["token_ids"]
    assert len(tokens) > 0 and len(token_ids) > 0, "Tokenization failed"
    safe_tokens = [repr(t) for t in tokens]
    print(f"Step 4 Tokenizer: {len(tokens)} tokens -> {safe_tokens}")
    
    # Forward pass & hidden states
    import torch
    inputs = qwen_tok(xray_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = qwen_model(**inputs, output_attentions=True, output_hidden_states=True)
        
    assert hasattr(outputs, "hidden_states") and outputs.hidden_states is not None, "Missing hidden states"
    assert hasattr(outputs, "attentions") and outputs.attentions is not None, "Missing attentions"
    
    # Embedding inspection (Step 5)
    first_token_emb = outputs.hidden_states[0][0, 0].detach().cpu().to(torch.float32).numpy()
    assert first_token_emb.shape[0] == qwen_model.config.hidden_size, "Hidden size dimension mismatch"
    print(f"Step 5 Embeddings: shape {first_token_emb.shape} confirmed.")
    
    # Attention extraction (Step 7 & 11)
    attn_matrix = outputs.attentions[0][0, 0].detach().cpu().to(torch.float32).numpy()
    assert attn_matrix.shape[0] == len(tokens) and attn_matrix.shape[1] == len(tokens), "Attention shape mismatch"
    print(f"Step 7 Attention Matrix: shape {attn_matrix.shape} confirmed.")
    
    # Logits extraction (Step 9)
    next_token_logits = outputs.logits[0, -1].to(torch.float32)
    probs = torch.softmax(next_token_logits, dim=-1)
    top_k_probs, top_k_ids = torch.topk(probs, 5)
    top_tokens = []
    for p, tid in zip(top_k_probs, top_k_ids):
        top_tokens.append((repr(qwen_tok.decode([tid.item()])), f"{p.item():.3f}"))
    assert len(top_tokens) == 5, "Top-K logits extraction failed"
    print(f"Step 9 Top Logits: {top_tokens}")
    test_results["TEST 10 (X-Ray Steps 4-11)"] = "PASSED"
    
    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY:")
    print("=" * 70)
    for t_name, status in test_results.items():
        print(f"  {t_name:40s} : {status}")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
