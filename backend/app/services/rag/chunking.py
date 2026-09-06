"""
Document Chunking Service for Phase 3 RAG Pipeline.
Reuses and builds upon Phase 2 chunking logic:
- Sliding word window (default 500 words, 50-word overlap)
- Multi-page and single-page span preservation (page_start, page_end)
- Structured metadata tagging with user_id and document_id ownership
"""

import os
import sys
from typing import List, Dict, Any, Optional

DEFAULT_CHUNK_SIZE = 500      # Words per chunk
DEFAULT_CHUNK_OVERLAP = 50    # Overlapping words between consecutive chunks


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Split plain text into overlapping word chunks while preserving original words.
    """
    if not text or not text.strip():
        return []

    words = text.strip().split()
    if len(words) <= chunk_size:
        return [text.strip()]

    chunks = []
    step = max(1, chunk_size - chunk_overlap)
    start_idx = 0

    while start_idx < len(words):
        end_idx = min(start_idx + chunk_size, len(words))
        chunk_words = words[start_idx:end_idx]
        chunk_str = " ".join(chunk_words).strip()
        if chunk_str:
            chunks.append(chunk_str)
        if end_idx >= len(words):
            break
        start_idx += step

    return chunks


def chunk_document(
    document: Dict[str, Any],
    user_id: Optional[str] = None,
    document_id: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    start_chunk_id: int = 0,
) -> List[Dict[str, Any]]:
    """
    Split a structured document (from document_loader) into structured chunks with metadata,
    including user and document ownership.
    """
    if not document or not document.get("text", "").strip():
        return []

    filename = document.get("filename", "unknown_document")
    filepath = document.get("filepath", "")
    file_type = document.get("file_type", "TXT")
    doc_id = document_id or document.get("document_id", filename)

    pages = document.get("pages", [])
    chunks_data: List[Dict[str, Any]] = []

    # If the document has structured page-level text (e.g. from PDF loader or multi-page CSV)
    if pages and len(pages) > 1:
        # Build running word stream with page mapping
        word_page_map = []
        for p in pages:
            p_num = p.get("page_number", 1)
            p_text = p.get("text", "")
            for w in p_text.split():
                word_page_map.append((w, p_num))

        if not word_page_map:
            return []

        if len(word_page_map) <= chunk_size:
            all_text = " ".join(w for w, _ in word_page_map)
            return [{
                "chunk_id": start_chunk_id,
                "document_id": doc_id,
                "user_id": user_id,
                "filename": filename,
                "filepath": filepath,
                "file_type": file_type,
                "page_start": word_page_map[0][1],
                "page_end": word_page_map[-1][1],
                "text": all_text,
                "char_count": len(all_text),
                "word_count": len(word_page_map),
            }]

        step = max(1, chunk_size - chunk_overlap)
        start_idx = 0
        current_id = start_chunk_id

        while start_idx < len(word_page_map):
            end_idx = min(start_idx + chunk_size, len(word_page_map))
            slice_data = word_page_map[start_idx:end_idx]
            chunk_str = " ".join(w for w, _ in slice_data).strip()

            if chunk_str:
                p_start = slice_data[0][1]
                p_end = slice_data[-1][1]
                chunks_data.append({
                    "chunk_id": current_id,
                    "document_id": doc_id,
                    "user_id": user_id,
                    "filename": filename,
                    "filepath": filepath,
                    "file_type": file_type,
                    "page_start": p_start,
                    "page_end": p_end,
                    "text": chunk_str,
                    "char_count": len(chunk_str),
                    "word_count": len(slice_data),
                })
                current_id += 1

            if end_idx >= len(word_page_map):
                break
            start_idx += step

    else:
        # Single page / TXT / DOCX chunking
        p_num = 1
        if pages and len(pages) == 1:
            p_num = pages[0].get("page_number", 1)

        raw_text_chunks = chunk_text(document["text"], chunk_size, chunk_overlap)
        for i, c_text in enumerate(raw_text_chunks):
            w_count = len(c_text.split())
            chunks_data.append({
                "chunk_id": start_chunk_id + i,
                "document_id": doc_id,
                "user_id": user_id,
                "filename": filename,
                "filepath": filepath,
                "file_type": file_type,
                "page_start": p_num,
                "page_end": p_num,
                "text": c_text,
                "char_count": len(c_text),
                "word_count": w_count,
            })

    return chunks_data
