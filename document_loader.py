"""
Document Loader Module for LocalGPT (Phase 2 - Step 3: Document Upload & Loader).
Supports safe text extraction from PDF, DOCX, and TXT files with structured metadata.
"""

import os
import re
from typing import Dict, Any, List, Optional


DOCUMENTS_DIR = os.path.join("data", "documents")


def get_safe_filename(filename: str) -> str:
    """Generate a clean, filesystem-safe filename."""
    # Remove any path traversal components
    base_name = os.path.basename(filename)
    # Replace non-alphanumeric (except dot, dash, underscore)
    safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', base_name)
    return safe_name or "uploaded_document"


def save_uploaded_file(uploaded_file, target_dir: str = DOCUMENTS_DIR) -> str:
    """
    Save a Streamlit UploadedFile object into the target directory safely.
    Returns the absolute path of the saved file.
    """
    os.makedirs(target_dir, exist_ok=True)
    safe_name = get_safe_filename(uploaded_file.name)
    target_path = os.path.join(target_dir, safe_name)
    
    # Avoid accidental overwrite if file exists with same name
    if os.path.exists(target_path):
        name_part, ext_part = os.path.splitext(safe_name)
        counter = 1
        while os.path.exists(os.path.join(target_dir, f"{name_part}_{counter}{ext_part}")):
            counter += 1
        target_path = os.path.join(target_dir, f"{name_part}_{counter}{ext_part}")

    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return target_path


def load_pdf(file_path: str) -> Dict[str, Any]:
    """
    Extract text and page-level metadata from a PDF file using pypdf.
    """
    filename = os.path.basename(file_path)
    result = {
        "filename": filename,
        "filepath": file_path,
        "file_type": "PDF",
        "text": "",
        "pages": [],
        "num_pages": 0,
        "char_count": 0,
        "word_count": 0,
        "status": "success",
        "error_message": None,
    }

    try:
        import pypdf
        reader = pypdf.PdfReader(file_path)
        
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                result["status"] = "error"
                result["error_message"] = "PDF is password-protected or encrypted."
                return result

        pages_data: List[Dict[str, Any]] = []
        full_text_parts: List[str] = []

        for idx, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            page_text_clean = page_text.strip()
            pages_data.append({
                "page_number": idx,
                "text": page_text_clean,
                "char_count": len(page_text_clean),
                "word_count": len(page_text_clean.split()),
            })
            if page_text_clean:
                full_text_parts.append(page_text_clean)

        combined_text = "\n\n".join(full_text_parts)
        result["text"] = combined_text
        result["pages"] = pages_data
        result["num_pages"] = len(reader.pages)
        result["char_count"] = len(combined_text)
        result["word_count"] = len(combined_text.split())

        if not combined_text.strip():
            result["status"] = "warning"
            result["error_message"] = "PDF loaded, but no selectable text was found (it may be a scanned image)."

    except Exception as e:
        result["status"] = "error"
        result["error_message"] = f"Failed to extract PDF text: {str(e)}"

    return result


def load_docx(file_path: str) -> Dict[str, Any]:
    """
    Extract text from a DOCX document using python-docx.
    """
    filename = os.path.basename(file_path)
    result = {
        "filename": filename,
        "filepath": file_path,
        "file_type": "DOCX",
        "text": "",
        "pages": [],
        "num_pages": 1,
        "char_count": 0,
        "word_count": 0,
        "status": "success",
        "error_message": None,
    }

    try:
        import docx
        doc = docx.Document(file_path)
        
        paragraphs_text = []
        for p in doc.paragraphs:
            text = p.text.strip()
            if text:
                paragraphs_text.append(text)

        # Also extract table text
        for table in doc.tables:
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    paragraphs_text.append(" | ".join(row_cells))

        combined_text = "\n\n".join(paragraphs_text)
        result["text"] = combined_text
        result["pages"] = [{
            "page_number": 1,
            "text": combined_text,
            "char_count": len(combined_text),
            "word_count": len(combined_text.split()),
        }]
        result["num_pages"] = 1
        result["char_count"] = len(combined_text)
        result["word_count"] = len(combined_text.split())

        if not combined_text.strip():
            result["status"] = "warning"
            result["error_message"] = "DOCX document is empty."

    except Exception as e:
        result["status"] = "error"
        result["error_message"] = f"Failed to extract DOCX text: {str(e)}"

    return result


def load_txt(file_path: str) -> Dict[str, Any]:
    """
    Extract text from a plain text file.
    """
    filename = os.path.basename(file_path)
    result = {
        "filename": filename,
        "filepath": file_path,
        "file_type": "TXT",
        "text": "",
        "pages": [],
        "num_pages": 1,
        "char_count": 0,
        "word_count": 0,
        "status": "success",
        "error_message": None,
    }

    try:
        # Try UTF-8 first, fallback to latin-1
        content = ""
        for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        combined_text = content.strip()
        result["text"] = combined_text
        result["pages"] = [{
            "page_number": 1,
            "text": combined_text,
            "char_count": len(combined_text),
            "word_count": len(combined_text.split()),
        }]
        result["num_pages"] = 1
        result["char_count"] = len(combined_text)
        result["word_count"] = len(combined_text.split())

        if not combined_text:
            result["status"] = "warning"
            result["error_message"] = "TXT file is empty."

    except Exception as e:
        result["status"] = "error"
        result["error_message"] = f"Failed to read TXT file: {str(e)}"

    return result


def load_document(file_path: str) -> Dict[str, Any]:
    """
    Universal document loader: dispatches to appropriate loader based on file extension.
    """
    if not os.path.exists(file_path):
        return {
            "filename": os.path.basename(file_path),
            "filepath": file_path,
            "file_type": "UNKNOWN",
            "text": "",
            "pages": [],
            "num_pages": 0,
            "char_count": 0,
            "word_count": 0,
            "status": "error",
            "error_message": f"File does not exist: {file_path}",
        }

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext in [".docx", ".doc"]:
        return load_docx(file_path)
    elif ext in [".txt", ".md", ".csv", ".json", ".py"]:
        return load_txt(file_path)
    else:
        return {
            "filename": os.path.basename(file_path),
            "filepath": file_path,
            "file_type": ext.upper().lstrip(".") or "UNKNOWN",
            "text": "",
            "pages": [],
            "num_pages": 0,
            "char_count": 0,
            "word_count": 0,
            "status": "error",
            "error_message": f"Unsupported file extension: '{ext}'. Supported formats are PDF, DOCX, and TXT.",
        }
