"""
SQLite Conversation Database Module for LocalGPT (Phase 2 - Step 7).
Provides local persistent storage for multi-turn conversations, messages, and RAG source metadata.
"""

import os
import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional


DEFAULT_DB_DIR = os.path.join("data", "conversations")
DEFAULT_DB_PATH = os.path.join(DEFAULT_DB_DIR, "localgpt.db")


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Get a connection to the SQLite database with row factory for dictionary-like access.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_database(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Initialize SQLite database tables for conversations and messages.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # 1. Conversations table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # 2. Messages table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            message_order INTEGER NOT NULL,
            feedback TEXT,
            sources_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
        """)
        
        # Create index on conversation_id for fast lookup
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_conversation_id 
        ON messages(conversation_id)
        """)
        
        conn.commit()


def create_conversation(
    title: str = "New Chat",
    conversation_id: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> str:
    """
    Create a new conversation record and return its unique conversation_id.
    """
    init_database(db_path)
    conv_id = conversation_id or str(uuid.uuid4())
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conv_id, title, now_str, now_str)
        )
        conn.commit()

    return conv_id


def get_conversations(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """
    Get all conversation records ordered by most recently updated.
    """
    init_database(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM conversations ORDER BY updated_at DESC, rowid DESC")
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_conversation(
    conversation_id: str,
    db_path: str = DEFAULT_DB_PATH
) -> Optional[Dict[str, Any]]:
    """
    Get a single conversation record by ID.
    """
    init_database(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def update_conversation_title(
    conversation_id: str,
    title: str,
    db_path: str = DEFAULT_DB_PATH
) -> bool:
    """
    Update the title of a conversation.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, now_str, conversation_id)
        )
        conn.commit()
        return cursor.rowcount > 0


def update_conversation_timestamp(
    conversation_id: str,
    db_path: str = DEFAULT_DB_PATH
) -> None:
    """
    Update the updated_at timestamp of a conversation to current time.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now_str, conversation_id)
        )
        conn.commit()


def delete_conversation(
    conversation_id: str,
    db_path: str = DEFAULT_DB_PATH
) -> bool:
    """
    Delete a conversation and all its associated messages.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
        return cursor.rowcount > 0


def clear_all_conversations(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Clear all conversations and messages from the database.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages")
        cursor.execute("DELETE FROM conversations")
        conn.commit()


def save_message(
    conversation_id: str,
    role: str,
    content: str,
    feedback: Optional[str] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """
    Save a message into the database for the given conversation.
    """
    init_database(db_path)
    sources_json = json.dumps(sources) if sources else None
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Calculate next message_order
        cursor.execute(
            "SELECT COALESCE(MAX(message_order), 0) + 1 FROM messages WHERE conversation_id = ?",
            (conversation_id,)
        )
        next_order = cursor.fetchone()[0]

        cursor.execute("""
        INSERT INTO messages (
            conversation_id, role, content, message_order, feedback, sources_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (conversation_id, role, content, next_order, feedback, sources_json, now_str))

        msg_id = cursor.lastrowid

        # Update conversation updated_at
        cursor.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now_str, conversation_id)
        )

        conn.commit()
        return msg_id


def get_conversation_messages(
    conversation_id: str,
    db_path: str = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    """
    Retrieve all messages for a conversation ordered chronologically.
    Decodes sources_json back into a list of dicts.
    """
    init_database(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY message_order ASC",
            (conversation_id,)
        )
        rows = cursor.fetchall()
        
        messages = []
        for r in rows:
            msg_dict = dict(r)
            if msg_dict.get("sources_json"):
                try:
                    msg_dict["sources"] = json.loads(msg_dict["sources_json"])
                except Exception:
                    msg_dict["sources"] = None
            else:
                msg_dict["sources"] = None
            messages.append(msg_dict)
            
        return messages


def update_last_assistant_message(
    conversation_id: str,
    new_content: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    db_path: str = DEFAULT_DB_PATH
) -> bool:
    """
    Update the content and sources of the most recent assistant message (e.g. for Regenerate).
    """
    sources_json = json.dumps(sources) if sources else None
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        # Find ID of the last assistant message
        cursor.execute("""
        SELECT id FROM messages 
        WHERE conversation_id = ? AND role = 'assistant' 
        ORDER BY message_order DESC LIMIT 1
        """, (conversation_id,))
        row = cursor.fetchone()
        
        if not row:
            return False

        last_id = row["id"]
        cursor.execute("""
        UPDATE messages 
        SET content = ?, sources_json = ?, created_at = ? 
        WHERE id = ?
        """, (new_content, sources_json, now_str, last_id))

        cursor.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now_str, conversation_id)
        )

        conn.commit()
        return True


def remove_last_assistant_message(
    conversation_id: str,
    db_path: str = DEFAULT_DB_PATH
) -> bool:
    """
    Delete the most recent assistant message for the conversation.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT id FROM messages 
        WHERE conversation_id = ? AND role = 'assistant' 
        ORDER BY message_order DESC LIMIT 1
        """, (conversation_id,))
        row = cursor.fetchone()
        
        if not row:
            return False

        last_id = row["id"]
        cursor.execute("DELETE FROM messages WHERE id = ?", (last_id,))
        conn.commit()
        return True


def edit_user_message_and_truncate(
    conversation_id: str,
    message_order: int,
    new_content: str,
    db_path: str = DEFAULT_DB_PATH
) -> bool:
    """
    Update a user message at message_order with new_content and delete all subsequent messages.
    """
    init_database(db_path)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clean_content = new_content.strip()

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # 1. Update the target message
        cursor.execute("""
        UPDATE messages 
        SET content = ?, created_at = ? 
        WHERE conversation_id = ? AND message_order = ?
        """, (clean_content, now_str, conversation_id, message_order))
        
        # 2. Delete all subsequent messages in this conversation
        cursor.execute("""
        DELETE FROM messages 
        WHERE conversation_id = ? AND message_order > ?
        """, (conversation_id, message_order))
        
        # 3. Update conversation updated_at
        cursor.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now_str, conversation_id)
        )
        
        # 4. If this was the first message (message_order == 1), auto-update title if it's default
        if message_order == 1:
            cursor.execute("SELECT title FROM conversations WHERE id = ?", (conversation_id,))
            conv_row = cursor.fetchone()
            if conv_row:
                first_line = clean_content.split("\n")[0].strip()
                auto_title = first_line[:48].strip() + ("..." if len(first_line) > 48 else "")
                cursor.execute(
                    "UPDATE conversations SET title = ? WHERE id = ?",
                    (auto_title or "Chat", conversation_id)
                )
            
        conn.commit()
        return True


def clear_conversation_messages(
    conversation_id: str,
    db_path: str = DEFAULT_DB_PATH
) -> bool:
    """
    Delete all messages belonging to a conversation while keeping the conversation entry.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        cursor.execute(
            "UPDATE conversations SET title = 'New Chat', updated_at = ? WHERE id = ?",
            (now_str, conversation_id)
        )
        conn.commit()
        return True


def format_timestamp(dt_str: Optional[str]) -> str:
    """
    Format SQLite timestamp (e.g. '2026-08-31 12:10:00') into 'Aug 31, 2026 12:10 PM'.
    """
    if not dt_str:
        return "N/A"
    try:
        clean_str = str(dt_str).replace("T", " ")
        if "." in clean_str:
            clean_str = clean_str.split(".")[0]
        dt = datetime.strptime(clean_str, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%b %d, %Y %I:%M %p")
    except Exception:
        return str(dt_str)
