"""
Conversation Memory Management for LocalGPT (Phase 2 - Step 7: Persistence + Database).
Coordinates fast in-memory Streamlit state with local SQLite persistent storage.
"""

from typing import List, Dict, Any, Optional
import streamlit as st

import database


SESSION_KEY = "localgpt_conversation_history"
ACTIVE_CONV_KEY = "active_conversation_id"
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant.\nExplain concepts clearly with examples."


def init_memory() -> None:
    """
    Initialize conversation memory and ensure an active conversation is loaded from SQLite.
    """
    database.init_database()

    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT

    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = []

    if ACTIVE_CONV_KEY not in st.session_state or not st.session_state[ACTIVE_CONV_KEY]:
        # Check for existing conversations in database
        convs = database.get_conversations()
        if convs:
            # Load the most recent conversation
            latest_id = convs[0]["id"]
            st.session_state[ACTIVE_CONV_KEY] = latest_id
            st.session_state[SESSION_KEY] = database.get_conversation_messages(latest_id)
        else:
            # Create a brand new conversation
            new_id = database.create_conversation(title="New Chat")
            st.session_state[ACTIVE_CONV_KEY] = new_id
            st.session_state[SESSION_KEY] = []


def get_active_conversation_id() -> str:
    """Get the active conversation ID."""
    init_memory()
    return st.session_state[ACTIVE_CONV_KEY]


def switch_conversation(conversation_id: str) -> None:
    """
    Switch active conversation, loading its messages from SQLite into session state.
    """
    init_memory()
    st.session_state[ACTIVE_CONV_KEY] = conversation_id
    st.session_state[SESSION_KEY] = database.get_conversation_messages(conversation_id)


def start_new_conversation(title: str = "New Chat") -> str:
    """
    Start a fresh conversation and set it as active.
    """
    init_memory()
    new_id = database.create_conversation(title=title)
    st.session_state[ACTIVE_CONV_KEY] = new_id
    st.session_state[SESSION_KEY] = []
    return new_id


def rename_conversation(conversation_id: str, new_title: str) -> bool:
    """
    Rename a conversation in SQLite if new_title is non-empty.
    """
    init_memory()
    clean_title = new_title.strip() if new_title else ""
    if not clean_title:
        return False
    return database.update_conversation_title(conversation_id, clean_title)


def delete_conversation_and_switch(conversation_id: str) -> None:
    """
    Delete a conversation from SQLite. If it's the active conversation, switch to another or create a new one.
    """
    init_memory()
    database.delete_conversation(conversation_id)
    
    if st.session_state.get(ACTIVE_CONV_KEY) == conversation_id:
        convs = database.get_conversations()
        if convs:
            switch_conversation(convs[0]["id"])
        else:
            start_new_conversation("New Chat")


def get_messages() -> List[Dict[str, Any]]:
    """Retrieve the current list of conversation messages."""
    init_memory()
    return st.session_state[SESSION_KEY]


def add_user_message(content: str) -> None:
    """
    Add a user message to in-memory conversation and persist to SQLite.
    Auto-updates conversation title on first message.
    """
    init_memory()
    conv_id = st.session_state[ACTIVE_CONV_KEY]
    clean_content = content.strip()

    # 1. Save to in-memory state
    msg_obj = {
        "id": len(st.session_state[SESSION_KEY]),
        "role": "user",
        "content": clean_content,
    }
    st.session_state[SESSION_KEY].append(msg_obj)

    # 2. Persist to SQLite
    database.save_message(
        conversation_id=conv_id,
        role="user",
        content=clean_content
    )

    # 3. Auto-generate title from first user prompt if current title is default
    current_conv = database.get_conversation(conv_id)
    if current_conv and current_conv.get("title") in ["New Chat", ""]:
        first_line = clean_content.split("\n")[0].strip()
        auto_title = first_line[:48].strip() + ("..." if len(first_line) > 48 else "")
        database.update_conversation_title(conv_id, auto_title or "Chat")


def add_assistant_message(
    content: str,
    feedback: Optional[str] = None,
    sources: Optional[List[Dict[str, Any]]] = None
) -> None:
    """
    Add a complete assistant response to in-memory conversation and persist to SQLite.
    """
    init_memory()
    conv_id = st.session_state[ACTIVE_CONV_KEY]

    # 1. Save to in-memory state
    msg_obj = {
        "id": len(st.session_state[SESSION_KEY]),
        "role": "assistant",
        "content": content,
        "feedback": feedback,
        "sources": sources,
    }
    st.session_state[SESSION_KEY].append(msg_obj)

    # 2. Persist to SQLite
    database.save_message(
        conversation_id=conv_id,
        role="assistant",
        content=content,
        feedback=feedback,
        sources=sources
    )


def remove_last_assistant_message() -> Optional[Dict[str, Any]]:
    """
    Remove the most recent assistant message from in-memory state and SQLite (for Regenerate).
    """
    init_memory()
    conv_id = st.session_state[ACTIVE_CONV_KEY]

    # 1. Remove from SQLite
    database.remove_last_assistant_message(conv_id)

    # 2. Remove from session state
    messages = st.session_state[SESSION_KEY]
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            return messages.pop(i)
    return None


def edit_user_message(message_index: int, new_content: str) -> None:
    """
    Edit a user message at message_index, truncate all subsequent messages in memory and SQLite.
    """
    init_memory()
    conv_id = st.session_state[ACTIVE_CONV_KEY]
    clean_content = new_content.strip()

    messages = st.session_state[SESSION_KEY]
    if message_index < 0 or message_index >= len(messages):
        return

    # 1. Update in-memory user message and truncate subsequent messages
    messages[message_index]["content"] = clean_content
    st.session_state[SESSION_KEY] = messages[:message_index + 1]

    # 2. Persist edit and truncation in SQLite
    message_order = message_index + 1
    if "message_order" in messages[message_index] and messages[message_index]["message_order"] is not None:
        message_order = messages[message_index]["message_order"]

    database.edit_user_message_and_truncate(
        conversation_id=conv_id,
        message_order=message_order,
        new_content=clean_content
    )


def clear_messages() -> None:
    """
    Clear all messages for the active conversation in memory and SQLite.
    """
    init_memory()
    conv_id = st.session_state[ACTIVE_CONV_KEY]
    database.clear_conversation_messages(conv_id)
    st.session_state[SESSION_KEY] = []


def get_qwen_chat_messages(
    system_prompt: Optional[str] = None,
    latest_user_override: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Format conversation history into standard chat role/content dictionaries
    including the system prompt for Qwen tokenization.
    """
    init_memory()
    sys_prompt = system_prompt if system_prompt is not None else st.session_state.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    
    formatted_messages = []
    if sys_prompt and sys_prompt.strip():
        formatted_messages.append({"role": "system", "content": sys_prompt.strip()})
    
    msgs = st.session_state[SESSION_KEY]
    for i, msg in enumerate(msgs):
        role = msg.get("role")
        content = msg.get("content")
        
        if role == "user" and latest_user_override and i == len(msgs) - 1:
            content = latest_user_override

        if role in ["user", "assistant"] and content:
            formatted_messages.append({"role": role, "content": content})
            
    return formatted_messages
