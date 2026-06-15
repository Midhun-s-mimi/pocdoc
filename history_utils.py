import os
import json
from langchain_core.messages import HumanMessage, AIMessage

HISTORY_FILE = "chat_history.json"

def _load_all_history():
    if not os.path.exists(HISTORY_FILE):
        return {}
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_all_history(all_history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(all_history, f, indent=4)

def save_chat_history(email: str, history: list):
    """Saves the chat history for a specific user email."""
    all_history = _load_all_history()
    
    # Convert LangChain message objects to simple dictionaries for JSON storage
    serializable_history = []
    for msg in history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        serializable_history.append({"role": role, "content": msg.content})
    
    all_history[email] = serializable_history
    _save_all_history(all_history)

def load_chat_history(email: str) -> list:
    """Loads the chat history for a specific user email."""
    all_history = _load_all_history()
    if email not in all_history:
        return []
    
    loaded_history = []
    for msg in all_history[email]:
        if msg["role"] == "user":
            loaded_history.append(HumanMessage(content=msg["content"]))
        else:
            loaded_history.append(AIMessage(content=msg["content"]))
    return loaded_history

def clear_chat_history(email: str):
    """Deletes the saved chat history for a specific user."""
    all_history = _load_all_history()
    if email in all_history:
        del all_history[email]
        _save_all_history(all_history)