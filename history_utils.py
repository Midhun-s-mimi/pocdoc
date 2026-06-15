import os
import hashlib
import streamlit as st
from pymongo import MongoClient
from langchain_core.messages import HumanMessage, AIMessage

# ============================================================
# HIPAA COMPLIANCE: DATA DE-IDENTIFICATION
# We hash the email so the database never stores real PII (Personally Identifiable Info)
# ============================================================
def get_anonymized_user_id(email: str) -> str:
    """Converts a real email into an irreversible, anonymous ID."""
    return hashlib.sha256(email.encode('utf-8')).hexdigest()

# ============================================================
# DATABASE CONNECTION
# ============================================================
@st.cache_resource
def get_db_client():
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI not found.")
    return MongoClient(uri)

def get_collection():
    client = get_db_client()
    return client.medical_assistant.chat_histories

# ============================================================
# CRUD OPERATIONS (Using Anonymized IDs)
# ============================================================
def save_chat_history(email: str, history: list):
    collection = get_collection()
    user_id = get_anonymized_user_id(email) # 🛡️ HIPAA FIX: Anonymize the ID
    
    serializable_history = []
    for msg in history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        serializable_history.append({"role": role, "content": msg.content})
    
    # Save using the anonymous ID instead of the real email
    collection.update_one(
        {"user_id": user_id},
        {"$set": {"history": serializable_history}},
        upsert=True
    )

def load_chat_history(email: str) -> list:
    collection = get_collection()
    user_id = get_anonymized_user_id(email) # 🛡️ HIPAA FIX: Hash the email to find the record
    
    doc = collection.find_one({"user_id": user_id})
    if not doc or "history" not in doc:
        return []
    
    loaded_history = []
    for msg in doc["history"]:
        if msg["role"] == "user":
            loaded_history.append(HumanMessage(content=msg["content"]))
        else:
            loaded_history.append(AIMessage(content=msg["content"]))
    return loaded_history

def clear_chat_history(email: str):
    collection = get_collection()
    user_id = get_anonymized_user_id(email) # 🛡️ HIPAA FIX
    collection.delete_one({"user_id": user_id})