import os
import time
import streamlit as st
import requests
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions
from groq import Groq, RateLimitError as GroqRateLimitError
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from streamlit_mic_recorder import mic_recorder
from history_utils import save_chat_history, load_chat_history, clear_chat_history
from file_utils import extract_file_content
from email_utils import send_emergency_email

load_dotenv()
st.set_page_config(page_title="Medical Assistant - Chat", page_icon="💬", layout="wide")

# ============================================================
# 1. AUTH GUARD
# ============================================================
if not st.session_state.get("authenticated") or "user_profile" not in st.session_state:
    st.warning("🔒 Please log in to access the consultation.")
    if st.button("Go to Login"):
        st.switch_page("app.py")
    st.stop()

profile = st.session_state.user_profile

# ============================================================
# 2. PROVIDER INITIALIZATION
# ============================================================
@st.cache_resource
def get_groq_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

@st.cache_resource
def get_groq_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, api_key=os.getenv("GROQ_API_KEY"))

@st.cache_resource
def get_gemini_client():
    key = os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=key) if key else None

groq_client = get_groq_client()
groq_llm = get_groq_llm()
gemini_client = get_gemini_client()

# ============================================================
# 3. STATE INITIALIZATION (Split into Past History & Current Session)
# ============================================================
user_email = profile.get("email", "unknown_user")

if "past_history" not in st.session_state:
    st.session_state.past_history = load_chat_history(user_email)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "conversation_summary" not in st.session_state:
    if len(st.session_state.past_history) > 0:
        recent_past = st.session_state.past_history[-4:]
        summary_text = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content[:100]}..." for m in recent_past])
        st.session_state.conversation_summary = f"Previous context:\n{summary_text}"
    else:
        st.session_state.conversation_summary = "No previous conversation."

if "initial_greeting_done" not in st.session_state:
    st.session_state.initial_greeting_done = False
if "emergency_alert_sent" not in st.session_state:
    st.session_state.emergency_alert_sent = False

# ============================================================
# 4. CONVERSATION SUMMARIZATION LOGIC
# ============================================================
def summarize_history():
    if len(st.session_state.chat_history) <= 4:
        return
    messages_to_summarize = st.session_state.chat_history[:-2]
    recent_messages = st.session_state.chat_history[-2:]
    summary_prompt = f"Summarize the following medical chat history concisely:\n{messages_to_summarize}"
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.2
        )
        st.session_state.conversation_summary = response.choices[0].message.content
        st.session_state.chat_history = recent_messages
    except Exception:
        pass
    # ============================================================
# 4.5 AUTO-EMERGENCY TRIAGE FUNCTION
# ============================================================
def check_if_critical(symptoms: str) -> tuple:
    """Asks the AI if the symptoms indicate a life-threatening emergency."""
    try:
        triage_prompt = f"""
        Act as a medical triage nurse. Analyze these symptoms: "{symptoms}".
        Respond ONLY in valid JSON format with two keys:
        - "is_critical": boolean (true ONLY if symptoms indicate a life-threatening emergency like heart attack, stroke, severe bleeding, unconsciousness, etc.)
        - "reason": string (brief medical explanation)
        """
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": triage_prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("is_critical", False), result.get("reason", "")
    except Exception as e:
        return False, "Triage check failed."

# ============================================================
# 5. AI GENERATION ROUTER
# ============================================================
def generate_response(user_input: str, file_context: str = "", image_bytes: bytes = None, mime_type: str = None):
    has_image = bool(image_bytes)
    system_template = """You are a professional AI medical assistant.
PATIENT PROFILE:
- Name: {name}
- Age: {age}
- Gender: {gender}
- Location: {location}
CONVERSATION SUMMARY:
{summary}
CURRENT FILE CONTEXT:
{file_context}
GUIDELINES:
1. Use the patient profile and summary to personalize your response.
2. If images/reports are provided, analyze them carefully.
3. NEVER prescribe specific medications.
4. ALWAYS end with: "Disclaimer: I am an AI, not a substitute for professional medical advice."
"""
    if has_image and gemini_client:
        contents = []
        for msg in st.session_state.chat_history:
            role = "user" if isinstance(msg, HumanMessage) else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.content)]))
        parts = [types.Part.from_text(text=user_input)]
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
        contents.append(types.Content(role="user", parts=parts))
        
        # FIXED: Changed to 1.5-flash to prevent 503 UNAVAILABLE errors
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=system_template.format(
                name=profile.get('name'), age=str(profile.get('age')),
                gender=profile.get('gender'), location=profile.get('location'),
                summary=st.session_state.conversation_summary,
                file_context=file_context if file_context else "No files."
            ), temperature=0.3)
        )
        return resp.text
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        chain = prompt | groq_llm | StrOutputParser()
        return chain.invoke({
            "name": profile.get('name'), "age": str(profile.get('age')),
            "gender": profile.get('gender'), "location": profile.get('location'),
            "summary": st.session_state.conversation_summary,
            "file_context": file_context if file_context else "No files.",
            "input": user_input, "history": st.session_state.chat_history
        })

# ============================================================
# 6. UI: SIDEBAR & HISTORY VIEWER
# ============================================================
with st.sidebar:
    st.markdown("### 👤 User Profile")
    st.markdown(f"**Name:**  {profile.get('name')}")
    st.markdown(f"**Age:** {profile.get('age')} | **Gender:** {profile.get('gender')}")
    st.markdown(f"**Location:** {profile.get('location')}")
    st.divider()

    # ==========================================
    # 📜 PAST HISTORY VIEWER
    # ==========================================
    st.markdown("#### 📜 Your Past Queries")
    
    # Extract only the messages sent by the user from the PAST history
    user_queries = [msg.content for msg in st.session_state.past_history if isinstance(msg, HumanMessage)]
    
    if not user_queries:
        st.caption("No past queries yet. Start chatting below!")
    else:
        with st.expander(f"🗂️ View History ({len(user_queries)} questions)", expanded=False):
            for i, query in enumerate(user_queries):
                clean_query = query.replace("\n", " ").strip()
                short_query = clean_query[:80] + "..." if len(clean_query) > 80 else clean_query
                st.markdown(f"**{i+1}.** {short_query}")
                
    st.divider()

    # --- CLEAR HISTORY BUTTON ---
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        clear_chat_history(user_email)
        st.session_state.past_history = []
        st.session_state.chat_history = []
        st.session_state.conversation_summary = "No previous conversation."
        st.session_state.initial_greeting_done = False
        st.success("Chat history cleared!")
        st.rerun()

    st.divider()

    # --- LOGOUT BUTTON ---
    if st.button("🚪 Logout", use_container_width=True):
        # Added 'emergency_alert_sent' to the cleanup list just in case
        for key in ["authenticated", "user_profile", "chat_history", "conversation_summary", "initial_greeting_done", "processed_files", "uploader_key", "pending_voice_text", "pending_voice_audio", "mic_key", "emergency_alert_sent"]:
            st.session_state.pop(key, None)
        st.switch_page("app.py")

# ============================================================
# 7. UI: INITIAL GREETING
# ============================================================
if not st.session_state.initial_greeting_done:
    greeting = f"Hello {profile.get('name')}. I see you are {profile.get('age')} years old from {profile.get('location')}. How can I assist you with your health today?"
    st.session_state.chat_history.append(AIMessage(content=greeting))
    st.session_state.initial_greeting_done = True

# ============================================================
# 8. UI: CHAT INTERFACE (Inputs First, Results Last)
# ============================================================
st.title("💬 Medical Consultation Chat")

if "last_chat_error" in st.session_state:
    st.error(f"❌ {st.session_state.last_chat_error}")
    del st.session_state.last_chat_error

# --- 1. VOICE INPUT ---
st.markdown("##### 🎤 Voice Input (Optional)")
if "pending_voice_text" not in st.session_state:
    st.session_state.pending_voice_text = ""
if "pending_voice_audio" not in st.session_state:
    st.session_state.pending_voice_audio = None
if "mic_key" not in st.session_state:
    st.session_state.mic_key = 0

audio = mic_recorder(start_prompt="🎙️ Start Recording", stop_prompt="⏹️ Stop", key=f"mic_{st.session_state.mic_key}")

if audio and "bytes" in audio:
    st.session_state.pending_voice_audio = audio["bytes"]
    with st.spinner("Transcribing voice..."):
        try:
            transcription = groq_client.audio.transcriptions.create(
                file=("audio.webm", audio["bytes"], "audio/webm"),
                model="whisper-large-v3", response_format="text"
            )
            clean_text = transcription.strip().lower().replace(".", "").replace(",", "")
            if clean_text in ["thank you", "thanks", "bye", ""] or len(clean_text) < 3:
                st.warning("⚠️ Silence detected. Please try again.")
                st.session_state.pending_voice_text = ""
                st.session_state.pending_voice_audio = None
            else:
                st.session_state.pending_voice_text = transcription
                st.success("✅ Transcribed! Review the audio and text below.")
        except Exception as e:
            st.error(f"Transcription failed: {e}")
    st.session_state.mic_key += 1
    st.rerun()

send_voice_btn = False
edited_text = ""
if st.session_state.pending_voice_text or st.session_state.pending_voice_audio:
    st.info("🎧 **Review your voice input:**")
    if st.session_state.pending_voice_audio:
        st.audio(st.session_state.pending_voice_audio, format="audio/webm")
    edited_text = st.text_area("📝 Edit transcribed text if needed (Text Format):", value=st.session_state.pending_voice_text, height=100, key="voice_text_area")
    col_send, col_clear = st.columns([1, 4])
    with col_send:
        send_voice_btn = st.button("📤 Send Voice Message", type="primary")
    with col_clear:
        if st.button("❌ Discard"):
            st.session_state.pending_voice_text = ""
            st.session_state.pending_voice_audio = None
            st.rerun()

# --- 2. FILE UPLOAD ---
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

st.markdown("##### 📎 Attach Files")
uploaded_files = st.file_uploader("Upload medical reports", type=["pdf", "docx", "txt", "png", "jpg", "jpeg"], accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}")

if uploaded_files:
    for f in uploaded_files:
        if not any(pf["name"] == f.name for pf in st.session_state.processed_files):
            extracted = extract_file_content(f)
            st.session_state.processed_files.append({"name": f.name, "text": extracted.get("text", ""), "image_bytes": extracted.get("image_bytes"), "mime_type": extracted.get("mime_type")})
    st.session_state.uploader_key += 1
    st.rerun()

file_context = ""
img_bytes, img_mime = None, None
analyze_btn = False

if st.session_state.processed_files:
    st.info(f"📎 **Currently Attached:** {', '.join([f['name'] for f in st.session_state.processed_files])}")
    for pf in st.session_state.processed_files:
        if pf["text"] and not pf["text"].startswith("[Unsupported"):
            file_context += f"\n--- File: {pf['name']} ---\n{pf['text']}"
        if pf["image_bytes"]:
            img_bytes = pf["image_bytes"]
            img_mime = pf["mime_type"]
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_btn = st.button("🔍 Analyze Files", type="primary")
    with col2:
        if st.button("🗑️ Clear Files"):
            st.session_state.processed_files = []
            st.rerun()

# --- 3. TEXT CHAT INPUT ---
st.markdown("##### ⌨️ Or Type Your Message")
prompt_text = st.chat_input("Type your symptoms or questions here...")

st.divider()

# --- 4. CHAT HISTORY DISPLAY ---
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# ============================================================
# PROCESS SUBMISSION LOGIC (WITH AUTO-EMERGENCY TRIAGE)
# ============================================================
final_text_input = ""

if send_voice_btn:
    final_text_input = edited_text.strip()
    st.session_state.pending_voice_text = ""
    st.session_state.pending_voice_audio = None
elif prompt_text or analyze_btn:
    if analyze_btn and not prompt_text:
        final_text_input = "Please analyze the attached files and provide a summary of the medical findings."
    else:
        final_text_input = prompt_text.strip()

if final_text_input:
    
    # 🚨 AUTO-EMERGENCY TRIAGE CHECK 🚨
    if not st.session_state.emergency_alert_sent:
        with st.spinner("🩺 Running emergency triage check..."):
            is_critical, triage_reason = check_if_critical(final_text_input)
        
        if is_critical:
            st.session_state.emergency_alert_sent = True # Prevent duplicate alerts
            
            # 1. Show massive warning to the user
            st.error(f"🚨 **CRITICAL CONDITION DETECTED**: {triage_reason}")
            st.warning("🚑 **Emergency alert is being AUTOMATICALLY sent to the nearest hospital!**")
            
            # 2. Automatically locate hospital and send email
            try:
                location = profile.get('location', 'Unknown') if profile else 'Unknown'
                patient_name = profile.get('name', 'Patient') if profile else 'Patient'
                
                headers = {'User-Agent': 'MedicalAssistantApp/1.0'}
                url = f"https://nominatim.openstreetmap.org/search?format=json&q=hospital+near+{location}&limit=1&extratags=1"
                geo_response = requests.get(url, headers=headers).json()
                
                                # 🚨 FIXED: Safely handle None values from the OpenStreetMap API
                if geo_response and len(geo_response) > 0:
                    # 'or {}' guarantees we never try to call .get() on a None value
                    extratags = geo_response[0].get("extratags") or {}
                    address = geo_response[0].get("address") or {}
                    
                    hospital_name = geo_response[0].get("display_name", "Local Hospital")
                    hospital_address = f"{address.get('road', '')} {address.get('city', location)}".strip()
                    hospital_email = extratags.get("contact:email") or os.getenv("DEFAULT_EMERGENCY_EMAIL", "hospital@example.com")
                else:
                    hospital_name = "Regional Emergency Hospital"
                    hospital_address = location
                    hospital_email = os.getenv("DEFAULT_EMERGENCY_EMAIL", "hospital@example.com")
                
                # 🚨 DEBUG LINES (Keep these to see exactly what is happening)
                st.info(f"🔍 DEBUG: Receiver Email is: {hospital_email}")
                st.info(f"🔍 DEBUG: Sender Email is: {os.getenv('SENDER_EMAIL')}")

                # Send the email automatically!
                success, msg = send_emergency_email(
                    patient_name, final_text_input, location, hospital_name, hospital_address, hospital_email
                )
                
                if success:
                    st.success(f"✅ AUTOMATIC ALERT SENT TO: {hospital_name}")
                else:
                    st.error(f"❌ Auto-alert email failed: {msg}")
                    
            except Exception as e:
                # This is the missing 'except' block that was causing your SyntaxError
                st.error(f"❌ Location lookup or email failed: {e}")
            
            st.divider() # Visual separator before the AI response

    # --- NORMAL AI RESPONSE GENERATION ---
    with st.chat_message("user"):
        st.write(final_text_input)

    response_text = ""
    error_msg = ""
    
    with st.chat_message("assistant"):
        with st.spinner("🧠 Analyzing with Llama-3 / Gemini..."):
            try:
                response_text = generate_response(final_text_input, file_context, img_bytes, img_mime)
                st.write(response_text)
            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ Error: {error_msg}")

    # 1. Add to current session history (Main Chat UI)
    st.session_state.chat_history.append(HumanMessage(content=final_text_input))
    if response_text:
        st.session_state.chat_history.append(AIMessage(content=response_text))
    elif error_msg:
        st.session_state.chat_history.append(AIMessage(content=f"⚠️ Failed to generate response: {error_msg}"))

    # 2. Add to persistent past history (Sidebar & AI Context)
    st.session_state.past_history.append(HumanMessage(content=final_text_input))
    if response_text:
        st.session_state.past_history.append(AIMessage(content=response_text))
        
    # 3. Save the updated past history to the JSON file
    save_chat_history(user_email, st.session_state.past_history)
    
    # 4. Prevent infinite file growth: Keep only the last 10 messages
    if len(st.session_state.past_history) > 10:
        st.session_state.past_history = st.session_state.past_history[-10:]
        save_chat_history(user_email, st.session_state.past_history)
    
    summarize_history()
    
    if error_msg:
        st.session_state.last_chat_error = error_msg
    st.rerun()