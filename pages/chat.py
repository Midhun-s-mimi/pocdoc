import os
import time
import streamlit as st
import requests
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
# 3. STATE INITIALIZATION
# ============================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = "No previous conversation."
if "initial_greeting_done" not in st.session_state:
    st.session_state.initial_greeting_done = False

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
# 6. UI: SIDEBAR & EMERGENCY
# ============================================================
with st.sidebar:
    st.markdown(f"### 👤 {profile.get('name')}")
    st.markdown(f"**Age:** {profile.get('age')} | **Gender:** {profile.get('gender')}")
    st.markdown(f"**Location:** {profile.get('location')}")
    st.divider()
    st.markdown("### 🚨 Emergency")
    if st.button("Send Emergency Alert", type="primary", use_container_width=True):
        consent = st.checkbox("I consent to sharing my details.")
        if consent:
            with st.spinner("Locating hospital..."):
                try:
                    location = profile.get('location', '')
                    headers = {'User-Agent': 'MedicalAssistantApp/1.0'}
                    url = f"https://nominatim.openstreetmap.org/search?format=json&q=hospital+near+{location}&limit=1&extratags=1"
                    geo_response = requests.get(url, headers=headers).json()
                    hospital_name = geo_response[0].get("display_name", "Local Hospital") if geo_response else "Regional Hospital"
                    hospital_email = geo_response[0].get("extratags", {}).get("contact:email", os.getenv("DEFAULT_EMERGENCY_EMAIL", "hospital@example.com")) if geo_response else os.getenv("DEFAULT_EMERGENCY_EMAIL")
                    success, msg = send_emergency_email(profile.get('name'), st.session_state.conversation_summary, location, hospital_name, location, hospital_email)
                    if success:
                        st.success(f"✅ Alert sent to: {hospital_name}")
                    else:
                        st.error(f"Email failed: {msg}")
                except Exception as e:
                    st.error(f"Location lookup failed: {e}")
    st.divider()
    if st.button("🚪 Logout", use_container_width=True):
        for key in ["authenticated", "user_profile", "chat_history", "conversation_summary", "initial_greeting_done", "processed_files", "uploader_key", "pending_voice_text", "pending_voice_audio", "mic_key"]:
            st.session_state.pop(key, None)
        st.switch_page("app.py")

# ============================================================
# 7. UI: INITIAL GREETING
# ============================================================
if not st.session_state.initial_greeting_done:
    with st.spinner("🤖 Preparing your personalized consultation..."):
        greeting = f"Hello {profile.get('name')}. I see you are {profile.get('age')} years old from {profile.get('location')}. How can I assist you with your health today?"
        st.session_state.chat_history.append(AIMessage(content=greeting))
        st.session_state.initial_greeting_done = True
        st.rerun()

# ============================================================
# 8. UI: CHAT INTERFACE (Inputs First, Results Last)
# ============================================================
st.title("💬 Medical Consultation Chat")

if "last_chat_error" in st.session_state:
    st.error(f"❌ {st.session_state.last_chat_error}")
    del st.session_state.last_chat_error

# ---------------------------------------------------------
# 1. VOICE INPUT (Audio Player + Editable Text Box)
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 2. FILE UPLOAD (Persistent Memory)
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 3. TEXT CHAT INPUT
# ---------------------------------------------------------
st.markdown("##### ⌨️ Or Type Your Message")
prompt_text = st.chat_input("Type your symptoms or questions here...")

st.divider()

# ---------------------------------------------------------
# 4. CHAT HISTORY (Displayed at the bottom)
# ---------------------------------------------------------
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# ============================================================
# PROCESS SUBMISSION LOGIC
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

    st.session_state.chat_history.append(HumanMessage(content=final_text_input))
    if response_text:
        st.session_state.chat_history.append(AIMessage(content=response_text))
    elif error_msg:
        st.session_state.chat_history.append(AIMessage(content=f"⚠️ Failed to generate response: {error_msg}"))

    summarize_history()
    if error_msg:
        st.session_state.last_chat_error = error_msg
    st.rerun()
