import os
import time
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions
from groq import RateLimitError as GroqRateLimitError
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from file_utils import extract_file_content

load_dotenv()
st.set_page_config(page_title="Medical Assistant - Chat", page_icon="💬")

if "patient_data" not in st.session_state:
    st.error("❌ No patient data found. Please return to the input page.")
    if st.button("Go to Input Page"):
        st.switch_page("app.py")
    st.stop()

patient_data = st.session_state.patient_data

BASE_SYSTEM_PROMPT = """You are a professional AI medical assistant.
PATIENT CONTEXT:
Age: {age}
Gender: {gender}
Initial Symptoms: {symptoms}
Duration: {days_suffering} days
Pain Level: {pain_level}
{report_section}

GUIDELINES:
Use ALL provided context including uploaded documents and images.
For images (X-rays, lab results, charts), describe relevant findings carefully and note limitations.
Never prescribe specific medications or dosages.
If pain is HIGH or emergency signs present, advise immediate medical attention.
Keep responses concise, empathetic, and professional.
ALWAYS send your last response with a disclaimer that you are an AI, NOT a substitute for professional medical advice.
"""

INITIAL_PROMPT = (
    "Based on all patient context (including any attached report/image), "
    "provide an initial health consultation. Do not ask user to repeat information."
)

@st.cache_resource
def get_groq_llm():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        st.error("❌ GROQ_API_KEY missing"); st.stop()
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, groq_api_key=key)

@st.cache_resource
def get_gemini_client():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    return genai.Client(api_key=key)

groq_llm = get_groq_llm()
gemini_client = get_gemini_client()

report_text = patient_data.get("report_text", "")
report_section = f"\nMEDICAL REPORT EXTRACT:\n{report_text}" if report_text.strip() else ""

# ✅ FIX: Removed trailing spaces in LangChain roles and variables
groq_prompt = ChatPromptTemplate.from_messages([
    ("system", BASE_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
]).partial(
    age=patient_data["age"], gender=patient_data["gender"],
    symptoms=patient_data["symptoms"], days_suffering=patient_data["days_suffering"],
    pain_level=patient_data["pain_level"], report_section=report_section
)
groq_chain = groq_prompt | groq_llm | StrOutputParser()

GEMINI_SYSTEM_INSTRUCTION = BASE_SYSTEM_PROMPT.format(
    age=patient_data["age"], gender=patient_data["gender"],
    symptoms=patient_data["symptoms"], days_suffering=patient_data["days_suffering"],
    pain_level=patient_data["pain_level"], report_section=report_section
)

def get_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    return st.session_state.chat_history

def build_gemini_contents(user_input, image_bytes=None, mime_type=None, include_initial_image=False):
    contents = []
    if include_initial_image and patient_data.get("report_image_bytes"):
        contents.append(types.Part.from_bytes(
            data=patient_data["report_image_bytes"],
            mime_type=patient_data["report_mime_type"]
        ))
    
    for msg in get_chat_history():
        role = "user" if isinstance(msg, HumanMessage) else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.content)]))
        
    parts = [types.Part.from_text(text=user_input)]
    if image_bytes and mime_type:
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
    contents.append(types.Content(role="user", parts=parts))
    return contents

def call_gemini(user_input, image_bytes=None, mime_type=None, include_initial_image=False):
    if gemini_client is None:
        return ("⚠️ Image analysis requires a Gemini API key. "
                "Please add GEMINI_API_KEY to your .env file, or describe the image content in text.")
    
    contents = build_gemini_contents(user_input, image_bytes, mime_type, include_initial_image)
    resp = gemini_client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=GEMINI_SYSTEM_INSTRUCTION, temperature=0.3
        )
    )
    return resp.text

def generate_hybrid(user_input, image_bytes=None, mime_type=None,
                    include_initial_image=False, max_retries=3):
    use_gemini = bool(image_bytes) or include_initial_image
    for attempt in range(max_retries):
        try:
            if use_gemini:
                return call_gemini(user_input, image_bytes, mime_type, include_initial_image)
            else:
                return groq_chain.invoke({
                    "input": user_input,
                    "history": get_chat_history()[:-1] if not include_initial_image else []
                })
        except (google_exceptions.ResourceExhausted, GroqRateLimitError):
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                provider = "Gemini" if use_gemini else "Groq"
                st.warning(f"⏳ {provider} rate limited. Retrying in {wait}s... ({attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
        except Exception:
            raise

def add_to_history(role: str, content: str):
    msg = HumanMessage(content=content) if role == "user" else AIMessage(content=content)
    st.session_state.chat_history.append(msg)

if "initial_consultation_done" not in st.session_state:
    has_img = bool(patient_data.get("has_initial_image"))
    provider_label = "Gemini (image)" if has_img else "Llama 3.3 70B"
    with st.spinner(f"Generating initial consultation via {provider_label}..."):
        try:
            initial_response = generate_hybrid(
                INITIAL_PROMPT, include_initial_image=has_img
            )
            add_to_history("assistant", initial_response)
            st.session_state.initial_consultation_done = True
        except (google_exceptions.ResourceExhausted, GroqRateLimitError):
            st.error("❌ API quota exhausted. Please wait ~1 minute and refresh.")
        except Exception as e:
            st.error(f"Error: {e}")

# ✅ FIX: Removed the infinite st.rerun() loop
if st.session_state.get("initial_consultation_done"):
    pass 

st.title("💬 Medical Consultation Chat")
has_report = bool(report_text.strip()) or bool(patient_data.get("report_image_bytes"))
st.caption(
    f"Patient: {patient_data['age']}y/o {patient_data['gender']} | "
    f"Pain: {patient_data['pain_level']} | "
    f"Duration: {patient_data['days_suffering']}d "
    + (" | 📎 Report/Image attached" if has_report else "")
    + " | 🦙 Llama + 🔷 Gemini Hybrid"
)

for msg in get_chat_history():
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

st.divider()

uploaded_files = st.file_uploader(
    "📎 Attach documents or images (Optional)",
    type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

prompt_text = st.chat_input("Ask a follow-up question or describe the attached files...")

if prompt_text or uploaded_files:
    combined_context, img_bytes, img_mime = "", None, None
    
    if uploaded_files:
        for f in uploaded_files:
            extracted = extract_file_content(f)
            # ✅ FIX: Removed trailing spaces in dictionary lookups
            if extracted.get("is_image"):
                img_bytes = extracted.get("image_bytes")
                img_mime = extracted.get("mime_type")
            if extracted.get("text") and not str(extracted["text"]).startswith("[Unsupported"):
                combined_context += f"\n\n[{f.name}]:\n{extracted['text']}"

    full_input = (prompt_text or "Please analyze the attached files and provide insights based on the patient context.") + combined_context

    with st.chat_message("user"):
        if prompt_text:
            st.write(prompt_text)
        if uploaded_files:
            labels = []
            for f in uploaded_files:
                ext = f.name.rsplit(".", 1)[-1].lower() if "." in f.name else ""
                is_img = ext in ["png", "jpg", "jpeg"]
                labels.append(f"{f.name} {'🖼️' if is_img else '📄'}")
            st.caption(f"📎 Attached: {', '.join(labels)}")

    add_to_history("user", full_input)

    route = "🔷 Gemini" if img_bytes else "🦙 Llama"
    with st.chat_message("assistant"):
        with st.spinner(f"Analyzing via {route}..."):
            try:
                response = generate_hybrid(full_input, img_bytes, img_mime)
                st.write(response)
                add_to_history("assistant", response)
            except (google_exceptions.ResourceExhausted, GroqRateLimitError):
                st.error("❌ Rate limit exceeded. Wait ~1 min or upgrade tier.")
            except Exception as e:
                st.error(f"Error: {e}")

st.divider()
if st.button("🔄 New Consultation"):
    for k in ["patient_data", "chat_history", "initial_consultation_done"]:
        st.session_state.pop(k, None)
    st.switch_page("app.py")