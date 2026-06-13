import os
import io
import json
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from streamlit_mic_recorder import mic_recorder
import requests
from email_utils import send_emergency_email

load_dotenv()
st.set_page_config(page_title="Medical Assistant - Input", page_icon="🏥")

# Initialize Groq Client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.title("🏥 Medical Assistant - Patient Intake")
st.write("Please provide your details. You can type or use your voice.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎤 Voice or Text Input")
    
    # Use the stable, native Streamlit mic recorder
    audio_state = mic_recorder(
        start_prompt="🎙️ Start Recording",
        stop_prompt="⏹️ Stop & Transcribe",
        key="native_recorder"
    )
    
    # Process the recorded audio
    if audio_state:
        with st.spinner("🔄 Transcribing audio..."):
            try:
                # ✅ CRITICAL FIXES APPLIED HERE:
                # 1. Wrapped in io.BytesIO() so Groq can read the stream
                # 2. ZERO trailing spaces in any string
                # 3. language="ta" locks it to Tamil to prevent "hello everyone" hallucinations
                transcription = groq_client.audio.transcriptions.create(
                    file=("audio.webm", io.BytesIO(audio_state["bytes"]), "audio/webm"),
                    model="whisper-large-v3",
                    response_format="text",
                    language="ta"  # Change to "en" if you are testing in English
                )
                st.success("✅ Transcription successful!")
                st.session_state["transcribed_text"] = transcription
            except Exception as e:
                st.error(f"❌ Transcription failed: {e}")

    # Text Input (pre-filled if voice was used)
    default_text = st.session_state.get("transcribed_text", "")
    symptoms = st.text_area(
        "Describe your symptoms:", 
        value=default_text, 
        height=150,
        help="You can edit the transcribed text here if needed."
    )

with col2:
    st.subheader("📍 Patient Details & Location")
    patient_name = st.text_input("Full Name")
    age = st.text_input("Age")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    pain_level = st.slider("Pain Level (1-10)", 1, 10, 5)
    days_suffering = st.number_input("Days suffering", min_value=0, max_value=365, value=1)
    location = st.text_input(
        "Your City or Pincode (e.g., 'Chennai' or '600001')", 
        help="Used to find the nearest hospital in case of emergency."
    )

st.divider()

if st.button("🔍 Analyze Symptoms & Proceed", type="primary"):
    if not symptoms or not location or not patient_name:
        st.warning("⚠️ Please fill in your name, location, and symptoms.")
    else:
        with st.spinner("🧠 Analyzing for critical conditions..."):
            try:
                triage_prompt = f"""
Act as a medical triage nurse. Analyze these symptoms: "{symptoms}".
Respond ONLY in valid JSON format with two keys:
- "is_critical": boolean (true if symptoms indicate a medical emergency)
- "reason": string (brief explanation)
"""
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": triage_prompt}],
                    response_format={"type": "json_object"}
                )
                
                triage_result = json.loads(response.choices[0].message.content)
                is_critical = triage_result.get("is_critical", False)
                reason = triage_result.get("reason", "")

                if is_critical:
                    st.error(f"⚠️ **CRITICAL ALERT**: {reason}")
                    st.warning("Your symptoms require immediate medical attention.")
                    
                    consent = st.checkbox("✅ I consent to sharing my details with a nearby hospital for emergency assistance.")
                    
                    if consent:
                        with st.spinner("📍 Locating nearest hospital and sending alert..."):
                            try:
                                headers = {'User-Agent': 'MedicalAssistantApp/1.0'}
                                url = f"https://nominatim.openstreetmap.org/search?format=json&q=hospital+near+{location}&limit=1"
                                geo_response = requests.get(url, headers=headers).json()
                                
                                if geo_response:
                                    hospital_name = geo_response[0].get("display_name", "Local Hospital")
                                    hospital_address = geo_response[0].get("display_name", "Address unavailable")
                                else:
                                    hospital_name = "Nearest Regional Hospital"
                                    hospital_address = location

                                success, msg = send_emergency_email(
                                    patient_name, symptoms, location, hospital_name, hospital_address
                                )
                                
                                if success:
                                    st.success(f"✅ Emergency alert sent to: {hospital_name}")
                                else:
                                    st.error(f"❌ Email failed: {msg}. Please call 108 immediately.")
                            except Exception as e:
                                st.error(f"❌ Location lookup failed: {e}. Please call 108 immediately.")
                    else:
                        st.info("Alert not sent. Please seek medical help manually.")
                else:
                    st.success(f"✅ **Non-Critical**: {reason}")
                    st.info("Proceeding to detailed consultation...")

                # Save to session state for the chat page
                st.session_state.patient_data = {
                    "age": age, 
                    "gender": gender, 
                    "symptoms": symptoms,
                    "days_suffering": days_suffering, 
                    "pain_level": pain_level,
                    "location": location, 
                    "report_text": "", 
                    "report_image_bytes": None
                }
                
                st.switch_page("pages/chat.py")

            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")