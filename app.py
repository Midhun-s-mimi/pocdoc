import streamlit as st
from streamlit_mic_recorder import mic_recorder
from auth_utils import register_user, login_user

st.set_page_config(page_title="Medical Assistant - Auth", page_icon="🏥", layout="centered")

st.title("🏥 Medical Assistant")
st.markdown("### Secure Health Consultation Portal")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if st.session_state.authenticated:
    st.switch_page("pages/chat.py")
    st.stop()

tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])

with tab1:
    st.subheader("Welcome Back")
    login_email = st.text_input("Email Address", key="login_email")
    login_password = st.text_input("Password", type="password", key="login_password")
    
    if st.button("Login", type="primary", use_container_width=True):
        if login_email and login_password:
            success, profile, message = login_user(login_email, login_password)
            if success:
                st.session_state.authenticated = True
                st.session_state.user_profile = profile
                st.success(message)
                st.switch_page("pages/chat.py")
            else:
                st.error(message)
        else:
            st.warning("Please fill in all fields.")

with tab2:
    st.subheader("Create an Account")
    col1, col2 = st.columns(2)
Recordewith col1:
    st.subheader("🎤 Voice or Text Input")
    
    # Initialize session state to hold the audio and text
    if "recorded_audio_bytes" not in st.session_state:
        st.session_state.recorded_audio_bytes = None
    if "transcribed_text" not in st.session_state:
        st.session_state.transcribed_text = ""
    if "symptoms_text" not in st.session_state:
        st.session_state.symptoms_text = ""

    # 1. Voice r Component
    audio_state = mic_recorder(
        start_prompt="🎙️ Start Recording (Tamil/English)",
        stop_prompt="⏹️ Stop",
        key="recorder"
    )
    
    # 2. Process Audio when recording stops
    if audio_state:
        # Save the audio bytes so we can play it back
        st.session_state.recorded_audio_bytes = audio_state["bytes"]
        
        with st.spinner("Transcribing audio..."):
            try:
                transcription = groq_client.audio.transcriptions.create(
                    file=("audio.webm", audio_state["bytes"], "audio/webm"),
                    model="whisper-large-v3",
                    response_format="text"
                )
                # Update the text state with the transcription
                st.session_state.transcribed_text = transcription
                st.session_state.symptoms_text = transcription 
                st.success("✅ Transcription successful!")
            except Exception as e:
                st.error(f"❌ Transcription failed: {e}")

    # 3. Display the Audio Player (Audio Format)
    if st.session_state.recorded_audio_bytes:
        st.markdown("##### 🎧 Your Recording (Audio Format)")
        st.audio(st.session_state.recorded_audio_bytes, format="audio/webm")
        
        # Optional: Add a button to clear the recording and text if they want to start over
        if st.button("🗑️ Clear Recording & Text"):
            st.session_state.recorded_audio_bytes = None
            st.session_state.transcribed_text = ""
            st.session_state.symptoms_text = ""
            st.rerun()

    # 4. Display the Text Box (Text Format)
    st.markdown("##### 📝 Symptoms (Text Format)")
    symptoms = st.text_area(
        "Review the transcribed text or type manually:", 
        value=st.session_state.symptoms_text, 
        height=150,
        key="symptoms_input"
    )
    
    # Keep the session state updated with whatever the user types/edits in the box
    st.session_state.symptoms_text = symptoms
    with col2:
        reg_age = st.text_input("Age", key="reg_age")
        reg_gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"], key="reg_gender")
        reg_location = st.text_input("City or Location", key="reg_location")
    
    if st.button("Sign Up", type="primary", use_container_width=True):
        if all([reg_name, reg_email, reg_password, reg_age, reg_gender, reg_location]):
            success, message = register_user(reg_name, reg_email, reg_password, reg_age, reg_gender, reg_location)
            if success:
                st.success(message)
                st.info("Please switch to the Login tab.")
            else:
                st.error(message)
        else:
            st.warning("Please fill in all fields.")
