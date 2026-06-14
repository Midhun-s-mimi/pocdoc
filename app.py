import streamlit as st
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
    with col1:
        reg_name = st.text_input("Full Name", key="reg_name")
        reg_email = st.text_input("Email Address", key="reg_email")
        reg_password = st.text_input("Password", type="password", key="reg_password")
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