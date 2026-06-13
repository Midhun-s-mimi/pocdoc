import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

def send_emergency_email(patient_name, symptoms, location, hospital_name, hospital_address, receiver_email):
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_APP_PASSWORD")
    
    # Fallback if the dynamic email is somehow empty
    if not receiver_email or receiver_email == "hospital.emergency@example.com":
        receiver_email = os.getenv("DEFAULT_EMERGENCY_EMAIL", "regional.dispatch@example.com")

    subject = f"🚨 URGENT: Critical Patient Alert - {patient_name}"
    body = f"""
    URGENT MEDICAL ALERT

    A patient has reported critical symptoms and consented to share their details.

    Patient Name: {patient_name}
    Location: {location}
    Reported Symptoms: {symptoms}

    Nearest Identified Facility: {hospital_name}
    Facility Address: {hospital_address}

    Please prepare for potential intake or provide immediate tele-guidance.
    """

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True, f"Emergency email sent successfully to {receiver_email}."
    except Exception as e:
        return False, f"Failed to send email: {str(e)}"