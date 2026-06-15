import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

def send_emergency_email(patient_name, symptoms, location, hospital_name, hospital_address, receiver_email):
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_APP_PASSWORD")
    
    # 🚨 TESTING OVERRIDE: Force the email to go to YOUR inbox instead of the hospital's
    # We use the DEFAULT_EMERGENCY_EMAIL from your .env file
    forced_receiver = os.getenv("DEFAULT_EMERGENCY_EMAIL", sender_email)
    
    print(f"🚨 DEBUG: Bypassing hospital email ({receiver_email}) and sending to: {forced_receiver}")

    subject = f"Medical Alert: {patient_name}"

    body = f"""Medical Assistant Alert

A patient has reported critical symptoms.

Patient Name: {patient_name}
Location: {location}
Reported Symptoms: {symptoms}

Nearest Identified Facility: {hospital_name}
Facility Address: {hospital_address}

Please prepare for potential intake.
"""

    msg = MIMEMultipart()
    msg['From'] = f"Medical Assistant <{sender_email}>"
    msg['To'] = forced_receiver
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True, f"Email sent to {forced_receiver}."
    except Exception as e:
        return False, f"Failed: {str(e)}"