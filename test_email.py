import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

# Force load the .env file
load_dotenv()

sender = os.getenv("SENDER_EMAIL")
password = os.getenv("SENDER_APP_PASSWORD")
receiver = os.getenv("DEFAULT_EMERGENCY_EMAIL")

print("--- EMAIL DIAGNOSTIC ---")
print(f"1. Sender Email: {sender}")
print(f"2. Password Length: {len(password) if password else 0} (Should be exactly 16)")
print(f"3. Receiver Email: {receiver}")
print("------------------------")

if not sender or not password or not receiver:
    print("❌ FAILED: One or more variables are missing from your .env file!")
else:
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    msg['Subject'] = "✅ MEDICAL APP EMAIL TEST SUCCESSFUL"
    msg.attach(MIMEText("If you are reading this, your Gmail App Password and SMTP setup are 100% correct!", 'plain'))

    try:
        print("Connecting to Gmail SMTP...")
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        print("✅ SUCCESS! Email sent. Please check your inbox (and Spam/Promotions folder).")
    except smtplib.SMTPAuthenticationError:
        print("❌ FAILED: Authentication Error. Your SENDER_APP_PASSWORD is incorrect or has spaces.")
    except Exception as e:
        print(f"❌ FAILED: {e}")