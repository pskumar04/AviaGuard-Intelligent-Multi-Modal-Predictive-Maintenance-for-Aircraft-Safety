# test_emailT51.py - COMPLETE WORKING VERSION
from flask import Flask
import os
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

load_dotenv()

# Create minimal Flask app for context
app = Flask(__name__)
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME', 'your-email@gmail.com')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD', 'your-password')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER', 'noreply@test.com')

def send_test_email(recipient):
    """Send a test email directly using SMTP"""
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = 'Test Email - Aircraft Predictive Maintenance'
        msg['From'] = app.config['MAIL_DEFAULT_SENDER']
        msg['To'] = recipient
        
        # HTML content
        html_content = f"""
        <html>
            <body>
                <h2 style="color: #667eea;">✈️ Aircraft Predictive Maintenance System</h2>
                <p>This is a test email to verify email configuration.</p>
                <p><strong>Time:</strong> {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <hr>
                <p style="color: green;">✅ If you're receiving this, email is working correctly!</p>
            </body>
        </html>
        """
        
        msg.attach(MIMEText(html_content, 'html'))
        
        # Send email
        print(f"📧 Sending test email to {recipient}...")
        with smtplib.SMTP(app.config['MAIL_SERVER'], app.config['MAIL_PORT']) as server:
            if app.config['MAIL_USE_TLS']:
                server.starttls()
            if app.config['MAIL_USERNAME'] and app.config['MAIL_PASSWORD']:
                server.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
            server.send_message(msg)
        
        print("✅ Email sent successfully!")
        print("📧 Check your inbox")
        return True
        
    except Exception as e:
        print(f"❌ Email sending failed: {e}")
        return False

# Run with app context
with app.app_context():
    # Change this to YOUR email address
    result = send_test_email('satishpanduru7013@gmail.com')  # Use your email
    
    if result:
        print("\n✅ Test passed! Email system is working.")
    else:
        print("\n❌ Test failed. Check your email configuration in .env file")