# pdf_previewT53.py - COMPLETE WORKING VERSION
from flask import Flask
from datetime import datetime
import os
import json

# Create minimal Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'test-key'
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME', 'test@gmail.com')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD', 'password')
app.config['MAIL_DEFAULT_SENDER'] = 'noreply@test.com'

# Create dummy prediction class with proper attributes
class DummyPrediction:
    def __init__(self):
        self.id = 12345
        self.flight_number = "AI-2024"
        self.registration_number = "VT-ABC"
        self.model_type = "Boeing 737"
        self.flight_condition = "normal"
        self.confidence_score = 0.962
        self.timestamp = datetime.now()
        self.owner_email = "test@example.com"
        self.management_email = "mgmt@example.com"
        self.maintenance_email = "maint@example.com"
        self.email_sent = False
        # Use dictionaries instead of strings for JSON fields
        self.fault_detections = {"total_faults": 0, "critical_faults": 0}
        self.prediction_result = {"confidence": 0.962}
        self.survey_data = {}
        
    def get_condition_display(self):
        return "✅ Working Good"

# Simple PDF generation function (bypassing the complex email_sender)
def generate_simple_pdf(prediction):
    """Generate a simple PDF report"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        
        # Create PDF directory
        pdf_dir = 'pdf_reports'
        os.makedirs(pdf_dir, exist_ok=True)
        
        # Filename
        filename = f"{pdf_dir}/prediction_{prediction.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        # Create PDF
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title = Paragraph("Aircraft Safety Prediction Report", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Flight info
        story.append(Paragraph(f"Flight: {prediction.flight_number}", styles['Heading2']))
        story.append(Paragraph(f"Registration: {prediction.registration_number}", styles['Normal']))
        story.append(Paragraph(f"Aircraft: {prediction.model_type}", styles['Normal']))
        story.append(Paragraph(f"Condition: {prediction.get_condition_display()}", styles['Normal']))
        story.append(Paragraph(f"Confidence: {prediction.confidence_score*100:.1f}%", styles['Normal']))
        story.append(Paragraph(f"Date: {prediction.timestamp.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        return filename
        
    except Exception as e:
        print(f"PDF generation error: {e}")
        return None

# Run with app context
with app.app_context():
    print("\n" + "="*50)
    print("📄 PDF REPORT GENERATION")
    print("="*50)
    
    try:
        prediction = DummyPrediction()
        
        # Generate PDF
        pdf_path = generate_simple_pdf(prediction)
        
        if pdf_path and os.path.exists(pdf_path):
            print(f"✅ PDF generated successfully!")
            print(f"📁 File location: {os.path.abspath(pdf_path)}")
            print(f"📁 File size: {os.path.getsize(pdf_path)} bytes")
            print("\n📄 You can open this file to show the receipt")
        else:
            print("❌ PDF generation failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("="*50)