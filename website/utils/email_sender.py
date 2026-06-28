"""
Email Sending Utility for Aircraft Predictive Maintenance
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from flask import render_template, current_app
from threading import Thread
import os
import json
from datetime import datetime
from pathlib import Path

class EmailSender:
    """Handles sending emails for prediction results"""
    
    def __init__(self, app=None):
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize with Flask app"""
        self.app = app
        self.mail_server = app.config['MAIL_SERVER']
        self.mail_port = app.config['MAIL_PORT']
        self.mail_use_tls = app.config['MAIL_USE_TLS']
        self.mail_username = app.config['MAIL_USERNAME']
        self.mail_password = app.config['MAIL_PASSWORD']
        self.mail_sender = app.config['MAIL_DEFAULT_SENDER']
    
    def send_prediction_email(self, prediction_result, flight_details, condition):
        """Send prediction result email to stakeholders"""
        try:
            # Render email template
            html_content = render_template(
                'email_template.html',
                prediction=prediction_result,
                flight_details=flight_details,
                condition=condition,
                now=datetime.utcnow()
            )
            
            # Create email message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"Aircraft Safety Prediction - {prediction_result.flight_number} - {prediction_result.get_condition_display()}"
            msg['From'] = self.mail_sender
            
            # Add recipients
            recipients = []
            if prediction_result.owner_email:
                recipients.append(prediction_result.owner_email)
            if prediction_result.management_email:
                recipients.append(prediction_result.management_email)
            if prediction_result.maintenance_email:
                recipients.append(prediction_result.maintenance_email)
            
            msg['To'] = ', '.join(recipients)
            msg['Cc'] = 'operations@aircraft-pm.com'
            
            # Attach HTML content
            msg.attach(MIMEText(html_content, 'html'))
            
            # Add PDF attachment (if generated)
            pdf_path = self.generate_prediction_pdf(prediction_result)
            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, 'rb') as f:
                    attachment = MIMEApplication(f.read(), _subtype='pdf')
                    attachment.add_header(
                        'Content-Disposition',
                        'attachment',
                        filename=f'Prediction_Report_{prediction_result.flight_number}_{datetime.now().strftime("%Y%m%d")}.pdf'
                    )
                    msg.attach(attachment)
            
            # Send email
            self._send_email(msg, recipients)
            
            # Update prediction result
            prediction_result.email_sent = True
            prediction_result.email_sent_at = datetime.utcnow()
            
            current_app.logger.info(f"Email sent for prediction {prediction_result.id}")
            return True
            
        except Exception as e:
            current_app.logger.error(f"Failed to send email: {str(e)}")
            return False
    
    def send_alert_email(self, alert_type, data):
        """Send alert email for critical conditions"""
        try:
            alert_templates = {
                'critical_failure': 'critical_alert.html',
                'maintenance_due': 'maintenance_alert.html',
                'safety_warning': 'safety_alert.html'
            }
            
            template = alert_templates.get(alert_type, 'alert.html')
            html_content = render_template(template, data=data)
            
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"Aircraft Alert: {alert_type.replace('_', ' ').title()}"
            msg['From'] = self.mail_sender
            msg['To'] = 'alerts@aircraft-pm.com'
            msg['Cc'] = 'operations@aircraft-pm.com'
            
            msg.attach(MIMEText(html_content, 'html'))
            
            self._send_email(msg, ['alerts@aircraft-pm.com', 'operations@aircraft-pm.com'])
            return True
            
        except Exception as e:
            current_app.logger.error(f"Failed to send alert email: {str(e)}")
            return False
    
    def send_test_email(self, recipient):
        """Send test email"""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = 'Test Email - Aircraft Predictive Maintenance'
            msg['From'] = self.mail_sender
            msg['To'] = recipient
            
            html_content = """
            <html>
                <body>
                    <h2>Test Email - Aircraft Predictive Maintenance System</h2>
                    <p>This is a test email to verify email configuration.</p>
                    <p>If you're receiving this, email is properly configured.</p>
                    <p>Time: {}</p>
                </body>
            </html>
            """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            msg.attach(MIMEText(html_content, 'html'))
            
            self._send_email(msg, [recipient])
            return True
            
        except Exception as e:
            current_app.logger.error(f"Test email failed: {str(e)}")
            return False
    
    def _send_email(self, msg, recipients):
        """Send email using SMTP"""
        with smtplib.SMTP(self.mail_server, self.mail_port) as server:
            if self.mail_use_tls:
                server.starttls()
            if self.mail_username and self.mail_password:
                server.login(self.mail_username, self.mail_password)
            server.send_message(msg)
    
    def generate_prediction_pdf(self, prediction_result):
        """Generate PDF report for prediction"""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors
            from reportlab.lib.units import inch
            from reportlab.pdfgen import canvas
            
            # Create PDF directory if not exists
            pdf_dir = Path('website/static/pdfs')
            pdf_dir.mkdir(exist_ok=True)
            
            filename = pdf_dir / f'prediction_{prediction_result.id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
            
            # Create PDF document
            doc = SimpleDocTemplate(str(filename), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=1  # Center alignment
            )
            title = Paragraph(f"Aircraft Safety Prediction Report", title_style)
            story.append(title)
            
            # Subtitle
            subtitle = Paragraph(
                f"Flight: {prediction_result.flight_number} | "
                f"Registration: {prediction_result.registration_number} | "
                f"Date: {prediction_result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                styles['Heading3']
            )
            story.append(subtitle)
            story.append(Spacer(1, 20))
            
            # Condition
            condition_style = ParagraphStyle(
                'Condition',
                parent=styles['Normal'],
                fontSize=16,
                textColor=colors.green if prediction_result.flight_condition == 'normal' else
                         colors.orange if prediction_result.flight_condition == 'warning' else
                         colors.red,
                alignment=1
            )
            condition = Paragraph(f"Condition: {prediction_result.get_condition_display()}", condition_style)
            story.append(condition)
            story.append(Spacer(1, 20))
            
            # Summary Table
            summary_data = [
                ['Parameter', 'Value'],
                ['Aircraft Model', prediction_result.model_type],
                ['Confidence Score', f"{prediction_result.confidence_score:.1%}"],
                ['Total Faults Detected', prediction_result.fault_detections.get('total_faults', 0)],
                ['Critical Faults', prediction_result.fault_detections.get('critical_faults', 0)],
                ['Email Sent', 'Yes' if prediction_result.email_sent else 'No'],
                ['Report Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            ]
            
            summary_table = Table(summary_data, colWidths=[2*inch, 3*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 30))
            
            # Recommendations
            recommendations_title = Paragraph("Recommendations", styles['Heading2'])
            story.append(recommendations_title)
            
            if prediction_result.flight_condition == 'critical':
                rec_text = "GROUND AIRCRAFT IMMEDIATELY. Do not attempt to fly until all critical issues are resolved."
            elif prediction_result.flight_condition == 'warning':
                rec_text = "Schedule maintenance before next flight. Monitor affected systems closely."
            else:
                rec_text = "Continue with regular maintenance schedule. All systems operating normally."
            
            recommendations = Paragraph(rec_text, styles['Normal'])
            story.append(recommendations)
            
            # Build PDF
            doc.build(story)
            
            return filename
            
        except Exception as e:
            current_app.logger.error(f"PDF generation failed: {str(e)}")
            return None
    
    def send_batch_emails(self, predictions):
        """Send emails for multiple predictions"""
        results = []
        for prediction in predictions:
            result = self.send_prediction_email_async(prediction)
            results.append(result)
        return results
    
    def send_prediction_email_async(self, prediction_result):
        """Send email asynchronously"""
        thread = Thread(
            target=self.send_prediction_email,
            args=(prediction_result,)
        )
        thread.daemon = True
        thread.start()
        return thread
    

# Global email sender instance
email_sender = EmailSender()

def send_prediction_email(prediction_result, flight_details, condition):
    """Send prediction email (compatibility function)"""
    return email_sender.send_prediction_email(prediction_result, flight_details, condition)