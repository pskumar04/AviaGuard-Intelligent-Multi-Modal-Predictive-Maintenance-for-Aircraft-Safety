# email_previewT52.py - COMPLETE WORKING VERSION
from flask import Flask, render_template
from datetime import datetime
import os
import json

# Tell Flask where to find templates
template_dir = os.path.abspath('website/templates')
app = Flask(__name__, template_folder=template_dir)

# Add custom Jinja2 filter for from_json
@app.template_filter('from_json')
def from_json_filter(value):
    try:
        if isinstance(value, str):
            return json.loads(value)
        return value
    except:
        return {}

# Add custom url_for for email template (bypass actual routing)
@app.context_processor
def utility_processor():
    def email_url_for(endpoint, **values):
        # Return dummy URLs for email preview
        if endpoint == 'prediction_result':
            return f"http://localhost:5000/prediction-result/{values.get('prediction_id', 12345)}"
        elif endpoint == 'index':
            return "http://localhost:5000/"
        else:
            return "#"
    return dict(url_for=email_url_for)

@app.route('/preview-email')
def preview_email():
    # Create dummy prediction data with all required attributes
    class DummyPrediction:
        def __init__(self):
            self.flight_number = "AI-2024"
            self.registration_number = "VT-ABC"
            self.model_type = "boeing_737"
            self.flight_condition = "normal"
            self.confidence_score = 0.962
            self.id = 12345
            self.timestamp = datetime.now()
            # Make sure fault_detections is a valid JSON string
            self.fault_detections = json.dumps({
                "total_faults": 0, 
                "critical_faults": 0,
                "predictions": []
            })
            self.prediction_result = json.dumps({"confidence": 0.962})
            self.survey_data = "{}"
            self.owner_email = "test@example.com"
            self.management_email = "mgmt@example.com"
            self.maintenance_email = "maint@example.com"
            self.email_sent = False
            self.email_sent_at = None
            
        def get_condition_display(self):
            return "✅ Working Good"
        
        def get_condition_color(self):
            return "success"
    
    prediction = DummyPrediction()
    
    flight_details = {
        'flight_number': 'AI-2024',
        'registration': 'VT-ABC',
        'owner_name': 'John Doe',
        'owner_email': 'john@example.com'
    }
    
    print(f"📧 Rendering email template...")
    return render_template('email_template.html', 
                         prediction=prediction,
                         flight_details=flight_details,
                         condition='normal',
                         now=datetime.now())

if __name__ == '__main__':
    print("\n" + "="*50)
    print("📧 EMAIL PREVIEW SERVER")
    print("="*50)
    print(f"📧 URL: http://localhost:5001/preview-email")
    print(f"📁 Templates folder: {template_dir}")
    print("="*50 + "\n")
    app.run(debug=True, port=5001)