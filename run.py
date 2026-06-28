"""
AIRCRAFT PREDICTIVE MAINTENANCE SYSTEM - COMPLETE VERSION
"""

from flask import Flask, render_template, render_template_string, jsonify, request, session, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
import os
import json
import threading
import re
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.template_folder = 'website/templates'
app.secret_key = os.getenv('SECRET_KEY', 'dev-key-12345')
app.config['SESSION_TYPE'] = 'filesystem'



from urllib.parse import quote_plus

# Your password with special characters
password = "Panduru@7013"

# URL encode the password (converts @ to %40)
encoded_password = quote_plus(password)

# Use encoded password in connection string
app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://postgres:{encoded_password}@localhost:5432/aircraft_pm'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 10,
    'pool_recycle': 3600,
    'pool_pre_ping': True
}
db = SQLAlchemy(app)

# ========== ADD THIS ==========
@app.context_processor
def inject_now():
    """Inject current datetime into templates"""
    from datetime import datetime
    return {'now': datetime.now()}
# ========== END ADD ==========

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# Simple User model for login (if you want to keep authentication)
class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    # Return a simple user object (for demo purposes)
    return User(user_id, 'demo_user', 'demo@example.com')

# Email configuration
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER', 'noreply@aviaguard.com')

print(f"Email configured for: {app.config['MAIL_USERNAME']}")



# PostgreSQL Database Configuration
POSTGRES_USER = 'postgres'
POSTGRES_PASSWORD = 'Panduru@7013'  # <-- CHANGE THIS TO YOUR PASSWORD
POSTGRES_HOST = 'localhost'
POSTGRES_PORT = '5432'
POSTGRES_DB = 'aircraft_pm'

app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}'

# ========== DATABASE MODEL ==========
class FlightSurvey(db.Model):
    """Flight survey data model for PostgreSQL"""
    __tablename__ = 'flight_surveys'
    
    id = db.Column(db.Integer, primary_key=True)
    flight_number = db.Column(db.String(20), unique=True, nullable=False, index=True)
    registration_number = db.Column(db.String(20))
    aircraft_model_type = db.Column(db.String(20), index=True)  # 'boeing', 'cessna', 'airbus'
    
    # Management team details
    management_team = db.Column(db.String(100))
    management_team_id = db.Column(db.String(50))
    management_contact = db.Column(db.String(100))
    management_email = db.Column(db.String(120))
    
    # Maintenance team details
    maintenance_team = db.Column(db.String(100))
    maintenance_team_id = db.Column(db.String(50))
    maintenance_manager = db.Column(db.String(100))
    maintenance_email = db.Column(db.String(120))
    
    # Aircraft details
    total_flight_hours = db.Column(db.Integer, default=0)
    aircraft_model = db.Column(db.String(50))
    
    # Survey data (JSON)
    survey_data = db.Column(db.JSON)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    last_updated = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    def __repr__(self):
        return f'<FlightSurvey {self.flight_number}>'

# Create tables
with app.app_context():
    try:
        db.session.execute(text('SELECT 1'))
        print("✅ PostgreSQL database connected successfully!")
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        print("Please check your PostgreSQL credentials and make sure PostgreSQL is running.")
# ========== END DATABASE MODEL ==========

# ========== SIMPLE PREDICTION RESULT CLASS ==========
class SimplePredictionResult:
    def __init__(self, flight_number, registration_number, model_type, flight_condition, confidence_score, owner_email):
        self.flight_number = flight_number
        self.registration_number = registration_number
        self.model_type = model_type
        self.flight_condition = flight_condition
        self.confidence_score = confidence_score
        self.owner_email = owner_email
        self.id = int(datetime.now().timestamp())
        self.timestamp = datetime.now()
        self.email_sent = False
        self.fault_detections = json.dumps({'total_faults': 0, 'critical_faults': 0})
        self.prediction_result = json.dumps({'confidence': confidence_score})
        self.survey_data = json.dumps({})
    
    def get_condition_display(self):
        conditions = {
            'normal': '✅ Working Good',
            'warning': '⚠️ Working with Issues',
            'critical': '❌ Not Working'
        }
        return conditions.get(self.flight_condition, 'Unknown')
# ========== END PREDICTION RESULT CLASS ==========

# ========== ROUTES ==========

# Home page
@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Aircraft Predictive Maintenance</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding-top: 50px;
            }
            .container { max-width: 1200px; }
            .hero { 
                background: white; 
                border-radius: 20px; 
                padding: 50px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                margin-bottom: 30px;
            }
            .model-card {
                background: white;
                border-radius: 15px;
                padding: 30px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s;
                height: 100%;
                box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            }
            .model-card:hover {
                transform: translateY(-10px);
                box-shadow: 0 15px 40px rgba(0,0,0,0.2);
            }
            .btn-custom {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px 30px;
                border-radius: 25px;
                text-decoration: none;
                font-weight: bold;
            }
            .btn-custom:hover { color: white; opacity: 0.9; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="hero text-center">
                <h1 class="display-4 mb-4">✈️ Aircraft Predictive Maintenance</h1>
                <p class="lead mb-4">AI-powered system to predict flight safety and detect potential failures</p>
                <a href="/model-selection" class="btn btn-custom btn-lg">Select Aircraft Model →</a>
            </div>
            
            <div class="row">
                <div class="col-md-4 mb-4">
                    <div class="model-card">
                        <div style="font-size: 48px; color: #667eea;">✈️</div>
                        <h3 class="mt-3">Boeing 737 NG</h3>
                        <p class="text-muted">Commercial Jet</p>
                        <p>CFM56-7B Engine<br>189 Passengers</p>
                        <button onclick="location.href='/select-model/boeing_737'" class="btn btn-primary">Select</button>
                    </div>
                </div>
                <div class="col-md-4 mb-4">
                    <div class="model-card">
                        <div style="font-size: 48px; color: #28a745;">✈️</div>
                        <h3 class="mt-3">Cessna 172</h3>
                        <p class="text-muted">Light Aircraft</p>
                        <p>Lycoming IO-360<br>4 Passengers</p>
                        <button onclick="location.href='/select-model/cessna_172'" class="btn btn-success">Select</button>
                    </div>
                </div>
                <div class="col-md-4 mb-4">
                    <div class="model-card">
                        <div style="font-size: 48px; color: #17a2b8;">🚁</div>
                        <h3 class="mt-3">Airbus H125</h3>
                        <p class="text-muted">Helicopter</p>
                        <p>Turbomeca Arriel 2B<br>6 Passengers</p>
                        <button onclick="location.href='/select-model/airbus_h125'" class="btn btn-info">Select</button>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    '''

# Model selection page
@app.route('/model-selection')
def model_selection():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Select Aircraft Model</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding-top: 50px;
            }
            .container { max-width: 1200px; }
            .page-header {
                background: white;
                border-radius: 20px;
                padding: 30px;
                margin-bottom: 30px;
                text-align: center;
            }
            .model-card {
                background: white;
                border-radius: 15px;
                padding: 30px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s;
                height: 100%;
                box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            }
            .model-card:hover {
                transform: translateY(-10px);
                box-shadow: 0 15px 40px rgba(0,0,0,0.2);
            }
            .btn-back {
                background: #6c757d;
                color: white;
                padding: 12px 30px;
                border-radius: 25px;
                text-decoration: none;
                margin-top: 20px;
                display: inline-block;
            }
            .btn-back:hover { color: white; opacity: 0.9; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="page-header">
                <h1 class="display-4 mb-4">✈️ Select Your Aircraft Model</h1>
                <p class="lead mb-0">Choose from our supported aircraft models to begin the safety prediction process</p>
            </div>
            
            <div class="row">
                <div class="col-md-4 mb-4">
                    <div class="model-card" onclick="location.href='/select-model/boeing_737'">
                        <div style="font-size: 48px; color: #667eea;">✈️</div>
                        <h3 class="mt-3">Boeing 737 NG</h3>
                        <p class="text-muted">Commercial Jet</p>
                        <p>CFM56-7B Engine<br>189 Passengers</p>
                        <button class="btn btn-primary">Select</button>
                    </div>
                </div>
                <div class="col-md-4 mb-4">
                    <div class="model-card" onclick="location.href='/select-model/cessna_172'">
                        <div style="font-size: 48px; color: #28a745;">✈️</div>
                        <h3 class="mt-3">Cessna 172</h3>
                        <p class="text-muted">Light Aircraft</p>
                        <p>Lycoming IO-360<br>4 Passengers</p>
                        <button class="btn btn-success">Select</button>
                    </div>
                </div>
                <div class="col-md-4 mb-4">
                    <div class="model-card" onclick="location.href='/select-model/airbus_h125'">
                        <div style="font-size: 48px; color: #17a2b8;">🚁</div>
                        <h3 class="mt-3">Airbus H125</h3>
                        <p class="text-muted">Helicopter</p>
                        <p>Turbomeca Arriel 2B<br>6 Passengers</p>
                        <button class="btn btn-info">Select</button>
                    </div>
                </div>
            </div>
            
            <div class="text-center mt-4">
                <a href="/" class="btn-back">← Back to Home</a>
            </div>
        </div>
    </body>
    </html>
    '''

# Model selection
@app.route('/select-model/<model_name>')
def select_model(model_name):
    session['selected_model'] = model_name
    return redirect(url_for('flight_details'))

# ========== FLIGHT DETAILS ROUTE WITH DATABASE ==========
@app.route('/flight-details', methods=['GET', 'POST'])
def flight_details():
    # Predefined teams
    management_teams = [
        ('operations', 'Operations Department'),
        ('maintenance_control', 'Maintenance Control'),
        ('quality_assurance', 'Quality Assurance'),
        ('engineering', 'Engineering Department'),
        ('safety', 'Safety & Compliance')
    ]
    
    maintenance_teams = [
        ('line_maintenance', 'Line Maintenance'),
        ('base_maintenance', 'Base Maintenance'),
        ('avionics', 'Avionics Team'),
        ('engine_shop', 'Engine Shop'),
        ('components', 'Components Repair')
    ]
    
    if request.method == 'POST':
        flight_number = request.form.get('flight_number', '').strip().upper()
        
        # Validate flight number
        flight_pattern = r'^[A-Z]{2,3}-\d{3,5}$'
        if not re.match(flight_pattern, flight_number):
            flash('Flight number must be in format: Airline Code (2-3 letters) + Hyphen + Numbers (3-5 digits). Example: AI-2024, 6E-1234', 'danger')
            return redirect(url_for('flight_details'))
        
        registration = request.form.get('registration', '').strip().upper()
        
        # Validate registration number
        reg_pattern = r'^VT-[A-Z]{3}$|^[A-Z]{2}-[A-Z]{3}$|^N\d{3}[A-Z]{2}$'
        if not re.match(reg_pattern, registration):
            flash('Registration number must be in valid format. Examples: VT-ABC (India), N123AB (USA)', 'danger')
            return redirect(url_for('flight_details'))
        
        # Validate aircraft hours
        aircraft_hours = request.form.get('aircraft_hours', '')
        try:
            aircraft_hours = int(aircraft_hours)
            if aircraft_hours < 0 or aircraft_hours > 50000:
                flash('Aircraft hours must be between 0 and 50,000', 'danger')
                return redirect(url_for('flight_details'))
        except ValueError:
            flash('Please enter a valid number for aircraft hours', 'danger')
            return redirect(url_for('flight_details'))
        
        # Get selected teams
        management_team = request.form.get('management_team', '')
        maintenance_team = request.form.get('maintenance_team', '')
        
        if not management_team:
            flash('Please select a management team', 'danger')
            return redirect(url_for('flight_details'))
        if not maintenance_team:
            flash('Please select a maintenance team', 'danger')
            return redirect(url_for('flight_details'))
        
        management_dict = dict(management_teams)
        maintenance_dict = dict(maintenance_teams)
        
        # Get selected model type
        selected_model = session.get('selected_model', '')
        if 'boeing' in selected_model:
            aircraft_model_type = 'boeing'
        elif 'cessna' in selected_model:
            aircraft_model_type = 'cessna'
        elif 'airbus' in selected_model:
            aircraft_model_type = 'airbus'
        else:
            aircraft_model_type = 'unknown'
        
        # Check if flight exists for this model
        existing_flight = FlightSurvey.query.filter_by(
            flight_number=flight_number,
            aircraft_model_type=aircraft_model_type
        ).first()
        
        if existing_flight:
            # UPDATE existing record
            existing_flight.registration_number = registration
            existing_flight.management_team = management_dict.get(management_team, management_team)
            existing_flight.management_team_id = management_team
            existing_flight.management_email = request.form.get('management_email', '')
            existing_flight.management_contact = request.form.get('management_name', '')
            existing_flight.maintenance_team = maintenance_dict.get(maintenance_team, maintenance_team)
            existing_flight.maintenance_team_id = maintenance_team
            existing_flight.maintenance_email = request.form.get('maintenance_email', '')
            existing_flight.maintenance_manager = request.form.get('maintenance_manager', '')
            existing_flight.total_flight_hours = aircraft_hours
            existing_flight.last_updated = datetime.now(timezone.utc)
            db.session.commit()
            flash('Flight details updated successfully!', 'success')
        else:
            # CREATE new flight record
            new_flight = FlightSurvey(
                flight_number=flight_number,
                registration_number=registration,
                aircraft_model_type=aircraft_model_type,
                management_team=management_dict.get(management_team, management_team),
                management_team_id=management_team,
                management_email=request.form.get('management_email', ''),
                management_contact=request.form.get('management_name', ''),
                maintenance_team=maintenance_dict.get(maintenance_team, maintenance_team),
                maintenance_team_id=maintenance_team,
                maintenance_email=request.form.get('maintenance_email', ''),
                maintenance_manager=request.form.get('maintenance_manager', ''),
                total_flight_hours=aircraft_hours,
                created_at=datetime.now(timezone.utc)
            )
            db.session.add(new_flight)
            db.session.commit()
            flash('Flight registered successfully!', 'success')
        
        # Store in session
        session['flight_details'] = {
            'management_name': request.form.get('management_name', ''),
            'management_email': request.form.get('management_email', ''),
            'management_team': management_dict.get(management_team, management_team),
            'maintenance_manager': request.form.get('maintenance_manager', ''),
            'maintenance_email': request.form.get('maintenance_email', ''),
            'maintenance_team': maintenance_dict.get(maintenance_team, maintenance_team),
            'flight_number': flight_number,
            'registration': registration,
            'aircraft_hours': aircraft_hours
        }
        
        # After saving flight details, redirect to route selection
        model = session.get('selected_model')
        if model == 'boeing_737':
            return redirect(url_for('flight_route'))
        elif model == 'cessna_172':
            return redirect(url_for('flight_route'))
        elif model == 'airbus_h125':
            return redirect(url_for('flight_route'))
    
    # GET request - show form
    return render_template('flight_details.html', 
                         management_teams=management_teams,
                         maintenance_teams=maintenance_teams)

# ========== FLIGHT ROUTE SELECTION ==========
# ========== FLIGHT ROUTE SELECTION ==========
@app.route('/flight-route', methods=['GET', 'POST'])
def flight_route():
    """Select flight route and calculate travel time"""
    
    # Predefined routes with distances and flight times
    routes = {
        'HYD-DEL': {
            'from': 'Hyderabad (HYD)',
            'to': 'New Delhi (DEL)',
            'distance': 1250,
            'flight_time_hours': 2.25,
            'flight_time_minutes': 135
        },
        'HYD-BOM': {
            'from': 'Hyderabad (HYD)',
            'to': 'Mumbai (BOM)',
            'distance': 620,
            'flight_time_hours': 1.33,
            'flight_time_minutes': 80
        },
        'DEL-BOM': {
            'from': 'New Delhi (DEL)',
            'to': 'Mumbai (BOM)',
            'distance': 1150,
            'flight_time_hours': 2.0,
            'flight_time_minutes': 120
        },
        'DEL-BLR': {
            'from': 'New Delhi (DEL)',
            'to': 'Bengaluru (BLR)',
            'distance': 1750,
            'flight_time_hours': 2.75,
            'flight_time_minutes': 165
        },
        'BOM-CCU': {
            'from': 'Mumbai (BOM)',
            'to': 'Kolkata (CCU)',
            'distance': 1650,
            'flight_time_hours': 2.5,
            'flight_time_minutes': 150
        },
        'HYD-CHE': {
            'from': 'Hyderabad (HYD)',
            'to': 'Chennai (MAA)',
            'distance': 630,
            'flight_time_hours': 1.33,
            'flight_time_minutes': 80
        }
    }
    
    if request.method == 'POST':
        selected_route = request.form.get('route', '')
        
        if selected_route in routes:
            route_info = routes[selected_route]
            travel_time_hours = route_info['flight_time_hours']
            
            # Get current flight details from session
            flight_details = session.get('flight_details', {})
            flight_number = flight_details.get('flight_number', '')
            
            print(f"🔍 Looking for flight number: {flight_number}")  # Debug
            
            if not flight_number:
                flash('No flight number found. Please start over.', 'danger')
                return redirect(url_for('flight_details'))
            
            # Query the flight (without model filter for now to debug)
            flight = FlightSurvey.query.filter_by(flight_number=flight_number).first()
            
            print(f"🔍 Flight found: {flight}")  # Debug
            
            if flight:
                # Update aircraft hours
                current_hours = flight.total_flight_hours or 0
                new_hours = current_hours + travel_time_hours
                flight.total_flight_hours = round(new_hours, 2)
                flight.last_updated = datetime.now(timezone.utc)
                db.session.commit()
                
                print(f"✅ Updated flight {flight_number}: {current_hours} → {new_hours} hours")  # Debug
                
                # Update session
                session['flight_details']['aircraft_hours'] = new_hours
                session['route_info'] = route_info
                
                flash(f'✈️ Flight from {route_info["from"]} to {route_info["to"]} completed!', 'success')
                flash(f'⏱️ Flight time: {int(route_info["flight_time_hours"])} hours {int((route_info["flight_time_hours"] % 1) * 60)} minutes', 'info')
                flash(f'📊 Total aircraft hours updated: {new_hours:.2f} hours', 'info')
                
                # After route selection, go to survey
                model = session.get('selected_model')
                if model == 'boeing_737':
                    return redirect(url_for('survey_boeing_737'))
                elif model == 'cessna_172':
                    return redirect(url_for('survey_cessna_172'))
                elif model == 'airbus_h125':
                    return redirect(url_for('survey_airbus_h125'))
            else:
                print(f"❌ Flight {flight_number} NOT found in database!")  # Debug
                flash(f'Flight {flight_number} not found. Please register first.', 'danger')
                return redirect(url_for('flight_details'))
        else:
            flash('Please select a valid route', 'danger')
            return redirect(url_for('flight_route'))
    
    # GET request - show route selection form
    return render_template('flight_route.html', routes=routes)

@app.route('/survey/boeing_737', methods=['GET', 'POST'])
def survey_boeing_737():
    import json
    from data_processing.multi_sensor_loader import MultiSensorDataLoader
    from website.forms import BoeingSurveyForm
    
    if request.method == 'POST':
        # Collect ALL sensor data from form
        survey_data = {
            'n1_rpm': float(request.form.get('n1_rpm', 98.5)),
            'n2_rpm': float(request.form.get('n2_rpm', 100.2)),
            'egt': float(request.form.get('egt', 550)),
            'oil_temperature': float(request.form.get('oil_temperature', 85)),
            'oil_pressure': float(request.form.get('oil_pressure', 60)),
            'vibration_level': float(request.form.get('vibration_level', 0.3)),
            'wing_vibration': float(request.form.get('wing_vibration', 0.15)),
            'fuselage_strain': float(request.form.get('fuselage_strain', 250)),
            'hydraulic_pressure': float(request.form.get('hydraulic_pressure', 3000)),
            'cabin_pressure_altitude': float(request.form.get('cabin_pressure_altitude', 5000)),
            'bleed_air_temperature': float(request.form.get('bleed_air_temperature', 200)),
            'aileron_position': float(request.form.get('aileron_position', 0)),
            'elevator_position': float(request.form.get('elevator_position', 0))
        }
        
        session['survey_data'] = survey_data
        
        # Save to database
        flight_details = session.get('flight_details', {})
        flight_number = flight_details.get('flight_number', '')
        
        if flight_number:
            flight = FlightSurvey.query.filter_by(flight_number=flight_number).first()
            if flight:
                flight.survey_data = survey_data  # Store as dict, will be JSON
                flight.aircraft_model = 'boeing_737'
                flight.last_updated = datetime.now(timezone.utc)
                db.session.commit()
                print(f"✅ Survey data saved for flight {flight_number}")
                flash('Survey data saved successfully!', 'success')
        
        return redirect(url_for('processing'))
    
    # GET request - show the survey form
    loader = MultiSensorDataLoader()
    form = BoeingSurveyForm()
    
    flight_details = session.get('flight_details', {})
    flight_number = flight_details.get('flight_number', '')
    
    existing_data = {}
    flight = None
    
    if flight_number:
        flight = FlightSurvey.query.filter_by(flight_number=flight_number).first()
        if flight and flight.survey_data:
            # Survey data is already a dict from PostgreSQL JSON column
            existing_data = flight.survey_data
            print(f"📊 Loaded existing survey data for {flight_number}")
    
    try:
        normal_data = loader.generate_realistic_sensor_data("Boeing 737 NG", "normal")
    except:
        normal_data = {}
    
    is_disabled = flight and flight.survey_data is not None
    
    return render_template('survey_boeing.html', 
                         config=app.config.get('BOEING_CONFIG', {}),
                         existing_data=existing_data,
                         normal_data=normal_data,
                         form=form,
                         is_disabled=is_disabled)

@app.route('/survey/cessna_172', methods=['GET', 'POST'])
def survey_cessna_172():
    import json
    from data_processing.multi_sensor_loader import MultiSensorDataLoader
    from website.forms import CessnaSurveyForm  # <-- ADD THIS IMPORT
    
    if request.method == 'POST':
        survey_data = {
            'vibration': {
                'engine_vibration': float(request.form.get('engine_vibration', 0.4)),
                'propeller_vibration': float(request.form.get('propeller_vibration', 0.2))
            },
            'thermal': {
                'cylinder_head_temp': float(request.form.get('cylinder_head_temp', 380)),
                'oil_temperature': float(request.form.get('oil_temperature', 180)),
                'exhaust_gas_temp': float(request.form.get('exhaust_gas_temp', 1300))
            },
            'acoustic': {
                'engine_acoustic': float(request.form.get('engine_acoustic', 80)),
                'propeller_acoustic': float(request.form.get('propeller_acoustic', 82))
            },
            'pressure': {
                'oil_pressure': float(request.form.get('oil_pressure', 40)),
                'fuel_pressure': float(request.form.get('fuel_pressure', 4.5))
            }
        }
        
        session['survey_data'] = survey_data
        
        # Save to database
        flight_details = session.get('flight_details', {})
        flight_number = flight_details.get('flight_number', '')
        
        if flight_number:
            flight = FlightSurvey.query.filter_by(flight_number=flight_number).first()
            if flight:
                flight.survey_data = json.dumps(survey_data)
                flight.aircraft_model = 'cessna_172'
                flight.last_updated = datetime.now(timezone.utc)
                db.session.commit()
                print(f"✅ Survey data saved for flight {flight_number}")
        
        return redirect(url_for('processing'))
    
    # GET request - show the survey form
    loader = MultiSensorDataLoader()
    form = CessnaSurveyForm()  # <-- ADD THIS
    
    flight_details = session.get('flight_details', {})
    flight_number = flight_details.get('flight_number', '')
    
    existing_data = {}
    if flight_number:
        flight = FlightSurvey.query.filter_by(flight_number=flight_number).first()
        if flight and flight.survey_data:
            existing_data = json.loads(flight.survey_data) if isinstance(flight.survey_data, str) else flight.survey_data
    
    try:
        normal_data = loader.generate_realistic_sensor_data("Cessna 172 Skyhawk", "normal")
    except:
        normal_data = {}
    
    # Check if survey already exists to disable inputs
    is_disabled = flight and flight.survey_data is not None
    
    return render_template('survey_cessna.html', 
                         config=app.config.get('CESSNA_CONFIG', {}),
                         existing_data=existing_data,
                         normal_data=normal_data,
                         form=form,  # <-- ADD THIS
                         is_disabled=is_disabled)  # <-- ADD THIS

@app.route('/survey/airbus_h125', methods=['GET', 'POST'])
def survey_airbus_h125():
    import json
    from website.forms import AirbusSurveyForm  # <-- ADD THIS IMPORT
    
    if request.method == 'POST':
        survey_data = {
            'vibration': {
                'rotor_vibration': float(request.form.get('rotor_vibration', 0.2)),
                'gearbox_vibration': float(request.form.get('gearbox_vibration', 0.3)),
                'tail_rotor_vibration': float(request.form.get('tail_rotor_vibration', 0.15))
            },
            'thermal': {
                't4_temperature': float(request.form.get('t4_temperature', 650)),
                'oil_temperature': float(request.form.get('oil_temperature', 80)),
                'gearbox_temp': float(request.form.get('gearbox_temp', 70))
            },
            'acoustic': {
                'rotor_acoustic': float(request.form.get('rotor_acoustic', 88)),
                'gearbox_acoustic': float(request.form.get('gearbox_acoustic', 78)),
                'bearing_acoustic': float(request.form.get('bearing_acoustic', 72))
            },
            'rotor_system': {
                'rotor_rpm': float(request.form.get('rotor_rpm', 350)),
                'track_and_balance': float(request.form.get('track_and_balance', 0.04))
            }
        }
        
        session['survey_data'] = survey_data
        
        # Save to database
        flight_details = session.get('flight_details', {})
        flight_number = flight_details.get('flight_number', '')
        
        if flight_number:
            flight = FlightSurvey.query.filter_by(flight_number=flight_number).first()
            if flight:
                flight.survey_data = json.dumps(survey_data)
                flight.aircraft_model = 'airbus_h125'
                flight.last_updated = datetime.now(timezone.utc)
                db.session.commit()
                print(f"✅ Survey data saved for flight {flight_number}")
        
        return redirect(url_for('processing'))
    
    # GET request - show the survey form
    form = AirbusSurveyForm()  # <-- ADD THIS
    
    flight_details = session.get('flight_details', {})
    flight_number = flight_details.get('flight_number', '')
    
    existing_data = {}
    if flight_number:
        flight = FlightSurvey.query.filter_by(flight_number=flight_number).first()
        if flight and flight.survey_data:
            existing_data = json.loads(flight.survey_data) if isinstance(flight.survey_data, str) else flight.survey_data
    
    # Check if survey already exists to disable inputs
    is_disabled = flight and flight.survey_data is not None
    
    return render_template('survey_airbus.html', 
                         config=app.config.get('AIRBUS_CONFIG', {}),
                         existing_data=existing_data,
                         form=form,  # <-- ADD THIS
                         is_disabled=is_disabled)  # <-- ADD THIS

# Processing page with REAL multi-sensor condition detection
@app.route('/processing')
def processing():
    # Get data from session
    survey = session.get('survey_data', {})
    model = session.get('selected_model', 'boeing_737')
    flight_details = session.get('flight_details', {})
    
    # Initialize counters
    warning_count = 0
    critical_count = 0
    problematic_sensors = []  # Track which sensors are problematic
    
    # Load sensor ranges based on aircraft model
    from data_processing.multi_sensor_loader import MultiSensorDataLoader
    loader = MultiSensorDataLoader()
    
    try:
        config = loader.load_aircraft_config(model.replace('_', ' ').title())
        sensor_ranges = config.get('real_sensor_ranges', {})
    except:
        sensor_ranges = {}
    
    # Check VIBRATION sensors
    if 'vibration' in survey:
        for sensor, value in survey['vibration'].items():
            # Get thresholds from config or use defaults
            if model == 'boeing_737':
                if value > 0.7:
                    critical_count += 1
                    problematic_sensors.append({'sensor': f'vibration_{sensor}', 'value': value, 'status': 'critical', 'range': '>0.7'})
                elif value > 0.5:
                    warning_count += 1
                    problematic_sensors.append({'sensor': f'vibration_{sensor}', 'value': value, 'status': 'warning', 'range': '0.5-0.7'})
            elif model == 'cessna_172':
                if value > 1.0:
                    critical_count += 1
                    problematic_sensors.append({'sensor': f'vibration_{sensor}', 'value': value, 'status': 'critical', 'range': '>1.0'})
                elif value > 0.8:
                    warning_count += 1
                    problematic_sensors.append({'sensor': f'vibration_{sensor}', 'value': value, 'status': 'warning', 'range': '0.8-1.0'})
            else:  # airbus_h125
                if value > 0.5:
                    critical_count += 1
                    problematic_sensors.append({'sensor': f'vibration_{sensor}', 'value': value, 'status': 'critical', 'range': '>0.5'})
                elif value > 0.4:
                    warning_count += 1
                    problematic_sensors.append({'sensor': f'vibration_{sensor}', 'value': value, 'status': 'warning', 'range': '0.4-0.5'})
    
    # Check THERMAL sensors
    if 'thermal' in survey:
        for sensor, value in survey['thermal'].items():
            if 'egt' in sensor or 't4' in sensor or 'exhaust' in sensor:
                # EGT type sensors
                if value > 700:
                    critical_count += 1
                    problematic_sensors.append({'sensor': f'thermal_{sensor}', 'value': value, 'status': 'critical', 'range': '>700°C'})
                elif value > 650:
                    warning_count += 1
                    problematic_sensors.append({'sensor': f'thermal_{sensor}', 'value': value, 'status': 'warning', 'range': '650-700°C'})
            elif 'oil' in sensor:
                # Oil temperature
                if model == 'boeing_737':
                    if value > 120:
                        critical_count += 1
                        problematic_sensors.append({'sensor': f'thermal_{sensor}', 'value': value, 'status': 'critical', 'range': '>120°C'})
                    elif value > 110:
                        warning_count += 1
                        problematic_sensors.append({'sensor': f'thermal_{sensor}', 'value': value, 'status': 'warning', 'range': '110-120°C'})
                elif model == 'cessna_172':
                    if value > 245:
                        critical_count += 1
                        problematic_sensors.append({'sensor': f'thermal_{sensor}', 'value': value, 'status': 'critical', 'range': '>245°F'})
                    elif value > 220:
                        warning_count += 1
                        problematic_sensors.append({'sensor': f'thermal_{sensor}', 'value': value, 'status': 'warning', 'range': '220-245°F'})
                else:  # airbus
                    if value > 110:
                        critical_count += 1
                        problematic_sensors.append({'sensor': f'thermal_{sensor}', 'value': value, 'status': 'critical', 'range': '>110°C'})
                    elif value > 100:
                        warning_count += 1
                        problematic_sensors.append({'sensor': f'thermal_{sensor}', 'value': value, 'status': 'warning', 'range': '100-110°C'})
            elif 'cylinder' in sensor or 'head' in sensor:
                # CHT
                if model == 'boeing_737':
                    if value > 475:
                        critical_count += 1
                        problematic_sensors.append({'sensor': f'thermal_{sensor}', 'value': value, 'status': 'critical', 'range': '>475°C'})
                    elif value > 450:
                        warning_count += 1
                        problematic_sensors.append({'sensor': f'thermal_{sensor}', 'value': value, 'status': 'warning', 'range': '450-475°C'})
                else:  # cessna
                    if value > 500:
                        critical_count += 1
                        problematic_sensors.append({'sensor': f'thermal_{sensor}', 'value': value, 'status': 'critical', 'range': '>500°F'})
                    elif value > 450:
                        warning_count += 1
                        problematic_sensors.append({'sensor': f'thermal_{sensor}', 'value': value, 'status': 'warning', 'range': '450-500°F'})
    
    # Check ACOUSTIC sensors
    if 'acoustic' in survey:
        for sensor, value in survey['acoustic'].items():
            if model == 'boeing_737':
                if value > 95:
                    critical_count += 1
                    problematic_sensors.append({'sensor': f'acoustic_{sensor}', 'value': value, 'status': 'critical', 'range': '>95dB'})
                elif value > 85:
                    warning_count += 1
                    problematic_sensors.append({'sensor': f'acoustic_{sensor}', 'value': value, 'status': 'warning', 'range': '85-95dB'})
            elif model == 'cessna_172':
                if value > 100:
                    critical_count += 1
                    problematic_sensors.append({'sensor': f'acoustic_{sensor}', 'value': value, 'status': 'critical', 'range': '>100dB'})
                elif value > 90:
                    warning_count += 1
                    problematic_sensors.append({'sensor': f'acoustic_{sensor}', 'value': value, 'status': 'warning', 'range': '90-100dB'})
            else:  # airbus
                if 'bearing' in sensor:
                    if value > 90:
                        critical_count += 1
                        problematic_sensors.append({'sensor': f'acoustic_{sensor}', 'value': value, 'status': 'critical', 'range': '>90dB'})
                    elif value > 80:
                        warning_count += 1
                        problematic_sensors.append({'sensor': f'acoustic_{sensor}', 'value': value, 'status': 'warning', 'range': '80-90dB'})
                else:
                    if value > 100:
                        critical_count += 1
                        problematic_sensors.append({'sensor': f'acoustic_{sensor}', 'value': value, 'status': 'critical', 'range': '>100dB'})
                    elif value > 90:
                        warning_count += 1
                        problematic_sensors.append({'sensor': f'acoustic_{sensor}', 'value': value, 'status': 'warning', 'range': '90-100dB'})
    
    # Check PRESSURE sensors
    if 'pressure' in survey:
        for sensor, value in survey['pressure'].items():
            if 'oil' in sensor:
                if model == 'boeing_737':
                    if value < 30 or value > 90:
                        critical_count += 1
                        problematic_sensors.append({'sensor': f'pressure_{sensor}', 'value': value, 'status': 'critical', 'range': '<30 or >90 PSI'})
                    elif value < 40 or value > 80:
                        warning_count += 1
                        problematic_sensors.append({'sensor': f'pressure_{sensor}', 'value': value, 'status': 'warning', 'range': '30-40 or 80-90 PSI'})
                elif model == 'cessna_172':
                    if value < 15 or value > 70:
                        critical_count += 1
                        problematic_sensors.append({'sensor': f'pressure_{sensor}', 'value': value, 'status': 'critical', 'range': '<15 or >70 PSI'})
                    elif value < 20 or value > 60:
                        warning_count += 1
                        problematic_sensors.append({'sensor': f'pressure_{sensor}', 'value': value, 'status': 'warning', 'range': '15-20 or 60-70 PSI'})
            elif 'hydraulic' in sensor:
                if value < 2500 or value > 3400:
                    critical_count += 1
                    problematic_sensors.append({'sensor': f'pressure_{sensor}', 'value': value, 'status': 'critical', 'range': '<2500 or >3400 PSI'})
                elif value < 2800 or value > 3200:
                    warning_count += 1
                    problematic_sensors.append({'sensor': f'pressure_{sensor}', 'value': value, 'status': 'warning', 'range': '2500-2800 or 3200-3400 PSI'})
            elif 'fuel' in sensor:
                if value < 0.2 or value > 10:
                    critical_count += 1
                    problematic_sensors.append({'sensor': f'pressure_{sensor}', 'value': value, 'status': 'critical', 'range': '<0.2 or >10 PSI'})
                elif value < 0.5 or value > 8:
                    warning_count += 1
                    problematic_sensors.append({'sensor': f'pressure_{sensor}', 'value': value, 'status': 'warning', 'range': '0.2-0.5 or 8-10 PSI'})
    
    # Check ROTOR SYSTEM for helicopter
    if 'rotor_system' in survey:
        for sensor, value in survey['rotor_system'].items():
            if 'rpm' in sensor:
                if value < 280 or value > 400:
                    critical_count += 1
                    problematic_sensors.append({'sensor': f'rotor_{sensor}', 'value': value, 'status': 'critical', 'range': '<280 or >400 RPM'})
                elif value < 300 or value > 380:
                    warning_count += 1
                    problematic_sensors.append({'sensor': f'rotor_{sensor}', 'value': value, 'status': 'warning', 'range': '280-300 or 380-400 RPM'})
            elif 'track' in sensor or 'balance' in sensor:
                if value > 0.1:
                    critical_count += 1
                    problematic_sensors.append({'sensor': f'rotor_{sensor}', 'value': value, 'status': 'critical', 'range': '>0.1 units'})
                elif value > 0.08:
                    warning_count += 1
                    problematic_sensors.append({'sensor': f'rotor_{sensor}', 'value': value, 'status': 'warning', 'range': '0.08-0.1 units'})
    
    # Determine overall condition based on ALL sensors
    if critical_count > 0:
        condition = 'critical'
        message = f'❌ CRITICAL: Aircraft is NOT safe to fly! {critical_count} critical issues detected'
        color = 'danger'
        confidence = 0.75
    elif warning_count >= 3:
        condition = 'critical'
        message = f'❌ CRITICAL: Multiple warnings ({warning_count}) - Do NOT fly!'
        color = 'danger'
        confidence = 0.80
    elif warning_count > 0:
        condition = 'warning'
        message = f'⚠️ WARNING: {warning_count} sensors indicate potential issues - Schedule maintenance'
        color = 'warning'
        confidence = 0.90
    else:
        condition = 'normal'
        message = '✅ SAFE: All sensors within normal range - Aircraft is safe to fly'
        color = 'success'
        confidence = 0.98
    
    # Store in session
    session['prediction_result'] = {
        'condition': condition,
        'message': message,
        'color': color,
        'confidence': confidence,
        'warning_count': warning_count,
        'critical_count': critical_count,
        'problematic_sensors': problematic_sensors
    }
    
    # Send email to management (not owner)
    try:
        management_email = flight_details.get('management_email', '')
        maintenance_email = flight_details.get('maintenance_email', '')
        
        # Send to management
        if management_email:
            def send_management_thread():
                with app.app_context():
                    try:
                        send_alert_email(
                            management_email, 
                            condition, 
                            warning_count, 
                            critical_count,
                            flight_details,
                            model
                        )
                    except Exception as e:
                        print(f"Management email error: {e}")
            
            thread = threading.Thread(target=send_management_thread)
            thread.daemon = True
            thread.start()
            print(f"✅ Alert sent to management: {management_email}")
        
        # Send to maintenance team (more detailed)
        if maintenance_email and (condition == 'warning' or condition == 'critical'):
            def send_maintenance_thread():
                with app.app_context():
                    try:
                        send_maintenance_alert(
                            maintenance_email,
                            condition,
                            warning_count,
                            critical_count,
                            problematic_sensors,  # Send problematic sensors
                            flight_details,
                            model
                        )
                    except Exception as e:
                        print(f"Maintenance email error: {e}")
            
            thread2 = threading.Thread(target=send_maintenance_thread)
            thread2.daemon = True
            thread2.start()
            print(f"✅ Detailed alert sent to maintenance: {maintenance_email}")
            
    except Exception as e:
        print(f"❌ Email setup failed: {e}")
    
    return redirect(url_for('prediction_result'))

# Helper function for management alerts (simple)
def send_alert_email(email, condition, warning_count, critical_count, flight_details, model):
    """Send simple alert to management"""
    import smtplib
    from email.mime.text import MIMEText
    
    status_emoji = '✅' if condition == 'normal' else '⚠️' if condition == 'warning' else '❌'
    
    html_content = f"""
    <html>
    <body>
        <h2>{status_emoji} Aircraft Safety Alert</h2>
        <p><strong>Flight:</strong> {flight_details.get('flight_number', 'N/A')}</p>
        <p><strong>Aircraft:</strong> {model.replace('_', ' ').title()}</p>
        <p><strong>Status:</strong> <span style="color:{'green' if condition=='normal' else 'orange' if condition=='warning' else 'red'}">{condition.upper()}</span></p>
        <p><strong>Warnings:</strong> {warning_count}</p>
        <p><strong>Critical Issues:</strong> {critical_count}</p>
        <p><a href="http://localhost:5000/prediction">View Full Report</a></p>
    </body>
    </html>
    """
    
    msg = MIMEText(html_content, 'html')
    msg['Subject'] = f"Aircraft Alert - Flight {flight_details.get('flight_number', 'N/A')} - {condition.upper()}"
    msg['From'] = app.config['MAIL_DEFAULT_SENDER']
    msg['To'] = email
    
    with smtplib.SMTP(app.config['MAIL_SERVER'], app.config['MAIL_PORT']) as server:
        if app.config['MAIL_USE_TLS']:
            server.starttls()
        server.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
        server.send_message(msg)

# Helper function for maintenance alerts (detailed)
def send_maintenance_alert(email, condition, warning_count, critical_count, problematic_sensors, flight_details, model):
    """Send detailed alert to maintenance team"""
    import smtplib
    from email.mime.text import MIMEText
    
    # Format problematic sensors
    sensor_html = ""
    for sensor in problematic_sensors:
        status_color = 'red' if sensor['status'] == 'critical' else 'orange'
        sensor_html += f"""
        <li style="color: {status_color}; font-weight: bold;">
            {sensor['sensor']}: {sensor['value']} - {sensor['status'].upper()} 
            (outside normal range {sensor['range']})
        </li>
        """
    
    html_content = f"""
    <html>
    <body>
        <h2>🔧 MAINTENANCE TEAM ALERT</h2>
        <h3>Flight: {flight_details.get('flight_number', 'N/A')}</h3>
        <p><strong>Aircraft:</strong> {model.replace('_', ' ').title()}</p>
        <p><strong>Status:</strong> <span style="color:{'green' if condition=='normal' else 'orange' if condition=='warning' else 'red'}">{condition.upper()}</span></p>
        <p><strong>Warnings:</strong> {warning_count}</p>
        <p><strong>Critical Issues:</strong> {critical_count}</p>
        
        <h3 style="color: red;">⚠️ PROBLEMATIC SENSORS:</h3>
        <ul>
            {sensor_html}
        </ul>
        
        <h3>Recommended Actions:</h3>
        <ul>
            {f'<li style="color: red; font-weight: bold;">IMMEDIATE inspection required for {critical_count} critical sensors</li>' if critical_count > 0 else ''}
            {f'<li style="color: orange;">Schedule maintenance for {warning_count} sensors showing warnings</li>' if warning_count > 0 else ''}
            {f'<li>No immediate action needed - routine maintenance only</li>' if condition == 'normal' else ''}
        </ul>
        
        <p><a href="http://localhost:5000/prediction">View Detailed Report</a></p>
    </body>
    </html>
    """
    
    msg = MIMEText(html_content, 'html')
    msg['Subject'] = f"🔧 MAINTENANCE ALERT - Flight {flight_details.get('flight_number', 'N/A')} - {critical_count} Critical, {warning_count} Warnings"
    msg['From'] = app.config['MAIL_DEFAULT_SENDER']
    msg['To'] = email
    
    with smtplib.SMTP(app.config['MAIL_SERVER'], app.config['MAIL_PORT']) as server:
        if app.config['MAIL_USE_TLS']:
            server.starttls()
        server.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
        server.send_message(msg)

# Prediction result
@app.route('/prediction')
def prediction_result():
    result = session.get('prediction_result', {})
    flight = session.get('flight_details', {})
    model = session.get('selected_model', 'Unknown')
    survey = session.get('survey_data', {})
    problematic_sensors = result.get('problematic_sensors', [])
    
    # Create a set of problematic sensor names for quick lookup
    problematic_names = {p['sensor'] for p in problematic_sensors}
    
    # Format sensor data for display with highlighting
    sensor_display = ""
    
    if 'vibration' in survey:
        sensor_display += "<h5>🔴 Vibration:</h5><ul>"
        for k, v in survey['vibration'].items():
            sensor_name = f'vibration_{k}'
            if sensor_name in problematic_names:
                # Find the status for this sensor
                status = next((p['status'] for p in problematic_sensors if p['sensor'] == sensor_name), 'warning')
                color = 'red' if status == 'critical' else 'orange'
                sensor_display += f'<li style="color: {color}; font-weight: bold;">{k}: {v} ⚠️ ({status.upper()} - outside range)</li>'
            else:
                sensor_display += f"<li>{k}: {v}</li>"
        sensor_display += "</ul>"
    
    if 'thermal' in survey:
        sensor_display += "<h5>🔥 Thermal:</h5><ul>"
        for k, v in survey['thermal'].items():
            sensor_name = f'thermal_{k}'
            if sensor_name in problematic_names:
                status = next((p['status'] for p in problematic_sensors if p['sensor'] == sensor_name), 'warning')
                color = 'red' if status == 'critical' else 'orange'
                sensor_display += f'<li style="color: {color}; font-weight: bold;">{k}: {v}°C ⚠️ ({status.upper()} - outside range)</li>'
            else:
                sensor_display += f"<li>{k}: {v}°C</li>"
        sensor_display += "</ul>"
    
    if 'acoustic' in survey:
        sensor_display += "<h5>🎤 Acoustic:</h5><ul>"
        for k, v in survey['acoustic'].items():
            sensor_name = f'acoustic_{k}'
            if sensor_name in problematic_names:
                status = next((p['status'] for p in problematic_sensors if p['sensor'] == sensor_name), 'warning')
                color = 'red' if status == 'critical' else 'orange'
                sensor_display += f'<li style="color: {color}; font-weight: bold;">{k}: {v}dB ⚠️ ({status.upper()} - outside range)</li>'
            else:
                sensor_display += f"<li>{k}: {v}dB</li>"
        sensor_display += "</ul>"
    
    if 'pressure' in survey:
        sensor_display += "<h5>⏲️ Pressure:</h5><ul>"
        for k, v in survey['pressure'].items():
            sensor_name = f'pressure_{k}'
            if sensor_name in problematic_names:
                status = next((p['status'] for p in problematic_sensors if p['sensor'] == sensor_name), 'warning')
                color = 'red' if status == 'critical' else 'orange'
                sensor_display += f'<li style="color: {color}; font-weight: bold;">{k}: {v}PSI ⚠️ ({status.upper()} - outside range)</li>'
            else:
                sensor_display += f"<li>{k}: {v}PSI</li>"
        sensor_display += "</ul>"
    
    if 'rotor_system' in survey:
        sensor_display += "<h5>🚁 Rotor System:</h5><ul>"
        for k, v in survey['rotor_system'].items():
            sensor_name = f'rotor_{k}'
            if sensor_name in problematic_names:
                status = next((p['status'] for p in problematic_sensors if p['sensor'] == sensor_name), 'warning')
                color = 'red' if status == 'critical' else 'orange'
                sensor_display += f'<li style="color: {color}; font-weight: bold;">{k}: {v} ⚠️ ({status.upper()} - outside range)</li>'
            else:
                sensor_display += f"<li>{k}: {v}</li>"
        sensor_display += "</ul>"
    
    # Create a summary of issues
    issues_summary = ""
    if problematic_sensors:
        issues_summary = "<div class='alert alert-danger'><h5>⚠️ ISSUES DETECTED:</h5><ul>"
        for p in problematic_sensors:
            color = 'red' if p['status'] == 'critical' else 'orange'
            issues_summary += f'<li style="color: {color};">{p["sensor"]}: {p["value"]} - {p["status"].upper()} (outside {p["range"]})</li>'
        issues_summary += "</ul></div>"
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prediction Result</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>
        <div class="container mt-5">
            <div class="row justify-content-center">
                <div class="col-md-8">
                    <div class="card shadow border-{result.get('color', 'primary')}">
                        <div class="card-header bg-{result.get('color', 'primary')} text-white text-center">
                            <h2 class="mb-0">✈️ Multi-Sensor Analysis Result</h2>
                        </div>
                        <div class="card-body">
                            <div class="text-center mb-4">
                                <div style="font-size: 64px;">
                                    {'✅' if result.get('condition') == 'normal' else '⚠️' if result.get('condition') == 'warning' else '❌'}
                                </div>
                                <h3 class="mb-3">{result.get('message', 'No result')}</h3>
                                <p class="lead">Aircraft: {model.replace('_', ' ').title()}</p>
                                <p>Flight: {flight.get('flight_number', 'N/A')}</p>
                                <p>Management: {flight.get('management_name', 'N/A')}</p>
                            </div>
                            
                            {issues_summary}
                            
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="card mb-3">
                                        <div class="card-header bg-secondary text-white">
                                            <h5 class="mb-0">📊 Summary</h5>
                                        </div>
                                        <div class="card-body">
                                            <p><strong>Warnings:</strong> {result.get('warning_count', 0)}</p>
                                            <p><strong>Critical Issues:</strong> {result.get('critical_count', 0)}</p>
                                            <p><strong>Confidence:</strong> {float(result.get('confidence', 0))*100:.1f}%</p>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="card mb-3">
                                        <div class="card-header bg-secondary text-white">
                                            <h5 class="mb-0">📋 Recommendations</h5>
                                        </div>
                                        <div class="card-body">
                                            <ul>
                                                {f'<li class="text-danger fw-bold">IMMEDIATE ACTION required for {result.get("critical_count", 0)} critical issues</li>' if result.get('critical_count', 0) > 0 else ''}
                                                {f'<li class="text-warning fw-bold">Schedule maintenance for {result.get("warning_count", 0)} warnings</li>' if result.get('warning_count', 0) > 0 else ''}
                                                {f'<li class="text-success">No action needed - all systems normal</li>' if result.get('condition') == 'normal' else ''}
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="card">
                                <div class="card-header bg-info text-white">
                                    <h5 class="mb-0">🔍 Detailed Sensor Readings</h5>
                                </div>
                                <div class="card-body">
                                    {sensor_display}
                                </div>
                            </div>
                            
                            <div class="text-center mt-4">
                                <a href="/" class="btn btn-primary">New Prediction</a>
                                <a href="/model-selection" class="btn btn-secondary">Select Different Aircraft</a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    '''

# API Route to fetch flight data
@app.route('/api/get-flight-data', methods=['POST'])
def get_flight_data():
    try:
        data = json.loads(request.data)
        flight_number = data.get('flight_number', '').strip().upper()
        
        # Get current selected model from session
        selected_model = session.get('selected_model', '')
        
        # Map to model type
        if 'boeing' in selected_model:
            model_type = 'boeing'
        elif 'cessna' in selected_model:
            model_type = 'cessna'
        elif 'airbus' in selected_model:
            model_type = 'airbus'
        else:
            model_type = None
        
        if not flight_number:
            return jsonify({'success': False, 'error': 'No flight number provided'})
        
        # Query with model type filter
        if model_type:
            flight = FlightSurvey.query.filter_by(
                flight_number=flight_number,
                aircraft_model_type=model_type
            ).first()
        else:
            flight = FlightSurvey.query.filter_by(flight_number=flight_number).first()
        
        if flight:
            return jsonify({
                'success': True,
                'data': {
                    'flight_number': flight.flight_number,
                    'registration_number': flight.registration_number,
                    'management_contact': flight.management_contact,
                    'management_email': flight.management_email,
                    'management_team_id': flight.management_team_id,
                    'management_team': flight.management_team,
                    'maintenance_manager': flight.maintenance_manager,
                    'maintenance_email': flight.maintenance_email,
                    'maintenance_team_id': flight.maintenance_team_id,
                    'maintenance_team': flight.maintenance_team,
                    'total_flight_hours': flight.total_flight_hours
                }
            })
        else:
            return jsonify({'success': False, 'data': None, 'message': 'No flight found for this model'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("✅ AIRCRAFT PREDICTIVE MAINTENANCE SYSTEM - COMPLETE VERSION")
    print("="*60)
    print("🚀 Server running at: http://localhost:5000")
    print("📱 Open this URL in your browser")
    print("⏸️  Press CTRL+C to stop")
    print("="*60 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000)