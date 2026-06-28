"""
Database Models for Aircraft Predictive Maintenance Website
"""

from website import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timezone

class User(UserMixin, db.Model):
    """User model for authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    role = db.Column(db.String(20), default='user')  # admin, operator, viewer
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    surveys = db.relationship('FlightSurvey', backref='creator', lazy='dynamic')
    predictions = db.relationship('PredictionResult', backref='user', lazy='dynamic')
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password"""
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class FlightSurvey(db.Model):
    """Flight survey data model"""
    __tablename__ = 'flight_surveys'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    aircraft_model = db.Column(db.String(50), nullable=False)
    flight_details = db.Column(db.JSON, nullable=False)
    survey_data = db.Column(db.JSON, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    total_flight_hours = db.Column(db.Integer, default=0)
    
    # NEW FIELDS FOR FLIGHT DETAILS STORAGE
    flight_number = db.Column(db.String(20), unique=True, nullable=False)
    registration_number = db.Column(db.String(20))
    management_team = db.Column(db.String(100))
    management_team_id = db.Column(db.String(50))
    management_contact = db.Column(db.String(100))
    management_email = db.Column(db.String(120))
    maintenance_team = db.Column(db.String(100))
    maintenance_team_id = db.Column(db.String(50))
    maintenance_manager = db.Column(db.String(100))
    maintenance_email = db.Column(db.String(120))
    total_flight_hours = db.Column(db.Integer, default=0)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    prediction = db.relationship('PredictionResult', backref='survey', uselist=False)
    
    def __repr__(self):
        return f'<FlightSurvey {self.aircraft_model} - {self.timestamp}>'

class PredictionResult(db.Model):
    """Prediction result model"""
    __tablename__ = 'prediction_results'
    
    id = db.Column(db.Integer, primary_key=True)
    survey_id = db.Column(db.Integer, db.ForeignKey('flight_surveys.id'))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    
    # Aircraft information
    model_type = db.Column(db.String(50), nullable=False)
    flight_number = db.Column(db.String(20))
    registration_number = db.Column(db.String(20))
    
    # Prediction data
    survey_data = db.Column(db.JSON, nullable=False)
    prediction_result = db.Column(db.JSON, nullable=False)
    fault_detections = db.Column(db.JSON, nullable=False)
    flight_condition = db.Column(db.String(20), nullable=False)  # normal, warning, critical
    confidence_score = db.Column(db.Float, default=0.0)
    
    # XAI explanations
    shap_explanations = db.Column(db.JSON)
    lime_explanations = db.Column(db.JSON)
    
    # Contact information
    owner_email = db.Column(db.String(120))
    management_email = db.Column(db.String(120))
    maintenance_email = db.Column(db.String(120))
    
    # Timestamps
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    email_sent = db.Column(db.Boolean, default=False)
    email_sent_at = db.Column(db.DateTime)
    
    def __repr__(self):
        return f'<PredictionResult {self.model_type} - {self.flight_condition}>'
    
    def get_condition_display(self):
        """Get display text for condition"""
        conditions = {
            'normal': '✅ Working Good',
            'warning': '⚠️ Working with Issues',
            'critical': '❌ Not Working'
        }
        return conditions.get(self.flight_condition, 'Unknown')
    
    def get_condition_color(self):
        """Get Bootstrap color for condition"""
        colors = {
            'normal': 'success',
            'warning': 'warning',
            'critical': 'danger'
        }
        return colors.get(self.flight_condition, 'secondary')
    
