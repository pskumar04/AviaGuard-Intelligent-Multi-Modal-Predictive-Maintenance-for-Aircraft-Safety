"""
Web Forms for Aircraft Predictive Maintenance Website
"""

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, SelectField
from wtforms import TextAreaField, FloatField, IntegerField, DateField, EmailField
from wtforms.validators import DataRequired, Email, EqualTo, Length, NumberRange, Optional
from wtforms import validators

class LoginForm(FlaskForm):
    """User login form"""
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')

class RegistrationForm(FlaskForm):
    """User registration form"""
    username = StringField('Username', validators=[DataRequired(), Length(min=2, max=64)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', 
                                     validators=[DataRequired(), EqualTo('password')])
    role = SelectField('Role', choices=[
        ('operator', 'Flight Operator'),
        ('maintenance', 'Maintenance Technician'),
        ('viewer', 'Viewer')
    ])
    submit = SubmitField('Register')

class FlightDetailsForm(FlaskForm):
    """Flight owner and management details form"""
    # Flight information
    flight_number = StringField('Flight Number', validators=[DataRequired(), Length(max=20)])
    registration_number = StringField('Registration Number', validators=[DataRequired(), Length(max=20)])
    total_flight_hours = IntegerField('Total Flight Hours', validators=[DataRequired(), NumberRange(min=0)])
    last_maintenance_date = DateField('Last Maintenance Date', validators=[Optional()])
    
    # Owner information
    owner_name = StringField('Flight Owner Name', validators=[DataRequired(), Length(max=100)])
    owner_email = EmailField('Owner Email', validators=[DataRequired(), Email()])
    
    # Crew information
    pilot_name = StringField('Pilot Name', validators=[DataRequired(), Length(max=100)])
    co_pilot_name = StringField('Co-pilot Name', validators=[Optional(), Length(max=100)])
    
    # Management team
    management_team = StringField('Management Team', validators=[DataRequired(), Length(max=200)])
    management_email = EmailField('Management Email', validators=[DataRequired(), Email()])
    
    # Maintenance team
    maintenance_manager = StringField('Maintenance Manager', validators=[DataRequired(), Length(max=100)])
    maintenance_email = EmailField('Maintenance Email', validators=[DataRequired(), Email()])
    
    submit = SubmitField('Continue to Survey')

class FlightRouteForm(FlaskForm):
    """Flight route selection form"""
    route = SelectField('Select Flight Route', choices=[
        ('HYD-DEL', 'Hyderabad (HYD) → New Delhi (DEL) - 2h 15m'),
        ('HYD-BOM', 'Hyderabad (HYD) → Mumbai (BOM) - 1h 20m'),
        ('DEL-BOM', 'New Delhi (DEL) → Mumbai (BOM) - 2h 00m'),
        ('DEL-BLR', 'New Delhi (DEL) → Bengaluru (BLR) - 2h 45m'),
        ('BOM-CCU', 'Mumbai (BOM) → Kolkata (CCU) - 2h 30m'),
        ('HYD-CHE', 'Hyderabad (HYD) → Chennai (MAA) - 1h 20m')
    ], validators=[DataRequired()])
    submit = SubmitField('Confirm Flight')

# Boeing 737 Survey Form
class BoeingSurveyForm(FlaskForm):
    """Boeing 737 survey questions based on JSON configuration"""
    
    # Engine parameters
    n1_rpm = FloatField('N1 RPM (%)', 
                        validators=[DataRequired(), NumberRange(min=0, max=120)],
                        description="Normal range: 95-101%")
    n2_rpm = FloatField('N2 RPM (%)',
                        validators=[DataRequired(), NumberRange(min=0, max=120)],
                        description="Normal range: 98-102%")
    egt = FloatField('EGT (°C)',
                     validators=[DataRequired(), NumberRange(min=0, max=1000)],
                     description="Normal range: 400-650°C")
    oil_temperature = FloatField('Oil Temperature (°C)',
                                 validators=[DataRequired(), NumberRange(min=0, max=200)],
                                 description="Normal range: 60-110°C")
    oil_pressure = FloatField('Oil Pressure (PSI)',
                              validators=[DataRequired(), NumberRange(min=0, max=150)],
                              description="Normal range: 40-80 PSI")
    vibration_level = FloatField('Vibration Level (ips)',
                                 validators=[DataRequired(), NumberRange(min=0, max=2)],
                                 description="Normal range: 0-0.5 ips")
    
    # Structural parameters
    wing_vibration = FloatField('Wing Vibration (g)',
                                validators=[DataRequired(), NumberRange(min=0, max=1)],
                                description="Normal range: 0-0.3g")
    fuselage_strain = FloatField('Fuselage Strain (με)',
                                 validators=[DataRequired(), NumberRange(min=0, max=1000)],
                                 description="Normal range: 0-500 με")
    
    # Systems parameters
    hydraulic_pressure = FloatField('Hydraulic Pressure (PSI)',
                                    validators=[DataRequired(), NumberRange(min=0, max=4000)],
                                    description="Normal range: 2800-3200 PSI")
    cabin_pressure_altitude = FloatField('Cabin Pressure Altitude (ft)',
                                         validators=[DataRequired(), NumberRange(min=0, max=15000)],
                                         description="Normal range: 0-8000 ft")
    bleed_air_temperature = FloatField('Bleed Air Temperature (°C)',
                                       validators=[DataRequired(), NumberRange(min=0, max=300)],
                                       description="Normal range: 150-250°C")
    
    # Flight controls
    aileron_position = FloatField('Aileron Position (deg)',
                                  validators=[DataRequired(), NumberRange(min=-30, max=30)],
                                  description="Normal range: -25 to 25 deg")
    elevator_position = FloatField('Elevator Position (deg)',
                                   validators=[DataRequired(), NumberRange(min=-25, max=25)],
                                   description="Normal range: -20 to 20 deg")
    
    submit = SubmitField('Predict Flight Safety')

# Cessna 172 Survey Form
class CessnaSurveyForm(FlaskForm):
    """Cessna 172 survey questions"""
    
    rpm = FloatField('Engine RPM',
                     validators=[DataRequired(), NumberRange(min=0, max=3000)],
                     description="Normal range: 2000-2700 RPM")
    manifold_pressure = FloatField('Manifold Pressure (inHg)',
                                   validators=[DataRequired(), NumberRange(min=0, max=40)],
                                   description="Normal range: 15-30 inHg")
    oil_temperature = FloatField('Oil Temperature (°F)',
                                 validators=[DataRequired(), NumberRange(min=0, max=300)],
                                 description="Normal range: 100-245°F")
    oil_pressure = FloatField('Oil Pressure (PSI)',
                              validators=[DataRequired(), NumberRange(min=0, max=100)],
                              description="Normal range: 20-60 PSI")
    fuel_pressure = FloatField('Fuel Pressure (PSI)',
                               validators=[DataRequired(), NumberRange(min=0, max=10)],
                               description="Normal range: 0.5-8 PSI")
    cylinder_head_temp = FloatField('Cylinder Head Temperature (°F)',
                                    validators=[DataRequired(), NumberRange(min=0, max=600)],
                                    description="Normal range: 200-500°F")
    
    submit = SubmitField('Predict Flight Safety')

# Airbus H125 Survey Form
class AirbusSurveyForm(FlaskForm):
    """Airbus H125 survey questions"""
    
    # Rotor system
    rotor_rpm = FloatField('Rotor RPM',
                           validators=[DataRequired(), NumberRange(min=0, max=500)],
                           description="Normal range: 300-400 RPM")
    vibration_level = FloatField('Vibration Level (ips)',
                                 validators=[DataRequired(), NumberRange(min=0, max=1)],
                                 description="Normal range: 0-0.5 ips")
    track_and_balance = FloatField('Track and Balance (units)',
                                   validators=[DataRequired(), NumberRange(min=0, max=0.5)],
                                   description="Normal range: 0-0.1 units")
    
    # Engine
    torque = FloatField('Torque (%)',
                        validators=[DataRequired(), NumberRange(min=0, max=100)],
                        description="Normal range: 20-80%")
    ng = FloatField('NG (%)',
                    validators=[DataRequired(), NumberRange(min=0, max=120)],
                    description="Normal range: 98-102%")
    t4 = FloatField('T4 Temperature (°C)',
                    validators=[DataRequired(), NumberRange(min=0, max=1000)],
                    description="Normal range: 500-850°C")
    
    submit = SubmitField('Predict Flight Safety')